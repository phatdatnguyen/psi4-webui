"""Psi4 execution, in a child process.

This module is the **only** place ``psi4`` is imported. The web server never imports it;
the Calculation tab launches it as

    python -m psi4_webui.runner run <job.json>
    python -m psi4_webui.runner cubeprop <cube.json>

and waits on the process. Running out-of-process buys three things that an in-process
thread cannot provide:

* **Cancellation.** Psi4's driver is a long-running C++ call with no interrupt hook, so
  the only reliable Stop is to kill the process tree (see ``on_stop_calculation``).
* **Crash isolation.** Psi4 aborts the process outright on some failures (out of memory,
  an internal assertion). In-process, that would take the whole web server with it.
* **Clean state.** Psi4's options, active molecule and scratch are process-global, so
  successive runs in one process leak settings into each other. A fresh child cannot.

The child communicates only through files in the working directory: the job spec going
in, and ``<base>.log`` (native Psi4 output), ``<base>.result.json`` (structured results)
plus optional ``.npy``/``.xyz`` artifacts coming back. Exit code 0 means the result file
holds a completed calculation; non-zero means it holds ``status: "failed"`` with an error
message. A result file is written in **both** cases, so the Result tab can always explain
what happened rather than showing an empty dropdown.

``import psi4`` deliberately happens inside the functions rather than at module scope, so
the module (and its file-naming helpers) can be imported and unit-tested in an
environment where Psi4 is not installed.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from typing import Any

from .utils import (
    EMISSION,
    FREQUENCY,
    GEOMETRY_OPTIMIZATION,
    JOB_SUFFIX,
    RESULT_SUFFIX,
    TDDFT,
    read_json,
    write_json,
)


def _solvation_block(solvent: str) -> str:
    """PCMSolver input block for an IEF-PCM calculation in ``solvent``.

    Psi4 has no SMD model, so the Calculation tab offers PCM only. The solvent name is
    passed straight through to PCMSolver, which knows a fixed list of named solvents and
    raises if it does not recognise one.
    """
    return f"""
        units = angstrom
        medium {{
            solvertype = iefpcm
            solvent = {solvent}
        }}
        cavity {{
            type = gepol
            area = 0.3
            radiiset = bondi
            scaling = true
        }}
    """


def job_base_name(job_path: str) -> str:
    """Strip the job-spec suffix from ``job_path`` to get the artifact base name.

    ``os.path.splitext`` alone would turn ``opt.job.json`` into ``opt.job``, so every
    artifact of that run would be named ``opt.job.log``, ``opt.job.npy`` and so on.
    """
    name = os.path.basename(job_path)
    for suffix in (JOB_SUFFIX, ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def _atom_symbols(molecule) -> list[str]:
    """Element symbols of ``molecule`` in Psi4's atom order."""
    return [molecule.symbol(i) for i in range(molecule.natom())]


def _geometry_angstrom(molecule, bohr2angstroms: float) -> list[list[float]]:
    """Current geometry of ``molecule`` as a plain nested list, in angstrom."""
    import numpy as np

    coords = np.asarray(molecule.geometry()) * bohr2angstroms
    return [[float(x) for x in row] for row in coords]


def _write_xyz(file_path: str, symbols, coords, charge: int, multiplicity: int) -> None:
    """Write a single geometry in this app's XYZ convention.

    The first line is ``"<charge> <multiplicity>"`` rather than an atom count, matching
    :func:`psi4_webui.utils.conformer_to_xyz_file`, so an optimized geometry can be fed
    straight back into a follow-up calculation with its charge and spin state intact.
    """
    lines = [f"{int(charge)} {int(multiplicity)}"]
    for symbol, (x, y, z) in zip(symbols, coords):
        lines.append(f"{symbol} {float(x):>14.8f} {float(y):>14.8f} {float(z):>14.8f}")
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_trajectory_xyz(file_path: str, symbols, frames) -> None:
    """Write an optimization trajectory as a standard multi-frame XYZ file.

    Standard format (atom count + comment line per frame) rather than this app's own
    convention, because the point of a trajectory file is to open it in an external
    viewer that can animate the frames. The in-app reader understands it too and shows
    the final frame.
    """
    lines = []
    for step, coords in enumerate(frames):
        lines.append(str(len(symbols)))
        lines.append(f"Step {step}")
        for symbol, (x, y, z) in zip(symbols, coords):
            lines.append(f"{symbol} {float(x):>14.8f} {float(y):>14.8f} {float(z):>14.8f}")
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _dipole_debye(psi4, wfn):
    """Total dipole moment in Debye, as ``(magnitude, [x, y, z])``.

    ``CURRENT DIPOLE`` tracks whichever method actually ran (SCF, MP2, CC...), so it is
    preferred over the method-specific variables. Psi4 stores it in atomic units.
    """
    import numpy as np

    for source in (wfn, psi4.core):
        try:
            vector = np.asarray(source.variable("CURRENT DIPOLE"), dtype=float).ravel()
        except Exception:
            continue
        if vector.size == 3:
            vector = vector * psi4.constants.dipmom_au2debye
            return float(np.linalg.norm(vector)), [float(v) for v in vector]
    return None, None


def _orbital_energies(wfn) -> dict[str, Any]:
    """Alpha MO energies (hartree) plus the occupation counts that locate the HOMO."""
    import numpy as np

    energies = np.asarray(wfn.epsilon_a_subset("AO", "ALL").to_array(), dtype=float).ravel()
    n_alpha = int(wfn.nalpha())
    return {
        "mo_energies_hartree": [float(e) for e in energies],
        # 0-based index of the highest occupied alpha orbital. -1 when there are no
        # electrons at all, which the Result tab reads as "no HOMO to label".
        "homo_index": n_alpha - 1,
        "n_alpha": n_alpha,
        "n_beta": int(wfn.nbeta()),
    }


def _frequency_results(psi4, wfn, spec: dict[str, Any]) -> dict[str, Any]:
    """Vibrational frequencies, IR intensities and thermochemistry.

    The subtle part is the alignment of the two arrays. ``wfn.frequencies()`` returns only
    the vibrational modes, but ``frequency_analysis['IR_intensity']`` is indexed over all
    3N modes, translations and rotations included. The ``TRV`` mask says which is which,
    and selecting ``'V'`` from it is what makes intensity *i* belong to frequency *i*.
    Zipping the two raw arrays -- as the previous version of this app did -- silently
    pairs each frequency with some other mode's intensity.

    None of ``frequency_analysis`` survives ``wfn.to_file()``, so it has to be read out
    here or it is lost when the child exits.
    """
    import numpy as np

    frequencies = np.asarray(wfn.frequencies().to_array(), dtype=float).ravel()
    analysis = wfn.frequency_analysis
    trv = np.asarray(analysis["TRV"].data)
    intensities = np.asarray(analysis["IR_intensity"].data, dtype=float).ravel()
    vibrational = intensities[trv == "V"]
    if vibrational.size != frequencies.size:
        # Should not happen, but a length mismatch would mis-assign every peak, so
        # normalise explicitly rather than letting zip() quietly drop the tail.
        vibrational = np.resize(vibrational, frequencies.size)

    temperature = float(spec.get("temperature", 298.15))
    thermo: dict[str, Any] = {
        "temperature_k": temperature,
        "pressure_pa": float(spec.get("pressure", 101325.0)),
    }
    for label, key in (
        ("ZPVE", "zpve_hartree"),
        ("THERMAL ENERGY", "thermal_energy_hartree"),
        ("ENTHALPY", "enthalpy_hartree"),
        ("GIBBS FREE ENERGY", "gibbs_hartree"),
    ):
        try:
            thermo[key] = float(psi4.core.variable(label))
        except Exception:
            thermo[key] = None

    enthalpy, gibbs = thermo.get("enthalpy_hartree"), thermo.get("gibbs_hartree")
    if enthalpy is not None and gibbs is not None and temperature > 0:
        # S = (H - G) / T, converted from hartree/K to cal/(mol K).
        thermo["entropy_cal_mol_k"] = (
            (enthalpy - gibbs) / temperature * psi4.constants.hartree2kcalmol * 1000.0
        )
    else:
        thermo["entropy_cal_mol_k"] = None

    return {
        "frequency": {
            "frequencies_cm1": [float(f) for f in frequencies],
            "ir_intensities_km_mol": [float(i) for i in vibrational],
            # Imaginary modes come back negative; a true minimum has none, a transition
            # state exactly one. Surfaced so the Result tab can warn, rather than leaving
            # the user to notice a minus sign in the table.
            "n_imaginary": int(np.sum(frequencies < 0)),
        },
        "thermochemistry": thermo,
    }


def _excitation_results(psi4, wfn, spec: dict[str, Any]) -> dict[str, Any]:
    """Run TDSCF on a converged SCF wavefunction and tabulate the excited states.

    Only the scalar spectroscopic observables are kept: the raw return also carries the
    full eigenvectors, which are large, not JSON-serializable, and of no use to the UI.
    """
    from psi4.driver.procrouting.response.scf_response import tdscf_excitations

    states = tdscf_excitations(
        wfn, states=int(spec.get("n_states", 10)), tda=bool(spec.get("tda", False))
    )

    excitations = []
    for index, state in enumerate(states, start=1):
        energy_hartree = float(state["EXCITATION ENERGY"])
        wavenumber = energy_hartree * psi4.constants.hartree2wavenumbers
        excitations.append(
            {
                "index": index,
                "energy_hartree": energy_hartree,
                "energy_ev": energy_hartree * psi4.constants.hartree2ev,
                "wavenumber_cm1": wavenumber,
                # 1e7 nm per cm; a non-positive root would be unphysical, so guard the divide.
                "wavelength_nm": (1.0e7 / wavenumber) if wavenumber > 0 else None,
                "oscillator_strength": float(state["OSCILLATOR STRENGTH (LEN)"]),
                "rotatory_strength": float(state["ROTATORY STRENGTH (LEN)"]),
                "spin": str(state.get("SPIN", "")),
                "symmetry": str(state.get("SYMMETRY", "")),
            }
        )
    return {"excitations": excitations}


def _emission_results(psi4, spec: dict[str, Any], molecule, excited_energy: float) -> dict[str, Any]:
    """Vertical emission from the relaxed excited state, at that relaxed geometry.

    Called *after* the excited state has been optimized, so ``molecule`` already carries
    the relaxed geometry and ``excited_energy`` is its EOM-CCSD energy. What remains is
    the ground state at that same geometry: the gap between the two is the vertical
    emission, and the difference between it and the vertical *absorption* at the ground
    geometry is the Stokes shift.

    ``ROOTS_PER_IRREP`` is cleared first, otherwise the ground-state CCSD call would try
    to solve for excited roots again.
    """
    psi4.set_options({"ROOTS_PER_IRREP": [0]})
    ground_energy = psi4.energy("ccsd", molecule=molecule)

    gap_hartree = float(excited_energy) - float(ground_energy)
    wavenumber = gap_hartree * psi4.constants.hartree2wavenumbers
    return {
        "emission": {
            "root": int(spec.get("root", 1)),
            "excited_energy_hartree": float(excited_energy),
            "ground_energy_at_excited_geometry_hartree": float(ground_energy),
            "emission_energy_hartree": gap_hartree,
            "emission_energy_ev": gap_hartree * psi4.constants.hartree2ev,
            "wavenumber_cm1": wavenumber,
            "wavelength_nm": (1.0e7 / wavenumber) if wavenumber > 0 else None,
        }
    }


def _configure(psi4, spec: dict[str, Any], scratch_dir: str) -> None:
    """Apply the job spec's resource and method settings to the child's Psi4 session."""
    os.makedirs(scratch_dir, exist_ok=True)
    # Keeps psi.*.clean / timer.dat inside the working directory instead of wherever the
    # server happens to have been started from.
    psi4.core.IOManager.shared_object().set_default_path(scratch_dir)

    psi4.set_memory(int(float(spec["memory_gb"]) * 1024 ** 3))
    psi4.set_num_threads(int(spec["n_threads"]))

    options: dict[str, Any] = {
        "BASIS": spec["basis_set"],
        "REFERENCE": spec["reference"],
        # DF is dramatically faster and is Psi4's default for DFT; the conventional PK
        # algorithm is used for HF and the correlated methods, matching the old app and
        # avoiding density-fitting error in small reference calculations.
        "SCF_TYPE": "DF" if spec["method_type"] == "DFT" else "PK",
    }
    if spec["calculation_type"] in (GEOMETRY_OPTIMIZATION, FREQUENCY, EMISSION):
        options["G_CONVERGENCE"] = spec.get("g_convergence", "QCHEM")
    if spec["calculation_type"] == EMISSION:
        # Which excited state to follow. The geometry block forces C1 (see
        # psi4_geometry_string), so there is a single irrep and the list has one entry.
        options["ROOTS_PER_IRREP"] = [int(spec.get("root", 1))]
    if spec["calculation_type"] == FREQUENCY:
        options["T"] = float(spec.get("temperature", 298.15))
        options["P"] = float(spec.get("pressure", 101325.0))
    if spec["calculation_type"] == TDDFT:
        # TDSCF reuses the converged SCF's integrals; without this Psi4 releases the JK
        # object at the end of the SCF and tdscf_excitations has nothing to work from.
        options["SAVE_JK"] = True
    if spec.get("use_solvation"):
        options["PCM"] = True
        options["PCM_SCF_TYPE"] = "TOTAL"

    psi4.set_options(options)
    if spec.get("use_solvation"):
        psi4.pcm_helper(_solvation_block(spec.get("solvent", "Water")))


def _save_wavefunction(wfn, path: str) -> str | None:
    """Serialize ``wfn`` for later cubeprop; return the file name, or ``None`` on refusal.

    ``to_file`` rejects anonymous/custom basis sets. That is not a reason to fail a
    calculation that has otherwise succeeded, so the failure is swallowed and the Result
    tab simply hides the orbital viewer for this job.
    """
    try:
        wfn.to_file(path)
        return os.path.basename(path)
    except Exception as exc:
        print(f"Could not save the wavefunction ({exc}); orbital visualization "
              f"will be unavailable for this calculation.", file=sys.stderr)
        return None


def run_job(job_path: str) -> int:
    """Execute the job described by ``job_path``; return the process exit code.

    Always writes ``<base>.result.json``, whether the calculation succeeded or failed, so
    the Result tab has something to explain.
    """
    spec = read_json(job_path)
    working_directory = os.path.dirname(os.path.abspath(job_path))
    base_name = job_base_name(job_path)

    log_path = os.path.join(working_directory, base_name + ".log")
    result_path = os.path.join(working_directory, base_name + RESULT_SUFFIX)

    result: dict[str, Any] = {
        "version": 1,
        "status": "failed",
        "error": None,
        "calculation_type": spec.get("calculation_type"),
        "method_type": spec.get("method_type"),
        "method": spec.get("method"),
        "functional": spec.get("functional"),
        "basis_set": spec.get("basis_set"),
        "reference": spec.get("reference"),
        "charge": spec.get("charge"),
        "multiplicity": spec.get("multiplicity"),
        "structure_file": spec.get("structure_file"),
        "solvent": spec.get("solvent") if spec.get("use_solvation") else None,
        "log_file": os.path.basename(log_path),
    }

    started = time.time()
    psi4 = None
    try:
        import psi4 as _psi4

        psi4 = _psi4
        psi4.core.set_output_file(log_path, False)
        _configure(psi4, spec, os.path.join(working_directory, "scratch"))

        molecule = psi4.geometry(spec["geometry"])
        method = spec["method"]
        calculation_type = spec["calculation_type"]
        geom_maxiter = int(spec.get("geom_maxiter", 50))

        if calculation_type == GEOMETRY_OPTIMIZATION:
            energy, wfn, history = psi4.optimize(
                method, molecule=molecule, optking__geom_maxiter=geom_maxiter,
                return_wfn=True, return_history=True,
            )
            # optking works in atomic units, so every trajectory frame is in Bohr. The
            # previous version of this app wrote them out unconverted, making every
            # optimization structure it produced 1.89x too large.
            frames = [
                [[float(v) * psi4.constants.bohr2angstroms for v in atom] for atom in step]
                for step in history["coordinates"]
            ]
            result["optimization"] = {
                "energies_hartree": [float(e) for e in history["energy"]],
                "n_steps": len(history["energy"]),
            }
            trajectory_path = os.path.join(working_directory, base_name + "_trajectory.xyz")
            _write_trajectory_xyz(trajectory_path, _atom_symbols(molecule), frames)
            result["trajectory_xyz_file"] = os.path.basename(trajectory_path)
        elif calculation_type == FREQUENCY:
            # A frequency analysis is only meaningful at a stationary point, so the
            # geometry is optimized first and the Hessian taken there. Taking it at an
            # arbitrary input geometry is what produces spurious imaginary modes and
            # renders the thermochemistry meaningless.
            psi4.optimize(method, molecule=molecule, optking__geom_maxiter=geom_maxiter)
            energy, wfn = psi4.frequency(method, molecule=molecule, return_wfn=True)
            result.update(_frequency_results(psi4, wfn, spec))
        elif calculation_type == TDDFT:
            energy, wfn = psi4.energy(method, molecule=molecule, return_wfn=True)
            result.update(_excitation_results(psi4, wfn, spec))
        elif calculation_type == EMISSION:
            # Optimize the *excited* state, then read the emission off that geometry.
            # EOM-CCSD is the only method in Psi4 with an excited-state gradient, so this
            # branch ignores the method choice entirely.
            energy, wfn = psi4.optimize("eom-ccsd", molecule=molecule,
                                        optking__geom_maxiter=geom_maxiter, return_wfn=True)
            result.update(_emission_results(psi4, spec, molecule, energy))
        else:  # SINGLE_POINT
            energy, wfn = psi4.energy(method, molecule=molecule, return_wfn=True)

        symbols = _atom_symbols(molecule)
        result["symbols"] = symbols
        result["geometry_angstrom"] = _geometry_angstrom(molecule, psi4.constants.bohr2angstroms)
        result["energy_hartree"] = float(energy)
        magnitude, vector = _dipole_debye(psi4, wfn)
        result["dipole_debye"] = magnitude
        result["dipole_vector_debye"] = vector
        result.update(_orbital_energies(wfn))

        if calculation_type in (GEOMETRY_OPTIMIZATION, FREQUENCY, EMISSION):
            optimized_path = os.path.join(working_directory, base_name + "_optimized.xyz")
            # Taken from the molecule Psi4 actually built rather than from the spec: the
            # geometry block carries its own charge/multiplicity header, and that header
            # is what the calculation ran with. Trusting the spec's separate fields would
            # mislabel the saved structure if the two ever disagreed.
            _write_xyz(optimized_path, symbols, result["geometry_angstrom"],
                       int(molecule.molecular_charge()), int(molecule.multiplicity()))
            result["optimized_xyz_file"] = os.path.basename(optimized_path)

        if spec.get("save_wavefunction", True):
            # Serialized so the Result tab can render orbitals and densities on demand
            # (see run_cubeprop) without re-running the calculation.
            result["wavefunction_file"] = _save_wavefunction(
                wfn, os.path.join(working_directory, base_name + ".npy")
            )

        result["status"] = "completed"
        exit_code = 0
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        print(result["traceback"], file=sys.stderr)
        exit_code = 1
    finally:
        if psi4 is not None:
            try:
                psi4.core.clean()
            except Exception:
                pass  # best-effort scratch cleanup; never mask the real outcome

    result["duration_seconds"] = time.time() - started
    write_json(result_path, result)
    return exit_code


def run_cubeprop(spec_path: str) -> int:
    """Generate cube files from a saved wavefunction.

    Kept separate from :func:`run_job` so orbitals can be explored *after* a calculation
    has finished, without paying to generate every cube up front or re-running the
    calculation just to get the wavefunction back.
    """
    spec = read_json(spec_path)
    psi4 = None
    try:
        import psi4 as _psi4

        psi4 = _psi4
        working_directory = os.path.dirname(os.path.abspath(spec_path))
        scratch_dir = os.path.join(working_directory, "scratch")
        os.makedirs(scratch_dir, exist_ok=True)
        psi4.core.set_output_file(os.path.join(working_directory, "cubeprop.log"), False)
        psi4.core.IOManager.shared_object().set_default_path(scratch_dir)
        psi4.set_memory(int(float(spec.get("memory_gb", 2)) * 1024 ** 3))
        psi4.set_num_threads(int(spec.get("n_threads", 1)))

        wfn = psi4.core.Wavefunction.from_file(spec["wavefunction_path"])

        if "ESP" in spec["tasks"]:
            # An electrostatic potential is evaluated by density fitting, and the fitting
            # basis is not part of what to_file() serializes -- without it cubeprop aborts
            # with "Auxiliary basis is required for ESP computations". Rebuild the JKFIT
            # set that matches the orbital basis and attach it.
            orbital_basis = wfn.basisset().name()
            wfn.set_basisset("DF_BASIS_SCF", psi4.core.BasisSet.build(
                wfn.molecule(), "DF_BASIS_SCF", "", "JKFIT", orbital_basis))

        spacing = float(spec.get("grid_spacing", 0.1))
        options: dict[str, Any] = {
            "CUBEPROP_TASKS": spec["tasks"],
            "CUBIC_GRID_SPACING": [spacing, spacing, spacing],
            "CUBEPROP_FILEPATH": spec["output_directory"],
        }
        if spec.get("orbitals"):
            options["CUBEPROP_ORBITALS"] = spec["orbitals"]
        psi4.set_options(options)

        os.makedirs(spec["output_directory"], exist_ok=True)
        psi4.cubeprop(wfn)
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if psi4 is not None:
            try:
                psi4.core.clean()
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``run`` / ``cubeprop`` to the matching handler."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in ("run", "cubeprop"):
        print("usage: python -m psi4_webui.runner {run|cubeprop} <spec.json>", file=sys.stderr)
        return 2
    mode, spec_path = argv
    return run_job(spec_path) if mode == "run" else run_cubeprop(spec_path)


if __name__ == "__main__":
    sys.exit(main())
