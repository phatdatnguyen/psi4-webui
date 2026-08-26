"""End-to-end tests that invoke the real Psi4 engine through the runner.

Marked ``psi4`` and skipped when Psi4 is not importable, so the rest of the suite stays
fast and installable-anywhere. Every system here is deliberately tiny (water or H2, with
STO-3G and one thread) so the whole module runs in seconds -- these exist to prove the
plumbing and the unit conversions are right, not to benchmark Psi4.
"""
import math
import os
import subprocess
import sys

import pytest

from psi4_webui.utils import JOB_SUFFIX, RESULT_SUFFIX, build_job_spec, read_json, write_json

pytestmark = pytest.mark.psi4

WATER = (
    "0 1\n"
    "O 0.000 0.000 0.117\n"
    "H 0.000 0.757 -0.469\n"
    "H 0.000 -0.757 -0.469\n"
    "units angstrom\nno_reorient\nno_com"
)


def run_job(working_directory, name="job", **overrides):
    """Write a job spec, run the real runner on it, and return ``(exit_code, result)``."""
    kwargs = dict(
        calculation_type="Single-Point",
        geometry=WATER,
        structure_file="water.xyz",
        method_type="HF",
        functional="B3LYP",
        basis_set="STO-3G",
        reference="RHF",
        charge=0,
        multiplicity=1,
        n_threads=1,
        memory_gb=1,
    )
    kwargs.update(overrides)
    job_path = os.path.join(working_directory, name + JOB_SUFFIX)
    write_json(job_path, build_job_spec(**kwargs))

    completed = subprocess.run(
        [sys.executable, "-m", "psi4_webui.runner", "run", job_path],
        cwd=working_directory, capture_output=True, text=True, timeout=900,
    )
    result_path = os.path.join(working_directory, name + RESULT_SUFFIX)
    result = read_json(result_path) if os.path.isfile(result_path) else None
    return completed.returncode, result


@pytest.fixture
def working_directory(tmp_path, psi4_available):
    """A scratch working directory, with Psi4 confirmed available."""
    return str(tmp_path)


class TestSinglePoint:
    def test_produces_the_expected_energy_and_artifacts(self, working_directory):
        exit_code, result = run_job(working_directory)

        assert exit_code == 0
        assert result["status"] == "completed"
        # HF/STO-3G water at this geometry. Pinned tightly: a change here means the
        # geometry or the method stopped being what the spec asked for.
        assert result["energy_hartree"] == pytest.approx(-74.962946, abs=1e-5)
        assert os.path.isfile(os.path.join(working_directory, "job.log"))
        assert os.path.isfile(os.path.join(working_directory, "job.npy"))

    def test_the_log_is_real_psi4_output(self, working_directory):
        # psi4.set_output_file (as opposed to psi4.core.set_output_file) attaches a
        # Python logging handler to the same path and clobbers it, leaving a log with no
        # Psi4 content at all. These markers come from Psi4's own C++ output stream.
        run_job(working_directory)
        log = open(os.path.join(working_directory, "job.log"), encoding="utf-8").read()
        assert "Nuclear repulsion" in log
        assert "Total Energy" in log
        assert "@RHF iter" in log

    def test_dipole_is_reported_in_debye(self, working_directory):
        _, result = run_job(working_directory)
        # Water's HF/STO-3G dipole is ~1.7 D. In atomic units it would be ~0.68, so this
        # also catches a missing unit conversion.
        assert result["dipole_debye"] == pytest.approx(1.73, abs=0.05)

    def test_homo_index_points_at_the_highest_occupied_orbital(self, working_directory):
        _, result = run_job(working_directory)
        # Water has 10 electrons, so 5 doubly-occupied orbitals: the HOMO is index 4.
        assert result["n_alpha"] == 5
        assert result["homo_index"] == 4
        assert len(result["mo_energies_hartree"]) == 7  # STO-3G water has 7 basis functions

    def test_geometry_is_returned_in_the_input_frame(self, working_directory):
        # no_reorient/no_com in the geometry block is what keeps this true. Without it
        # Psi4 re-centres and rotates, and every structure written back out -- including
        # the frame cube files are computed on -- would be in that rotated frame.
        _, result = run_job(working_directory)
        assert result["geometry_angstrom"][0] == pytest.approx([0.0, 0.0, 0.117], abs=1e-6)

    def test_dft_runs_and_differs_from_hartree_fock(self, working_directory):
        _, hf = run_job(working_directory, name="hf", method_type="HF")
        exit_code, dft = run_job(working_directory, name="dft", method_type="DFT", functional="B3LYP")

        assert exit_code == 0
        assert dft["status"] == "completed"
        # Correlation lowers the energy substantially; identical values would mean the
        # functional never reached Psi4.
        assert dft["energy_hartree"] < hf["energy_hartree"] - 0.1


class TestOptimization:
    def test_converges_to_a_physical_geometry(self, working_directory):
        exit_code, result = run_job(
            working_directory, calculation_type="Geometry Optimization",
            geometry=WATER.replace("0.117", "0.200").replace("0.757", "0.800"),
        )

        assert exit_code == 0
        geometry = result["geometry_angstrom"]
        bond_length = math.dist(geometry[0], geometry[1])
        # HF/STO-3G gives an O-H bond of ~0.99 A. If the Bohr->angstrom conversion on the
        # optimizer's coordinates were missing, this would come out near 1.87.
        assert bond_length == pytest.approx(0.99, abs=0.05)

    def test_energy_decreases_monotonically(self, working_directory):
        _, result = run_job(working_directory, calculation_type="Geometry Optimization",
                            geometry=WATER.replace("0.117", "0.200"))
        energies = result["optimization"]["energies_hartree"]
        assert len(energies) > 1
        assert all(b <= a + 1e-9 for a, b in zip(energies, energies[1:]))

    def test_writes_a_reusable_optimized_structure(self, working_directory):
        from psi4_webui.utils import mol_from_xyz_file

        _, result = run_job(working_directory, calculation_type="Geometry Optimization")
        path = os.path.join(working_directory, result["optimized_xyz_file"])

        # The optimized geometry has to be loadable as an input structure for a follow-up
        # calculation, charge and multiplicity included. Both are read back from the
        # molecule Psi4 actually ran, not from the job spec.
        mol, charge, multiplicity = mol_from_xyz_file(path, return_charge_and_multiplicity=True)
        assert mol.GetNumAtoms() == 3
        assert (charge, multiplicity) == (0, 1)


    def test_open_shell_optimization_records_its_spin_state(self, working_directory):
        from psi4_webui.utils import mol_from_xyz_file

        # The hydroxyl radical: 9 electrons, so a genuine doublet.
        _, result = run_job(
            working_directory, calculation_type="Geometry Optimization",
            geometry="\n".join([
                "0 2", "O 0.0 0.0 0.0", "H 0.0 0.0 0.97",
                "units angstrom", "no_reorient", "no_com",
            ]),
            reference="UHF", multiplicity=2,
        )
        path = os.path.join(working_directory, result["optimized_xyz_file"])
        _, charge, multiplicity = mol_from_xyz_file(path, return_charge_and_multiplicity=True)

        assert (charge, multiplicity) == (0, 2)

    def test_trajectory_holds_every_step(self, working_directory):
        _, result = run_job(working_directory, calculation_type="Geometry Optimization",
                            geometry=WATER.replace("0.117", "0.200"))
        path = os.path.join(working_directory, result["trajectory_xyz_file"])
        text = open(path, encoding="utf-8").read()
        assert text.count("Step ") == result["optimization"]["n_steps"]


class TestFrequency:
    def test_water_has_three_real_modes_and_valid_thermochemistry(self, working_directory):
        exit_code, result = run_job(working_directory, calculation_type="Frequency")

        assert exit_code == 0
        frequency = result["frequency"]
        # Water is non-linear with 3 atoms: 3N-6 = 3 vibrational modes.
        assert len(frequency["frequencies_cm1"]) == 3
        # The runner optimizes before taking the Hessian, so a minimum is expected.
        assert frequency["n_imaginary"] == 0
        assert all(f > 0 for f in frequency["frequencies_cm1"])

    def test_each_frequency_has_its_own_intensity(self, working_directory):
        # frequency_analysis['IR_intensity'] covers all 3N modes while frequencies()
        # covers only the vibrational ones, so these can silently differ in length.
        _, result = run_job(working_directory, calculation_type="Frequency")
        frequency = result["frequency"]
        assert len(frequency["ir_intensities_km_mol"]) == len(frequency["frequencies_cm1"])
        assert all(i >= 0 for i in frequency["ir_intensities_km_mol"])

    def test_thermochemistry_is_internally_consistent(self, working_directory):
        _, result = run_job(working_directory, calculation_type="Frequency")
        thermo = result["thermochemistry"]

        assert thermo["gibbs_hartree"] < thermo["enthalpy_hartree"], "G = H - TS must be below H"
        assert thermo["zpve_hartree"] > 0
        # Water's standard molar entropy is ~45 cal/(mol K).
        assert thermo["entropy_cal_mol_k"] == pytest.approx(45.0, abs=3.0)


class TestExcitations:
    def test_tddft_produces_plottable_excited_states(self, working_directory):
        exit_code, result = run_job(
            working_directory, calculation_type="Time-Dependent Density Functional Theory",
            method_type="DFT", functional="B3LYP", n_states=3,
        )

        assert exit_code == 0
        excitations = result["excitations"]
        assert len(excitations) == 3
        for state in excitations:
            assert state["energy_hartree"] > 0
            assert state["wavelength_nm"] > 0
            assert state["oscillator_strength"] >= 0
            # Present even for achiral molecules (where it is zero); the Result tab needs
            # the key to decide whether to enable the ECD button.
            assert "rotatory_strength" in state

    def test_excitation_energies_increase_with_state_index(self, working_directory):
        _, result = run_job(working_directory, calculation_type="Time-Dependent Density Functional Theory",
                            method_type="DFT", functional="B3LYP", n_states=3)
        energies = [s["energy_hartree"] for s in result["excitations"]]
        assert energies == sorted(energies)


class TestEmission:
    """Emission via EOM-CCSD, the only method in Psi4 with an excited-state gradient."""

    # Formaldehyde: its S1 (n -> pi*) is bound and is the textbook emission example.
    FORMALDEHYDE = "\n".join([
        "0 1",
        "C 0.000 0.000 -0.600", "O 0.000 0.000  0.600",
        "H 0.000 0.940 -1.180", "H 0.000 -0.940 -1.180",
        "units angstrom", "no_reorient", "no_com", "symmetry c1",
    ])

    def test_produces_a_physical_emission_energy(self, working_directory):
        exit_code, result = run_job(
            working_directory, calculation_type="Emission (EOM-CCSD)",
            geometry=self.FORMALDEHYDE, method_type="EOM-CCSD", root=1, geom_maxiter=30,
        )

        assert exit_code == 0, result and result.get("error")
        emission = result["emission"]
        # Emission is a real, positive gap down to the ground state.
        assert emission["emission_energy_hartree"] > 0
        assert emission["wavelength_nm"] > 0
        assert emission["excited_energy_hartree"] > emission["ground_energy_at_excited_geometry_hartree"]

    def test_the_excited_state_relaxes_the_way_the_orbital_says_it_should(self, working_directory):
        # The n -> pi* excitation populates a C=O antibonding orbital, so the bond must
        # lengthen substantially from its ~1.21 A ground-state value. This is what
        # distinguishes a genuine excited-state optimization from one that quietly
        # optimized the ground state instead.
        _, result = run_job(
            working_directory, calculation_type="Emission (EOM-CCSD)",
            geometry=self.FORMALDEHYDE, method_type="EOM-CCSD", root=1, geom_maxiter=30,
        )
        geometry = result["geometry_angstrom"]
        assert math.dist(geometry[0], geometry[1]) > 1.30

    def test_saves_the_relaxed_excited_state_structure(self, working_directory):
        _, result = run_job(
            working_directory, calculation_type="Emission (EOM-CCSD)",
            geometry=self.FORMALDEHYDE, method_type="EOM-CCSD", root=1, geom_maxiter=30,
        )
        assert os.path.isfile(os.path.join(working_directory, result["optimized_xyz_file"]))


class TestCubeprop:
    def _cubeprop(self, working_directory, tasks, orbitals=None):
        from psi4_webui.utils import write_json as _write_json

        output_directory = os.path.join(working_directory, "cubes", "test")
        spec_path = os.path.join(working_directory, "cube.json")
        _write_json(spec_path, {
            "wavefunction_path": os.path.join(working_directory, "job.npy"),
            "tasks": tasks, "orbitals": orbitals,
            "grid_spacing": 0.4, "output_directory": output_directory,
        })
        completed = subprocess.run(
            [sys.executable, "-m", "psi4_webui.runner", "cubeprop", spec_path],
            cwd=working_directory, capture_output=True, text=True, timeout=600,
        )
        return completed, output_directory

    def test_density_cube_round_trips_through_the_saved_wavefunction(self, working_directory):
        # This is the test that proves Wavefunction.from_file() gives cubeprop something
        # it can work with -- the whole on-demand orbital viewer depends on it.
        run_job(working_directory)
        completed, output_directory = self._cubeprop(working_directory, ["DENSITY"])

        assert completed.returncode == 0, completed.stdout[-2000:]
        assert os.path.isfile(os.path.join(output_directory, "Dt.cube"))

    def test_orbital_cube_is_found_by_the_viewer(self, working_directory):
        from psi4_webui.visualization import cube_files_for_selection

        run_job(working_directory)
        completed, output_directory = self._cubeprop(working_directory, ["ORBITALS"], [5, -5])

        assert completed.returncode == 0, completed.stdout[-2000:]
        # Psi4 embeds an irrep label in the orbital file name, so the viewer globs for it.
        assert cube_files_for_selection(output_directory, "MO 5")

    def test_esp_cube_needs_and_gets_a_fitting_basis(self, working_directory):
        # An ESP is evaluated by density fitting, and the fitting basis is not part of
        # what to_file() serializes; without rebuilding it, cubeprop aborts with
        # "Auxiliary basis is required for ESP computations".
        run_job(working_directory)
        completed, output_directory = self._cubeprop(working_directory, ["ESP"])

        assert completed.returncode == 0, completed.stdout[-2000:]
        assert os.path.isfile(os.path.join(output_directory, "ESP.cube"))


class TestFailurePath:
    def test_a_bad_basis_set_fails_cleanly_with_a_usable_message(self, working_directory):
        exit_code, result = run_job(working_directory, basis_set="NOSUCHBASIS")

        assert exit_code == 1
        # A result file is written even on failure so the Result tab can explain it.
        assert result is not None
        assert result["status"] == "failed"
        # Assert on Psi4's own diagnostic rather than just "something failed", so the
        # test cannot pass for the wrong reason.
        assert "NOSUCHBASIS" in result["error"].upper()
        assert result["traceback"]

    def test_an_impossible_spin_state_is_reported_not_crashed(self, working_directory):
        # Water has 10 electrons, which cannot be arranged into a doublet. Psi4 rejects
        # the combination while parsing the geometry, before any SCF starts.
        exit_code, result = run_job(
            working_directory,
            geometry=WATER.replace("0 1", "0 2", 1),
            multiplicity=2, reference="UHF",
        )
        assert exit_code == 1
        assert result["status"] == "failed"
        assert "chg/mult" in result["error"] or "mult" in result["error"].lower()
