"""Tests for the runner's side of the contract, without running Psi4.

A fake ``psi4`` module is injected into ``sys.modules`` so the runner's control flow --
which artifacts it writes, what the result file contains, which exit code it returns --
can be pinned in milliseconds. The real-Psi4 behaviour is covered separately in
``test_psi4_integration.py``; this file is about the shape of the contract, which is what
the Result tab depends on.
"""
import json
import os
import sys
import types

import pytest

from psi4_webui.runner import job_base_name, main
from psi4_webui.utils import JOB_SUFFIX, RESULT_SUFFIX, build_job_spec, read_json, write_json


class _Datum:
    """Stand-in for a qcdb.vib Datum, which exposes its payload as ``.data``."""

    def __init__(self, data, units=""):
        self.data = data
        self.units = units


class _Array:
    """Stand-in for a psi4 Matrix/Vector, which converts via ``.to_array()``."""

    def __init__(self, values):
        self._values = values

    def to_array(self):
        return self._values


class _FakeWavefunction:
    def __init__(self, fail_to_file=False):
        self._fail_to_file = fail_to_file
        self.saved_to = None
        # Nine 3N modes, of which six are vibrational -- the same shape Psi4 produces,
        # and the reason the TRV mask matters.
        self.frequency_analysis = {
            "TRV": _Datum(["TR", "TR", "TR", "V", "V", "V", "V", "V", "V"]),
            "IR_intensity": _Datum([0.0, 0.0, 0.0, 11.0, 22.0, 33.0, 44.0, 55.0, 66.0], "km/mol"),
        }

    def variable(self, name):
        if name == "CURRENT DIPOLE":
            return [0.0, 0.0, -0.6789427]
        raise KeyError(name)

    def epsilon_a_subset(self, *_):
        return _Array([-20.24, -1.27, -0.62, -0.45, -0.39, 0.60, 0.74])

    def frequencies(self):
        return _Array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0])

    def nalpha(self):
        return 5

    def nbeta(self):
        return 5

    def to_file(self, path):
        if self._fail_to_file:
            raise RuntimeError("Cannot serialize a wavefunction with a custom basis set")
        self.saved_to = path
        open(path, "wb").close()


class _FakeMolecule:
    def __init__(self, charge=0, multiplicity=1):
        self._symbols = ["O", "H", "H"]
        # Bohr, as Psi4 stores geometry internally.
        self._geometry = [[0.0, 0.0, 0.2211], [0.0, 1.4306, -0.8863], [0.0, -1.4306, -0.8863]]
        self._charge = charge
        self._multiplicity = multiplicity

    def natom(self):
        return 3

    def symbol(self, index):
        return self._symbols[index]

    def geometry(self):
        return self._geometry

    def molecular_charge(self):
        return self._charge

    def multiplicity(self):
        return self._multiplicity


def _install_fake_psi4(monkeypatch, *, raise_on=None, fail_to_file=False):
    """Register a fake ``psi4`` module and return the record of calls made to it."""
    # "options" is the accumulated final state; "option_sets" keeps each call separately,
    # which matters where the runner deliberately overwrites an option partway through.
    calls = {"options": {}, "option_sets": [], "driver": [], "cleaned": False, "output_file": None}
    wavefunction = _FakeWavefunction(fail_to_file=fail_to_file)

    core = types.SimpleNamespace()

    def set_output_file(path, append):
        calls["output_file"] = path
        open(path, "w", encoding="utf-8").write("fake psi4 output\n")

    core.set_output_file = set_output_file
    core.variable = lambda name: {
        "ZPVE": 0.024, "THERMAL ENERGY": -74.93, "ENTHALPY": -74.92, "GIBBS FREE ENERGY": -74.95,
    }[name]
    core.clean = lambda: calls.__setitem__("cleaned", True)
    core.IOManager = types.SimpleNamespace(
        shared_object=lambda: types.SimpleNamespace(set_default_path=lambda p: None))

    def _driver(name_of_call):
        def call(method, molecule=None, return_wfn=False, return_history=False, **kwargs):
            calls["driver"].append(name_of_call)
            if raise_on == name_of_call:
                raise RuntimeError("SCF failed to converge")
            energy = -74.96
            if return_history:
                history = {
                    "energy": [-74.95, -74.96],
                    # Bohr, exactly as optking reports it.
                    "coordinates": [[[0.0, 0.0, 1.0]] * 3, [[0.0, 0.0, 2.0]] * 3],
                }
                return energy, wavefunction, history
            if return_wfn:
                return energy, wavefunction
            return energy
        return call

    fake = types.ModuleType("psi4")
    fake.core = core
    fake.constants = types.SimpleNamespace(
        bohr2angstroms=0.529177210903, dipmom_au2debye=2.541746451895026,
        hartree2kcalmol=627.5094740631, hartree2ev=27.21138602,
        hartree2wavenumbers=219474.6313702)
    fake.set_memory = lambda n: calls.__setitem__("memory", n)
    fake.set_num_threads = lambda n: calls.__setitem__("threads", n)
    def set_options(opts):
        calls["option_sets"].append(dict(opts))
        calls["options"].update(opts)

    fake.set_options = set_options
    fake.pcm_helper = lambda block: calls.__setitem__("pcm_block", block)
    def geometry(text):
        # Psi4 reads charge and multiplicity from the geometry block's first line, not
        # from anything passed alongside it -- the fake does the same so the runner's use
        # of molecule.molecular_charge()/multiplicity() is exercised honestly.
        charge, multiplicity = 0, 1
        fields = text.strip().splitlines()[0].split()
        if len(fields) == 2:
            charge, multiplicity = int(fields[0]), int(fields[1])
        return _FakeMolecule(charge, multiplicity)

    fake.geometry = geometry
    fake.energy = _driver("energy")
    fake.optimize = _driver("optimize")
    fake.frequency = _driver("frequency")

    monkeypatch.setitem(sys.modules, "psi4", fake)
    return calls


def _write_job(tmp_path, name="job", **overrides):
    """Write a job spec into ``tmp_path`` and return its path."""
    kwargs = dict(
        calculation_type="Single-Point",
        geometry="0 1\nO 0.0 0.0 0.0",
        structure_file="water.xyz",
        method_type="HF",
        functional="B3LYP",
        basis_set="STO-3G",
        reference="RHF",
        charge=0,
        multiplicity=1,
        n_threads=2,
        memory_gb=2,
    )
    kwargs.update(overrides)
    path = tmp_path / (name + JOB_SUFFIX)
    write_json(str(path), build_job_spec(**kwargs))
    return str(path)


class TestArtifactNaming:
    def test_strips_the_compound_job_suffix(self):
        # Getting this wrong names every artifact "job.job.log", "job.job.npy", ...
        assert job_base_name("/tmp/wd/opt" + JOB_SUFFIX) == "opt"

    def test_handles_a_plain_json_spec(self):
        assert job_base_name("/tmp/wd/opt.json") == "opt"


class TestSinglePoint:
    def test_writes_a_completed_result_and_exits_zero(self, tmp_path, monkeypatch):
        _install_fake_psi4(monkeypatch)
        assert main(["run", _write_job(tmp_path)]) == 0

        result = read_json(str(tmp_path / ("job" + RESULT_SUFFIX)))
        assert result["status"] == "completed"
        assert result["energy_hartree"] == pytest.approx(-74.96)
        assert result["error"] is None

    def test_records_the_fields_the_result_tab_reads(self, tmp_path, monkeypatch):
        _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path)])
        result = read_json(str(tmp_path / ("job" + RESULT_SUFFIX)))
        for key in ("symbols", "geometry_angstrom", "mo_energies_hartree", "homo_index",
                    "n_alpha", "n_beta", "dipole_debye", "log_file", "duration_seconds"):
            assert key in result, f"result file is missing {key!r}"

    def test_converts_the_dipole_to_debye(self, tmp_path, monkeypatch):
        _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path)])
        result = read_json(str(tmp_path / ("job" + RESULT_SUFFIX)))
        # 0.6789427 a.u. * 2.5417 = 1.7257 D
        assert result["dipole_debye"] == pytest.approx(1.7257, abs=1e-3)

    def test_homo_index_is_zero_based(self, tmp_path, monkeypatch):
        _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path)])
        # 5 alpha electrons occupy MOs 1-5, so the HOMO is at 0-based index 4.
        assert read_json(str(tmp_path / ("job" + RESULT_SUFFIX)))["homo_index"] == 4

    def test_converts_geometry_from_bohr_to_angstrom(self, tmp_path, monkeypatch):
        _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path)])
        geometry = read_json(str(tmp_path / ("job" + RESULT_SUFFIX)))["geometry_angstrom"]
        assert geometry[0][2] == pytest.approx(0.2211 * 0.529177210903, abs=1e-6)

    def test_saves_the_wavefunction_for_later_visualization(self, tmp_path, monkeypatch):
        _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path)])
        result = read_json(str(tmp_path / ("job" + RESULT_SUFFIX)))
        assert result["wavefunction_file"] == "job.npy"
        assert (tmp_path / "job.npy").exists()

    def test_a_wavefunction_that_cannot_be_saved_does_not_fail_the_job(self, tmp_path, monkeypatch):
        # to_file() refuses custom basis sets. Throwing away an otherwise-successful
        # calculation over that would be absurd; the viewer just stays hidden.
        _install_fake_psi4(monkeypatch, fail_to_file=True)
        assert main(["run", _write_job(tmp_path)]) == 0
        result = read_json(str(tmp_path / ("job" + RESULT_SUFFIX)))
        assert result["status"] == "completed"
        assert result["wavefunction_file"] is None


class TestOptimization:
    def test_writes_trajectory_and_optimized_geometry(self, tmp_path, monkeypatch):
        _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path, calculation_type="Geometry Optimization")])
        result = read_json(str(tmp_path / ("job" + RESULT_SUFFIX)))
        assert result["optimization"]["n_steps"] == 2
        assert (tmp_path / "job_trajectory.xyz").exists()
        assert (tmp_path / "job_optimized.xyz").exists()

    def test_trajectory_coordinates_are_converted_from_bohr(self, tmp_path, monkeypatch):
        # optking works in atomic units. The previous version of this app wrote these
        # straight out, making every structure it produced 1.89x too large.
        _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path, calculation_type="Geometry Optimization")])
        lines = (tmp_path / "job_trajectory.xyz").read_text(encoding="utf-8").splitlines()
        first_atom = lines[2].split()
        assert float(first_atom[3]) == pytest.approx(1.0 * 0.529177210903, abs=1e-6)

    def test_optimized_xyz_starts_with_charge_and_multiplicity(self, tmp_path, monkeypatch):
        # That first line is what lets the optimized geometry be fed straight back into a
        # follow-up calculation with its charge and spin state intact. It is taken from
        # the geometry block the calculation actually ran, not from the spec's separate
        # charge/multiplicity fields, which would mislabel the file if the two disagreed.
        _install_fake_psi4(monkeypatch)
        main(["run", _write_job(
            tmp_path, calculation_type="Geometry Optimization",
            geometry="-1 2\nO 0.0 0.0 0.0", charge=-1, multiplicity=2)])
        assert (tmp_path / "job_optimized.xyz").read_text(encoding="utf-8").splitlines()[0] == "-1 2"

    def test_optimized_xyz_follows_the_geometry_not_the_spec(self, tmp_path, monkeypatch):
        # If the spec's fields and the geometry header ever diverge, the geometry wins --
        # it is what Psi4 read.
        _install_fake_psi4(monkeypatch)
        main(["run", _write_job(
            tmp_path, calculation_type="Geometry Optimization",
            geometry="0 3\nO 0.0 0.0 0.0", charge=-1, multiplicity=1)])
        assert (tmp_path / "job_optimized.xyz").read_text(encoding="utf-8").splitlines()[0] == "0 3"


class TestFrequency:
    def test_optimizes_before_taking_the_hessian(self, tmp_path, monkeypatch):
        # A frequency analysis is only meaningful at a stationary point; taking the
        # Hessian at an arbitrary input geometry produces spurious imaginary modes.
        calls = _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path, calculation_type="Frequency")])
        assert calls["driver"] == ["optimize", "frequency"]

    def test_selects_vibrational_intensities_with_the_trv_mask(self, tmp_path, monkeypatch):
        # IR_intensity is indexed over all 3N modes but frequencies() returns only the
        # vibrational ones. Zipping the raw arrays pairs each frequency with the wrong
        # mode's intensity -- the bug this app previously shipped.
        _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path, calculation_type="Frequency")])
        frequency = read_json(str(tmp_path / ("job" + RESULT_SUFFIX)))["frequency"]
        assert frequency["ir_intensities_km_mol"] == [11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
        assert len(frequency["frequencies_cm1"]) == len(frequency["ir_intensities_km_mol"])

    def test_derives_entropy_from_enthalpy_and_gibbs(self, tmp_path, monkeypatch):
        _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path, calculation_type="Frequency", temperature=298.15)])
        thermo = read_json(str(tmp_path / ("job" + RESULT_SUFFIX)))["thermochemistry"]
        expected = (-74.92 - -74.95) / 298.15 * 627.5094740631 * 1000.0
        assert thermo["entropy_cal_mol_k"] == pytest.approx(expected)

    def test_counts_imaginary_frequencies(self, tmp_path, monkeypatch):
        _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path, calculation_type="Frequency")])
        # The fake returns six positive frequencies.
        assert read_json(str(tmp_path / ("job" + RESULT_SUFFIX)))["frequency"]["n_imaginary"] == 0


class TestEmission:
    def test_optimizes_the_excited_state_then_drops_to_the_ground_state(self, tmp_path, monkeypatch):
        # Emission is E(S1 relaxed) - E(S0 at that same geometry), so it takes exactly two
        # driver calls: an excited-state optimization and a ground-state energy.
        calls = _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path, calculation_type="Emission (EOM-CCSD)")])
        assert calls["driver"] == ["optimize", "energy"]

    def test_follows_the_requested_root(self, tmp_path, monkeypatch):
        calls = _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path, calculation_type="Emission (EOM-CCSD)", root=2)])
        # The geometry block forces C1, so there is one irrep and one entry.
        roots = [o["ROOTS_PER_IRREP"] for o in calls["option_sets"] if "ROOTS_PER_IRREP" in o]
        assert roots[0] == [2], "the excited-state optimization did not follow the chosen root"

    def test_clears_the_root_before_the_ground_state_energy(self, tmp_path, monkeypatch):
        # Left set, the ground-state CCSD call would go solving for excited roots again
        # instead of returning the S0 energy the emission gap is measured against.
        calls = _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path, calculation_type="Emission (EOM-CCSD)", root=2)])
        roots = [o["ROOTS_PER_IRREP"] for o in calls["option_sets"] if "ROOTS_PER_IRREP" in o]
        assert roots[-1] == [0]

    def test_records_the_emission_energy_and_wavelength(self, tmp_path, monkeypatch):
        _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path, calculation_type="Emission (EOM-CCSD)")])
        emission = read_json(str(tmp_path / ("job" + RESULT_SUFFIX)))["emission"]
        # The fake returns the same energy for both calls, so the gap is zero and the
        # wavelength is undefined -- what matters here is that the fields exist and the
        # divide is guarded rather than raising.
        assert emission["emission_energy_hartree"] == pytest.approx(0.0)
        assert emission["wavelength_nm"] is None

    def test_writes_the_relaxed_excited_state_geometry(self, tmp_path, monkeypatch):
        # The relaxed excited-state structure is the whole point of the calculation.
        _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path, calculation_type="Emission (EOM-CCSD)")])
        result = read_json(str(tmp_path / ("job" + RESULT_SUFFIX)))
        assert result["optimized_xyz_file"] == "job_optimized.xyz"
        assert (tmp_path / "job_optimized.xyz").exists()


class TestOptionsAndSolvation:
    def test_dft_uses_density_fitting_and_hf_uses_pk(self, tmp_path, monkeypatch):
        calls = _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path, method_type="DFT", functional="B3LYP")])
        assert calls["options"]["SCF_TYPE"] == "DF"

        calls = _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path, name="hf", method_type="HF")])
        assert calls["options"]["SCF_TYPE"] == "PK"

    def test_tddft_keeps_the_jk_object_alive(self, tmp_path, monkeypatch):
        # Without SAVE_JK, Psi4 releases the integrals at the end of the SCF and
        # tdscf_excitations has nothing to work from.
        calls = _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path, calculation_type="Time-Dependent Density Functional Theory")])
        assert calls["options"]["SAVE_JK"] is True

    def test_solvation_sets_pcm_and_supplies_the_solver_block(self, tmp_path, monkeypatch):
        calls = _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path, use_solvation=True, solvent="Acetonitrile")])
        assert calls["options"]["PCM"] is True
        assert "Acetonitrile" in calls["pcm_block"]
        assert "iefpcm" in calls["pcm_block"]

    def test_no_pcm_block_when_solvation_is_off(self, tmp_path, monkeypatch):
        calls = _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path)])
        assert "PCM" not in calls["options"]
        assert "pcm_block" not in calls

    def test_memory_and_threads_reach_psi4(self, tmp_path, monkeypatch):
        calls = _install_fake_psi4(monkeypatch)
        main(["run", _write_job(tmp_path, n_threads=6, memory_gb=3)])
        assert calls["threads"] == 6
        assert calls["memory"] == 3 * 1024 ** 3


class TestFailurePaths:
    def test_a_raising_calculation_exits_one_and_still_writes_a_result(self, tmp_path, monkeypatch):
        # The Result tab must be able to explain the failure, so a result file is written
        # even when the calculation blew up.
        _install_fake_psi4(monkeypatch, raise_on="energy")
        assert main(["run", _write_job(tmp_path)]) == 1

        result = read_json(str(tmp_path / ("job" + RESULT_SUFFIX)))
        assert result["status"] == "failed"
        assert "SCF failed to converge" in result["error"]
        assert "Traceback" in result["traceback"]

    def test_scratch_is_cleaned_up_even_on_failure(self, tmp_path, monkeypatch):
        calls = _install_fake_psi4(monkeypatch, raise_on="energy")
        main(["run", _write_job(tmp_path)])
        assert calls["cleaned"] is True

    def test_bad_usage_exits_two(self, tmp_path):
        assert main([]) == 2
        assert main(["frobnicate", "x.json"]) == 2


@pytest.mark.parametrize("contents", ["{broken", "[]", "null"])
def test_invalid_job_still_writes_a_failure_result(tmp_path, contents):
    path = tmp_path / "invalid.job.json"
    path.write_text(contents, encoding="utf-8")
    assert main(["run", str(path)]) == 1
    result = read_json(str(tmp_path / "invalid.result.json"))
    assert result["status"] == "failed"
    assert result["error"]


def test_result_charge_and_spin_follow_geometry(tmp_path, monkeypatch):
    _install_fake_psi4(monkeypatch)
    assert main(["run", _write_job(tmp_path, geometry="-1 2\nO 0 0 0")]) == 0
    result = read_json(str(tmp_path / "job.result.json"))
    assert result["charge"] == -1
    assert result["multiplicity"] == 2


def test_mismatched_ir_arrays_fail_instead_of_repeating_intensities(tmp_path, monkeypatch):
    _install_fake_psi4(monkeypatch)
    monkeypatch.setattr(_FakeWavefunction, "frequencies", lambda self: _Array([1000.0] * 7))
    assert main(["run", _write_job(tmp_path, calculation_type="Frequency")]) == 1
    result = read_json(str(tmp_path / "job.result.json"))
    assert "different lengths" in result["error"]
    assert "frequency" not in result


def test_invalid_cube_spec_returns_failure_exit_code(tmp_path):
    path = tmp_path / "cube.json"
    path.write_text("{broken", encoding="utf-8")
    assert main(["cubeprop", str(path)]) == 1
