"""Tests for the job-spec layer: what the UI hands to the runner.

This is the psi4-webui analogue of orca-webui's ``test_orca_input.py``. It is pure --
no Psi4, no Gradio -- because every decision about *what* to compute is made here, and
pinning those decisions is what stops the Calculation tab and the runner drifting apart.
"""
import json

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from psi4_webui.utils import (
    FREQUENCY,
    GEOMETRY_OPTIMIZATION,
    JOB_SUFFIX,
    RESULT_SUFFIX,
    SINGLE_POINT,
    TDDFT,
    build_job_spec,
    psi4_geometry_string,
    psi4_method_keyword,
    read_json,
    result_base_name,
    write_json,
)


def _spec(**overrides):
    """A valid job spec with sensible defaults, overridable per test."""
    kwargs = dict(
        calculation_type=SINGLE_POINT,
        geometry="0 1\nO 0.0 0.0 0.0\nunits angstrom",
        structure_file="water.xyz",
        method_type="DFT",
        functional="B3LYP",
        basis_set="6-31G(d)",
        reference="RHF",
        charge=0,
        multiplicity=1,
        n_threads=4,
        memory_gb=8,
    )
    kwargs.update(overrides)
    return build_job_spec(**kwargs)


class TestMethodKeyword:
    """The string handed to psi4.energy/optimize/frequency as its method name."""

    def test_dft_uses_the_functional_name(self):
        # Psi4 has no "DFT" method; a DFT calculation is requested by naming the
        # functional, so the functional must survive into the spec verbatim.
        assert psi4_method_keyword("DFT", "wB97X-D") == "wB97X-D"

    def test_hartree_fock_maps_to_scf(self):
        # Psi4 spells Hartree-Fock "SCF"; passing "HF" through unchanged would be a
        # method Psi4 does not recognise.
        assert psi4_method_keyword("HF", "B3LYP") == "SCF"

    @pytest.mark.parametrize("method_type", ["MP2", "CCSD", "CCSD(T)"])
    def test_correlated_methods_pass_through(self, method_type):
        assert psi4_method_keyword(method_type, "B3LYP") == method_type

    def test_unknown_method_type_is_passed_through_not_dropped(self):
        # A method added to the UI but not to the table should still reach Psi4 and fail
        # loudly there, rather than being silently replaced by something else.
        assert psi4_method_keyword("SAPT0", "B3LYP") == "SAPT0"


class TestGeometryString:
    """The geometry block Psi4 parses."""

    def test_first_line_is_charge_and_multiplicity(self, water_mol):
        geometry = psi4_geometry_string(water_mol, -1, 2)
        assert geometry.splitlines()[0] == "-1 2"

    def test_one_line_per_atom_in_molecule_order(self, water_mol):
        geometry = psi4_geometry_string(water_mol, 0, 1)
        atom_lines = [ln for ln in geometry.splitlines() if len(ln.split()) == 4]
        assert len(atom_lines) == water_mol.GetNumAtoms()
        assert [ln.split()[0] for ln in atom_lines] == [a.GetSymbol() for a in water_mol.GetAtoms()]

    def test_orientation_is_frozen(self, water_mol):
        # Without these, Psi4 shifts the molecule to its centre of mass and rotates it
        # onto its principal axes. Every geometry written back out -- optimized
        # structures, trajectories, and the frame cube files are computed on -- would
        # then be in that rotated frame rather than the one the viewer showed.
        geometry = psi4_geometry_string(water_mol, 0, 1)
        assert "no_reorient" in geometry
        assert "no_com" in geometry

    def test_symmetry_is_kept_by_default(self, water_mol):
        # Symmetry makes the calculation faster; it is only dropped where it breaks things.
        assert "symmetry c1" not in psi4_geometry_string(water_mol, 0, 1)

    def test_symmetry_can_be_dropped_for_excited_state_optimization(self, water_mol):
        # Psi4 selects excited roots per irrep, so a molecule that distorts out of its
        # starting point group mid-optimization aborts with "Point group changed!".
        assert "symmetry c1" in psi4_geometry_string(water_mol, 0, 1, force_c1=True)

    def test_units_are_declared(self, water_mol):
        # RDKit conformers are in angstrom, but Psi4's default for a bare geometry block
        # is also angstrom only by convention -- stating it removes the ambiguity.
        assert "units angstrom" in psi4_geometry_string(water_mol, 0, 1)

    def test_coordinates_match_the_conformer(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("O"))
        AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
        conformer = mol.GetConformer()
        geometry = psi4_geometry_string(mol, 0, 1)
        atom_lines = [ln for ln in geometry.splitlines() if len(ln.split()) == 4]
        for index, line in enumerate(atom_lines):
            position = conformer.GetAtomPosition(index)
            x, y, z = (float(v) for v in line.split()[1:4])
            assert (x, y, z) == pytest.approx((position.x, position.y, position.z), abs=1e-6)


class TestJobSpec:
    """The spec dict itself."""

    def test_is_json_serializable(self):
        # The spec crosses a process boundary as JSON; a numpy scalar sneaking in from a
        # Gradio slider would break the run at launch.
        json.dumps(_spec())

    def test_numeric_fields_are_coerced(self):
        # Gradio sliders hand back floats even for integer-valued controls, and Psi4
        # rejects a float where it wants a count.
        spec = _spec(charge=-1.0, multiplicity=3.0, n_threads=4.0)
        assert isinstance(spec["charge"], int)
        assert isinstance(spec["multiplicity"], int)
        assert isinstance(spec["n_threads"], int)

    def test_dft_spec_carries_the_functional_as_the_method(self):
        spec = _spec(method_type="DFT", functional="PBE0")
        assert spec["method"] == "PBE0"
        assert spec["functional"] == "PBE0"

    def test_hf_spec_ignores_the_functional(self):
        # The functional dropdown is hidden for HF but still holds a value; it must not
        # leak into the method.
        assert _spec(method_type="HF", functional="B3LYP")["method"] == "SCF"

    @pytest.mark.parametrize("calculation_type", [SINGLE_POINT, GEOMETRY_OPTIMIZATION, FREQUENCY, TDDFT])
    def test_every_calculation_type_round_trips(self, calculation_type, tmp_path):
        path = tmp_path / ("job" + JOB_SUFFIX)
        spec = _spec(calculation_type=calculation_type)
        write_json(str(path), spec)
        assert read_json(str(path)) == spec

    def test_solvation_is_off_by_default(self):
        assert _spec()["use_solvation"] is False

    def test_solvent_is_recorded_when_solvation_is_on(self):
        spec = _spec(use_solvation=True, solvent="Acetonitrile")
        assert spec["use_solvation"] is True
        assert spec["solvent"] == "Acetonitrile"


class TestResultBaseName:
    """Deriving a job's artifact names from one of its files."""

    def test_strips_the_compound_result_suffix(self):
        # os.path.splitext would leave "opt.result", so the sibling .log and .npy of the
        # job could not be found from its result file.
        assert result_base_name("opt" + RESULT_SUFFIX) == "opt"

    def test_strips_the_compound_job_suffix(self):
        assert result_base_name("opt" + JOB_SUFFIX) == "opt"

    def test_leaves_a_plain_name_alone(self):
        assert result_base_name("opt.log") == "opt"

    def test_preserves_dots_inside_the_name(self):
        assert result_base_name("scan.step.2" + RESULT_SUFFIX) == "scan.step.2"
