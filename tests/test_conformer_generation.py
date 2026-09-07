"""Conformer generation rejects invalid output paths and unreliable minima."""
import math

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from psi4_webui import conformer_generation as conformers
from psi4_webui.utils import mol_from_xyz_file


class Progress:
    def __call__(self, *args, **kwargs):
        pass

    def tqdm(self, iterable, **kwargs):
        return iterable


def generate(directory, **overrides):
    values = dict(working_directory_path=str(directory), input_smiles="O",
                  charge=0, multiplicity=1, num_confs=1, energy_threshold=0.1,
                  rms_threshold=0.5, file_name="water", file_type="xyz",
                  progress=Progress())
    values.update(overrides)
    return conformers.on_generate_conformers(**values)


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", r"..\outside", "", " "])
def test_invalid_filename_is_rejected_before_embedding(tmp_path, monkeypatch, name):
    def unexpected_embedding(*args, **kwargs):
        pytest.fail("Invalid filenames must be rejected before embedding")

    monkeypatch.setattr(conformers, "generate_unique_conformers", unexpected_embedding)
    status, files, table = generate(tmp_path, file_name=name)
    assert "color:red" in status
    assert "without a directory path" in status
    assert files == [] and table.empty


def test_output_symlink_cannot_overwrite_file_outside_working_directory(tmp_path):
    directory = tmp_path / "work"
    directory.mkdir()
    outside = tmp_path / "outside.xyz"
    outside.write_text("preserve this")
    (directory / "water_1.xyz").symlink_to(outside)
    status, _, table = generate(directory)
    assert "color:red" in status and table.empty
    assert outside.read_text() == "preserve this"


@pytest.mark.parametrize("directory", [None, "missing"])
def test_missing_directory_is_reported_without_raising(tmp_path, directory):
    path = None if directory is None else str(tmp_path / directory)
    status, files, table = generate(tmp_path, working_directory_path=path)
    assert "color:red" in status
    assert "open a working directory" in status
    assert files == [] and table.empty


def test_empty_smiles_is_rejected(tmp_path):
    status, files, table = generate(tmp_path, input_smiles="")
    assert "color:red" in status
    assert "containing atoms" in status
    assert files == [] and table.empty


@pytest.mark.parametrize("smiles", ["[Na+].[Cl-]", "O.O"])
def test_disconnected_fragments_do_not_generate_overlapping_nuclei(tmp_path, monkeypatch, smiles):
    def unexpected_embedding(*args, **kwargs):
        pytest.fail("Disconnected fragments must not be independently embedded at the same origin")

    monkeypatch.setattr(AllChem, "EmbedMultipleConfs", unexpected_embedding)
    status, files, table = generate(tmp_path, input_smiles=smiles)
    assert "color:red" in status and "single connected molecule" in status
    assert files == [] and table.empty


def test_clearing_sketch_clears_smiles():
    assert conformers.on_draw_molecule(None) == ""
    assert conformers.on_draw_molecule("") == ""


def test_only_converged_finite_energies_are_accepted(water_mol, monkeypatch):
    monkeypatch.setattr(AllChem, "MMFFOptimizeMoleculeConfs", lambda *args, **kwargs:
                        [(0, -1.0), (1, -2.0), (-1, -3.0), (0, math.nan)])
    assert conformers.optimize_conformers(water_mol) == [-1.0, None, None, None]


def test_missing_force_field_parameters_are_reported(monkeypatch, water_mol):
    monkeypatch.setattr(AllChem, "MMFFGetMoleculeProperties", lambda mol: None)
    monkeypatch.setattr(AllChem, "UFFHasAllMoleculeParams", lambda mol: False)

    def unexpected_optimization(*args, **kwargs):
        pytest.fail("A force field with missing parameters must not be used")

    monkeypatch.setattr(AllChem, "UFFOptimizeMoleculeConfs", unexpected_optimization)
    with pytest.raises(ValueError, match="parameters for every atom"):
        conformers.optimize_conformers(water_mol)


def test_failed_minima_are_skipped_without_shifting_conformer_ids(water_mol, monkeypatch):
    monkeypatch.setattr(conformers, "optimize_conformers", lambda mol:
                        [None, -4.0] + [None] * (mol.GetNumConformers() - 2))
    molecule, kept, discarded = conformers.generate_unique_conformers(water_mol, 1, 0.1, 0.5)
    assert kept == [(-4.0, 0)]
    assert molecule.GetNumConformers() == 1
    assert discarded == 0


def test_all_failed_minima_produce_an_error_and_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(conformers, "optimize_conformers", lambda mol:
                        [None] * mol.GetNumConformers())
    status, files, table = generate(tmp_path)
    assert "color:red" in status and "No minimized conformers" in status
    assert files == [] and table.empty


def test_successful_generation_preserves_slider_charge_and_multiplicity(tmp_path):
    status, files, table = generate(tmp_path, charge=-1.0, multiplicity=2.0, num_confs=1.0)
    assert "color:green" in status
    assert files == ["water_1.xyz"]
    assert len(table) == 1
    molecule, charge, multiplicity = mol_from_xyz_file(str(tmp_path / files[0]), True)
    assert molecule.GetNumAtoms() == 3
    assert (charge, multiplicity) == (-1, 2)


@pytest.mark.parametrize("file_type", ["mol", "pdb"])
def test_other_conformer_formats_keep_their_extension(tmp_path, file_type):
    status, files, _ = generate(tmp_path, file_type=file_type)
    assert "color:green" in status
    assert files == [f"water_1.{file_type}"]
