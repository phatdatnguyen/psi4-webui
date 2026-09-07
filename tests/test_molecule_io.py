"""Tests for molecule I/O helpers: xyz round-trip, bond perception, file listing."""
import os

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from psi4_webui import utils


# --- get_files_in_working_directory -----------------------------------------

def test_get_files_excludes_zone_identifier(tmp_path):
    (tmp_path / "a.xyz").write_text("x")
    (tmp_path / "b.log").write_text("y")
    (tmp_path / "a.xyz.Zone.Identifier").write_text("junk")
    files = utils.get_files_in_working_directory(str(tmp_path))
    assert "a.xyz" in files
    assert "b.log" in files
    assert all(not f.endswith("Zone.Identifier") for f in files)


def test_get_files_handles_a_removed_or_invalid_working_directory(tmp_path):
    assert utils.get_files_in_working_directory(str(tmp_path / "missing")) == []
    file_path = tmp_path / "file.txt"
    file_path.write_text("not a directory")
    assert utils.get_files_in_working_directory(str(file_path)) == []


# --- conformer_to_xyz_file + mol_from_xyz_file round trip --------------------

def test_xyz_roundtrip_preserves_atoms_and_charge(tmp_path):
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))  # methanol: C, O, 4 H
    assert AllChem.EmbedMolecule(mol, randomSeed=1) == 0
    path = str(tmp_path / "mol.xyz")
    utils.conformer_to_xyz_file(mol, 0, path, charge=-1, multiplicity=2)

    parsed, charge, mult = utils.mol_from_xyz_file(path, return_charge_and_multiplicity=True)
    assert charge == -1
    assert mult == 2
    assert parsed.GetNumAtoms() == mol.GetNumAtoms()
    # element multiset preserved
    orig = sorted(a.GetSymbol() for a in mol.GetAtoms())
    got = sorted(a.GetSymbol() for a in parsed.GetAtoms())
    assert orig == got


def test_xyz_coordinates_preserved(tmp_path):
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    assert AllChem.EmbedMolecule(mol, randomSeed=2) == 0
    path = str(tmp_path / "w.xyz")
    utils.conformer_to_xyz_file(mol, 0, path)
    parsed = utils.mol_from_xyz_file(path)

    conf_a = mol.GetConformer()
    conf_b = parsed.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pa, pb = conf_a.GetAtomPosition(i), conf_b.GetAtomPosition(i)
        assert abs(pa.x - pb.x) < 1e-4
        assert abs(pa.y - pb.y) < 1e-4
        assert abs(pa.z - pb.z) < 1e-4


def test_xyz_defaults_charge_multiplicity_when_absent(tmp_path):
    # An xyz whose first line is a bare count (not "charge multiplicity")
    path = tmp_path / "plain.xyz"
    path.write_text("3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n")
    mol, charge, mult = utils.mol_from_xyz_file(str(path), return_charge_and_multiplicity=True)
    assert (charge, mult) == (0, 1)
    assert mol.GetNumAtoms() == 3


# --- add_bonds ---------------------------------------------------------------

def test_add_bonds_infers_connectivity():
    # methanol geometry: RDKit builds bonds; strip them, re-perceive from distance.
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=3) == 0

    # Rebuild a bond-free mol carrying only atoms + conformer.
    rw = Chem.RWMol()
    for atom in mol.GetAtoms():
        rw.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    rw.AddConformer(mol.GetConformer(), assignId=True)
    bare = rw.GetMol()
    assert bare.GetNumBonds() == 0

    bonded = utils.add_bonds(bare)
    # methanol has 5 bonds (C-O, O-H, 3x C-H)
    assert bonded.GetNumBonds() == 5


def test_add_bonds_charges_tetravalent_boron(tmp_path):
    """A 4-coordinate boron (borate) must receive a -1 formal charge so the molecule
    passes RDKit sanitization instead of raising "Explicit valence for atom B, 4, is
    greater than permitted". This is the BF2 bridge case in BODIPY dyes; modeled here
    with a tetrahedral BF4- so the test needs no external file.
    """
    path = tmp_path / "bf4.xyz"
    path.write_text(
        "0 1\nBF4\n"
        "B  0.000  0.000  0.000\n"
        "F  0.810  0.810  0.810\n"
        "F -0.810 -0.810  0.810\n"
        "F -0.810  0.810 -0.810\n"
        "F  0.810 -0.810 -0.810\n"
    )
    mol = utils.add_bonds(utils.mol_from_xyz_file(str(path)))
    boron = next(a for a in mol.GetAtoms() if a.GetSymbol() == "B")
    assert boron.GetDegree() == 4
    assert boron.GetFormalCharge() == -1
    Chem.SanitizeMol(mol)  # must not raise


def test_add_bonds_leaves_normal_valences_uncharged():
    """Ordinary (non-over-coordinated) atoms keep a zero formal charge."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))  # methanol: normal C, O valences
    assert AllChem.EmbedMolecule(mol, randomSeed=5) == 0
    rw = Chem.RWMol()
    for atom in mol.GetAtoms():
        rw.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    rw.AddConformer(mol.GetConformer(), assignId=True)
    bonded = utils.add_bonds(rw.GetMol())
    assert all(a.GetFormalCharge() == 0 for a in bonded.GetAtoms())


def test_add_bonds_preserves_conformer_geometry():
    """add_bonds must keep the input geometry (exactly one conformer, same coords).

    Regression guard: the structure viewer only re-embeds when GetNumConformers()==0,
    so a parsed file keeps its REAL coordinates. If add_bonds dropped the conformer,
    the viewer would fall back to a fresh (all-single-bond) embedding that puckers
    planar/aromatic systems into a bogus sp3-looking geometry.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))  # benzene: planar
    assert AllChem.EmbedMolecule(mol, randomSeed=11) == 0

    rw = Chem.RWMol()
    for atom in mol.GetAtoms():
        rw.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    rw.AddConformer(mol.GetConformer(), assignId=True)
    bonded = utils.add_bonds(rw.GetMol())

    assert bonded.GetNumConformers() == 1  # geometry retained, viewer won't re-embed
    src, dst = mol.GetConformer(), bonded.GetConformer()
    for i in range(mol.GetNumAtoms()):
        a, b = src.GetAtomPosition(i), dst.GetAtomPosition(i)
        assert abs(a.x - b.x) < 1e-6 and abs(a.y - b.y) < 1e-6 and abs(a.z - b.z) < 1e-6


# --- mol_from_symbols_and_coords (rebuilding a molecule from a result file) --

def test_mol_from_symbols_and_coords_preserves_order_and_geometry():
    """The orbital viewer rebuilds the molecule from the result file's own geometry.

    Atom order and coordinates have to survive exactly: the cube grid is computed in
    this frame, so a reordered or shifted molecule would draw the isosurface in the
    wrong place relative to the nuclei.
    """
    symbols = ["O", "H", "H"]
    coords = [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]]

    mol = utils.mol_from_symbols_and_coords(symbols, coords)

    assert [a.GetSymbol() for a in mol.GetAtoms()] == symbols
    conformer = mol.GetConformer()
    for index, (x, y, z) in enumerate(coords):
        position = conformer.GetAtomPosition(index)
        assert abs(position.x - x) < 1e-6
        assert abs(position.y - y) < 1e-6
        assert abs(position.z - z) < 1e-6


def test_mol_from_symbols_and_coords_perceives_bonds():
    """A result file stores geometry only, so connectivity has to be inferred."""
    mol = utils.mol_from_symbols_and_coords(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]],
    )
    assert mol.GetNumBonds() == 2


# --- multi-frame XYZ (optimization trajectories) -----------------------------

def test_reading_a_trajectory_returns_the_final_frame(tmp_path):
    """A trajectory holds many frames; reading every line would fuse them all.

    The final frame is the converged geometry, which is what someone clicking the
    trajectory in the file browser wants to see.
    """
    path = tmp_path / "traj.xyz"
    path.write_text(
        "\n".join([
            "2", "Step 0", "H 0.0 0.0 0.0", "H 0.0 0.0 1.0",
            "2", "Step 1", "H 0.0 0.0 0.0", "H 0.0 0.0 0.74",
            "",
        ]),
        encoding="utf-8",
    )

    mol = utils.mol_from_xyz_file(str(path))

    assert mol.GetNumAtoms() == 2, "frames were concatenated instead of read separately"
    assert mol.GetConformer().GetAtomPosition(1).z == pytest.approx(0.74)


def test_standard_xyz_defaults_to_neutral_singlet(tmp_path):
    """A standard XYZ carries no charge/multiplicity, so the neutral defaults apply."""
    path = tmp_path / "std.xyz"
    path.write_text(
        "\n".join(["2", "water fragment", "H 0.0 0.0 0.0", "H 0.0 0.0 0.74", ""]),
        encoding="utf-8",
    )

    mol, charge, multiplicity = utils.mol_from_xyz_file(str(path), return_charge_and_multiplicity=True)

    assert (charge, multiplicity) == (0, 1)
    assert mol.GetNumAtoms() == 2


# --- mol_from_structure_file -------------------------------------------------

def test_unsupported_extension_raises_a_clear_error(tmp_path):
    """A bad extension should say so, not fail later inside RDKit."""
    path = tmp_path / "notes.txt"
    path.write_text("not a structure", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported structure file type"):
        utils.mol_from_structure_file(str(path))


def test_xyz_roundtrip_accepts_integer_valued_slider_floats(tmp_path, water_mol):
    path = str(tmp_path / "charged.xyz")
    utils.conformer_to_xyz_file(water_mol, 0, path, charge=-1.0, multiplicity=2.0)
    _, charge, multiplicity = utils.mol_from_xyz_file(path, return_charge_and_multiplicity=True)
    assert (charge, multiplicity) == (-1, 2)


def test_xyz_trajectory_does_not_splice_an_incomplete_final_frame(tmp_path):
    path = tmp_path / "truncated.xyz"
    path.write_text("2\nStep 0\nH 0 0 0\nH 0 0 1\n2\nStep 1\nH 0 0 0.1\n")
    with pytest.raises(ValueError, match="Incomplete XYZ frame"):
        utils.mol_from_xyz_file(str(path))


def test_xyz_frame_comments_are_never_read_as_atoms(tmp_path):
    path = tmp_path / "comments.xyz"
    path.write_text("2\nO 8 8 8\nH 0 0 0\nH 0 0 1\n1\nO 9 9 9\nHe 1 2 3\n")
    mol = utils.mol_from_xyz_file(str(path))
    assert [atom.GetSymbol() for atom in mol.GetAtoms()] == ["He"]
    assert tuple(mol.GetConformer().GetAtomPosition(0)) == (1, 2, 3)


@pytest.mark.parametrize("contents, message", [
    ("", "empty"),
    ("0\nempty\n", "atom count must be positive"),
    ("2\ncomment\nH 0 0 0\nH 0 0\n", "Invalid XYZ atom line"),
    ("0 1\nH 0 0 0\nH 0 0\n", "Invalid XYZ atom line"),
    ("1\ncomment\nH nan 0 0\n", "must be finite"),
    ("1\ncomment\nH 0 inf 0\n", "must be finite"),
    ("1\ncomment\nUnknown 0 0 0\n", "Unknown XYZ element"),
    ("0 1\nUnknown 0 0 0\nH 0 0 1\n", "Unknown XYZ element"),
    ("0 0\nH 0 0 0\n", "multiplicity must be positive"),
])
def test_invalid_xyz_geometry_is_rejected(tmp_path, contents, message):
    path = tmp_path / "invalid.xyz"
    path.write_text(contents)
    with pytest.raises(ValueError, match=message):
        utils.mol_from_xyz_file(str(path))


def test_bare_xyz_coordinate_block_keeps_first_atom(tmp_path):
    path = tmp_path / "bare.xyz"
    path.write_text("H 0 0 0\nH 0 0 0.74\n")
    assert utils.mol_from_xyz_file(str(path)).GetNumAtoms() == 2


@pytest.mark.parametrize("coords, message", [
    ([[0, 0, 0]], "one coordinate triplet per atom"),
    ([[0, 0, 0], [0, 0]], "one numeric coordinate triplet per atom"),
    ([[0, 0, 0], [0, float("nan"), 1]], "must be finite"),
])
def test_result_geometry_rejects_missing_or_invalid_coordinates(coords, message):
    with pytest.raises(ValueError, match=message):
        utils.mol_from_symbols_and_coords(["H", "H"], coords)
