"""Core, UI-free engine for psi4-webui.

Groups three responsibilities used by the Gradio tabs:

1. Molecule I/O (``mol_from_*``, ``add_bonds``, ``conformer_to_xyz_file``).
2. Psi4 job specs: the small JSON contract between the Calculation tab (which writes a
   spec) and :mod:`psi4_webui.runner` (which executes it in a child process) and the
   Result tab (which reads the results the runner wrote back).
3. Spectrum simulation (IR, UV-Vis absorption, ECD) as Plotly figures.

Everything here is pure Python with no Gradio dependency -- and, importantly, **no
``psi4`` import**: this module is loaded by the web server, which must start and remain
usable even where Psi4 is missing. Psi4 is imported only inside the runner subprocess.
"""
from __future__ import annotations

import json
import os
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import plotly.graph_objects as go

# Suffixes of the two JSON artifacts the runner contract is built on. A job spec is what
# the Calculation tab writes and the runner consumes; a result file is what the runner
# writes and the Result tab consumes. They are distinct suffixes (rather than one
# directory of ``.json``) so the file browser can label them and each tab's dropdown can
# filter to just the kind it accepts.
JOB_SUFFIX = ".job.json"
RESULT_SUFFIX = ".result.json"

# Calculation types. Module-level constants because the Calculation tab, the runner and
# the Result tab all branch on them and a typo in any one of the three would silently
# mis-route a job.
SINGLE_POINT = "Single-Point"
GEOMETRY_OPTIMIZATION = "Geometry Optimization"
FREQUENCY = "Frequency"
TDDFT = "Time-Dependent Density Functional Theory"
EMISSION = "Emission (EOM-CCSD)"
CALCULATION_TYPES = [SINGLE_POINT, GEOMETRY_OPTIMIZATION, FREQUENCY, TDDFT, EMISSION]

def get_files_in_working_directory(working_directory_path: str) -> list[str]:
    """Return the file names in ``working_directory_path``, or ``[]`` if it is unset.

    Windows ``Zone.Identifier`` alternate-data-stream files (created when files are
    downloaded) are filtered out so they never appear in the UI file list.

    Subdirectories are excluded: the Psi4 runner keeps its scratch in ``scratch/`` and
    cube files in ``cubes/`` inside the working directory, and listing those as if they
    were files would offer them to the viewers and the delete button, neither of which
    can handle a directory.

    The ``None`` guard matters: the shared path ``gr.State`` starts empty and is reset
    to ``None`` when a directory fails to open, and ``os.listdir(None)`` quietly lists
    the *current* directory instead of raising — which surfaced as callers building
    paths out of a ``None`` directory.
    """
    if not working_directory_path:
        return []
    return [f for f in os.listdir(working_directory_path)
            if not f.endswith('Zone.Identifier')
            and os.path.isfile(os.path.join(working_directory_path, f))]

def conformer_to_xyz_file(mol: Chem.Mol, conf_id: int, file_path: str,
                          charge: int = 0, multiplicity: int = 1) -> None:
    """Write one conformer of ``mol`` to an XYZ file.

    The first line is ``"<charge> <multiplicity>"`` (the convention this app's XYZ
    reader expects) rather than the standard atom count, followed by one
    ``"<symbol> <x> <y> <z>"`` line per atom taken from conformer ``conf_id``.
    """
    # Get atom information
    atoms = mol.GetAtoms()
    xyz_lines = []
    for atom in atoms:
        pos = mol.GetConformer(conf_id).GetAtomPosition(atom.GetIdx())
        xyz_lines.append(f"{atom.GetSymbol()} {pos.x} {pos.y} {pos.z}")

    # Construct the XYZ string
    xyz_string = f"{charge} {multiplicity}\n" + "\n".join(xyz_lines)
    
    with open(file_path, 'w') as file:
        file.write(xyz_string)

def add_bonds(mol: Chem.Mol, bond_factor: float = 1.25) -> Chem.Mol:
    """Return a copy of ``mol`` with bonds inferred from interatomic distances.

    Structure files such as XYZ carry only atoms and coordinates, so RDKit sees no
    connectivity. Two atoms are joined by a single bond when their distance is below
    ``(r_cov_i + r_cov_j) * bond_factor`` using RDKit's covalent radii. Bond orders
    are not perceived (all bonds are single); this is enough for visualization and
    for feeding coordinates into ORCA input generation.

    Formal charges are assigned to hypervalent main-group atoms so the result passes
    ``Chem.SanitizeMol`` — currently 4-coordinate boron (see ``_assign_formal_charges``),
    e.g. the ``BF2`` bridge in BODIPY dyes. These are per-atom bookkeeping charges only;
    the overall charge used for an ORCA calculation is set separately by the user.
    """
    # Create a new empty molecule
    mol_new = Chem.RWMol()

    # Add atoms
    for atom in mol.GetAtoms():
        mol_new.AddAtom(atom)

    # Add conformer
    conf = mol.GetConformer()
    mol_new.AddConformer(conf)

    # Add bonds based on covalent radii
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            dist = np.linalg.norm(np.array(conf.GetAtomPosition(i)) - np.array(conf.GetAtomPosition(j)))
            pt = Chem.GetPeriodicTable()
            r_cov_i = pt.GetRcovalent(mol.GetAtomWithIdx(i).GetSymbol())
            r_cov_j = pt.GetRcovalent(mol.GetAtomWithIdx(j).GetSymbol())
            if dist < (r_cov_i + r_cov_j) * bond_factor:
                mol_new.AddBond(i, j, Chem.BondType.SINGLE)

    # Reconcile valences that would otherwise fail sanitization (e.g. borate boron).
    _assign_formal_charges(mol_new)

    # Convert to a regular Mol object
    mol_new = mol_new.GetMol()

    return mol_new

# Neutral bonding valence for main-group elements whose over-coordination (with the
# all-single-bond perception used here) implies a formal charge. degree - valence gives
# the charge: 4-coordinate boron -> -1 (borate), a 4th bond on N -> +1 (ammonium), etc.
_NEUTRAL_VALENCE = {'B': 3, 'N': 3, 'O': 2}

def _assign_formal_charges(mol: Chem.RWMol) -> None:
    """Assign formal charges so over-coordinated main-group atoms pass sanitization.

    Only atoms bonded to MORE neighbors than their neutral valence are touched, and the
    charge is the difference (boron gains electrons -> negative; N/O lose -> positive).
    This fixes structures like BODIPY's 4-coordinate borate boron, which RDKit otherwise
    rejects with "Explicit valence for atom B, 4, is greater than permitted."
    """
    for atom in mol.GetAtoms():
        neutral = _NEUTRAL_VALENCE.get(atom.GetSymbol())
        if neutral is None:
            continue
        excess = atom.GetDegree() - neutral
        if excess > 0:
            # Group 13 (B) is electron-deficient: an extra bond makes it anionic;
            # groups 15/16 (N, O) become cationic.
            sign = -1 if atom.GetSymbol() == 'B' else 1
            atom.SetFormalCharge(sign * excess)

def mol_from_xyz_file(file_path: str, return_charge_and_multiplicity: bool = False):
    """Read an XYZ file into a bond-free RDKit ``Mol`` with one conformer.

    Handles both flavours the app deals with:

    * The first line is ``"<charge> <multiplicity>"`` (what
      :func:`conformer_to_xyz_file` writes, and what Psi4 geometry blocks look like).
    * Standard XYZ, whose first line is an atom count. These may hold **many frames**
      concatenated (an optimization trajectory), in which case the *last* frame is
      returned -- for a trajectory that is the converged geometry, which is what someone
      clicking the file in the browser wants to see. Reading every line as an atom would
      otherwise fuse all the frames into one nonsense molecule.

    Anything else defaults to charge/multiplicity ``0``/``1``. Only lines with exactly
    four whitespace-separated fields (symbol + x/y/z) are treated as atoms.

    Returns the ``Mol`` alone, or ``(mol, charge, multiplicity)`` when
    ``return_charge_and_multiplicity`` is True.
    """
    with open(file_path, 'r') as file:
        xyz_string = file.read()

    lines = xyz_string.split('\n')
    charge, multiplicity = 0, 1
    first_fields = lines[0].split()
    if len(first_fields) == 2:
        try:
            charge, multiplicity = int(first_fields[0]), int(first_fields[1])
        except ValueError:
            charge, multiplicity = 0, 1
    elif len(first_fields) == 1:
        # Standard XYZ: keep only the trailing ``n_atoms`` atom lines, i.e. the last frame.
        try:
            n_atoms = int(first_fields[0])
        except ValueError:
            n_atoms = 0
        if n_atoms > 0:
            atom_lines = [line for line in lines if len(line.split()) == 4]
            if len(atom_lines) >= n_atoms:
                lines = [""] + atom_lines[-n_atoms:]

    # Create a new empty molecule
    mol = Chem.RWMol()

    # Parse the atomic coordinates and add atoms
    elements = []
    for line in lines[1:]:
        if len(line.split())==4:
            parts = line.split()
            element = parts[0].capitalize()
            atom = Chem.Atom(element)
            mol.AddAtom(atom)
            elements.append(element)
        else:
            continue

    # Add 3D coordinates to the molecule
    conf = Chem.Conformer(mol.GetNumAtoms())
    atom_idx = 0
    for line in lines[1:]:
        if len(line.split())==4:
            parts = line.split()
            x, y, z = map(float, parts[1:4])
            conf.SetAtomPosition(atom_idx, (x, y, z))
            atom_idx += 1
        else:
            continue
    mol.AddConformer(conf)
                
    # Convert to a regular Mol object
    mol = mol.GetMol()
    
    if return_charge_and_multiplicity:
        return mol, charge, multiplicity
    else:
        return mol

def mol_from_structure_file(file_path: str) -> Chem.Mol:
    """Load any structure file the UI accepts into a sanitized ``Mol`` with a conformer.

    Shared by the structure viewer and the Calculation tab so that both see the same
    atoms in the same order. That ordering is load-bearing: the viewer labels each atom
    with its index, and :func:`psi4_geometry_string` emits the coordinate block in that
    same index order, so two different loaders here would make the on-screen labels
    disagree with the geometry Psi4 actually receives.

    Raises ``ValueError`` for an unsupported extension rather than returning ``None``,
    which would otherwise surface as an opaque RDKit error further down.
    """
    if file_path.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(file_path, sanitize=False, removeHs=False)
    elif file_path.endswith('.mol'):
        mol = Chem.MolFromMolFile(file_path, sanitize=False, removeHs=False)
    elif file_path.endswith('.mol2'):
        mol = Chem.MolFromMol2File(file_path, sanitize=False, removeHs=False)
    elif file_path.endswith('.xyz'):
        mol = add_bonds(mol_from_xyz_file(file_path))
    else:
        raise ValueError(f"unsupported structure file type: {os.path.basename(file_path)}")

    if mol is None:
        raise ValueError(f"could not read structure file: {os.path.basename(file_path)}")

    Chem.SanitizeMol(mol)
    # Display/use the file's REAL coordinates. Only embed a fresh conformer if the file
    # somehow carried none — re-embedding otherwise would discard the parsed (e.g.
    # Psi4-optimized) geometry and, because add_bonds perceives every bond as single,
    # pucker planar/aromatic systems into a bogus sp3-looking shape.
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol)

    return mol

def lorentzian_ir(wavenumber, position, intensity, width: float = 10):
    """Lorentzian band shape (peak height = ``intensity`` at ``position``).

    ``wavenumber`` may be a scalar or a numpy array (vectorized over the x-axis).
    ``width`` is the half-width at half-maximum in cm^-1.
    """
    return intensity * (width ** 2) / ((wavenumber - position) ** 2 + width ** 2)

def generate_ir_spectrum_interactive(
    frequencies,
    intensities,
    width: float = 10.0,
    points: int = 4000,
    plot_range: tuple[float, float] | None = None,
    normalize: bool = True,
    transmittance: bool = True,
) -> go.Figure | None:
    """Build an interactive IR spectrum from vibrational frequencies + IR intensities.

    Each mode is broadened with a Lorentzian (:func:`lorentzian_ir`) and summed onto
    a wavenumber grid. Optionally normalized and converted to transmittance
    (``1 - absorbance``, with the x-axis reversed per IR convention). Returns a Plotly
    figure, or ``None`` when no frequencies are supplied.
    """
    # Safety check
    if frequencies is None or len(frequencies) == 0:
        print("No IR frequencies detected.")
        return None

    # Define plotting range
    if plot_range:
        min_wn, max_wn = plot_range
    else:
        min_wn = np.min(frequencies) - 100
        max_wn = np.max(frequencies) + 100

    x = np.linspace(min_wn, max_wn, points)
    spectrum = np.zeros_like(x)

    # Generate Lorentzian broadened spectrum
    annotations = []
    for freq, inten in zip(frequencies, intensities):
        spectrum += lorentzian_ir(x, freq, inten, width)

        annotations.append({
            "freq": freq,
            "intensity": inten,
            "text": f"{freq:.1f}"
        })

    # Normalize if requested
    if normalize and np.max(spectrum) != 0:
        spectrum /= np.max(spectrum)

    # Convert to transmittance if requested
    if transmittance:
        spectrum = 1 - spectrum  # simple inversion

    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=spectrum,
        mode='lines',
        line=dict(color='black'),
        name="IR Spectrum"
    ))

    # Add peak annotations
    for ann in annotations:
        fig.add_annotation(
            x=ann["freq"],
            y=max(spectrum),
            text=ann["text"],
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-30,
            textangle=-90,
            font=dict(color='red', size=10)
        )

    # Layout
    fig.update_layout(
        title="IR Spectrum",
        xaxis=dict(
            title="Wavenumber (cm⁻¹)",
            autorange="reversed"  # Important for IR
        ),
        yaxis=dict(
            title="Transmittance" if transmittance else "Intensity (a.u.)"
        ),
        showlegend=False
    )

    # Apply manual range if provided
    if plot_range:
        fig.update_xaxes(range=plot_range[::-1])  # reverse manually

    return fig

def gaussian(x, x0, fwhm: float):
    """Unit-height Gaussian centered at ``x0`` with the given full width at half max.

    ``x`` may be a scalar or numpy array (vectorized over the x-axis).
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-(x - x0)**2 / (2 * sigma**2))

def generate_absorption_emission_spectrum_interactive(
    wavelengths, oscs, points: int = 10000,
    plot_range: tuple[float, float] | None = None,
) -> go.Figure | None:
    """Build an interactive UV-Vis spectrum from transition wavelengths + oscillator strengths.

    Each transition is broadened with a Gaussian whose width grows with its oscillator
    strength, weighted by that strength, and summed onto a wavelength grid. Returns a
    normalized Plotly figure, or ``None`` when no transitions are supplied.
    """
    # Check if wavelengths and oscs are valid
    if wavelengths is None or len(wavelengths) == 0:
        print(f"No absorption peak detected.")
        return
    
    # Create the chemical shift axis
    if plot_range:
        min_wavelength, max_wavelength = plot_range
    else:
        min_wavelength = np.min(wavelengths) - 10
        max_wavelength = np.max(wavelengths) + 10
    
    x = np.linspace(min_wavelength, max_wavelength, points)
    
    # Initialize the spectrum
    spectrum = np.zeros_like(x)
    
    # Generate annotations for each peak with Gaussian broadening
    annotations = []
    base_fwhm = 10.0
    for wavelength, osc in zip(wavelengths, oscs):
        fwhm = base_fwhm * (1 + osc)   # stronger osc → broader peak
        spectrum += osc * gaussian(x, wavelength, fwhm)
        annotations.append({
            'wavelength': wavelength,
            'oscilation strength': oscs,
            'text': f"{wavelength:.2f}",
        })

    # Normalize the spectrum (guard against all-zero oscillator strengths)
    max_intensity = np.max(spectrum)
    if max_intensity != 0:
        spectrum /= max_intensity

    # Create the Plotly figure
    fig = go.Figure()

    # Add the spectrum trace
    fig.add_trace(go.Scatter(
        x=x,
        y=spectrum,
        mode='lines',
        line=dict(color='black'),
        name=f'Spectrum'
    ))
    
    # Add peak annotations
    for ann in annotations:
        fig.add_annotation(
            x=ann['wavelength'],
            y=1.05,
            text=ann['text'],
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            yshift=10,
            textangle=-90,
            font=dict(color='red'),
        )

    fig.update_layout(
        xaxis=dict(
            title='Wavelength (nm)',
        ),
        yaxis=dict(
            title='Relative Intensity (a.u.)',
        ),
        title=f'Spectrum',
        showlegend=False,
    )

    # Adjust plot range if specified
    if plot_range:
        fig.update_xaxes(range=plot_range)

    return fig

def generate_ecd_spectrum_interactive(
    wavelengths, rotatory_strengths, points: int = 10000,
    plot_range: tuple[float, float] | None = None,
    fwhm_wavenumber: float = 3000.0,
) -> go.Figure | None:
    """Build an interactive ECD spectrum from wavelengths + rotatory strengths.

    Differs from :func:`generate_absorption_emission_spectrum_interactive` in two ways
    that matter physically:

    * Rotatory strengths are **signed**. They are summed with their sign so positive and
      negative Cotton effects appear above and below zero, and neighbouring bands of
      opposite sign partially cancel. Taking magnitudes here would destroy the
      information that makes ECD useful (two enantiomers differ only by the sign of
      every R, so their spectra are mirror images).
    * Broadening is done in **wavenumber space**, where a Gaussian band shape is the
      physically appropriate model, and only the display axis is wavelength. Broadening
      with a constant width directly in nm would make bands artificially narrow at short
      wavelengths and wide at long ones.

    ``fwhm_wavenumber`` is the band width in cm^-1 (~3000 cm^-1 is a conventional
    starting point for ECD). The curve is scaled by its largest absolute value, so the
    y-axis is relative and signed, in [-1, 1]. Returns ``None`` when no transitions are
    supplied.
    """
    if wavelengths is None or len(wavelengths) == 0:
        print("No ECD transitions to plot.")
        return None

    wavelengths = np.asarray(wavelengths, dtype=float)
    rotatory_strengths = np.asarray(rotatory_strengths, dtype=float)

    if plot_range:
        min_wavelength, max_wavelength = plot_range
    else:
        min_wavelength = np.min(wavelengths) - 50
        max_wavelength = np.max(wavelengths) + 50
    # The nm -> cm^-1 conversion below divides by the axis, so it must stay positive.
    min_wavelength = max(float(min_wavelength), 1.0)
    max_wavelength = max(float(max_wavelength), min_wavelength + 1.0)

    x = np.linspace(min_wavelength, max_wavelength, points)
    grid_wavenumbers = 1e7 / x

    spectrum = np.zeros_like(x)
    annotations = []
    for wavelength, rotatory_strength in zip(wavelengths, rotatory_strengths):
        if wavelength <= 0:
            continue
        spectrum += rotatory_strength * gaussian(grid_wavenumbers, 1e7 / wavelength, fwhm_wavenumber)
        annotations.append({
            'wavelength': wavelength,
            'rotatory_strength': rotatory_strength,
            'text': f"{wavelength:.2f}",
        })

    # Scale by the largest absolute value so the sign is preserved.
    max_intensity = np.max(np.abs(spectrum))
    if max_intensity != 0:
        spectrum /= max_intensity

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=spectrum,
        mode='lines',
        line=dict(color='black'),
        name='ECD',
    ))
    # ECD crosses zero, so the baseline is a meaningful reference and is drawn in.
    fig.add_hline(y=0, line_width=1, line_color='grey')

    # Annotate each transition on the side its own sign points to.
    for ann in annotations:
        above = ann['rotatory_strength'] >= 0
        fig.add_annotation(
            x=ann['wavelength'],
            y=1.05 if above else -1.05,
            text=ann['text'],
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40 if above else 40,
            yshift=10 if above else -10,
            textangle=-90,
            font=dict(color='red' if above else 'blue'),
        )

    fig.update_layout(
        xaxis=dict(title='Wavelength (nm)'),
        # Symmetric range so a mirror-image (opposite enantiomer) spectrum is directly
        # comparable, and so the zero crossing sits in the middle.
        yaxis=dict(title='Relative Δε (a.u.)', range=[-1.3, 1.3]),
        title='ECD Spectrum',
        showlegend=False,
    )

    if plot_range:
        fig.update_xaxes(range=[min_wavelength, max_wavelength])

    return fig

# ---------------------------------------------------------------------------
# Psi4 job specs
#
# The Calculation tab never builds a Psi4 *input file* -- there is nothing for a user to
# author or edit. It writes a machine-readable job spec and the runner turns that into
# direct psi4.energy/optimize/frequency API calls. The spec exists only because the
# calculation runs in a child process (so it can be cancelled, and so a Psi4 abort
# cannot take the web server down with it); it is an implementation detail that happens
# to be visible in the file browser, not a format anyone is expected to write by hand.
# ---------------------------------------------------------------------------

# Method families offered in the UI. "DFT" is the only one that consumes a functional.
METHOD_TYPES = ["HF", "DFT", "MP2", "CCSD", "CCSD(T)"]

# Emission is a special case: it needs the gradient of an *excited* state, and EOM-CCSD is
# the only method in Psi4 that has one -- there are no TD-DFT/TDSCF gradients at all. So
# the Emission calculation type offers no method choice; this is the method.
EMISSION_METHOD_TYPE = "EOM-CCSD"

# Psi4 spells the SCF method "SCF"; the correlated methods are named directly. DFT is
# requested by passing the functional name itself as the method.
_METHOD_KEYWORDS = {"HF": "SCF", "MP2": "MP2", "CCSD": "CCSD", "CCSD(T)": "CCSD(T)",
                    EMISSION_METHOD_TYPE: "eom-ccsd"}


def psi4_method_keyword(method_type: str, functional: str) -> str:
    """Return the string Psi4's driver takes as its ``name`` argument.

    DFT is requested by naming the functional (``psi4.energy("B3LYP")``); every other
    family has a fixed keyword. Unknown types fall through to the type itself so a
    method added to the UI still reaches Psi4 rather than being silently dropped.
    """
    if method_type == "DFT":
        return functional
    return _METHOD_KEYWORDS.get(method_type, method_type)


def psi4_geometry_string(mol: Chem.Mol, charge: int, multiplicity: int,
                        force_c1: bool = False) -> str:
    """Build the geometry block Psi4's ``psi4.geometry()`` parses.

    Atom order follows ``mol``'s atom order, which is the order the structure viewer
    labels on screen. ``no_reorient``/``no_com`` are set so the coordinates Psi4 works
    in match the ones the user saw: without them Psi4 shifts the molecule to its centre
    of mass and rotates it onto its principal axes, and every geometry written back out
    (optimized structures, trajectories, cube grids) would be in that rotated frame
    instead of the input frame.

    ``force_c1`` drops symmetry, which an excited-state optimization needs. Psi4 selects
    excited roots *per irrep*, so if the molecule distorts out of its starting point group
    partway through the optimization -- which is exactly what excited states tend to do --
    the root being followed stops meaning what it meant, and Psi4 aborts with "Point group
    changed!". Symmetry only makes the calculation faster, so it is left on everywhere else.
    """
    conformer = mol.GetConformer()
    lines = [f"{int(charge)} {int(multiplicity)}"]
    for atom in mol.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol()} {position.x:>14.8f} {position.y:>14.8f} {position.z:>14.8f}")
    lines.append("units angstrom")
    lines.append("no_reorient")
    lines.append("no_com")
    if force_c1:
        lines.append("symmetry c1")
    return "\n".join(lines)


def build_job_spec(
    *,
    calculation_type: str,
    geometry: str,
    structure_file: str,
    method_type: str,
    functional: str,
    basis_set: str,
    reference: str,
    charge: int,
    multiplicity: int,
    n_threads: int,
    memory_gb: float,
    geom_maxiter: int = 50,
    g_convergence: str = "QCHEM",
    temperature: float = 298.15,
    pressure: float = 101325.0,
    n_states: int = 10,
    tda: bool = False,
    root: int = 1,
    use_solvation: bool = False,
    solvent: str = "Water",
    save_wavefunction: bool = True,
) -> dict[str, Any]:
    """Assemble the JSON-serializable job spec handed to the runner.

    Keyword-only because the argument list is long and homogeneous (several ints, several
    strings) — positional calls would be easy to mis-order and hard to spot.
    """
    return {
        "version": 1,
        "calculation_type": calculation_type,
        "geometry": geometry,
        "structure_file": structure_file,
        "method_type": method_type,
        "method": psi4_method_keyword(method_type, functional),
        "functional": functional,
        "basis_set": basis_set,
        "reference": reference,
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "n_threads": int(n_threads),
        "memory_gb": float(memory_gb),
        "geom_maxiter": int(geom_maxiter),
        "g_convergence": g_convergence,
        "temperature": float(temperature),
        "pressure": float(pressure),
        "n_states": int(n_states),
        "tda": bool(tda),
        "root": int(root),
        "use_solvation": bool(use_solvation),
        "solvent": solvent,
        "save_wavefunction": bool(save_wavefunction),
    }


def write_json(file_path: str, payload: dict[str, Any]) -> None:
    """Write ``payload`` as UTF-8 JSON.

    ``default=str`` is a backstop: the runner builds results out of numpy scalars in a
    few places, and a stray ``np.float64`` would otherwise abort a finished calculation
    at the very last step, throwing away hours of compute over a formatting detail.
    """
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def read_json(file_path: str) -> dict[str, Any]:
    """Read a UTF-8 JSON file written by :func:`write_json`."""
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def result_base_name(file_name: str) -> str:
    """Strip the ``.result.json`` / ``.job.json`` suffix from a job artifact name.

    ``os.path.splitext`` would only remove ``.json`` and leave the ``.result`` behind, so
    the sibling ``.log`` / ``.npy`` of a job could not be located from its result file.
    """
    for suffix in (RESULT_SUFFIX, JOB_SUFFIX):
        if file_name.endswith(suffix):
            return file_name[: -len(suffix)]
    return os.path.splitext(file_name)[0]


def mol_from_symbols_and_coords(symbols, coords) -> Chem.Mol:
    """Build a bonded RDKit ``Mol`` from element symbols and angstrom coordinates.

    Used to rebuild the molecule from a result file for the orbital viewer, so the atoms
    drawn are in exactly the frame the cube grid was computed on. Bonds are perceived by
    :func:`add_bonds` because a result file stores geometry only, not connectivity.
    """
    mol = Chem.RWMol()
    for symbol in symbols:
        mol.AddAtom(Chem.Atom(str(symbol).capitalize()))

    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, (x, y, z) in enumerate(coords):
        conformer.SetAtomPosition(index, (float(x), float(y), float(z)))
    mol.AddConformer(conformer)

    return add_bonds(mol.GetMol())
