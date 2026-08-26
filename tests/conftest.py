"""Shared pytest fixtures and path setup for the psi4-webui test suite.

The application modules live in ``src/psi4_webui``, so ``src`` is added to sys.path here
(mirroring ``pythonpath = src`` in pytest.ini) to keep ``from psi4_webui import ...``
working no matter where pytest is invoked from.

The synthetic result-JSON fixtures below are hand-written on purpose. Real calculation
outputs live under ``data/`` and are gitignored, so they must not be relied on as test
inputs; and writing the fixtures by hand is what makes them a check on the *contract*
between the runner and the Result tab rather than a snapshot of whatever the runner
happens to emit today.
"""
import os
import subprocess
import sys

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def psi4_is_available() -> bool:
    """Whether ``import psi4`` succeeds in the interpreter the runner would use.

    Probed in a subprocess rather than with ``importorskip`` because that is exactly how
    the app invokes Psi4, and because importing Psi4 into the test process is slow and
    leaves global state behind.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-c", "import psi4"],
            capture_output=True, timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


@pytest.fixture(scope="session")
def psi4_available():
    """Skip the test unless Psi4 can actually be imported."""
    if not psi4_is_available():
        pytest.skip("psi4 is not installed in this interpreter")
    return True


@pytest.fixture
def water_mol():
    """A 3D water molecule (one conformer), for geometry/coordinate tests."""
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    assert AllChem.EmbedMolecule(mol, randomSeed=0xF00D) == 0
    return mol


# Water at a reasonable HF/STO-3G geometry, in angstrom. Used wherever a test needs a
# concrete geometry without paying for an embedding.
WATER_SYMBOLS = ["O", "H", "H"]
WATER_COORDS = [[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]]

WATER_GEOMETRY = (
    "0 1\n"
    "O     0.00000000     0.00000000     0.11700000\n"
    "H     0.00000000     0.75700000    -0.46900000\n"
    "H     0.00000000    -0.75700000    -0.46900000\n"
    "units angstrom\nno_reorient\nno_com"
)


def _base_result(calculation_type, **extra):
    """A minimal completed result file, with ``extra`` merged in."""
    result = {
        "version": 1,
        "status": "completed",
        "error": None,
        "calculation_type": calculation_type,
        "method_type": "HF",
        "method": "SCF",
        "functional": "B3LYP",
        "basis_set": "STO-3G",
        "reference": "RHF",
        "charge": 0,
        "multiplicity": 1,
        "structure_file": "water.xyz",
        "solvent": None,
        "log_file": "job.log",
        "symbols": WATER_SYMBOLS,
        "geometry_angstrom": WATER_COORDS,
        "energy_hartree": -74.96294665,
        "dipole_debye": 1.7257,
        "dipole_vector_debye": [0.0, 0.0, -1.7257],
        "mo_energies_hartree": [-20.24, -1.27, -0.62, -0.45, -0.39, 0.60, 0.74],
        "homo_index": 4,
        "n_alpha": 5,
        "n_beta": 5,
        "wavefunction_file": "job.npy",
        "duration_seconds": 1.2,
    }
    result.update(extra)
    return result


@pytest.fixture
def single_point_result():
    """A completed single-point result."""
    return _base_result("Single-Point")


@pytest.fixture
def optimization_result():
    """A completed geometry optimization, with a monotonically falling energy trace."""
    return _base_result(
        "Geometry Optimization",
        optimization={
            "energies_hartree": [-74.9554, -74.9650, -74.9657, -74.9659, -74.9659],
            "n_steps": 5,
        },
        optimized_xyz_file="job_optimized.xyz",
        trajectory_xyz_file="job_trajectory.xyz",
    )


@pytest.fixture
def frequency_result():
    """A completed frequency job: three real modes and full thermochemistry."""
    return _base_result(
        "Frequency",
        frequency={
            "frequencies_cm1": [2170.79, 4138.69, 4389.25],
            "ir_intensities_km_mol": [7.21, 44.35, 30.05],
            "n_imaginary": 0,
        },
        thermochemistry={
            "temperature_k": 298.15,
            "pressure_pa": 101325.0,
            "zpve_hartree": 0.0243735,
            "thermal_energy_hartree": -74.9386947,
            "enthalpy_hartree": -74.9377506,
            "gibbs_hartree": -74.9592658,
            "entropy_cal_mol_k": 45.485,
        },
    )


@pytest.fixture
def saddle_point_result(frequency_result):
    """A frequency job at a saddle point: one imaginary mode, stored as a negative."""
    frequency_result["frequency"] = {
        "frequencies_cm1": [-911.02, 2170.79, 4138.69],
        "ir_intensities_km_mol": [12.5, 7.21, 44.35],
        "n_imaginary": 1,
    }
    return frequency_result


@pytest.fixture
def tddft_result():
    """A completed TD-DFT job with three excited states, two of them ECD-active."""
    return _base_result(
        "Time-Dependent Density Functional Theory",
        excitations=[
            {"index": 1, "energy_hartree": 0.4835, "energy_ev": 13.16, "wavenumber_cm1": 106140.0,
             "wavelength_nm": 94.2, "oscillator_strength": 0.0033, "rotatory_strength": 1.5,
             "spin": "singlet", "symmetry": "1 B1"},
            {"index": 2, "energy_hartree": 0.5566, "energy_ev": 15.15, "wavenumber_cm1": 122160.0,
             "wavelength_nm": 81.9, "oscillator_strength": 0.0, "rotatory_strength": -2.0,
             "spin": "singlet", "symmetry": "1 A2"},
            {"index": 3, "energy_hartree": 0.6125, "energy_ev": 16.67, "wavenumber_cm1": 134430.0,
             "wavelength_nm": 74.4, "oscillator_strength": 0.0664, "rotatory_strength": 0.0,
             "spin": "singlet", "symmetry": "2 A1"},
        ],
    )


@pytest.fixture
def emission_result():
    """A completed emission job: a relaxed S1 geometry and the gap back down to S0."""
    return _base_result(
        "Emission (EOM-CCSD)",
        method_type="EOM-CCSD",
        method="eom-ccsd",
        emission={
            "root": 1,
            "excited_energy_hartree": -112.39733817,
            "ground_energy_at_excited_geometry_hartree": -112.47758000,
            "emission_energy_hartree": 0.08024183,
            "emission_energy_ev": 2.1837,
            "wavenumber_cm1": 17612.0,
            "wavelength_nm": 567.8,
        },
        optimized_xyz_file="job_optimized.xyz",
    )


@pytest.fixture
def failed_result():
    """A result file recording a calculation that raised."""
    return {
        "version": 1,
        "status": "failed",
        "error": "ValidationError: BASIS (NOSUCHBASIS) is not a recognized basis set",
        "traceback": "Traceback (most recent call last): ...",
        "calculation_type": "Single-Point",
        "log_file": "job.log",
        "duration_seconds": 0.4,
    }
