# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Overview

A Gradio web UI wrapping [Psi4](https://psicode.org/) for simple computational chemistry
workflows: conformer generation, then Single-Point / Geometry Optimization / Frequency /
TD-DFT / Emission calculations, then result visualization (energies, MOs, IR spectra, UV-Vis and ECD
spectra, and orbital/density isosurfaces).

Psi4 is driven through its **Python API**, not through input files. There is deliberately
no "generate an input file, then run it" step: the Calculation tab takes the structure and
the settings on screen straight to a calculation.

Psi4 is **not** a declared dependency — it is conda-only (`conda install psi4 -c
conda-forge`) and cannot be installed from PyPI, so declaring it would make the wheel
uninstallable. Everything except actually running a calculation works without it.

## Packaging & layout

A pip-installable package (import name `psi4_webui`) using a **src layout**: all modules
live under `src/psi4_webui/`. Metadata is in `pyproject.toml` (setuptools backend);
`styles.css` ships as package data. The `psi4-webui` console command maps to
`psi4_webui.app:main`, which serves on the first free port at/after 7860.

## Commands

```bash
conda create -p ./psi4-env python=3.12 && conda activate ./psi4-env
conda install psi4 -c conda-forge
pip install -e ".[dev]"

psi4-webui                                # or: python -m psi4_webui.app

pytest                                    # whole suite
pytest -m "not psi4"                      # skip tests that invoke real Psi4
python -m build && python -m twine check dist/*
```

`data/` (working directories) and `static/` (transient viewer HTML) are created relative
to the **current working directory**, never the install location.

## Architecture

FastAPI host with a Gradio Blocks UI mounted at `/` and a `StaticFiles` mount at
`/static`, assembled in `app.py`'s `build_app()`. The `/static` mount exists only because
nglview renders to standalone HTML that has to be embedded in an `<iframe>`.

### The execution model — this is the load-bearing design decision

Calculations run in a **child process** (`runner.py`), launched as
`python -m psi4_webui.runner run <job.json>`. `runner.py` is the only module that imports
`psi4`. Three reasons, all of which an in-process thread fails:

1. **Cancellation.** Psi4's driver is a long C++ call with no interrupt hook. The only
   reliable Stop is killing the process tree.
2. **Crash isolation.** Psi4 `abort()`s the process on some failures. In-process that
   takes down the web server and every browser session with it.
3. **Clean state.** Psi4's options, active molecule and scratch are process-global, so
   consecutive runs in one process leak settings into each other.

The job spec written into the working directory is the *argument to the child*, not an
input file — nobody is expected to read or edit it. Two consequences worth knowing:

- **Paths handed to the child must be absolute.** Working directories are named
  relatively (`./data/wd`) and the child is started *in* that directory, so a relative
  path gets resolved against itself twice.
- **The runner always writes a result file**, success or failure, so the Result tab can
  explain what happened. Exit 0 = completed, 1 = the calculation raised, 2 = bad usage.

`on_run_calculation` is a **generator handler**: it yields `(status, log_tail, file_list)`
every 0.5 s so the Psi4 log streams into the page while the calculation runs.

Running processes are tracked in `_processes`, keyed by **working directory** rather than
in a single global, so two sessions in two directories can run at once and Stop hits the
right one. `stop_button.click(..., concurrency_limit=None)` is load-bearing: without it
the Stop event queues behind the run it is meant to interrupt.

### The result JSON is the contract

`result.py` reads `<name>.result.json` — never the Psi4 log. Because this app owns both
producer and consumer there is no output-parsing layer to go stale, and some of what is
displayed (IR intensities, TD-DFT rotatory strengths) exists only on the live wavefunction
and is never printed. Units are canonical in the file (hartree, Å, cm⁻¹, km/mol, Debye);
display conversion happens in the UI.

When changing the schema, change `runner.py` and `result.py` together, and update the
synthetic fixtures in `tests/conftest.py` — those fixtures are the executable spec.

### Working-directory model

Everything is organized around a working directory: a subfolder of `data/` selected in the
left column. `working_directory.py` owns that column and the file browser/viewers.

State flows through `gr.State` objects created in `working_directory_blocks()`:

- `working_directory_path_state` — the current path.
- `working_directory_file_list_state` — **the central event bus.** Any handler that
  changes files on disk returns a fresh `get_files_in_working_directory(...)` into it; its
  `.change` event fires each tab's `on_working_directory_file_list_change`, which is how
  every dropdown stays in sync. When adding an operation that writes files, return the
  refreshed list into this state.
- `status_markdown` — a shared status line; handlers return coloured HTML spans.

### UI structure (three tabs, one file each)

- `conformer_generation.py` — SMILES (or a `gradio_molecule2d` sketch) → RDKit embeds
  candidates in oversampled rounds → MMFF/UFF minimize → discard duplicates (close in
  energy **and** superimposable) → write `.xyz`/`.pdb`/`.mol` per conformer.
- `calculation.py` — structure + method + type → job spec → run the child, streaming its
  log. The calculation-type radio drives which control groups are visible.
- `result.py` — loads a result file and conditionally reveals accordions (Energy,
  Geometry Optimization, Frequency, Absorption/ECD, Orbitals & Density) based on what the
  calculation produced. `on_load_result_file` returns one positional tuple that must stay
  index-aligned with `_result_outputs`; `RESULT_OUTPUT_COUNT` and a test in
  `test_result_loading.py` enforce it.
- `visualization.py` — nglview rendering, shared by the structure viewer and the cube
  viewer.
- `utils.py` — the UI-free engine: molecule I/O, job-spec building, spectrum simulation
  (Plotly). Deliberately does **not** import `psi4`, so the server starts without it.

### nglview rendering

Two non-obvious details, both enforced by `tests/test_visualization.py`:

- **`isolevelType="value"`.** NGL's surface representation defaults to `"sigma"`, which
  multiplies the given number by the RMS of the grid. A chemist's 0.02 a.u. contour then
  silently becomes ~0.0008 a.u. and renders a huge diffuse blob clipped by the grid box —
  which is what made the two lobes of a symmetric orbital appear unequal.
- **Transient widgets must be unregistered.** `nglview.write_html` reaches ipywidgets'
  `embed_data`, which serializes *every widget still registered in the process*. Without
  `_transient_widgets()` each render embeds all its predecessors: ~1 MB of growth per
  render, plus stale isosurfaces from earlier selections. Unregister, do not `close()` —
  closing severs the comm while leaving the object referenced, and the next render fails.

### Orbital/density visualization

The runner saves `wfn.to_file()` as `<name>.npy`. The Result tab then runs a second,
short-lived `runner cubeprop` child on demand, so any orbital can be inspected after the
fact without re-running the calculation or generating every cube up front. Cube output
directories are wiped before each render, so file discovery is unambiguous.

## Psi4 gotchas that shaped this code

Each of these was verified against Psi4 1.11 and is enforced by a test:

- **`psi4.core.set_output_file(path, False)`, never `psi4.set_output_file`.** The latter
  also attaches a Python logging handler to the same path and clobbers Psi4's own output.
- **`history['coordinates']` is in Bohr.** Multiply by `bohr2angstroms`. The previous
  version of this app did not, making every optimized structure it wrote 1.89× too large.
- **IR intensities are indexed over all 3N modes**, but `wfn.frequencies()` returns only
  the vibrational ones. Select with the `TRV == "V"` mask; zipping the raw arrays pairs
  each frequency with the wrong mode's intensity.
- **`no_reorient` / `no_com` in the geometry block.** Otherwise Psi4 re-centres and
  rotates the molecule, and cube grids no longer line up with the displayed atoms.
- **Charge/multiplicity live in the geometry block**, which is what Psi4 reads. The runner
  takes them back from `molecule.molecular_charge()` / `.multiplicity()`, not from the
  spec's separate fields.
- **`SAVE_JK` is required for TD-DFT**, or Psi4 releases the integrals before
  `tdscf_excitations` can use them.
- **ESP cubes need a JKFIT auxiliary basis**, which `to_file()` does not serialize; the
  runner rebuilds it.
- **`wfn.to_file()` refuses custom basis sets.** That must not fail an otherwise
  successful calculation — the orbital viewer just stays hidden.
- Psi4 accepts `RHF`/`UHF` as the reference for DFT (it maps them to `RKS`/`UKS`), but a
  restricted reference with multiplicity > 1 is rejected — the UI prevents that pairing.
- **Excited-state optimization needs `symmetry c1`.** Psi4 selects roots *per irrep*, so a
  molecule that distorts out of its starting point group mid-optimization aborts with
  "Point group changed!" — which is exactly what excited states tend to do. Only the
  Emission path drops symmetry; everywhere else it is left on for the speed.
- **`ROOTS_PER_IRREP` must be cleared before the ground-state energy** in an emission job,
  or the CCSD call solves for excited roots again instead of returning S0.

## What Psi4 cannot do

- **No NMR shielding.** SCF properties top out at charges, bond orders, ESP and
  multipoles. The ORCA sibling app's NMR pipeline has no counterpart here and was dropped.
- **No TD-DFT gradients.** Excited-state optimization is possible only through
  **EOM-CCSD**, which is the sole entry in Psi4's gradient table with an excited-state
  gradient. That is what the Emission calculation type uses, and why it offers no method
  choice. It scales as N⁶, so it is practical only for small molecules.
- **No EOM-CCSD transition dipoles** (`TRANSITION_DIPOLE` is unsupported and the
  `oscillator_strength` path aborts inside `cclambda`), so an emission spectrum is a
  single normalized band at the computed wavelength — which is what Kasha's rule implies
  anyway. Absorption and ECD keep their real oscillator/rotatory strengths, since those
  come from TDSCF.
- **No SMD solvation** — PCM only.

## Conventions

- **Handler signature:** every UI callback is a top-level `on_*` function taking Gradio
  component values positionally and returning positionally into the outputs list wired in
  the tab's `*_tab_content` / `*_blocks` builder. Keep the wiring at the bottom of each
  builder index-aligned with handler returns.
- **Errors surface to the user, never raise:** handlers wrap work in `try/except` and
  return a red status span (and a refreshed file list) rather than propagating.
- Generated artifacts and `data/`, `static/`, `psi4-env/` are gitignored — never commit
  them.
