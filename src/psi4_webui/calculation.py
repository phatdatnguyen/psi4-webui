"""Calculation tab: pick a structure, choose a method, run Psi4.

Unlike the ORCA app this is modelled on, there is **no input-file step**. ORCA is driven
by a text input file, so building one and running it are two separate user actions; Psi4
is driven by its Python API, so there is nothing for a user to author. One Run button
takes the structure and the settings on screen straight to a calculation.

The calculation itself happens in a child process (:mod:`psi4_webui.runner`) rather than
on the Gradio worker thread. The job spec written into the working directory is the
argument to that child -- an implementation detail that happens to be visible in the file
browser, not a format anyone is expected to edit.

Because the child is a real OS process, Stop can genuinely stop it, and the handler
streams the growing Psi4 log back to the browser while it runs.
"""
import math
import multiprocessing
import os
import subprocess
import sys
import threading
import time

import gradio as gr
import psutil
from rdkit.Chem import AllChem

from .utils import (
    CALCULATION_TYPES,
    EMISSION,
    EMISSION_METHOD_TYPE,
    FREQUENCY,
    GEOMETRY_OPTIMIZATION,
    JOB_SUFFIX,
    METHOD_TYPES,
    SINGLE_POINT,
    TDDFT,
    build_job_spec,
    get_files_in_working_directory,
    mol_from_structure_file,
    psi4_geometry_string,
    write_json,
)

# Physical RAM in whole GB, and the share of it a calculation may claim. Psi4's
# set_memory is a budget for the *whole* process (unlike ORCA's per-process %maxcore), so
# there is no per-core division to do here -- just leave the OS some headroom.
_TOTAL_MEMORY_GB = max(1, math.floor(psutil.virtual_memory().total / (1024 ** 3)))
MAX_MEMORY_GB = max(1, math.floor(_TOTAL_MEMORY_GB * 0.8))
DEFAULT_MEMORY_GB = max(1, min(4, MAX_MEMORY_GB))
# Psi4 parallelises with OpenMP threads, so the full logical CPU count is usable. (The
# ORCA app halves this only because ORCA refuses more MPI ranks than physical cores.)
MAX_THREADS = max(1, multiprocessing.cpu_count())
DEFAULT_THREADS = max(1, MAX_THREADS // 2)

FUNCTIONALS = ["B3LYP", "B3LYP-D3BJ", "PBE", "PBE0", "BP86", "BLYP", "M06-2X", "M06-L",
               "wB97X-D", "CAM-B3LYP", "TPSS", "SCAN"]
# TD-DFT quality depends strongly on the functional; range-separated hybrids are the
# defensible default for excitation energies, so the menu leads with them.
TDDFT_FUNCTIONALS = ["CAM-B3LYP", "wB97X-D", "PBE0", "B3LYP", "M06-2X", "TPSS", "PBE"]
BASIS_SETS = ["STO-3G", "3-21G", "6-31G", "6-31G(d)", "6-31G(d,p)", "6-31+G(d,p)",
              "6-311G(d,p)", "6-311+G(d,p)", "cc-pVDZ", "cc-pVTZ", "cc-pVQZ",
              "aug-cc-pVDZ", "aug-cc-pVTZ", "def2-SVP", "def2-TZVP", "def2-TZVPP"]
DEFAULT_BASIS_SET = "6-31G(d)"
# PCMSolver's named solvents. Psi4 has no SMD model, so this is the whole solvation story.
SOLVENTS = ["Water", "Methanol", "Ethanol", "Acetonitrile", "Acetone", "DMSO",
            "Dichloromethane", "Chloroform", "Tetrahydrofuran", "Toluene", "Benzene",
            "Cyclohexane", "N-heptane", "Carbon tetrachloride", "Aniline",
            "Chlorobenzene", "Nitromethane", "1,2-Dichloroethane"]
RESTRICTED_REFERENCES = ["RHF", "ROHF", "UHF"]
# A closed-shell (restricted) reference cannot represent an open-shell state, so anything
# with unpaired electrons must use an unrestricted or restricted-open reference.
OPEN_SHELL_REFERENCES = ["UHF", "ROHF"]
G_CONVERGENCES = ["QCHEM", "MOLPRO", "GAU", "GAU_LOOSE", "GAU_TIGHT", "GAU_VERYTIGHT",
                  "TURBOMOLE", "CFOUR", "NWCHEM_LOOSE"]

# Default job/artifact name per calculation type, so the name field tracks the choice.
_DEFAULT_JOB_NAMES = {
    SINGLE_POINT: "single_point",
    GEOMETRY_OPTIMIZATION: "geometry_optimization",
    FREQUENCY: "frequency",
    TDDFT: "tddft",
    EMISSION: "emission",
}

# How often the run handler re-reads the log to stream progress, in seconds. Short enough
# to feel live, long enough not to spend the worker thread's time on stat() calls.
_POLL_INTERVAL = 0.5
# Tail of the Psi4 log shown while a calculation runs. The full file is always one click
# away in the working-directory viewer; streaming all of it would push megabytes into the
# browser for a long CCSD run.
_LOG_TAIL_CHARS = 8000

# Running child processes, keyed by working directory rather than held in a single global
# as the ORCA app does. Two browser sessions working in two different directories can
# then run at the same time, and Stop kills the right one.
_processes: dict[str, subprocess.Popen] = {}
_stopped: set[str] = set()
_lock = threading.Lock()


def _status(message: str, color: str) -> str:
    """Colored status span, the convention every handler in this app reports through."""
    return f"<span style='color:{color};'>{message}</span>"


def _read_log_tail(log_path: str) -> str:
    """Return the last :data:`_LOG_TAIL_CHARS` characters of ``log_path``, or ``""``."""
    try:
        size = os.path.getsize(log_path)
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            if size > _LOG_TAIL_CHARS:
                handle.seek(size - _LOG_TAIL_CHARS)
            return handle.read()
    except OSError:
        return ""


def on_working_directory_file_list_change(working_directory_file_list):
    """Repopulate the input-structure dropdown from the working directory's files.

    The file-list state is ``None`` until a directory is opened, so it is normalised to an
    empty list here. Trajectory files are excluded: they hold many frames, and feeding one
    to a calculation would silently use only its final geometry, which is never what
    someone picking a "structure" intends.
    """
    working_directory_file_list = working_directory_file_list or []
    structure_file_names = sorted(
        (f for f in working_directory_file_list
         if f.endswith(('.xyz', '.pdb', '.mol', '.mol2')) and not f.endswith('_trajectory.xyz')),
        key=str.lower,
    )
    return gr.update(
        choices=structure_file_names,
        value=structure_file_names[0] if structure_file_names else None,
        interactive=True,
    )


def on_change_calculation_type(calculation_type, functional):
    """React to the calculation-type radio changing.

    Suggests a matching job name, swaps the functional list for TD-DFT, and toggles the
    control groups that only apply to some calculation types: optimizer settings for
    optimization, frequency and emission; thermochemistry conditions for frequency;
    the number of excited states for TD-DFT; the root to follow for emission.

    Emission also takes the method choice away. It optimizes an *excited* state, which
    needs an excited-state gradient, and EOM-CCSD is the only method in Psi4 that has one
    -- so offering HF or DFT here would only let the user pick something that cannot run.
    """
    is_tddft = calculation_type == TDDFT
    is_emission = calculation_type == EMISSION
    functionals = TDDFT_FUNCTIONALS if is_tddft else FUNCTIONALS
    # The dropdown allows custom values, so a functional left selected when switching into
    # TD-DFT could be one this menu no longer offers. Fall back to the list's default
    # rather than silently keeping a choice the user can no longer see.
    functional_value = functional if functional in functionals else functionals[0]

    if is_emission:
        method_update = gr.update(choices=[EMISSION_METHOD_TYPE], value=EMISSION_METHOD_TYPE,
                                  interactive=False)
    else:
        method_update = gr.update(choices=METHOD_TYPES, value="DFT", interactive=True)

    show_optimizer = calculation_type in (GEOMETRY_OPTIMIZATION, FREQUENCY, EMISSION)
    return (
        _DEFAULT_JOB_NAMES.get(calculation_type, "calculation"),
        gr.update(choices=functionals, value=functional_value),
        gr.update(visible=show_optimizer),
        gr.update(visible=calculation_type == FREQUENCY),
        gr.update(visible=is_tddft),
        method_update,
        gr.update(visible=is_emission),
        gr.update(visible=is_emission),
    )


def on_method_type_change(method_type):
    """Show the functional dropdown only for DFT; every other family ignores it."""
    return gr.update(visible=method_type == "DFT")


def on_multiplicity_change(multiplicity, reference):
    """Restrict the reference to open-shell options when the state has unpaired electrons.

    A restricted (RHF) reference forces electrons into doubly-occupied orbitals, which
    simply cannot represent a doublet or triplet -- Psi4 rejects the combination. Rather
    than letting the user discover that from a crashed job, the menu drops the restricted
    option and switches to UHF as soon as the multiplicity leaves 1.
    """
    if int(multiplicity) > 1:
        value = reference if reference in OPEN_SHELL_REFERENCES else "UHF"
        return gr.update(choices=OPEN_SHELL_REFERENCES, value=value)
    return gr.update(choices=RESTRICTED_REFERENCES, value=reference or "RHF")


def on_mm_checkbox_change(use_mm):
    """Show/hide the force-field and max-iterations controls with the MM checkbox."""
    return gr.update(visible=use_mm), gr.update(visible=use_mm)


def on_solvation_checkbox_change(use_solvation):
    """Show/hide the solvent picker with the solvation checkbox."""
    return gr.update(visible=use_solvation)


def on_calculation_started():
    """Grey out the Run button for the duration of a calculation.

    Chained *ahead* of :func:`on_run_calculation` rather than folded into it: that handler
    is a generator whose first yield only reaches the browser once the child has been
    launched, and the button needs to go dead the moment the click lands.
    """
    return gr.update(interactive=False)


def on_calculation_finished():
    """Re-enable the Run button once no calculation is running.

    Chained with ``.then`` (not ``.success``) so the button comes back even when the run
    failed or was stopped, which would otherwise leave it stuck.
    """
    return gr.update(interactive=True)


def on_run_calculation(working_directory_path, structure_file_name, calculation_type,
                       use_mm, force_field, max_iters,
                       method_type, functional, basis_set, reference,
                       charge, multiplicity,
                       geom_maxiter, g_convergence, temperature, pressure,
                       n_states, tda, root,
                       use_solvation, solvent,
                       n_threads, memory_gb, job_name):
    """Run Psi4 on the selected structure, streaming its log as it goes.

    A generator handler: it yields ``(status, log_tail, file_list)`` every
    :data:`_POLL_INTERVAL` seconds while the child runs, so the user watches the SCF
    iterate instead of staring at a frozen page. The final yield reports the outcome.

    Errors are reported in the status span rather than raised, matching every other
    handler in this app.
    """
    if not working_directory_path:
        gr.Warning("Please open a working directory first.")
        return
    if not structure_file_name:
        gr.Warning("Please select an input structure.")
        return
    if not job_name or not job_name.strip():
        gr.Warning("Please give the calculation a name.")
        return

    job_name = job_name.strip()
    file_list = get_files_in_working_directory(working_directory_path)

    try:
        # Same loader the structure viewer uses, so the atom indices labelled on screen
        # are the ones Psi4 receives.
        structure_path = os.path.join(working_directory_path, structure_file_name)
        mol = mol_from_structure_file(structure_path)

        if use_mm:
            # A cheap force-field cleanup before an expensive QM calculation: a strained
            # embedding costs optimizer steps, and MMFF removes most of that strain for
            # a fraction of a second of work.
            if force_field == "MMFF":
                AllChem.MMFFOptimizeMolecule(mol, maxIters=int(max_iters))
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=int(max_iters))

        spec = build_job_spec(
            calculation_type=calculation_type,
            # Emission optimizes an excited state, and Psi4 picks excited roots per
            # irrep -- so symmetry has to go, or a molecule that distorts out of its
            # starting point group mid-optimization aborts the run.
            geometry=psi4_geometry_string(mol, charge, multiplicity,
                                          force_c1=(calculation_type == EMISSION)),
            structure_file=structure_file_name,
            method_type=method_type,
            functional=functional,
            basis_set=basis_set,
            reference=reference,
            charge=charge,
            multiplicity=multiplicity,
            n_threads=n_threads,
            memory_gb=memory_gb,
            geom_maxiter=geom_maxiter,
            g_convergence=g_convergence,
            temperature=temperature,
            # The slider is in atm because that is how chemists quote standard state;
            # Psi4's P option is in pascal.
            pressure=float(pressure) * 101325.0,
            n_states=n_states,
            tda=tda,
            root=root,
            use_solvation=use_solvation,
            solvent=solvent,
        )

        job_path = os.path.join(working_directory_path, job_name + JOB_SUFFIX)
        log_path = os.path.join(working_directory_path, job_name + ".log")
        write_json(job_path, spec)

        # Opening the log below truncates it, so preserve a previous run's output first:
        # an accidental click on Run would otherwise destroy it. os.replace overwrites an
        # older backup atomically and works on Windows too.
        backed_up = False
        if os.path.isfile(log_path):
            os.replace(log_path, log_path + ".bak")
            backed_up = True
    except Exception as exc:
        yield _status(f"Could not start the calculation: {exc}", "red"), "", file_list
        return

    with _lock:
        existing = _processes.get(working_directory_path)
        if existing is not None and existing.poll() is None:
            yield (_status("A calculation is already running in this working directory.", "red"),
                   "", file_list)
            return

    # The child's own stderr (Python tracebacks, and any crash message from the OS) goes
    # to a separate file: <name>.log is owned exclusively by Psi4's C++ output stream, and
    # a segfault never reaches it.
    runner_log_path = os.path.join(working_directory_path, job_name + ".runner.log")
    started = time.time()
    try:
        with open(runner_log_path, "w", encoding="utf-8") as runner_log:
            process = subprocess.Popen(
                # The job path must be absolute: working directories are named relatively
                # ("./data/wd"), and the child is started *in* that directory, so a
                # relative path would be resolved a second time against itself.
                [sys.executable, "-m", "psi4_webui.runner", "run", os.path.abspath(job_path)],
                cwd=working_directory_path,
                stdout=runner_log,
                stderr=subprocess.STDOUT,
            )
            with _lock:
                _processes[working_directory_path] = process
                _stopped.discard(working_directory_path)

            try:
                while process.poll() is None:
                    elapsed = time.time() - started
                    yield (_status(f"Running {calculation_type}... ({elapsed:.0f} s)", "black"),
                           _read_log_tail(log_path),
                           file_list)
                    time.sleep(_POLL_INTERVAL)
                return_code = process.returncode
            finally:
                with _lock:
                    _processes.pop(working_directory_path, None)
                    stopped = working_directory_path in _stopped
                    _stopped.discard(working_directory_path)
    except Exception as exc:
        yield _status(f"Error running calculation: {exc}", "red"), "", get_files_in_working_directory(working_directory_path)
        return

    duration = time.time() - started
    file_list = get_files_in_working_directory(working_directory_path)
    log_tail = _read_log_tail(log_path)

    if stopped:
        yield (_status(f"Calculation stopped after {duration:.1f} s. Partial output in {job_name}.log.", "red"),
               log_tail, file_list)
        return
    if return_code != 0:
        # The runner writes a result file even when it fails, so point at the message it
        # recorded rather than making the user go hunting through two log files.
        detail = f"See {job_name}.log and {job_name}.runner.log."
        yield (_status(f"Calculation failed (exit code {return_code}). {detail}", "red"),
               log_tail, file_list)
        return

    message = f"Calculation finished ({duration:.1f} s)."
    if backed_up:
        message += f" Previous log saved as {job_name}.log.bak."
    yield _status(message, "green"), log_tail, file_list


def on_stop_calculation(working_directory_path):
    """Terminate the Psi4 child running in ``working_directory_path``, if any.

    Psi4 spawns OpenMP worker threads and can shell out further, so the whole process
    tree is signalled (SIGTERM first, SIGKILL for anything still alive after 5 s);
    terminating only the process we launched could leave the cores busy. The run handler
    reports the outcome once ``poll()`` returns, so this only needs to acknowledge the
    request.
    """
    with _lock:
        process = _processes.get(working_directory_path)
        if process is None or process.poll() is not None:
            gr.Warning("No calculation is running.")
            return ""
        _stopped.add(working_directory_path)

    try:
        parent = psutil.Process(process.pid)
        targets = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        targets = []

    for target in targets:
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=5)
    for target in alive:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass

    return _status("Stopping calculation...", "red")


def calculation_tab_content(working_directory_path_state, working_directory_file_list_state, status_markdown):
    """Build the "Calculation" tab and wire its events.

    Takes the shared path/file-list states and status line; returns the tab component.
    """
    with gr.Tab("Calculation") as calculation_tab:
        with gr.Accordion("Resources", open=False):
            with gr.Row():
                n_threads_slider = gr.Slider(label="Number of threads", value=DEFAULT_THREADS,
                                             minimum=1, maximum=MAX_THREADS, step=1,
                                             info="Psi4 parallelises with OpenMP threads.")
                memory_slider = gr.Slider(label="Memory (GB)", value=DEFAULT_MEMORY_GB,
                                          minimum=1, maximum=MAX_MEMORY_GB, step=1,
                                          info=f"Total budget for the calculation; this machine has {_TOTAL_MEMORY_GB} GB.")
        with gr.Accordion("Calculation"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_structure_file_dropdown = gr.Dropdown(label="Input structure", choices=[""], value="", interactive=False)
                with gr.Column(scale=4):
                    calculation_type_radio = gr.Radio(label="Type of calculation", value=SINGLE_POINT, choices=CALCULATION_TYPES)
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        mm_checkbox = gr.Checkbox(label="Pre-optimize geometry with molecular mechanics", value=False)
                    with gr.Row():
                        force_field_dropdown = gr.Dropdown(label="Force field", value="MMFF", choices=["MMFF", "UFF"], visible=False)
                        max_iters_slider = gr.Slider(label="Max iterations", value=200, minimum=0, maximum=1000, step=1, visible=False)
                    with gr.Row():
                        solvation_checkbox = gr.Checkbox(label="Solvation (PCM)", value=False,
                                                         info="Psi4 offers PCM only; there is no SMD model.")
                    with gr.Row():
                        solvent_dropdown = gr.Dropdown(label="Solvent", value="Water", choices=SOLVENTS, visible=False)
                    with gr.Group(visible=False) as optimizer_group:
                        geom_maxiter_slider = gr.Slider(label="Max optimization steps", value=50, minimum=1, maximum=500, step=1)
                        g_convergence_dropdown = gr.Dropdown(label="Convergence criteria", value="QCHEM", choices=G_CONVERGENCES)
                    with gr.Group(visible=False) as thermo_group:
                        temperature_slider = gr.Slider(label="Temperature (K)", value=298.15, minimum=1, maximum=1000, step=0.05)
                        pressure_slider = gr.Slider(label="Pressure (atm)", value=1.0, minimum=0.1, maximum=100, step=0.1)
                with gr.Column(scale=1):
                    with gr.Row():
                        method_type_dropdown = gr.Dropdown(label="Type of method", value="DFT", choices=METHOD_TYPES)
                        reference_dropdown = gr.Dropdown(label="Reference", value="RHF", choices=RESTRICTED_REFERENCES)
                    with gr.Row():
                        functional_dropdown = gr.Dropdown(label="Functional", value="B3LYP", choices=FUNCTIONALS, allow_custom_value=True)
                        basis_set_dropdown = gr.Dropdown(label="Basis set", value=DEFAULT_BASIS_SET, choices=BASIS_SETS, allow_custom_value=True)
                    with gr.Row():
                        charge_slider = gr.Slider(label="Charge", value=0, minimum=-2, maximum=2, step=1)
                        multiplicity_dropdown = gr.Dropdown(label="Multiplicity", value=1,
                                                            choices=[("Singlet", 1), ("Doublet", 2), ("Triplet", 3),
                                                                     ("Quartet", 4), ("Quintet", 5), ("Sextet", 6)])
                    with gr.Group(visible=False) as excited_state_group:
                        n_states_slider = gr.Slider(label="Number of excited states", value=10, minimum=1, maximum=50, step=1)
                        tda_checkbox = gr.Checkbox(label="Tamm-Dancoff approximation (TDA)", value=False,
                                                   info="Cheaper and more stable than full TDDFT/RPA, at some cost in accuracy.")
                    with gr.Group(visible=False) as emission_group:
                        root_slider = gr.Slider(label="Excited state to optimize", value=1, minimum=1, maximum=10, step=1,
                                                info="1 = S1. By Kasha's rule emission comes from S1, so leave this at 1 unless you know otherwise.")
                with gr.Column(scale=1):
                    job_name_textbox = gr.Textbox(label="Calculation name", value=_DEFAULT_JOB_NAMES[SINGLE_POINT],
                                                  info="Names the .log, result and wavefunction files.")
                    emission_warning_markdown = gr.Markdown(visible=False, value=(
                        "**Emission is expensive.** It optimizes the excited state with "
                        "EOM-CCSD -- the only method in Psi4 with an excited-state gradient "
                        "-- which scales as N&#8310;. Expect minutes for a handful of atoms "
                        "in a small basis, and far longer beyond that."
                    ))
                    run_button = gr.Button("Run", variant="primary")
                    stop_button = gr.Button("Stop", variant="stop")
        with gr.Accordion("Psi4 output", open=True):
            live_output_textarea = gr.TextArea(label="Psi4 output (live)", lines=18,
                                               elem_id="textfile_viewer", interactive=False)

        working_directory_file_list_state.change(
            on_working_directory_file_list_change,
            working_directory_file_list_state,
            input_structure_file_dropdown,
        )
        calculation_type_radio.change(
            on_change_calculation_type,
            [calculation_type_radio, functional_dropdown],
            [job_name_textbox, functional_dropdown, optimizer_group, thermo_group,
             excited_state_group, method_type_dropdown, emission_group, emission_warning_markdown],
        )
        method_type_dropdown.change(on_method_type_change, method_type_dropdown, functional_dropdown)
        multiplicity_dropdown.change(on_multiplicity_change, [multiplicity_dropdown, reference_dropdown], reference_dropdown)
        mm_checkbox.change(on_mm_checkbox_change, mm_checkbox, [force_field_dropdown, max_iters_slider])
        solvation_checkbox.change(on_solvation_checkbox_change, solvation_checkbox, solvent_dropdown)

        run_button.click(on_calculation_started, None, run_button) \
                  .then(on_run_calculation,
                        [working_directory_path_state, input_structure_file_dropdown, calculation_type_radio,
                         mm_checkbox, force_field_dropdown, max_iters_slider,
                         method_type_dropdown, functional_dropdown, basis_set_dropdown, reference_dropdown,
                         charge_slider, multiplicity_dropdown,
                         geom_maxiter_slider, g_convergence_dropdown, temperature_slider, pressure_slider,
                         n_states_slider, tda_checkbox, root_slider,
                         solvation_checkbox, solvent_dropdown,
                         n_threads_slider, memory_slider, job_name_textbox],
                        [status_markdown, live_output_textarea, working_directory_file_list_state]) \
                  .then(on_calculation_finished, None, run_button)
        # concurrency_limit=None: without it this event would queue behind the running
        # calculation it is meant to interrupt, and only fire once Psi4 had finished.
        stop_button.click(on_stop_calculation, working_directory_path_state, status_markdown, concurrency_limit=None)

    return calculation_tab
