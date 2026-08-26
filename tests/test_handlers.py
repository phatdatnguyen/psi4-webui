"""Tests for the Gradio handlers: dropdown population, dynamic visibility, run/stop.

These need Gradio (the handlers return ``gr.update`` objects) but never Psi4: the run
tests drive :func:`on_run_calculation` against a stub child process, which is what makes
it possible to test the process registry, the Stop path and the streaming loop without
waiting on an SCF.
"""
import os
import subprocess
import sys
import textwrap
import time

import pytest

pytest.importorskip("gradio")

from psi4_webui import calculation  # noqa: E402
from psi4_webui.calculation import (  # noqa: E402
    OPEN_SHELL_REFERENCES,
    on_change_calculation_type,
    on_method_type_change,
    on_multiplicity_change,
    on_run_calculation,
    on_stop_calculation,
)
from psi4_webui.result import on_working_directory_file_list_change as result_file_list_change  # noqa: E402
from psi4_webui.utils import (  # noqa: E402
    EMISSION,
    EMISSION_METHOD_TYPE,
    FREQUENCY,
    GEOMETRY_OPTIMIZATION,
    RESULT_SUFFIX,
    SINGLE_POINT,
    TDDFT,
)
from psi4_webui.working_directory import on_file_list_change  # noqa: E402


def _update(value):
    """Read a ``gr.update`` result as a plain dict."""
    return dict(value)


class TestFileBrowser:
    def test_rows_are_ordered_by_modification_time_newest_first(self, tmp_path):
        # Sorting the formatted ctime *string* would order by weekday name rather than
        # chronologically -- the regression this test exists to hold.
        old = tmp_path / "old.xyz"
        new = tmp_path / "new.xyz"
        old.write_text("x", encoding="utf-8")
        new.write_text("y", encoding="utf-8")
        os.utime(old, (1_600_000_000, 1_600_000_000))
        os.utime(new, (1_700_000_000, 1_700_000_000))

        table = on_file_list_change(str(tmp_path))

        assert list(table["File"]) == ["new.xyz", "old.xyz"]

    def test_classifies_psi4_artifacts(self, tmp_path):
        for name in ("job" + RESULT_SUFFIX, "job.job.json", "job.npy", "job.log", "mo.cube", "a.xyz"):
            (tmp_path / name).write_text("x", encoding="utf-8")

        table = on_file_list_change(str(tmp_path))
        types = dict(zip(table["File"], table["Type"]))

        assert types["job" + RESULT_SUFFIX] == "Result file"
        assert types["job.job.json"] == "Job spec file"
        assert types["job.npy"] == "Wavefunction file"
        assert types["job.log"] == "Log file"
        assert types["mo.cube"] == "Cube file"
        assert types["a.xyz"] == "Structure file"

    def test_empty_working_directory_state_yields_an_empty_table(self):
        # The shared path state is None until a directory is opened, and this fires on
        # its .change event.
        assert on_file_list_change(None).empty


class TestDropdownPopulation:
    def test_structure_dropdown_lists_only_structure_files(self):
        update = _update(calculation.on_working_directory_file_list_change(
            ["a.xyz", "b.pdb", "c.log", "d" + RESULT_SUFFIX, "e.npy"]))
        assert update["choices"] == ["a.xyz", "b.pdb"]

    def test_structure_dropdown_excludes_trajectories(self):
        # A trajectory holds many frames; feeding one to a calculation would silently use
        # only its last geometry, which is never what picking a "structure" means.
        update = _update(calculation.on_working_directory_file_list_change(
            ["opt_trajectory.xyz", "opt_optimized.xyz"]))
        assert update["choices"] == ["opt_optimized.xyz"]

    def test_result_dropdown_lists_only_result_files(self):
        update = _update(result_file_list_change(
            ["a" + RESULT_SUFFIX, "a.log", "a.job.json", "b" + RESULT_SUFFIX]))
        assert update["choices"] == ["a" + RESULT_SUFFIX, "b" + RESULT_SUFFIX]

    def test_none_file_list_is_tolerated(self):
        # Both dropdowns subscribe to a state that is None before a directory is opened.
        assert _update(calculation.on_working_directory_file_list_change(None))["choices"] == []
        assert _update(result_file_list_change(None))["choices"] == []


class TestDynamicVisibility:
    def test_calculation_type_drives_the_control_groups(self):
        _, _, optimizer, thermo, excited, _, _, _ = on_change_calculation_type(SINGLE_POINT, "B3LYP")
        assert _update(optimizer)["visible"] is False
        assert _update(thermo)["visible"] is False
        assert _update(excited)["visible"] is False

        _, _, optimizer, thermo, excited, _, _, _ = on_change_calculation_type(GEOMETRY_OPTIMIZATION, "B3LYP")
        assert _update(optimizer)["visible"] is True
        assert _update(thermo)["visible"] is False

        _, _, optimizer, thermo, excited, _, _, _ = on_change_calculation_type(FREQUENCY, "B3LYP")
        # A frequency job optimizes first, so it needs the optimizer settings too.
        assert _update(optimizer)["visible"] is True
        assert _update(thermo)["visible"] is True

        _, _, optimizer, thermo, excited, _, _, _ = on_change_calculation_type(TDDFT, "B3LYP")
        assert _update(excited)["visible"] is True

    def test_job_name_tracks_the_calculation_type(self):
        assert on_change_calculation_type(GEOMETRY_OPTIMIZATION, "B3LYP")[0] == "geometry_optimization"
        assert on_change_calculation_type(TDDFT, "B3LYP")[0] == "tddft"

    def test_tddft_swaps_in_its_own_functional_list(self):
        # B3LYP is a poor choice for excitation energies; the TD-DFT menu leads with
        # range-separated hybrids instead.
        functional = _update(on_change_calculation_type(TDDFT, "B3LYP")[1])
        assert "CAM-B3LYP" in functional["choices"]

    def test_a_functional_not_on_the_new_menu_falls_back_to_its_default(self):
        # The dropdown allows custom values, so a functional the new menu does not offer
        # would otherwise survive invisibly.
        functional = _update(on_change_calculation_type(TDDFT, "M06-L")[1])
        assert functional["value"] in functional["choices"]

    def test_emission_locks_the_method_to_eom_ccsd(self):
        # EOM-CCSD is the only method in Psi4 with an excited-state gradient, so any other
        # choice here could only produce a job that cannot run.
        outputs = on_change_calculation_type(EMISSION, "B3LYP")
        method = _update(outputs[5])
        assert method["choices"] == [EMISSION_METHOD_TYPE]
        assert method["value"] == EMISSION_METHOD_TYPE
        assert method["interactive"] is False

    def test_emission_reveals_the_root_picker_and_the_cost_warning(self):
        outputs = on_change_calculation_type(EMISSION, "B3LYP")
        assert _update(outputs[6])["visible"] is True
        assert _update(outputs[7])["visible"] is True

    def test_emission_needs_the_optimizer_settings(self):
        # It is an optimization, just of an excited state.
        assert _update(on_change_calculation_type(EMISSION, "B3LYP")[2])["visible"] is True

    def test_leaving_emission_restores_the_method_choice(self):
        method = _update(on_change_calculation_type(SINGLE_POINT, "B3LYP")[5])
        assert method["interactive"] is True
        assert len(method["choices"]) > 1

    def test_functional_is_shown_only_for_dft(self):
        assert _update(on_method_type_change("DFT"))["visible"] is True
        assert _update(on_method_type_change("HF"))["visible"] is False
        assert _update(on_method_type_change("CCSD(T)"))["visible"] is False


class TestReferenceGuard:
    def test_open_shell_multiplicity_forces_an_unrestricted_reference(self):
        # A restricted reference doubly occupies every orbital and simply cannot
        # represent a doublet; Psi4 rejects the combination outright.
        update = _update(on_multiplicity_change(2, "RHF"))
        assert update["choices"] == OPEN_SHELL_REFERENCES
        assert update["value"] == "UHF"

    def test_an_already_valid_open_shell_reference_is_kept(self):
        assert _update(on_multiplicity_change(3, "ROHF"))["value"] == "ROHF"

    def test_singlet_restores_the_restricted_options(self):
        update = _update(on_multiplicity_change(1, "UHF"))
        assert "RHF" in update["choices"]


# A stub standing in for psi4_webui.runner: it writes to the log the run handler tails,
# lives long enough to be stopped, and exits with a code the test chooses.
_STUB_CHILD = textwrap.dedent("""
    import sys, time
    log_path, seconds, exit_code = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    with open(log_path, "w") as handle:
        handle.write("stub psi4 output\\n")
        handle.flush()
        deadline = time.time() + seconds
        while time.time() < deadline:
            time.sleep(0.05)
    sys.exit(exit_code)
""")


@pytest.fixture
def stub_child(tmp_path, monkeypatch):
    """Replace the runner subprocess with a fast, controllable stub."""
    script = tmp_path / "stub_child.py"
    script.write_text(_STUB_CHILD, encoding="utf-8")

    state = {"seconds": 0.3, "exit_code": 0}
    real_popen = subprocess.Popen

    def fake_popen(cmd, **kwargs):
        # Patching subprocess.Popen reaches the shared subprocess module, so anything
        # else launching a process during the test would be intercepted too. Only a
        # runner invocation is stubbed; everything else runs for real.
        if "psi4_webui.runner" not in cmd:
            return real_popen(cmd, **kwargs)
        # cmd is [python, -m, psi4_webui.runner, run, <job path>]; the log the handler
        # tails is named after that job.
        job_path = cmd[-1]
        base = os.path.basename(job_path).replace(".job.json", "")
        log_path = os.path.join(os.path.dirname(job_path), base + ".log")
        return real_popen([sys.executable, str(script), log_path,
                           str(state["seconds"]), str(state["exit_code"])], **kwargs)

    monkeypatch.setattr(calculation.subprocess, "Popen", fake_popen)
    return state


def _run(tmp_path, **overrides):
    """Drive on_run_calculation with reasonable defaults; return every yielded tuple."""
    structure = tmp_path / "water.xyz"
    structure.write_text(
        "\n".join(["0 1", "O 0.0 0.0 0.117", "H 0.0 0.757 -0.469", "H 0.0 -0.757 -0.469", ""]),
        encoding="utf-8",
    )
    kwargs = dict(
        working_directory_path=str(tmp_path), structure_file_name="water.xyz",
        calculation_type=SINGLE_POINT, use_mm=False, force_field="MMFF", max_iters=200,
        method_type="HF", functional="B3LYP", basis_set="STO-3G", reference="RHF",
        charge=0, multiplicity=1, geom_maxiter=50, g_convergence="QCHEM",
        temperature=298.15, pressure=1.0, n_states=10, tda=False, root=1,
        use_solvation=False, solvent="Water", n_threads=1, memory_gb=1, job_name="job",
    )
    kwargs.update(overrides)
    return list(on_run_calculation(**kwargs))


class TestRunCalculation:
    def test_writes_a_job_spec_and_reports_success(self, tmp_path, stub_child):
        yields = _run(tmp_path)

        assert (tmp_path / "job.job.json").exists()
        final_status = yields[-1][0]
        assert "green" in final_status
        assert "finished" in final_status

    def test_streams_the_log_while_running(self, tmp_path, stub_child):
        # The whole point of the generator handler: the user watches Psi4 work instead of
        # staring at a frozen page.
        stub_child["seconds"] = 1.0
        yields = _run(tmp_path)

        assert len(yields) > 1, "handler did not stream intermediate updates"
        assert any("stub psi4 output" in (y[1] or "") for y in yields)

    def test_a_failing_child_is_reported_in_red(self, tmp_path, stub_child):
        stub_child["exit_code"] = 1
        final_status = _run(tmp_path)[-1][0]
        assert "red" in final_status
        assert "failed" in final_status

    def test_an_existing_log_is_backed_up_not_destroyed(self, tmp_path, stub_child):
        # An accidental second click on Run would otherwise throw away the previous run.
        (tmp_path / "job.log").write_text("precious previous output", encoding="utf-8")

        _run(tmp_path)

        assert (tmp_path / "job.log.bak").read_text(encoding="utf-8") == "precious previous output"

    def test_refuses_to_run_without_a_structure(self, tmp_path, stub_child):
        assert _run(tmp_path, structure_file_name=None) == []

    def test_refuses_to_run_without_a_working_directory(self, tmp_path, stub_child):
        assert _run(tmp_path, working_directory_path=None) == []

    def test_refuses_a_blank_job_name(self, tmp_path, stub_child):
        assert _run(tmp_path, job_name="   ") == []

    def test_works_when_the_working_directory_is_a_relative_path(self, tmp_path, stub_child, monkeypatch):
        # Working directories are named relatively ("./data/wd"), and the child is
        # started *in* that directory -- so a job path relative to it would be resolved
        # against itself a second time and the child would not find its spec. Every other
        # test here uses tmp_path, which is absolute, and so cannot catch this.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "wd").mkdir(parents=True)
        relative = os.path.join(".", "data", "wd")

        structure = tmp_path / "data" / "wd" / "water.xyz"
        structure.write_text("\n".join(["0 1", "O 0.0 0.0 0.0", ""]), encoding="utf-8")

        yields = _run(tmp_path, working_directory_path=relative)

        assert yields, "handler produced no output"
        assert "green" in yields[-1][0], yields[-1][0]

    def test_the_final_file_list_includes_the_new_artifacts(self, tmp_path, stub_child):
        file_list = _run(tmp_path)[-1][2]
        # This list is the app's event bus: every tab's dropdowns repopulate from it.
        assert "job.job.json" in file_list
        assert "job.log" in file_list


class TestStopCalculation:
    def test_warns_when_nothing_is_running(self, tmp_path):
        assert on_stop_calculation(str(tmp_path)) == ""

    def test_stopping_kills_the_child_and_reports_it(self, tmp_path, stub_child):
        stub_child["seconds"] = 30.0
        results = []

        import threading

        def run():
            results.extend(_run(tmp_path))

        thread = threading.Thread(target=run)
        thread.start()
        # Wait for the child to be registered before trying to stop it.
        deadline = time.time() + 10
        while time.time() < deadline:
            with calculation._lock:
                if str(tmp_path) in calculation._processes:
                    break
            time.sleep(0.05)

        status = on_stop_calculation(str(tmp_path))
        thread.join(timeout=15)

        assert "Stopping" in status
        assert results, "run handler produced no output"
        assert "stopped" in results[-1][0]

    def test_targets_the_right_working_directory(self, tmp_path, stub_child):
        # Keying the registry on the working directory (rather than one global handle) is
        # what lets two sessions run at once without Stop hitting the wrong job.
        other = tmp_path / "other"
        other.mkdir()
        with calculation._lock:
            calculation._processes[str(other)] = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            assert on_stop_calculation(str(tmp_path)) == ""
        finally:
            with calculation._lock:
                process = calculation._processes.pop(str(other))
            process.kill()
            process.wait()
