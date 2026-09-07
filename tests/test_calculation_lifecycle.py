"""Process ownership and cancellation regressions, using small real child processes."""
import json
import os
import subprocess
import sys
import threading

import pytest

from psi4_webui import calculation
from psi4_webui.utils import SINGLE_POINT


@pytest.fixture
def run_args(tmp_path):
    (tmp_path / "water.xyz").write_text(
        "0 1\nO 0 0 0\nH 0 0.757 0.586\nH 0 -0.757 0.586\n", encoding="utf-8")
    return dict(
        working_directory_path=str(tmp_path), structure_file_name="water.xyz",
        calculation_type=SINGLE_POINT, use_mm=False, force_field="MMFF", max_iters=200,
        method_type="HF", functional="B3LYP", basis_set="STO-3G", reference="RHF",
        charge=0, multiplicity=1, geom_maxiter=50, g_convergence="QCHEM",
        temperature=298.15, pressure=1.0, n_states=10, tda=False, root=1,
        use_solvation=False, solvent="Water", n_threads=1, memory_gb=1, job_name="job",
    )


@pytest.fixture
def child_processes(monkeypatch):
    real_popen = subprocess.Popen
    children = []
    state = {"seconds": 60, "exit_code": 0}

    def launch(command, **kwargs):
        child = real_popen([
            sys.executable, "-c",
            f"import time; time.sleep({state['seconds']}); raise SystemExit({state['exit_code']})",
        ], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(calculation.subprocess, "Popen", launch)
    monkeypatch.setattr(calculation, "_POLL_INTERVAL", 0.005)
    yield children, state
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait()


def test_duplicate_run_does_not_modify_active_artifacts(run_args, child_processes, tmp_path):
    first = calculation.on_run_calculation(**run_args)
    next(first)
    try:
        job_before = (tmp_path / "job.job.json").read_bytes()
        (tmp_path / "job.log").write_text("active output", encoding="utf-8")
        second = list(calculation.on_run_calculation(**dict(run_args, basis_set="cc-pVTZ")))
        assert "already running" in second[-1][0]
        assert (tmp_path / "job.job.json").read_bytes() == job_before
        assert (tmp_path / "job.log").read_text() == "active output"
        assert not (tmp_path / "job.log.bak").exists()
    finally:
        first.close()


def test_directory_is_reserved_during_preparation(run_args, child_processes, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    load = calculation.mol_from_structure_file
    errors = []

    def paused_load(path):
        entered.set()
        assert release.wait(timeout=5)
        return load(path)

    monkeypatch.setattr(calculation, "mol_from_structure_file", paused_load)
    first = calculation.on_run_calculation(**run_args)

    def advance():
        try:
            next(first)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=advance)
    thread.start()
    try:
        assert entered.wait(timeout=5)
        rejected = list(calculation.on_run_calculation(**run_args))
        assert "already running" in rejected[-1][0]
        assert not child_processes[0]
    finally:
        release.set()
        thread.join(timeout=5)
        first.close()
    assert not thread.is_alive()
    assert not errors


def test_stop_during_preparation_prevents_launch(run_args, child_processes, monkeypatch):
    load = calculation.mol_from_structure_file

    def stop_during_load(path):
        calculation.on_stop_calculation(run_args["working_directory_path"])
        return load(path)

    monkeypatch.setattr(calculation, "mol_from_structure_file", stop_during_load)
    outputs = list(calculation.on_run_calculation(**run_args))
    assert "stopped before launch" in outputs[-1][0]
    assert not child_processes[0]
    assert run_args["working_directory_path"] not in calculation._starting


def test_path_aliases_share_process_ownership(run_args, child_processes):
    first = calculation.on_run_calculation(**run_args)
    next(first)
    try:
        alias = os.path.join(run_args["working_directory_path"], ".")
        rejected = list(calculation.on_run_calculation(**dict(run_args, working_directory_path=alias)))
        assert "already running" in rejected[-1][0]
        assert "Stopping" in calculation.on_stop_calculation(alias)
        assert "stopped" in list(first)[-1][0]
        assert child_processes[0][0].poll() is not None
    finally:
        first.close()


def test_closing_generator_terminates_child(run_args, child_processes, tmp_path):
    stream = calculation.on_run_calculation(**run_args)
    next(stream)
    stream.close()
    assert child_processes[0][0].poll() is not None
    assert run_args["working_directory_path"] not in calculation._processes
    result = json.loads((tmp_path / "job.result.json").read_text())
    assert result["status"] == "failed"
    assert "stream was closed" in result["error"]


def test_native_failure_replaces_previous_completed_result(run_args, child_processes, tmp_path):
    result_path = tmp_path / "job.result.json"
    previous = {"status": "completed", "energy_hartree": -75.0}
    result_path.write_text(json.dumps(previous), encoding="utf-8")
    child_processes[1].update(seconds=0, exit_code=7)
    outputs = list(calculation.on_run_calculation(**run_args))
    assert "failed" in outputs[-1][0]
    result = json.loads(result_path.read_text())
    assert result["status"] == "failed"
    assert "code 7" in result["error"]
    assert json.loads((tmp_path / "job.result.json.bak").read_text()) == previous


@pytest.mark.parametrize("job_name", ["../escaped", "/tmp/escaped", r"..\escaped"])
def test_calculation_name_cannot_escape_directory(run_args, child_processes, job_name):
    outputs = list(calculation.on_run_calculation(**dict(run_args, job_name=job_name)))
    assert "red" in outputs[-1][0]
    assert not child_processes[0]


def test_deleted_directory_reports_error(run_args, child_processes, tmp_path):
    (tmp_path / "water.xyz").unlink()
    tmp_path.rmdir()
    outputs = list(calculation.on_run_calculation(**run_args))
    assert "red" in outputs[-1][0]
    assert outputs[-1][2] == []
    assert run_args["working_directory_path"] not in calculation._starting


@pytest.mark.parametrize("force_field", ["MMFF", "UFF"])
def test_preoptimization_requires_parameters(run_args, child_processes, tmp_path, force_field):
    # Helium is a valid atom but is not parameterized by either force field. UFF's
    # optimizer can otherwise report success with zero energy for this input.
    (tmp_path / "water.xyz").write_text("0 1\nHe 0 0 0\n", encoding="utf-8")
    outputs = list(calculation.on_run_calculation(**dict(
        run_args, use_mm=True, force_field=force_field)))
    assert f"{force_field} parameters are unavailable" in outputs[-1][0]
    assert not child_processes[0]


@pytest.mark.parametrize("force_field", ["MMFF", "UFF"])
def test_preoptimization_may_stop_at_iteration_limit(run_args, child_processes, monkeypatch, force_field):
    monkeypatch.setattr(calculation.AllChem, force_field + "OptimizeMolecule", lambda *a, **k: 1)
    stream = calculation.on_run_calculation(**dict(run_args, use_mm=True, force_field=force_field))
    try:
        assert "Running" in next(stream)[0]
        assert len(child_processes[0]) == 1
    finally:
        stream.close()


def test_calculation_error_text_is_escaped(run_args, child_processes, monkeypatch):
    def fail_load(path):
        raise ValueError('<img src=x onerror="alert(1)">')

    monkeypatch.setattr(calculation, "mol_from_structure_file", fail_load)
    status = list(calculation.on_run_calculation(**run_args))[-1][0]
    assert "&lt;img" in status
    assert "<img" not in status
