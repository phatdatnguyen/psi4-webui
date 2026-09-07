"""Regression coverage for file targeting, directory isolation, and runtime paths."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from psi4_webui import working_directory as wd
from psi4_webui._paths import file_in_directory, validate_name
from psi4_webui.app import build_app


@pytest.mark.parametrize("name", ["", " ", ".", "..", "../outside", "/outside",
                                   "nested/file", r"nested\file", r"C:\outside", "x\0y"])
def test_file_names_cannot_be_paths(name):
    with pytest.raises(ValueError):
        validate_name(name)


def test_file_target_cannot_follow_an_external_symlink(tmp_path):
    directory = tmp_path / "work"
    directory.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("keep")
    (directory / "linked.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="inside"):
        file_in_directory(directory, "linked.txt")


def test_open_directory_stays_under_data_and_retains_state_on_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opened = wd.on_open_working_directory("water")
    assert Path(opened[1]) == tmp_path / "data" / "water"
    assert opened[2] == []
    with pytest.warns(UserWarning, match="without a directory path"):
        rejected = wd.on_open_working_directory("../../outside")
    assert all(value == {"__type__": "update"} for value in rejected)
    assert wd.get_working_directories() == ["water"]


def test_blank_directory_does_not_clear_an_existing_session():
    with pytest.warns(UserWarning, match="specify a working directory"):
        rejected = wd.on_open_working_directory("")
    assert all(value == {"__type__": "update"} for value in rejected)


def test_save_targets_loaded_file_after_selecting_another_row(tmp_path):
    first = tmp_path / "first.xyz"
    second = tmp_path / "second.xyz"
    first.write_text("first")
    second.write_text("second")
    _, _, loaded = wd.on_view_text_file(str(tmp_path), first.name)
    selection = wd.on_select_file(SimpleNamespace(row_value=[second.name]))
    assert selection[2] == second.name

    with pytest.warns(UserWarning, match="saved successfully"):
        files = wd.on_save_loaded_text_file(str(tmp_path), loaded, "edited first")

    assert first.read_text() == "edited first"
    assert second.read_text() == "second"
    assert set(files) == {first.name, second.name}


def test_loaded_text_cannot_be_saved_into_a_different_directory(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        (directory / "water.xyz").write_text(directory.name)
    _, _, loaded = wd.on_view_text_file(str(first), "water.xyz")
    with pytest.warns(UserWarning, match="current working directory"):
        wd.on_save_loaded_text_file(str(second), loaded, "old editor contents")
    assert (first / "water.xyz").read_text() == "first"
    assert (second / "water.xyz").read_text() == "second"


def test_directory_change_clears_editor_target_and_stale_selections():
    updates = wd.on_working_directory_change()
    assert updates[:4] == (None, None, None, None)
    assert updates[8]["value"] == ""
    assert updates[9]["interactive"] is False


@pytest.mark.parametrize("operation", [wd.on_save_text_file, wd.on_delete_file])
def test_file_operations_do_not_modify_a_parent_directory(tmp_path, operation):
    directory = tmp_path / "work"
    directory.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("keep")
    args = [str(directory), "../outside.txt"]
    if operation is wd.on_save_text_file:
        args.append("overwrite")
    with pytest.warns(UserWarning, match="without a directory path"):
        assert operation(*args) == []
    assert outside.read_text() == "keep"


def test_clean_without_a_directory_does_not_clean_the_process_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scratch = tmp_path / "timer.dat"
    scratch.write_text("keep")
    with pytest.warns(UserWarning, match="open a working directory"):
        assert wd.on_clean_working_directory(None) == []
    assert scratch.exists()


def test_upload_missing_source_reports_error_and_refreshes(tmp_path):
    (tmp_path / "existing.xyz").write_text("keep")
    with pytest.warns(UserWarning, match="Error uploading file"):
        assert wd.on_upload_file(str(tmp_path), None) == ["existing.xyz"]


def test_file_disappearing_during_refresh_is_ignored(tmp_path, monkeypatch):
    (tmp_path / "water.xyz").write_text("structure")

    def removed(_):
        raise FileNotFoundError("removed during refresh")

    monkeypatch.setattr(wd.os.path, "getmtime", removed)
    assert wd.on_file_list_change(str(tmp_path)).empty


def test_custom_run_directory_is_used_by_ui_and_static_server(tmp_path, monkeypatch, water_mol):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    runtime = tmp_path / "runtime"
    app = build_app(runtime)
    blocks = app.routes[-1].app.get_blocks()
    handlers = {getattr(function.fn, "func", function.fn).__name__: function
                for function in blocks.fns.values()}
    opened = handlers["on_open_working_directory"].fn("water")
    directory = Path(opened[1])
    assert directory == runtime / "data" / "water"

    from psi4_webui.utils import conformer_to_xyz_file

    conformer_to_xyz_file(water_mol, 0, str(directory / "water.xyz"))
    html = handlers["on_view_structure_file"].fn(str(directory), "water.xyz")
    assert html is not None
    url = html.split('src="', 1)[1].split('"', 1)[0]
    static_app = next(route.app for route in app.routes if route.path == "/static")
    served_path, stat = static_app.lookup_path(url.removeprefix("/static/"))
    assert stat is not None
    assert Path(served_path).is_relative_to(runtime / "static")
    assert "nglview" in Path(served_path).read_text()
    assert not (elsewhere / "data").exists()
    assert not (elsewhere / "static").exists()

    # Save must use the loaded target state, rather than the newly selected file state.
    load_outputs = handlers["on_view_text_file"].outputs
    save_inputs = handlers["on_save_loaded_text_file"].inputs
    assert load_outputs[2] is save_inputs[1]
