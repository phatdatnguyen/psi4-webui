"""Tests for the nglview cube renderer.

The rendering itself cannot be checked without a browser, so these assert on the
parameters that reach the generated HTML -- which is where the interesting mistakes live.
"""
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import pytest

pytest.importorskip("nglview")

from psi4_webui.visualization import (  # noqa: E402
    ELECTROSTATIC_POTENTIAL,
    SPIN_DENSITY,
    TOTAL_DENSITY,
    cube_files_for_selection,
    render_cube_html,
    render_structure_html,
)


def _write_cube(path, values):
    """Write a minimal valid Gaussian cube file over a 2x2x2 grid."""
    header = [
        "test cube", "generated for tests",
        "    1    0.000000    0.000000    0.000000",
        "    2    0.500000    0.000000    0.000000",
        "    2    0.000000    0.500000    0.000000",
        "    2    0.000000    0.000000    0.500000",
        "    1    1.000000    0.000000    0.000000    0.000000",
    ]
    body = ["  ".join(f"{v:.6E}" for v in values[i:i + 6]) for i in range(0, len(values), 6)]
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")


@pytest.fixture
def cube_directory(tmp_path):
    """A directory holding one cube file per selection kind."""
    values = [0.1, -0.1, 0.05, -0.05, 0.2, -0.2, 0.01, -0.01]
    for name in ("Dt.cube", "Ds.cube", "ESP.cube", "Psi_a_2_2-A1.cube"):
        _write_cube(tmp_path / name, values)
    return str(tmp_path)


@pytest.fixture
def result():
    return {"symbols": ["H", "H"], "geometry_angstrom": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]}


def _render(tmp_path, monkeypatch, result, cube_directory, selection, isolevel=0.02):
    """Render into an isolated static/ directory and return the generated HTML."""
    monkeypatch.chdir(tmp_path)
    iframe = render_cube_html(result, cube_directory, selection, "#0000ff", "#ff0000", 0.8, isolevel)
    return _view_path(tmp_path, iframe).read_text(encoding="utf-8")


def _view_path(root, iframe):
    src = re.search(r'src="([^"]+)"', iframe).group(1)
    return root / urlsplit(src).path.lstrip("/")


def _isolevels(html):
    return [float(v) for v in re.findall(r'"isolevel":\s*([-\d.eE+]+)', html)]


class TestCubeFileDiscovery:
    @pytest.mark.parametrize("selection,expected", [
        (TOTAL_DENSITY, "Dt.cube"),
        (SPIN_DENSITY, "Ds.cube"),
        (ELECTROSTATIC_POTENTIAL, "ESP.cube"),
    ])
    def test_named_tasks_map_to_their_cube_file(self, cube_directory, selection, expected):
        found = cube_files_for_selection(cube_directory, selection)
        assert [os.path.basename(f) for f in found] == [expected]

    def test_orbital_is_found_despite_the_irrep_in_its_name(self, cube_directory):
        # Psi4 embeds an irrep label ("2-A1") that the caller cannot predict.
        found = cube_files_for_selection(cube_directory, "MO 2")
        assert [os.path.basename(f) for f in found] == ["Psi_a_2_2-A1.cube"]

    def test_a_missing_cube_yields_no_candidates(self, cube_directory):
        assert cube_files_for_selection(cube_directory, "MO 7") == []

    def test_an_unknown_selection_yields_no_candidates(self, cube_directory):
        assert cube_files_for_selection(cube_directory, "Nonsense") == []


class TestIsolevels:
    def test_isolevel_is_absolute_not_sigma(self, tmp_path, monkeypatch, result, cube_directory):
        # NGL defaults to isolevelType="sigma", which rescales the number by the RMS of
        # the grid. A 0.02 a.u. orbital contour then becomes ~0.0008 a.u. and renders a
        # huge diffuse blob clipped by the grid box -- which is what made the two lobes
        # of a symmetric orbital appear unequal.
        html = _render(tmp_path, monkeypatch, result, cube_directory, "MO 2")
        types = re.findall(r'"isolevelType":\s*"(\w+)"', html)
        assert types, "isolevelType was not emitted at all"
        assert set(types) == {"value"}

    def test_an_orbital_gets_mirrored_positive_and_negative_surfaces(self, tmp_path, monkeypatch, result, cube_directory):
        # The two lobes of an orbital must be contoured at exactly opposite levels, or a
        # symmetric orbital renders asymmetrically.
        levels = _isolevels(_render(tmp_path, monkeypatch, result, cube_directory, "MO 2", isolevel=0.03))
        assert sorted(levels) == [-0.03, 0.03]

    @pytest.mark.parametrize("selection", [ELECTROSTATIC_POTENTIAL, SPIN_DENSITY])
    def test_other_signed_quantities_also_get_both_lobes(self, tmp_path, monkeypatch, result, cube_directory, selection):
        levels = _isolevels(_render(tmp_path, monkeypatch, result, cube_directory, selection))
        assert sorted(levels) == [-0.02, 0.02]

    def test_total_density_gets_a_single_positive_surface(self, tmp_path, monkeypatch, result, cube_directory):
        # An electron density is unsigned; a negative contour would render nothing.
        levels = _isolevels(_render(tmp_path, monkeypatch, result, cube_directory, TOTAL_DENSITY))
        assert levels == [0.02]

    def test_a_negative_isolevel_from_the_slider_is_normalised(self, tmp_path, monkeypatch, result, cube_directory):
        # The slider cannot go negative today, but a negative value must not swap the
        # lobes' colours or collapse both surfaces onto the same sign.
        levels = _isolevels(_render(tmp_path, monkeypatch, result, cube_directory, "MO 2", isolevel=-0.03))
        assert sorted(levels) == [-0.03, 0.03]

    def test_the_chosen_colours_reach_the_viewer(self, tmp_path, monkeypatch, result, cube_directory):
        html = _render(tmp_path, monkeypatch, result, cube_directory, "MO 2")
        colors = re.findall(r'"color":\s*"(#\w+)"', html)
        assert "#0000ff" in colors and "#ff0000" in colors


def test_a_missing_cube_file_raises_a_clear_error(tmp_path, monkeypatch, result, cube_directory):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="no cube file"):
        render_cube_html(result, cube_directory, "MO 9")


def test_renders_do_not_accumulate_previous_views(tmp_path, monkeypatch, result, cube_directory):
    """Each render must embed only its own surfaces.

    nglview's write_html serializes every widget still registered in the process, so
    without pruning them the file grows by a full view per render (about 1 MB with real
    cube data) and drags stale isosurfaces from earlier selections into the current one.
    """
    monkeypatch.chdir(tmp_path)
    sizes = []
    for _ in range(3):
        iframe = render_cube_html(result, cube_directory, "MO 2", "#0000ff", "#ff0000", 0.8, 0.02)
        view_path = _view_path(tmp_path, iframe)
        html = view_path.read_text(encoding="utf-8")
        assert len(_isolevels(html)) == 2, "render embedded surfaces from a previous render"
        sizes.append(view_path.stat().st_size)

    assert len(set(sizes)) == 1, f"viewer HTML grew across renders: {sizes}"


def test_later_render_does_not_replace_an_existing_view(tmp_path, monkeypatch, result, cube_directory):
    monkeypatch.chdir(tmp_path)
    first = render_cube_html(result, cube_directory, "MO 2", isolevel=0.02)
    first_path = _view_path(tmp_path, first)
    original = first_path.read_bytes()

    second = render_cube_html(result, cube_directory, TOTAL_DENSITY, isolevel=0.05)
    assert first != second
    assert first_path.read_bytes() == original
    assert _isolevels(_view_path(tmp_path, second).read_text()) == [0.05]


def test_concurrent_renders_do_not_embed_each_others_widgets(tmp_path, monkeypatch, result, cube_directory):
    monkeypatch.chdir(tmp_path)
    ready = threading.Barrier(2)

    def render(level):
        ready.wait(timeout=10)
        iframe = render_cube_html(result, cube_directory, "MO 2", isolevel=level)
        return _view_path(tmp_path, iframe).read_text(encoding="utf-8")

    with ThreadPoolExecutor(max_workers=2) as pool:
        rendered = list(pool.map(render, [0.02, 0.03]))

    assert sorted(_isolevels(rendered[0])) == [-0.02, 0.02]
    assert sorted(_isolevels(rendered[1])) == [-0.03, 0.03]


def test_structure_views_are_isolated_and_use_the_configured_static_directory(tmp_path, water_mol):
    static_directory = tmp_path / "runtime" / "static"
    first = render_structure_html(water_mol, static_directory=static_directory)
    second = render_structure_html(water_mol, static_directory=static_directory)
    assert first != second
    assert _view_path(tmp_path / "runtime", first).is_file()
    assert _view_path(tmp_path / "runtime", second).is_file()


def test_standalone_renders_do_not_leave_permanent_notebook_threads(
        tmp_path, monkeypatch, water_mol, result, cube_directory):
    from nglview.remote_thread import RemoteCallThread

    monkeypatch.chdir(tmp_path)
    before = {thread for thread in threading.enumerate() if isinstance(thread, RemoteCallThread)}
    for _ in range(3):
        render_structure_html(water_mol)
        render_cube_html(result, cube_directory, "MO 2")
    after = {thread for thread in threading.enumerate() if isinstance(thread, RemoteCallThread)}
    assert after == before, "standalone renders leaked permanent threads holding their widgets alive"
