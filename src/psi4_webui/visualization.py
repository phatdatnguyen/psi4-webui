"""nglview rendering, shared by the structure viewer and the orbital/density viewer.

nglview is a Jupyter widget, so it cannot be embedded in a Gradio page directly. Both
viewers therefore take the same route: build the widget, write it to a standalone HTML
file under ``static/``, and return an ``<iframe>`` pointing at it. Each render gets its
own URL so simultaneous sessions cannot overwrite each other's molecule or surfaces.
"""
import contextlib
import glob
import os
import threading
import uuid

import nglview

from .utils import mol_from_symbols_and_coords

# Where the generated viewer HTML goes. Relative to the process working directory, which
# is what app.py mounts at /static.
_STATIC_DIR = os.path.join(".", "static")
_WIDGET_LOCK = threading.RLock()

# The non-orbital selections the viewer understands. Defined here rather than in the
# Result tab because this module owns both halves of the mapping: which cubeprop task a
# selection needs, and which file name that task produces.
TOTAL_DENSITY = "Electron density"
SPIN_DENSITY = "Spin density"
ELECTROSTATIC_POTENTIAL = "Electrostatic potential"

# Psi4 names each cube file after the task that produced it.
_CUBE_FILE_NAMES = {
    TOTAL_DENSITY: "Dt.cube",
    SPIN_DENSITY: "Ds.cube",
    ELECTROSTATIC_POTENTIAL: "ESP.cube",
}


class _StandaloneNGLWidget(nglview.NGLWidget):
    """An HTML-export widget with no live notebook communication threads."""

    def _initialize_threads(self):
        # nglview 4.0 starts permanent RemoteCallThreads that hold the whole view alive
        # and cannot be stopped. A standalone export never connects a notebook frontend:
        # loaded stays False and write_html serializes the recorded message archive.
        # No remote callbacks need executing in this process.
        pass


def _iframe(src: str, width: int = 600, height: int = 600) -> str:
    """Iframe pointing at a unique render under ``/static``."""
    return (f'<iframe src="{src}" height="{height}" width="{width}" '
            f'title="NGL View" style="border:none;"></iframe>')


@contextlib.contextmanager
def _transient_widgets():
    """Unregister any widgets created inside the block once it finishes.

    This is not optional housekeeping. ``nglview.write_html`` ends up in ipywidgets'
    ``embed_data``, which -- given no explicit state -- serializes **every widget still
    registered in the process**, not just the one it was handed. Each render would
    therefore embed all of its predecessors: the file grows by about a full view per
    render (roughly 1 MB with real cube data) and carries stale isosurfaces from earlier
    selections into the current one. In a long-lived server that compounds without bound.

    The widgets are unregistered rather than closed. ``close()`` severs the comm while
    leaving the object referenced, and the *next* render then fails serializing a widget
    whose ``model_id`` no longer resolves.

    The registry is a private ipywidgets detail, so a version that moves it degrades to
    the old accumulating behaviour rather than breaking the viewer.
    """
    try:
        from ipywidgets.widgets import widget as widget_module
        registry = widget_module._instances
    except Exception:  # pragma: no cover - depends on the ipywidgets version
        registry = None

    # Structure and cube viewers use different Gradio handlers and can run together.
    # Serialize widget creation through HTML export and cleanup: otherwise one render
    # embeds or unregisters the other render's widgets from this process-wide registry.
    with _WIDGET_LOCK:
        before = set(registry) if registry is not None else set()
        try:
            yield
        finally:
            if registry is not None:
                for key in set(registry) - before:
                    registry.pop(key, None)


def _write_view(view, relative_path: str, *, static_directory=None) -> str:
    """Write ``view`` to ``static/<relative_path>`` and return the iframe HTML."""
    stem, extension = os.path.splitext(relative_path)
    relative_path = f"{stem}-{uuid.uuid4().hex}{extension}"
    output_path = os.path.join(_STATIC_DIR if static_directory is None else static_directory, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nglview.write_html(output_path, [view])
    return _iframe("/static/" + relative_path.replace(os.sep, "/"))


def render_structure_html(mol, *, static_directory=None) -> str:
    """Render ``mol`` with per-atom index labels and return the embedding iframe.

    The labels are what let a user map what they see to the geometry Psi4 receives: this
    is the same loader the Calculation tab uses, so atom *i* here is atom *i* there.

    ``labelType="text"`` with an explicit ``labelText`` is what makes this work.
    ``labelType="atomname"`` reads the names out of the PDB block ``show_rdkit``
    generates, and RDKit numbers those per element (C1, C2, O1, ...) -- so the third atom
    of ethanol came out as "O1", the first oxygen, rather than "O3".
    """
    with _transient_widgets():
        view = _StandaloneNGLWidget(nglview.RdkitStructure(mol))
        atom_labels = [f"{atom.GetSymbol()}{atom.GetIdx()}" for atom in mol.GetAtoms()]
        view.add_representation("label", labelType="text", labelText=atom_labels,
                                color="black", showBackground=False)
        return _write_view(view, "structure.html", static_directory=static_directory)


def cube_files_for_selection(output_directory: str, selection: str) -> list[str]:
    """Locate the cube files ``cubeprop`` produced for ``selection``.

    Psi4 names its cube files by task: ``Dt.cube`` / ``Ds.cube`` for the total and spin
    density, ``ESP.cube`` for the electrostatic potential, and
    ``Psi_a_<n>_<n>-<irrep>.cube`` for orbital *n*. The orbital name embeds an irrep label
    that is not known here, hence the glob -- but the directory is wiped before each
    render, so unlike the old version of this app there is never more than one candidate
    and no "pick the newest file" guessing is involved.
    """
    if selection.startswith("MO "):
        pattern = os.path.join(output_directory, f"Psi_a_{int(selection.split()[1])}_*.cube")
        return sorted(glob.glob(pattern))
    name = _CUBE_FILE_NAMES.get(selection)
    if name is None:
        return []
    path = os.path.join(output_directory, name)
    return [path] if os.path.isfile(path) else []


def render_cube_html(result, output_directory: str, selection: str,
                     color1: str = "#0000ff", color2: str = "#ff0000",
                     opacity: float = 0.8, isolevel: float = 0.05, *, static_directory=None) -> str:
    """Render a cube-file isosurface over the molecule and return the embedding iframe.

    The molecule is rebuilt from the result's own geometry rather than from the original
    structure file, so the atoms are in the same frame as the cube grid. (The runner asks
    Psi4 not to reorient or re-centre the molecule, so the two frames coincide -- but
    taking the coordinates from the result is what keeps that true if it ever changes.)

    Signed quantities -- molecular orbitals and the electrostatic potential -- are drawn
    as two surfaces at ``+isolevel`` and ``-isolevel`` in the two chosen colours; a
    density is unsigned and gets a single surface.

    ``isolevelType="value"`` is essential and easy to miss. NGL's surface representation
    defaults to ``isolevelType="sigma"``, which multiplies the given number by the RMS of
    the grid rather than using it directly -- so a chemist's 0.02 a.u. contour silently
    becomes 0.02 sigma, roughly 0.0008 a.u., and renders a huge diffuse blob that clips
    against the edge of the grid box (which is what makes the two lobes of a symmetric
    orbital come out visibly unequal). Setting the type to "value" makes the number mean
    the wavefunction amplitude it appears to mean, and makes +/-isolevel enclose exactly
    mirrored volumes.
    """
    cube_files = cube_files_for_selection(output_directory, selection)
    if not cube_files:
        raise FileNotFoundError(f"Psi4 produced no cube file for '{selection}'.")

    mol = mol_from_symbols_and_coords(result["symbols"], result["geometry_angstrom"])
    cube_path = cube_files[0]
    # A total electron density is the only unsigned quantity here. Orbitals and the ESP
    # obviously change sign, and so does a spin density -- alpha minus beta goes negative
    # wherever spin polarization is inverted, which is exactly the interesting part.
    signed = selection != TOTAL_DENSITY

    with _transient_widgets():
        view = _StandaloneNGLWidget(nglview.RdkitStructure(mol))
        view.add_component(cube_path)
        view.component_1.update_surface(opacity=opacity, color=color1,
                                        isolevelType="value", isolevel=abs(isolevel))
        if signed:
            view.add_component(cube_path)
            view.component_2.update_surface(opacity=opacity, color=color2,
                                            isolevelType="value", isolevel=-abs(isolevel))
        view.camera = "orthographic"

        return _write_view(view, os.path.join("cubes", "view.html"), static_directory=static_directory)
