"""Application entry point.

Builds the FastAPI host, mounts the ``/static`` file server and the Gradio Blocks UI
(the three tabs: conformer generation, calculation, result), and serves everything on
the first free port at/after 7860.

Runtime directories (``data/`` and ``static/``) are created in the **current working
directory** so an installed copy of the package never writes into its own install
location; ``styles.css`` ships inside the package and is read from there. Transient
artifacts left in ``static/`` (and stray Psi4 scratch files) by a previous run are
cleaned up on startup.
"""
import socket
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .working_directory import working_directory_blocks
from .conformer_generation import conformer_generation_tab_content
from .calculation import calculation_tab_content
from .result import result_tab_content

# styles.css is packaged data, resolved relative to this module (read-only).
_STYLES_PATH = Path(__file__).parent / "styles.css"

# Glob patterns for transient files removed from the run directory on startup.
#
# ``timer.dat`` and ``*.clean`` are Psi4's own scratch droppings. Calculation logs are
# deliberately *not* swept: every one this app produces lives inside a working directory
# under ``data/``, so a ``*.log`` at the run root belongs to something else -- a server
# log, or whatever the user happened to leave there -- and deleting it would be pure
# collateral damage.
_TRANSIENT_PATTERNS = (
    "timer.dat",
    "*.clean",
    "static/*.html",
    "static/**/*.html",
    "static/**/*.cube",
    "static/**/*.xyz",
)


def _cleanup_transient_files(run_dir: Path) -> None:
    """Delete transient artifacts left in ``run_dir`` by a previous session."""
    for pattern in _TRANSIENT_PATTERNS:
        for filepath in run_dir.glob(pattern):
            try:
                filepath.unlink()
            except OSError:
                pass  # best-effort cleanup; ignore files we cannot remove


def find_available_port(start_port: int = 7860) -> int:
    """Return the first TCP port at/after ``start_port`` that can be bound on localhost."""
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port  # Available port found
            except OSError:
                port += 1  # Try next port


def build_app(run_dir: Path | None = None) -> FastAPI:
    """Build the FastAPI app with the Gradio UI mounted.

    ``run_dir`` (default: current working directory) is where the ``data/`` and
    ``static/`` directories are created and served from.
    """
    run_dir = Path.cwd() if run_dir is None else run_dir
    _cleanup_transient_files(run_dir)

    app = FastAPI()

    # Working directories live under ./data; transient viewer files under ./static.
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    static_dir = run_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    # mount FastAPI StaticFiles server
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    with gr.Blocks(css_paths=_STYLES_PATH) as blocks:
        with gr.Row():
            working_directory_path_state, working_directory_file_list_state = working_directory_blocks()
            with gr.Column(scale=2):
                with gr.Row(min_height=40):
                    status_markdown = gr.Markdown()
                with gr.Row():
                    with gr.Tabs():
                        conformer_generation_tab_content(working_directory_path_state, working_directory_file_list_state, status_markdown)
                        calculation_tab_content(working_directory_path_state, working_directory_file_list_state, status_markdown)
                        result_tab_content(working_directory_path_state, working_directory_file_list_state, status_markdown)

    # mount Gradio app to FastAPI app
    return gr.mount_gradio_app(app, blocks, path="/")


def main() -> None:
    """Console-script entry point: build the app and serve it on the first free port."""
    app = build_app()
    port = find_available_port()
    uvicorn.run(app, host="127.0.0.1", port=port, access_log=False)


if __name__ == "__main__":
    main()
