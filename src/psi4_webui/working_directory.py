"""Working-directory column: directory selection, file browser, and file viewers.

Owns the left-hand column of the UI. A "working directory" is a subfolder under
``data/`` in which all structure files, Psi4 job specs/logs/results, and exports live.
This module also holds the two shared ``gr.State`` objects (current path and file list)
that the other tabs subscribe to; any handler that changes files on disk returns a
fresh file list into that state so every tab's dropdowns stay in sync.
"""
import os
import shutil
import time
import pandas as pd
import gradio as gr
from .utils import JOB_SUFFIX, RESULT_SUFFIX, get_files_in_working_directory, mol_from_structure_file
from .visualization import render_structure_html

def get_working_directories() -> list[str]:
    """List the existing working-directory names (immediate subfolders of ``data/``)."""
    base_path = "./data/"
    return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

def on_open_working_directory(working_directory):
    """Create (if needed) and open ``working_directory`` under ``data/``.

    Returns updates for the directory dropdown, the path/file-list states, and the
    two buttons that become interactive once a directory is open. Warns and no-ops
    when the name is blank.
    """
    if working_directory is None or working_directory.strip() == "":
        gr.Warning("Please specify a working directory.")
        return gr.update(), None, None, gr.update(), gr.update()
    
    working_directory_path = os.path.join("./data/", working_directory)
    os.makedirs(working_directory_path, exist_ok=True)
    files = get_files_in_working_directory(working_directory_path)
    
    return gr.update(choices=get_working_directories(), value=working_directory), working_directory_path, files, gr.update(interactive=True), gr.update(interactive=True)

def on_file_list_change(working_directory_path) -> pd.DataFrame:
    """Build the file-browser DataFrame (File / Type / Modified) for the directory.

    Each file is classified by extension into a human-readable type, and rows are
    ordered by actual modification time (newest first). Yields an empty table when no
    working directory is open, since this fires on the shared state's ``.change`` and
    that state is ``None`` before the first directory is opened.
    """
    if not working_directory_path:
        return pd.DataFrame(columns=["File", "Type", "Modified"])

    files = get_files_in_working_directory(working_directory_path)

    # Update the file dataframe
    file_info = []
    for f in files:
        if f.endswith('.xyz') or f.endswith('.pdb') or f.endswith('.mol') or f.endswith('.mol2'):
            file_type = "Structure file"
        elif f.endswith(RESULT_SUFFIX):
            file_type = "Result file"
        elif f.endswith(JOB_SUFFIX):
            file_type = "Job spec file"
        elif f.endswith('.npy'):
            file_type = "Wavefunction file"
        elif f.endswith('.log'):
            file_type = "Log file"
        elif f.endswith('.cube'):
            file_type = "Cube file"
        elif f.endswith('.csv'):
            file_type = "Exported data file"
        elif f.endswith('.txt'):
            file_type = "Text file"
        elif f == 'timer.dat' or f.endswith('.clean'):
            # Psi4 leaves these behind; "Clean Working Directory" removes them.
            file_type = "Psi4 scratch file"
        else:
            file_type = "Other File"
        
        file_path = os.path.join(working_directory_path, f)
        mtime = os.path.getmtime(file_path)
        file_info.append([f, file_type, mtime])
    # Sort by the numeric modification time (newest first), then format for display.
    # Sorting the ctime *string* would order by weekday name, not chronologically.
    file_info.sort(key=lambda x: x[2], reverse=True)
    file_info = [[f, file_type, time.ctime(mtime)] for f, file_type, mtime in file_info]
    file_df = pd.DataFrame(file_info, columns=["File", "Type", "Modified"])

    return file_df

def on_select_file(evt: gr.SelectData):
    """Handle a row click in the file browser.

    Returns the selected file name plus two derived selections (a viewable structure
    file and a viewable text file, each ``None`` when the type doesn't apply) and
    enables the delete button. These drive which viewer buttons become active.
    """
    selected_file_name = evt.row_value[0]
    # Psi4's .log is plain prose, not a coordinate file, so (unlike ORCA's) it is not
    # offered to the 3D viewer -- optimized geometries are written as .xyz instead.
    if selected_file_name.endswith(('.xyz', '.pdb', '.mol', '.mol2')):
        selected_structure_file = selected_file_name
    else:
        selected_structure_file = None
    if selected_file_name.endswith(('.xyz', '.pdb', '.mol', '.mol2', '.log', '.json', '.csv', '.txt')):
        selected_text_file = selected_file_name
    else:
        selected_text_file = None
    
    return selected_file_name, selected_structure_file, selected_text_file, gr.update(interactive=True)

def on_selected_structure_file_state_change(state):
    """Enable the "View Structure File" button only when a structure file is selected."""
    return gr.update(interactive=(state is not None))

def on_selected_text_file_state_change(state):
    """Enable the "View Text File" button only when a text-viewable file is selected."""
    return gr.update(interactive=(state is not None))

def on_upload_file(working_directory_path, file_path):
    """Copy an uploaded file into the working directory; return the refreshed file list."""
    shutil.copy2(file_path, os.path.join(working_directory_path, os.path.basename(file_path)))
    return get_files_in_working_directory(working_directory_path)

def on_delete_file(working_directory_path, selected_file_name):
    """Delete the selected file; return the refreshed file list (warns on error)."""
    if selected_file_name is None:
        return get_files_in_working_directory(working_directory_path)
    
    file_path = os.path.join(working_directory_path, selected_file_name)
    try:
        os.remove(file_path)
        status = "File deleted successfully."
    except Exception as exc:
        status = "Error deleting file!\n" + str(exc)
    gr.Warning(status)
    
    return get_files_in_working_directory(working_directory_path)

def on_clean_working_directory(working_directory_path):
    """Remove scratch/leftover files (ORCA temp, ``.bibtex``, ``Zone.Identifier``, etc.).

    Returns the refreshed file list; warns on error.
    """
    try:
        # Psi4 scratch: psi.<pid>.<n>.clean / *.clean, timer.dat, and the usual temp files.
        files_to_clean = [f for f in os.listdir(working_directory_path)
                          if f.startswith('#') or f.endswith(".tmp") or f.endswith(".clean")
                          or f == "timer.dat" or f.endswith("Zone.Identifier") or ".tmp." in f]
        for f in files_to_clean:
            file_path = os.path.join(working_directory_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        status = "Working directory cleaned successfully."
    except Exception as exc:
        status = "Error cleaning working directory!\n" + str(exc)
    gr.Warning(status)
    
    return get_files_in_working_directory(working_directory_path)

def on_view_structure_file(working_directory_path, file_name):
    """Render a 3D viewer for the selected structure/output file.

    Loads the geometry with :func:`mol_from_structure_file` — the same loader the
    Calculation tab uses to build the geometry Psi4 receives, so the atom indices
    labelled here are the indices Psi4 works in. Rendering is delegated to
    :func:`psi4_webui.visualization.render_structure_html`, which the orbital viewer in
    the Result tab shares. Returns ``None`` and warns on error.
    """
    try:
        file_path = os.path.join(working_directory_path, file_name)
        return render_structure_html(mol_from_structure_file(file_path))
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))
        return None

def on_view_text_file(working_directory_path, text_file_name):
    """Load a text-viewable file into the editable text viewer.

    Returns updates for the text area (label + content, made editable) and the save
    button. Warns and no-ops on read error.
    """
    text_file_path = os.path.join(working_directory_path, text_file_name)
    try:
        with open(text_file_path, 'r') as file:
            content = file.read()
        return gr.update(label=f"Text File Viewer - {text_file_name}", value=content, interactive=True), gr.update(interactive=True)
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))
        return gr.update(), gr.update()

def on_save_text_file(working_directory_path, text_file_name, text_content):
    """Write the text-viewer contents back to the selected file.

    Returns the refreshed file list; warns if no file is selected or on write error.
    """
    if text_file_name is None:
        gr.Warning("Please select a text file to save.")
        return get_files_in_working_directory(working_directory_path)
    
    text_file_path = os.path.join(working_directory_path, text_file_name)
    try:
        with open(text_file_path, 'w') as file:
            file.write(text_content)
        status = "File saved successfully."
    except Exception as exc:
        status = "Error saving file!\n" + str(exc)
    gr.Warning(status)
    
    return get_files_in_working_directory(working_directory_path)

def working_directory_blocks():
    """Build the working-directory column and wire its events.

    Returns the two shared ``gr.State`` objects — ``(working_directory_path_state,
    working_directory_file_list_state)`` — which the calculation and result tabs
    subscribe to.
    """
    with gr.Column(scale=1):
        working_directory_dropdown = gr.Dropdown(label="Working Directory", choices=get_working_directories(), value="wd", allow_custom_value=True)
        working_directory_path_state = gr.State()
        open_working_directory_button = gr.Button(value="Create/Open Working Directory")
        working_directory_file_list_state = gr.State()
        working_directory_file_dataframe = gr.Dataframe(label="Files in Working Directory", headers=["File", "Type", "Modified"], max_height=360, wrap=True, interactive=False)
        selected_file_state = gr.State()
        selected_structure_file_state = gr.State()
        selected_text_file_state = gr.State()
        with gr.Row():
            add_file_upload_button = gr.UploadButton(label="Add File", file_types=[".xyz", ".pdb", ".mol", ".mol2"], interactive=False)
            delete_file_button = gr.Button(value="Delete Selected File", interactive=False)
            clean_working_directory_button = gr.Button(value="Clean Working Directory", interactive=False)
        view_structure_button = gr.Button(value="View Structure File", interactive=False)
        structure_viewer_html = gr.HTML()
        view_text_file_button = gr.Button(value="View Text File", interactive=False)
        text_file_viewer_textarea = gr.TextArea(label="Text File Viewer", lines=20, elem_id="textfile_viewer", interactive=False)
        save_text_file_button = gr.Button(value="Save Text File", interactive=False)
    
    working_directory_dropdown.change(on_open_working_directory, working_directory_dropdown, [working_directory_dropdown, working_directory_path_state, working_directory_file_list_state, add_file_upload_button, clean_working_directory_button])
    open_working_directory_button.click(on_open_working_directory, working_directory_dropdown, [working_directory_dropdown, working_directory_path_state, working_directory_file_list_state, add_file_upload_button, clean_working_directory_button])
    working_directory_file_list_state.change(on_file_list_change, working_directory_path_state, working_directory_file_dataframe)
    working_directory_file_dataframe.select(on_select_file, [], [selected_file_state, selected_structure_file_state, selected_text_file_state, delete_file_button])
    selected_structure_file_state.change(on_selected_structure_file_state_change, selected_structure_file_state, view_structure_button)
    selected_text_file_state.change(on_selected_text_file_state_change, selected_text_file_state, view_text_file_button)
    add_file_upload_button.upload(on_upload_file, [working_directory_path_state, add_file_upload_button], working_directory_file_list_state)
    delete_file_button.click(on_delete_file, [working_directory_path_state, selected_file_state], working_directory_file_list_state)
    clean_working_directory_button.click(on_clean_working_directory, working_directory_path_state, working_directory_file_list_state)
    view_structure_button.click(on_view_structure_file, [working_directory_path_state, selected_structure_file_state], structure_viewer_html)
    view_text_file_button.click(on_view_text_file, [working_directory_path_state, selected_text_file_state], [text_file_viewer_textarea, save_text_file_button])
    save_text_file_button.click(on_save_text_file, [working_directory_path_state, selected_text_file_state, text_file_viewer_textarea], working_directory_file_list_state)

    return working_directory_path_state, working_directory_file_list_state