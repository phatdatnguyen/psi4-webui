import os
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from gradio_molecule2d import molecule2d
from rdkit import Chem
from rdkit.Chem import AllChem
import psi4
import nglview
from utils import *

def on_create_molecule(molecule_editor: molecule2d):
    os.makedirs("./structures", exist_ok=True)
    file_path = "./structures/molecule_opt.pdb"
    try:
        global mol
        mol = Chem.MolFromSmiles(molecule_editor)
        mol = Chem.AddHs(mol)
        smiles = Chem.CanonSmiles(molecule_editor)
        AllChem.EmbedMolecule(mol)

        Chem.MolToPDBFile(mol, file_path)

        # Create the NGL view widget
        view = nglview.show_rdkit(mol)
        
        # Write the widget to HTML
        if os.path.exists('./static/molecule_opt.html'):
            os.remove('./static/molecule_opt.html')
        nglview.write_html('./static/molecule_opt.html', [view])

        # Read the HTML file
        timestamp = int(time.time())
        html = f'<iframe src="/static/molecule_opt.html?ts={timestamp}" height="300" width="400" title="NGL View"></iframe>'
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))
        return [None, None, None]
    
    return smiles, html, gr.update(interactive=True)

def on_upload_molecule(load_molecule_uploadbutton: gr.UploadButton):
    file_path = load_molecule_uploadbutton
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() != ".pdb":
        gr.Warning("Invalid file!\nFile must be in .pdb format.")
        return [None, None, None]

    try:
        global mol
        mol = Chem.MolFromPDBFile(file_path, sanitize=False, removeHs=False)
        smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))
        AllChem.EmbedMolecule(mol)

        Chem.MolToPDBFile(mol, file_path)

        # Create the NGL view widget
        view = nglview.show_rdkit(mol)
        
        # Write the widget to HTML
        if os.path.exists('./static/molecule_opt.html'):
            os.remove('./static/molecule_opt.html')
        nglview.write_html('./static/molecule_opt.html', [view])

        # Read the HTML file
        timestamp = int(time.time())
        html = f'<iframe src="/static/molecule_opt.html?ts={timestamp}" height="300" width="400" title="NGL View"></iframe>'
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))
        return [None, None, None]  
    
    return smiles, html, gr.update(interactive=True)

def on_method_change(method_radio: gr.Radio):
    if method_radio == "Density Functional Theory":
        return gr.Textbox(label="Functional", value="B3LYP", visible=True)
    return gr.Textbox(label="Functional", value="B3LYP", visible=False)

def on_optimizer_geometry(method_radio: gr.Radio, reference_dropdown: gr.Dropdown, basis_set_dropdown: gr.Dropdown,
                          functional_textbox: gr.Textbox, charge_slider: gr.Slider, multiplicity_dropdown: gr.Dropdown,
                          memory_slider: gr.Slider, num_threads_slider: gr.Slider, iterations_slider: gr.Slider,
                          step_type_dropdown: gr.Dropdown, full_hess_every_slider: gr.Slider, convergence_dropdown: gr.Dropdown):
    energy_textbox = "Not calculated"
    dipole_moment_textbox = "Not calculated"
    try:
        # Set calculation options
        psi4.set_memory(memory_slider*1024*1024*1024)
        psi4.set_num_threads(num_threads_slider)
        psi4.set_options({"REFERENCE": reference_dropdown})
        psi4.set_options({"BASIS": basis_set_dropdown})
        psi4.set_options({"GEOM_MAXITER": iterations_slider})
        psi4.set_options({"STEP_TYPE": step_type_dropdown})
        psi4.set_options({"FULL_HESS_EVERY": full_hess_every_slider})
        psi4.set_options({"G_CONVERGENCE": convergence_dropdown})

        # Write the geometry to XYZ string
        xyz_string = generate_xyz_string(mol, charge_slider, multiplicity_dropdown)

        # Get the psi4 geometry
        geometry = psi4.geometry(xyz_string)

        # Run calculation
        start = time.time()

        global wfn
        if method_radio == "Hartree-Fock":
            psi4.set_options({"SCF_TYPE": "PK"})
            energy, wfn, history = psi4.optimize("HF", molecule=geometry, optking__geom_maxiter=iterations_slider, return_wfn=True, return_history=True)
        elif method_radio == "Self-Consistent Field":
            psi4.set_options({"SCF_TYPE": "PK"})
            energy, wfn, history = psi4.optimize("SCF", molecule=geometry, optking__geom_maxiter=iterations_slider, return_wfn=True, return_history=True)
        else:
            psi4.set_options({"SCF_TYPE": "DF"})
            energy, wfn, history = psi4.optimize(functional_textbox, molecule=geometry, optking__geom_maxiter=iterations_slider, return_wfn=True, return_history=True)

        end = time.time()
        duration = end - start

        # Get results
        energy_values = history['energy']
        energy_plot = plt.figure()
        plt.plot(energy_values)
        plt.xlabel("Geometry")
        plt.ylabel("Energy (kcal/mol)")
        plt.title("Optimization")

        coordinates_list = history['coordinates']
        num_confs = len(coordinates_list)
        for conformer_idx, coordinates in enumerate(coordinates_list):
            conf = mol.GetConformer()
            for i, coord in enumerate(coordinates):
                conf.SetAtomPosition(i, coord)
            conformers_file_path = f"./structures/opt_conformer{conformer_idx}.pdb"
            Chem.MolToPDBFile(mol, conformers_file_path)
        conformer_dropdown = gr.Dropdown(label="Conformer", value=num_confs-1, choices=list(range(0, num_confs)), visible=True)

        energy = energy * psi4.constants.hartree2kcalmol
        energy_textbox = "{:.4f} (kcal/mol)".format(energy)
        dipole_moment = wfn.variable("CURRENT DIPOLE")
        dipole_moment_textbox = "{:.4f} (Debye)".format(np.linalg.norm(dipole_moment))

        MO_energies = wfn.epsilon_a_subset("AO", "ALL").to_array()
        visualization_dropdown_choices=["Electron density"]
        MO_df = pd.DataFrame(columns=["Molecular orbital", "Energy (kcal/mol)"])
        for i, MO_energy in enumerate(MO_energies):
            MO_energy = MO_energy * psi4.constants.hartree2kcalmol
            MO_name = f"MO {i+1}"
            MO_df = MO_df._append({"Molecular orbital": MO_name, "Energy (kcal/mol)": "{:.4f}".format(MO_energy)}, ignore_index=True)
            visualization_dropdown_choices.append(MO_name)
        visualization_dropdown = gr.Dropdown(label="Visualization", value="Electron density", choices=visualization_dropdown_choices)
    except Exception as exc:
        gr.Warning("Optimization error!\n" + str(exc))
        return [None, None, None, None, None, None, None, None, None, None]

    calculation_status = "Optimization complete. ({0:.3f} s)".format(duration)
    return calculation_status, energy_plot, conformer_dropdown, conformers_file_path, energy_textbox, dipole_moment_textbox, visualization_dropdown, gr.update(interactive=True), "", MO_df

def on_conformer_change(conformer_dropdown: gr.Dropdown):
    conformers_file_path = f"./structures/opt_conformer{conformer_dropdown}.pdb"

    Chem.MolToPDBFile(mol, conformers_file_path)

    # Create the NGL view widget
    view = nglview.show_rdkit(mol)
    
    # Write the widget to HTML
    if os.path.exists('./static/opt_conformer.html'):
        os.remove('./static/opt_conformer.html')
    nglview.write_html('./static/opt_conformer.html', [view])

    # Read the HTML file
    timestamp = int(time.time())
    html = f'<iframe src="/static/opt_conformer.html?ts={timestamp}" height="300" width="400" title="NGL View"></iframe>'

    return html

def on_visualization_change(visualization_dropdown: gr.Dropdown):
    if visualization_dropdown == "Electron density":
        return gr.Slider(label="Isolevel", value=0.5, minimum=0, maximum=1, step=0.01, visible=True)
    return gr.Slider(label="Isolevel", value=0.5, minimum=0, maximum=1, step=0.01, visible=False)

def on_visualization(visualization_dropdown: gr.Dropdown, visualization_color1: gr.ColorPicker, visualization_color2: gr.ColorPicker, visualization_opacity: gr.Slider, visualization_isolevel: gr.Slider):
    # Set options for cube file generation
    try:
        os.makedirs("./static/opt_visualization", exist_ok=True)
        if visualization_dropdown == "Electron density":
            psi4.set_options({'CUBEPROP_TASKS': ['DENSITY'],
                              'CUBIC_GRID_SPACING': [0.1, 0.1, 0.1],
                              'CUBEPROP_FILEPATH': './static/opt_visualization'})
            
            # Generate the cube file
            psi4.cubeprop(wfn)

            # Get the latest cube file
            view = nglview.show_rdkit(mol)
            view.add_component("./static/opt_visualization/Dt.cube")

            # Adjust visualization settings
            view.component_1.update_surface(opacity=visualization_opacity, color=visualization_color1, isolevel=visualization_isolevel)
            view.camera = 'orthographic'
        else:
            MO_index = int(visualization_dropdown.split(" ")[1])
            psi4.set_options({'CUBEPROP_TASKS': ['ORBITALS'],
                              'CUBEPROP_ORBITALS': [MO_index, -MO_index],
                              'CUBIC_GRID_SPACING': [0.1, 0.1, 0.1],
                              'CUBEPROP_FILEPATH': './static/opt_visualization'})
            
            # Generate the cube file
            psi4.cubeprop(wfn)

            # Get the latest cube file
            view = nglview.show_rdkit(mol)
            a_cube_filepath = max(glob.glob("./static/opt_visualization/Psi_a_{}_*.cube".format(MO_index)), key=os.path.getctime)
            view.add_component(a_cube_filepath)
            view.add_component(a_cube_filepath)

            # Adjust visualization settings
            view.component_1.update_surface(opacity=visualization_opacity, color=visualization_color1, isolevel=2)
            view.component_2.update_surface(opacity=visualization_opacity, color=visualization_color2, isolevel=-2)
            view.camera = 'orthographic'
        
        # Write the widget to HTML
        if os.path.exists('./static/opt_visualization/cube_file.html'):
            os.remove('./static/opt_visualization/cube_file.html')
        nglview.write_html('./static/opt_visualization/cube_file.html', [view])

        # Read the HTML file
        timestamp = int(time.time())
        html = f'<iframe src="/static/opt_visualization/cube_file.html?ts={timestamp}" height="400" width="600" title="NGL View"></iframe>'
    except Exception as exc:
        gr.Warning("Visualization error!\n" + str(exc))
        return None
    
    return html

def geometry_optimization_tab_content():
    with gr.Tab("Geometry Optimization") as geometry_optimization_tab:
        with gr.Accordion("Molecule"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    molecule_editor = molecule2d(label="Molecule")
                with gr.Column(scale=1):
                    create_molecule_button = gr.Button(value="Create molecule")
                    smiles_texbox = gr.Textbox(label="SMILES")
                    molecule_viewer = gr.HTML(label="Molecule")
                    load_molecule_uploadbutton = gr.UploadButton(label="Load molecule")
        with gr.Accordion("Geometry Optimization"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    method_radio = gr.Radio(label="Method", value="Hartree-Fock", choices=["Hartree-Fock", "Self-Consistent Field", "Density Functional Theory"])
                    reference_dropdown = gr.Dropdown(label="Reference", value="RHF", choices=["RHF", "UHF", "ROHF"])
                with gr.Column(scale=1):
                    basis_set_dropdown = gr.Dropdown(label="Basis set",
                                                     value="STO-3G",
                                                     allow_custom_value=True,
                                                     choices=["STO-3G", "3-21G", "6-31G", "6-31G*", "6-31G**", "6-311G", "6-311G*", "6-311G**",
                                                              "6-31+G", "6-31+G*", "6-31+G**", "6-311+G", "6-311+G*", "6-311+G**",
                                                              "6-311++G", "6-311++G*", "6-311++G**",
                                                              "cc-pV5Z", "cc-pVDZ", "cc-pVQZ", "cc-pVTZ",
                                                              "aug-cc-pV5Z", "aug-cc-pVDZ", "aug-cc-pVQZ", "aug-cc-pVTZ"])
                    functional_textbox = gr.Textbox(label="Functional", value="B3LYP", visible=False)
                    charge_slider = gr.Slider(label="Charge", value=0, minimum=-2, maximum=2, step=1)
                    multiplicity_dropdown = gr.Dropdown(label="Multiplicity", value=1, choices=[("Singlet", 1), ("Doublet", 2),
                                                                                                 ("Triplet", 3), ("Quartet", 4),
                                                                                                 ("Quintet", 5), ("Sextet ", 6)])
                with gr.Column(scale=1):
                    memory_slider = gr.Slider(label="Memory (GB)", value=16, minimum=1, maximum=32, step=1)
                    num_threads_slider = gr.Slider(label="Number of threads", value=16, minimum=1, maximum=32, step=1)
                    iterations_slider = gr.Slider(label="Max iterations", value=500, minimum=1, maximum=10000, step=1)
                    step_type_dropdown = gr.Dropdown(label="Step type", value="RFO",
                                                     choices=["RFO", "P_RFO", "NR", "SD", "LINESEARCH"])
                    full_hess_every_slider = gr.Slider(label="Full Hessian every", value=-1, minimum=-1, maximum=1000, step=1)
                    convergence_dropdown = gr.Dropdown(label="Convergence", value="GAU",
                                                       choices=["GAU", "GAU_LOOSE", "GAU_TIGHT", "GAU_VERYTIGHT", "NWCHEM_LOOSE",
                                                                "TURBOMOLE", "CFOUR", "QCHEM", "MOLPRO", "INTERFRAG_TIGHT"])
                    optimize_geometry_button = gr.Button(value="Optimize geometry", interactive=False)
        with gr.Accordion("Optimization Results"):
            with gr.Row(equal_height=True):
                status_markdown = gr.Markdown()
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    energy_plot = gr.Plot(label="Energy plot")
                with gr.Column(scale=1):
                    conformer_dropdown = gr.Dropdown(label="Conformer", value=0, choices=[0], visible=False)
                    conformers_viewer = gr.HTML(label="Conformer")
            with gr.Row(equal_height=True):        
                with gr.Column(scale=1):
                    with gr.Row():
                        energy_texbox = gr.Textbox(label="Energy", value="Not calculated")
                        dipole_moment_texbox = gr.Textbox(label="Dipole moment", value="Not calculated")
                    visualization_dropdown = gr.Dropdown(label="Visualization", value="Electron density", choices=["Electron density"])
                    with gr.Row():
                        visualization_color1 = gr.ColorPicker(label="Color 1", value="#0000ff")
                        visualization_color2 = gr.ColorPicker(label="Color 2", value="#ff0000")
                        visualization_opacity = gr.Slider(label="Opacity", value=0.8, minimum=0, maximum=1, step=0.01)
                        visualization_isolevel = gr.Slider(label="Isolevel", value=0.05, minimum=0, maximum=1, step=0.01)
                    visualize_button = gr.Button(value="Visualize", interactive=False)
                    visualization_html = gr.HTML(label="Visualization")
                with gr.Column(scale=1):
                    MO_dataframe = gr.DataFrame(label="Molecular orbitals")
                
        create_molecule_button.click(on_create_molecule, molecule_editor, [smiles_texbox, molecule_viewer, optimize_geometry_button])
        load_molecule_uploadbutton.upload(on_upload_molecule, load_molecule_uploadbutton, [smiles_texbox, molecule_viewer, optimize_geometry_button])
        method_radio.change(on_method_change, method_radio, functional_textbox)
        optimize_geometry_button.click(on_optimizer_geometry, [method_radio, reference_dropdown, basis_set_dropdown, functional_textbox,
                                                               charge_slider, multiplicity_dropdown, memory_slider, num_threads_slider,
                                                               iterations_slider, step_type_dropdown, full_hess_every_slider, convergence_dropdown],
                                                               [status_markdown, energy_plot, conformer_dropdown, conformers_viewer,
                                                                energy_texbox, dipole_moment_texbox, visualization_dropdown, visualize_button, visualization_html, MO_dataframe])
        conformer_dropdown.change(on_conformer_change, conformer_dropdown, conformers_viewer)
        visualization_dropdown.change(on_visualization_change, visualization_dropdown, visualization_isolevel)
        visualize_button.click(on_visualization, [visualization_dropdown, visualization_color1, visualization_color2, visualization_opacity, visualization_isolevel], [visualization_html])
        
    return geometry_optimization_tab