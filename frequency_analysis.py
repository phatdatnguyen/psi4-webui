import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from gradio_molecule2d import molecule2d
from gradio_molecule3d import Molecule3D
from rdkit import Chem
from rdkit.Chem import AllChem
import psi4
from utils import *

def on_create_molecule(molecule_editor: molecule2d):
    os.makedirs(".\\structures", exist_ok=True)
    file_path = ".\\structures\\molecule.pdb"
    try:
        mol = Chem.MolFromSmiles(molecule_editor)
        mol = Chem.AddHs(mol)
        smiles = Chem.CanonSmiles(molecule_editor)
        AllChem.EmbedMolecule(mol)

        Chem.MolToPDBFile(mol, file_path)
    except Exception as exc:
        gr.Warning("Error!\n" + str(exc))
        return [None, None]
    
    return smiles, file_path

reps = [
    {
      "model": 0,
      "chain": "",
      "resname": "",
      "style": "stick",
      "color": "whiteCarbon",
      "residue_range": "",
      "around": 0,
      "byres": False,
      "visible": False
    }
]

def on_method_change(method_radio = gr.Radio):
    if method_radio == "Density Functional Theory":
        return gr.Textbox(label="Functional", value="B3LYP", visible=True)
    return gr.Textbox(label="Functional", value="B3LYP", visible=False)

def on_run_geometry_optimization_change(run_geometry_optimization: gr.Checkbox):
    if run_geometry_optimization:
        iterations_slider = gr.Slider(label="Max iterations", value=500, minimum=1, maximum=10000, step=1, visible=True)
        step_type_dropdown = gr.Dropdown(label="Step type", value="RFO",
                                         choices=["RFO", "P_RFO", "NR", "SD", "LINESEARCH"], visible=True)
        full_hess_every_slider = gr.Slider(label="Full Hessian every", value=-1, minimum=-1, maximum=1000, step=1, visible=True)
        convergence_dropdown = gr.Dropdown(label="Convergence", value="GAU",
                                           choices=["GAU", "GAU_LOOSE", "GAU_TIGHT", "GAU_VERYTIGHT", "NWCHEM_LOOSE",
                                                    "TURBOMOLE", "CFOUR", "QCHEM", "MOLPRO", "INTERFRAG_TIGHT"],
                                           visible=True)
        return iterations_slider, step_type_dropdown, full_hess_every_slider, convergence_dropdown
    else:
        iterations_slider = gr.Slider(label="Max iterations", value=500, minimum=1, maximum=10000, step=1, visible=False)
        step_type_dropdown = gr.Dropdown(label="Step type", value="RFO",
                                         choices=["RFO", "P_RFO", "NR", "SD", "LINESEARCH"], visible=False)
        full_hess_every_slider = gr.Slider(label="Full Hessian every", value=-1, minimum=-1, maximum=1000, step=1, visible=False)
        convergence_dropdown = gr.Dropdown(label="Convergence", value="GAU",
                                           choices=["GAU", "GAU_LOOSE", "GAU_TIGHT", "GAU_VERYTIGHT", "NWCHEM_LOOSE",
                                                    "TURBOMOLE", "CFOUR", "QCHEM", "MOLPRO", "INTERFRAG_TIGHT"],
                                           visible=False)
        return iterations_slider, step_type_dropdown, full_hess_every_slider, convergence_dropdown

def lorentzian(wavenumber, position, intensity, width=10):
    return intensity * (width ** 2) / ((wavenumber - position) ** 2 + width ** 2)

def on_frequency_analysis(method_radio: gr.Radio, reference_dropdown: gr.Dropdown, basis_set_dropdown: gr.Dropdown,
                          functional_textbox: gr.Textbox, charge_slider: gr.Slider, multiplicity_dropdown: gr.Dropdown,
                          memory_slider: gr.Slider, num_threads_slider: gr.Slider, run_geometry_optimization_checkbox: gr.Checkbox,
                          iterations_slider: gr.Slider, step_type_dropdown: gr.Dropdown, full_hess_every_slider: gr.Slider,
                          convergence_dropdown: gr.Dropdown):
    file_path = '.\\structures\\molecule.pdb'
    energy_textbox = "Not calculated"
    dipole_moment_textbox = "Not calculated"
    try:
        # Load molecule
        mol = Chem.MolFromPDBFile(file_path)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        
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
        # Geometry optimization
        if run_geometry_optimization_checkbox:
            if method_radio == "Hartree-Fock":
                psi4.set_options({"SCF_TYPE": "PK"})
                _, wfn0, history = psi4.optimize("HF", molecule=geometry, optking__geom_maxiter=iterations_slider, return_wfn=True, return_history=True)
            elif method_radio == "Self-Consistent Field":
                psi4.set_options({"SCF_TYPE": "PK"})
                _, wfn0, history = psi4.optimize("SCF", molecule=geometry, optking__geom_maxiter=iterations_slider, return_wfn=True, return_history=True)
            else:
                psi4.set_options({"SCF_TYPE": "DF"})
                _, wfn0, history = psi4.optimize(functional_textbox, molecule=geometry, optking__geom_maxiter=iterations_slider, return_wfn=True, return_history=True)
       
        # Frequency analysis
        if method_radio == "Hartree-Fock":
            psi4.set_options({"SCF_TYPE": "PK"})
            energy, wfn = psi4.frequency("HF", return_wfn=True)
        elif method_radio == "Self-Consistent Field":
            psi4.set_options({"SCF_TYPE": "PK"})
            energy, wfn = psi4.frequency("SCF", return_wfn=True)
        else:
            psi4.set_options({"SCF_TYPE": "DF"})
            energy, wfn = psi4.frequency(functional_textbox, molecule=geometry, return_wfn=True)

        end = time.time()
        duration = end - start

        # Get results
        if run_geometry_optimization_checkbox:
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
                conformers_file_path = f".\\structures\\conformer{conformer_idx}.pdb"
                Chem.MolToPDBFile(mol, conformers_file_path)
            conformer_dropdown = gr.Dropdown(label="Conformer", value=num_confs-1, choices=list(range(0, num_confs)), visible=True)

            MO_energies = wfn0.epsilon_a_subset("AO", "ALL").to_array()
            MO_df = pd.DataFrame(columns=["Molecular orbital", "Energy (kcal/mol)"])
            for i, MO_energy in enumerate(MO_energies):
                MO_energy = MO_energy * psi4.constants.hartree2kcalmol
                if i == wfn.nalpha() - 1:
                    MO_df = MO_df._append({"Molecular orbital": f"MO {i+1} (HOMO)", "Energy (kcal/mol)": "{:.4f}".format(MO_energy)}, ignore_index=True)
                elif i == wfn.nalpha():
                    MO_df = MO_df._append({"Molecular orbital": f"MO {i+1} (LUMO)", "Energy (kcal/mol)": "{:.4f}".format(MO_energy)}, ignore_index=True)
                else:
                    MO_df = MO_df._append({"Molecular orbital": f"MO {i+1}", "Energy (kcal/mol)": "{:.4f}".format(MO_energy)}, ignore_index=True)
        else:
            energy_plot = None
            conformer_dropdown = gr.Dropdown(label="Conformer", visible=False)
            conformers_file_path = None
            MO_df = None
        
        energy = energy * psi4.constants.hartree2kcalmol
        energy_textbox = "{:.4f} (kcal/mol)".format(energy)
        dipole_moment = wfn.variable("CURRENT DIPOLE")
        dipole_moment_textbox = "{:.4f} (Debye)".format(np.linalg.norm(dipole_moment))

        frequencies = wfn.frequencies().to_array()
        vib_info = wfn.frequency_analysis
        ir_intensities = vib_info['IR_intensity'].data
        wavenumbers = np.linspace(min(frequencies) - 100, max(frequencies) + 100, 2000)
        spectrum = np.zeros_like(wavenumbers)
        for freq, intensity in zip(frequencies, ir_intensities):
            spectrum += lorentzian(wavenumbers, freq, intensity)
        ir_spectrum = plt.figure()
        plt.plot(wavenumbers, spectrum, 'k')
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.xlabel("Wavenumber ($cm^{-1}$)")
        plt.ylabel("Transmittance")
        plt.title("IR Spectrum")

        zpve = psi4.core.variable('ZPVE') * psi4.constants.hartree2kcalmol
        e_thermo = psi4.core.variable('THERMAL ENERGY') * psi4.constants.hartree2kcalmol
        h = psi4.core.variable('ENTHALPY') * psi4.constants.hartree2kcalmol
        g = psi4.core.variable('GIBBS FREE ENERGY') * psi4.constants.hartree2kcalmol
        s = (h - g) / 298.15 * 1000

        thermo_html = """
        <div>
            <table>
                <thead>
                    <tr><td><b>Thermodynamic quantity</b></td><td><b>Value</b></td></tr>
                </thead>
                <tbody>
                    <tr><td>Zero-point potential energy</td><td>{:.4f} kcal/mol</td></tr>
                    <tr><td>Thermal energy (295.15 K)</td><td>{:.4f} kcal/mol</td></tr>
                    <tr><td>Enthalpy (295.15 K)</td><td>{:.4f} kcal/mol</td></tr>
                    <tr><td>Entropy (295.15 K)</td><td>{:.4f} cal/(mol·K)</td></tr>
                    <tr><td>Gibbs free energy (295.15 K)</td><td>{:.4f} kcal/mol</td></tr>
                </tbody>
            </table>
        </div>
        """.format(zpve, e_thermo, h, s, g)
    except Exception as exc:
        gr.Warning("Optimization error!\n" + str(exc))
        return [None, None, None, None, None, None, None, None, None]

    calculation_status = "Analysis complete. ({0:.3f} s)".format(duration)
    return calculation_status, energy_plot, conformer_dropdown, conformers_file_path, energy_textbox, dipole_moment_textbox, MO_df, ir_spectrum, thermo_html

def on_conformer_change(conformer_dropdown: gr.Dropdown):
    if conformer_dropdown is not None:
        conformers_file_path = f".\\structures\\conformer{conformer_dropdown}.pdb"
        return conformers_file_path
    else:
        return None

def frequency_analysis_tab_content():
    with gr.Tab("Frequency Analysis") as frequency_analysis_tab:
        with gr.Accordion("Molecule"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    molecule_editor = molecule2d(label="Molecule")
                with gr.Column(scale=1):
                    create_molecule_button = gr.Button(value="Create molecule")
                    smiles_texbox = gr.Textbox(label="SMILES")
                    molecule_viewer = Molecule3D(label="Molecule", reps=reps)
        with gr.Accordion("Frequency Analysis"):
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
                    run_geometry_optimization_checkbox = gr.Checkbox(label="Run geometry optimization", value=True)
                    iterations_slider = gr.Slider(label="Max iterations", value=500, minimum=1, maximum=10000, step=1)
                    step_type_dropdown = gr.Dropdown(label="Step type", value="RFO",
                                                     choices=["RFO", "P_RFO", "NR", "SD", "LINESEARCH"])
                    full_hess_every_slider = gr.Slider(label="Full Hessian every", value=-1, minimum=-1, maximum=1000, step=1)
                    convergence_dropdown = gr.Dropdown(label="Convergence", value="GAU",
                                                       choices=["GAU", "GAU_LOOSE", "GAU_TIGHT", "GAU_VERYTIGHT", "NWCHEM_LOOSE",
                                                                "TURBOMOLE", "CFOUR", "QCHEM", "MOLPRO", "INTERFRAG_TIGHT"])
                    frequency_analysis_button = gr.Button(value="Frequency analysis")
        with gr.Accordion("Frequency Analysis Results"):
            with gr.Row(equal_height=True):
                status_markdown = gr.Markdown()
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    energy_plot = gr.Plot(label="Energy plot")
                with gr.Column(scale=1):
                    conformer_dropdown = gr.Dropdown(label="Conformer", visible=False)
                    conformers_viewer = Molecule3D(label="Conformer", reps=reps)  
            with gr.Row(equal_height=True):        
                with gr.Column(scale=1):
                    energy_texbox = gr.Textbox(label="Energy", value="Not calculated")
                    dipole_moment_texbox = gr.Textbox(label="Dipole moment", value="Not calculated")
                with gr.Column(scale=1):
                    MO_dataframe = gr.DataFrame(label="Molecular orbitals")
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    ir_spectrum_plot = gr.Plot(label="IR spectrum")
                with gr.Column(scale=1):
                    thermo_html = gr.HTML()
                
        create_molecule_button.click(on_create_molecule, molecule_editor, [smiles_texbox, molecule_viewer])
        method_radio.change(on_method_change, method_radio, functional_textbox)
        run_geometry_optimization_checkbox.change(on_run_geometry_optimization_change, run_geometry_optimization_checkbox,
                                                  [iterations_slider, step_type_dropdown, full_hess_every_slider, convergence_dropdown])
        frequency_analysis_button.click(on_frequency_analysis, [method_radio, reference_dropdown, basis_set_dropdown, functional_textbox,
                                                               charge_slider, multiplicity_dropdown, memory_slider, num_threads_slider,
                                                               run_geometry_optimization_checkbox, iterations_slider, step_type_dropdown, full_hess_every_slider, convergence_dropdown],
                                                               [status_markdown, energy_plot, conformer_dropdown, conformers_viewer,
                                                                energy_texbox, dipole_moment_texbox, MO_dataframe, ir_spectrum_plot, thermo_html])
        conformer_dropdown.change(on_conformer_change, conformer_dropdown, conformers_viewer)
        
    return frequency_analysis_tab