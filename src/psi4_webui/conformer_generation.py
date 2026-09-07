"""Conformer generation tab: SMILES (or a 2D sketch) -> N unique 3D conformers.

RDKit embeds candidates in oversampled rounds, minimizes each with MMFF94 (UFF
fallback), and discards duplicates -- a candidate counts as a duplicate only when it is
both close in energy AND superimposable on an accepted conformer. Survivors are written
into the working directory as .xyz/.pdb/.mol, ready for the Calculation tab.
"""
import os
import html
import math
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
import gradio as gr
from .utils import get_files_in_working_directory, conformer_to_xyz_file
from ._paths import file_in_directory, validate_name

# The 2D sketcher is optional: gradio_molecule2d 0.0.3 declares gradio<5.0, which cannot
# coexist with the gradio==5.50.0 this app targets, so it is installed separately with
# --no-deps (or not at all). Without it the tab still works -- users type SMILES.
try:
    from gradio_molecule2d import molecule2d
except ImportError:  # pragma: no cover - depends on the environment
    molecule2d = None

# maximum number of embedding rounds before giving up on reaching the requested count
MAX_EMBEDDING_ROUNDS = 10
# stop early when this many rounds in a row produce no new conformer
MAX_BARREN_ROUNDS = 2
# how many candidates to embed per round, relative to the number still missing
OVERSAMPLING_FACTOR = 4
MAX_BATCH_SIZE = 250

def on_draw_molecule(molecule_editor):
    if not molecule_editor:
        return ""
    mol = Chem.MolFromSmiles(molecule_editor)
    if mol is None:
        return ""

    return Chem.MolToSmiles(mol, canonical=True)

def optimize_conformers(mol):
    # Optimize every conformer with MMFF94 (UFF as fallback) and return the energies in kcal/mol.
    # The energies only mean something for minimized geometries, an unrelaxed embedding is dominated
    # by bond and angle strain rather than by the conformation.
    mmff_properties = AllChem.MMFFGetMoleculeProperties(mol)
    if mmff_properties is not None:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=1000, numThreads=0)
    else:
        if not AllChem.UFFHasAllMoleculeParams(mol):
            raise ValueError("Neither MMFF nor UFF has parameters for every atom in this molecule")
        results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=1000, numThreads=0)

    # Preserve the conformer/energy alignment while excluding unfinished or failed
    # minimizations. Their energies must not enter the table or duplicate comparison.
    return [energy if status == 0 and math.isfinite(energy) else None
            for status, energy in results]

def is_duplicate_conformer(energy, conf_id, kept, candidate_heavy_mol, kept_heavy_mol, energy_threshold, rms_threshold):
    # A candidate is a duplicate of an accepted conformer only when it is both close in energy and
    # geometrically superimposable. Energy alone discards distinct conformers that happen to be
    # isoenergetic (mirror-image gauche forms, equivalent group rotations).
    for kept_energy, kept_conf_id in kept:
        if abs(energy - kept_energy) >= energy_threshold:
            continue
        # heavy atoms only, symmetry-aware; note this aligns the probe conformer in place, which is
        # harmless because RMSD is orientation independent and the output geometry comes from the
        # full molecule
        rms = rdMolAlign.GetBestRMS(candidate_heavy_mol, kept_heavy_mol, prbId=conf_id, refId=kept_conf_id)
        if rms < rms_threshold:
            return True

    return False

def generate_unique_conformers(mol, num_confs, energy_threshold, rms_threshold, progress=None):
    # Embed conformers repeatedly, discarding the duplicates, until num_confs unique conformers
    # are collected or the molecule runs out of distinct conformations
    if mol.GetNumAtoms() == 0:
        raise ValueError("Please enter a SMILES string containing atoms")
    if len(Chem.GetMolFrags(mol)) != 1:
        # RDKit embeds disconnected fragments independently at the same origin and
        # the force fields ignore their interactions, leaving overlapping nuclei.
        raise ValueError("Conformer generation requires a single connected molecule; "
                         "use a prepared 3D structure file for multiple fragments")
    if not math.isfinite(num_confs) or num_confs < 1 or int(num_confs) != num_confs:
        raise ValueError("The number of conformers must be a positive integer")
    num_confs = int(num_confs)
    if any(not math.isfinite(value) or value < 0 for value in (energy_threshold, rms_threshold)):
        raise ValueError("Duplicate thresholds must be finite and non-negative")
    unique_mol = Chem.Mol(mol)
    unique_mol.RemoveAllConformers()
    # heavy atom copy of the accepted conformers, kept in sync with unique_mol, used for the RMSD test
    kept_heavy_mol = Chem.RemoveHs(Chem.Mol(mol))
    kept_heavy_mol.RemoveAllConformers()

    kept = []  # (energy, conf id) of the conformers stored in unique_mol
    num_discarded = 0
    num_barren_rounds = 0

    for embedding_round in range(MAX_EMBEDDING_ROUNDS):
        num_missing = num_confs - len(kept)
        if num_missing <= 0 or num_barren_rounds >= MAX_BARREN_ROUNDS:
            break

        # oversample, a large share of the candidates is expected to collapse onto known minima
        batch_size = min(num_missing * OVERSAMPLING_FACTOR, MAX_BATCH_SIZE)
        candidate_mol = Chem.Mol(mol)
        # a different seed per round guarantees a different set of candidates
        AllChem.EmbedMultipleConfs(candidate_mol, numConfs=batch_size, randomSeed=embedding_round + 1, numThreads=0)
        if candidate_mol.GetNumConformers() == 0:
            break

        energies = optimize_conformers(candidate_mol)
        candidate_heavy_mol = Chem.RemoveHs(Chem.Mol(candidate_mol))

        num_accepted_this_round = 0
        for conformer, energy in zip(candidate_mol.GetConformers(), energies):
            if energy is None:
                continue
            conf_id = conformer.GetId()
            if is_duplicate_conformer(energy, conf_id, kept, candidate_heavy_mol, kept_heavy_mol, energy_threshold, rms_threshold):
                num_discarded += 1
                continue

            new_conf_id = unique_mol.AddConformer(conformer, assignId=True)
            kept_heavy_mol.AddConformer(candidate_heavy_mol.GetConformer(conf_id), assignId=True)
            kept.append((energy, new_conf_id))
            num_accepted_this_round += 1
            if progress is not None:
                progress(len(kept) / num_confs, desc="Generating")
            if len(kept) == num_confs:
                break

        num_barren_rounds = 0 if num_accepted_this_round > 0 else num_barren_rounds + 1

    if not kept:
        raise ValueError("No minimized conformers could be generated; embedding or force-field optimization failed")
    # lowest energy first
    return unique_mol, sorted(kept), num_discarded

def on_generate_conformers(working_directory_path, input_smiles, charge, multiplicity, num_confs, energy_threshold, rms_threshold, file_name, file_type, progress=gr.Progress()):
    empty_dataframe = pd.DataFrame(columns=["ID", "Energy (kcal/mol)"])
    try:
        if not working_directory_path or not os.path.isdir(working_directory_path):
            raise ValueError("Please open a working directory first")
        file_name = validate_name(file_name)
        if file_type not in {"xyz", "pdb", "mol"}:
            raise ValueError("Unsupported conformer file type")
        mol = Chem.MolFromSmiles(input_smiles)
        if mol is None:
            raise ValueError(f'invalid SMILES "{input_smiles}"')
        mol = Chem.AddHs(mol)

        # Generate conformers, removing the duplicates and replacing them by new ones
        unique_mol, conformers, num_discarded = generate_unique_conformers(mol, num_confs, energy_threshold, rms_threshold, progress)

        conformer_rows = []
        for index, (energy, conf_id) in enumerate(progress.tqdm(conformers, total=len(conformers), desc="Writing")):
            conformer_id = index + 1
            # Create a unique file name for each conformer
            conf_file_path = file_in_directory(
                working_directory_path, f'{file_name}_{conformer_id}.{file_type}')
            # Write conformers geometry to file
            if file_type == 'xyz':
                conformer_to_xyz_file(unique_mol, conf_id, conf_file_path, charge, multiplicity)
            elif file_type == 'pdb':
                Chem.MolToPDBFile(unique_mol, conf_file_path, confId=conf_id)
            else: # file_type_dropdown == 'mol'
                Chem.MolToMolFile(unique_mol, conf_file_path, confId=conf_id)
            conformer_rows.append([conformer_id, round(energy, 4)])

        conformer_dataframe = pd.DataFrame(conformer_rows, columns=["ID", "Energy (kcal/mol)"])

        status = f'{len(conformer_rows)} conformers generated, {num_discarded} duplicates discarded.'
        if len(conformer_rows) < num_confs:
            status += ' The search stopped before reaching the requested count. Try lower duplicate thresholds to keep more conformers.'
            return f"<span style='color:orange;'>{status}</span>", get_files_in_working_directory(working_directory_path), conformer_dataframe
        return f"<span style='color:green;'>{status}</span>", get_files_in_working_directory(working_directory_path), conformer_dataframe
    except Exception as exc:
        status = f'Error generating conformers: {exc}'
        return f"<span style='color:red;'>{html.escape(status)}</span>", get_files_in_working_directory(working_directory_path), empty_dataframe

def show_selected_file(selected_file):
    gr.Warning(selected_file)
    return selected_file

def conformer_generation_tab_content(working_directory_path_state, working_directory_file_list_state, status_markdown):
    with gr.Tab("Conformer generation") as conformer_generation_tab:
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Accordion("Molecular Structure"):
                    if molecule2d is not None:
                        molecule_editor = molecule2d(label="Molecule")
                    else:
                        molecule_editor = None
                        gr.Markdown(
                            "The 2D structure editor is not installed. Enter a SMILES "
                            "string on the right, or install the editor with "
                            "`pip install gradio_molecule2d --no-deps`."
                        )
            with gr.Column(scale=1):
                with gr.Accordion("Generate Conformers"):
                    input_smiles_texbox = gr.Textbox(label="SMILES")
                    charge_slider = gr.Slider(label="Charge", value=0, minimum=-2, maximum=2, step=1)
                    multiplicity_dropdown = gr.Dropdown(label="Multiplicity", value=1, choices=[("Singlet", 1), ("Doublet", 2), ("Triplet", 3), ("Quartet", 4), ("Quintet", 5), ("Sextet", 6)])
                    num_confs_slider = gr.Slider(label="Number of conformers", value=1, minimum=1, maximum=100, step=1)
                    energy_threshold_slider = gr.Slider(label="Energy threshold (kcal/mol)", info="Duplicates are conformers within this energy AND below the RMSD threshold", value=0.1, minimum=0, maximum=5, step=0.1)
                    rms_threshold_slider = gr.Slider(label="RMSD threshold (A)", info="Heavy atom RMSD below which two conformers of equal energy count as the same", value=0.5, minimum=0, maximum=3, step=0.05)
                    file_name_textbox = gr.Textbox(label="File name", value="conformer")
                    file_type_dropdown = gr.Dropdown(label="File type", value="xyz", choices=["xyz", "pdb", "mol"])
                    generate_button = gr.Button(value="Generate")
                    conformer_dataframe = gr.Dataframe(label="Generated conformers", headers=["ID", "Energy (kcal/mol)"], datatype=["number", "number"], max_height=360, interactive=False)

    if molecule_editor is not None:
        molecule_editor.change(on_draw_molecule, molecule_editor, input_smiles_texbox)
    generate_button.click(on_generate_conformers, [working_directory_path_state, input_smiles_texbox, charge_slider, multiplicity_dropdown, num_confs_slider, energy_threshold_slider, rms_threshold_slider, file_name_textbox, file_type_dropdown], [status_markdown, working_directory_file_list_state, conformer_dataframe])

    return conformer_generation_tab
