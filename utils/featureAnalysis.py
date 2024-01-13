import numpy as np
import os
import io
from flask import Response
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure

from utils.features import getDescriptors

def get_important_fingerprints(mol, rdkbi):
    bit_list = [495, 433, 382, 338, 350, 193, 74, 417, 263, 308]
    substructure_list = []
    substructure_numbers = []
    for i in bit_list:
        if i in rdkbi:
            substructure = Draw.IPythonConsole.DrawRDKitBit(mol, i ,rdkbi)
            substructure_list.append(substructure)
            substructure_numbers.append(i)
    sub_file_names = []
    for index, substructure_img in enumerate(substructure_list):
        sub_file_name = 'sub' + str(index) + '.png'
        sub_file_names.append(sub_file_name)
        sub_path = os.path.join('static', 'imgs', sub_file_name)
        substructure_img.save(sub_path)
    if len(substructure_numbers) < 4:
        img_width = 150
    elif len(substructure_numbers) < 11:
        img_width = 90
    return sub_file_names, substructure_numbers, img_width, len(substructure_numbers)


def graph_important_descriptors(smiles, mol, rdkbi):
    all_descriptors = getDescriptors(mol)[1]
    important_descriptor_names = [
        "NumSaturatedHeterocycles",
        "NumSaturatedRings",
        "SMR_VSA6",
        "VSA_EState3",
        "BCUT2D_CHGHI", 
        "SlogP_VSA2",
        "BCUT2D_CHGLO",
        "PEOE_VSA12",
        "BCUT2D_MRHI",
        "PEOE_VSA8"
    ]
    molecular_weight = Descriptors.MolWt(mol)
    important_descriptors = [int(all_descriptors[descriptor]) for descriptor in important_descriptor_names]
    
    plt.figure(figsize=(10, 6))
    plt.bar(important_descriptor_names, important_descriptors, color='blue')
    for i, value in enumerate(important_descriptors):
        plt.text(i, value + 1, str(value), ha='center', va='bottom', rotation=45)
    plt.xlabel('Molecular Descriptors')
    plt.ylabel('Descriptor Values')
    plt.title(f"{smiles}\nMolecular Weight: {molecular_weight} g/mol")
    plt.ylim(0, max(important_descriptors) + 20) 
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('static/imgs/descriptor_plot.png')