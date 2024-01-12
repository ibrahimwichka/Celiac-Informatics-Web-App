import numpy as np
import matplotlib.pyplot
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw

from utils.features import getDescriptors

def get_important_fingerprints(mol, rdkbi):
    bit_list = [495, 433, 382, 338, 350, 193, 74, 417, 263, 308]
    substructure_list = []
    substructure_numbers = []
    for i in bit_list:
        if i in rdkbi:
            nums+=1
            substructure = Draw.DrawRDKitBit(mol, i ,rdkbi)
            substructure_list.append(substructure)
            substructure_numbers.append(i)
    return

def graph_important_descriptors(mol, rdkbi):
    all_descriptors = getDescriptors(mol)
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
    important_descriptors = int(all_descriptors)
