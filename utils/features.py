import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors

# import pubchempy as pcp


def getFingerprints(mol):
    rdkbi = {}
    all_drops = [330, 342, 2, 3, 16, 36, 40, 51, 64, 81, 87, 103, 112, 117, 141, 142, 161, 166, 186, 194, 209, 233, 236, 238, 248, 265, 280, 284, 290, 294, 297, 312, 341, 351, 369, 371, 372, 379, 386, 402, 403, 410, 416, 417, 428, 439, 443, 447, 460, 479, 480, 491, 504, 506, 509]
    
    fingerprints = pd.DataFrame(np.array(AllChem.RDKFingerprint(mol, maxPath=5, fpSize=512, bitInfo=rdkbi)))
    fingerprints.drop(all_drops, inplace = True)
    fingerprints_model_input = np.array(fingerprints)
    fingerprints_model_input = fingerprints_model_input.reshape(1, 457)
    return fingerprints_model_input, rdkbi

def getDescriptors(mol, missingVal = None):
    descriptor_dict = {}
    for nm,fn in Descriptors._descList:
        try:
            val = fn(mol)
        except:
            import traceback
            traceback.print_exc()
            val = missingVal
        descriptor_dict[nm] = val
    descriptors_df = pd.DataFrame({'Values': pd.Series(list(descriptor_dict.values()))})
    descriptors_model_input = np.array(descriptors_df)
    descriptors_model_input = descriptors_model_input.reshape((1, 210))
    return descriptors_model_input, descriptor_dict

def getFeatures(mol):
    fing_input, rdkbi = getFingerprints(mol)
    desc_input = getDescriptors(mol)[0]
    features_model_input = np.concatenate([fing_input, desc_input], axis = 1)
    return features_model_input, rdkbi


# def getMoleculeInfo(smiles):
#     pcp_compounds = pcp.get_compounds(smiles, 'smiles')
#     molecule = pcp_compounds[0]
#     molecule_name = molecule.to_dict()['title']
#     molecule_cid = molecule.cid
#     molecular_formula = molecule.to_dict()['molecular_formula']
#     return molecule_name, molecule_cid, molecular_formula