from flask import Flask, render_template, url_for, request
from flask_material import Material

import os
import uuid
import numpy as np
import pandas as pd
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
from rdkit.Chem.Draw import IPythonConsole

import pickle
from PIL import Image

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from rdkit.Chem import Crippen

from utils.moleculeInputs import check_smiles, check_organic
from utils.features import getDescriptors, getFingerprints, getFeatures, getMoleculeInfo
from utils.prediction import scale_input, predict_activity
from utils.featureAnalysis import get_important_fingerprints, graph_important_descriptors
from utils.drugLikeness import lipinski_report, muegge_report, ghose_report, veber_report, egan_report, check_num_violations

app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/results', methods=['POST'])
def results():
    if request.method == 'POST':
        smiles = request.form['smiles_input']
        smiles = str(smiles)
        mol = Chem.MolFromSmiles(smiles)
        
        isValidSmiles, message = check_smiles(mol, smiles)
        if not isValidSmiles:
            error_message = message 
            return render_template('index.html', error_message = error_message)
        
        isOrganic, message = check_organic(mol, smiles)
        if not isOrganic:
            error_message = message
            return render_template('index.html', error_message = error_message)
        
        
        smiles = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smiles)

        molecular_formula = getMoleculeInfo(mol)
        
        molecule_img = Draw.MolToImage(mol)
        img_path = os.path.join('static', 'imgs', 'molecule.png')
        molecule_img.save(img_path)

        lipinski, color_lip = lipinski_report(smiles)
        muegge, color_mue = muegge_report(smiles)
        ghose, color_gho = ghose_report(smiles)
        veber, color_veb = veber_report(smiles)
        egan, color_eg = egan_report(smiles)
        violation_report, viol_report_color = check_num_violations(smiles)

        features, rdkbi = getFeatures(mol)
        activity_result, pred_color = predict_activity(features)

        if viol_report_color == "red":
            activity_result = "Molecule is not Drug-Like"
            num_of_sub = 0
            sub_file_names = 0
            substructure_numbers = 0
            img_width = 0
            key_colors = 0
            isGraph = False
        else:
            sub_file_names, substructure_numbers, img_width, num_of_sub, key_colors = get_important_fingerprints(mol, rdkbi)
            graph_important_descriptors(smiles, mol, rdkbi)
            isGraph = True
        
        return render_template(
            'results.html', 
            smiles_input=smiles, 
            activity_result = activity_result, 
            pred_color = pred_color, 
            sub_file_names = sub_file_names, 
            substructure_numbers = substructure_numbers,
            img_width = img_width,
            num_of_sub = num_of_sub,
            molecular_formula = molecular_formula,
            key_colors = key_colors,
            lipinski = lipinski,
            muegge = muegge,
            ghose = ghose,
            veber = veber,
            egan = egan,
            color_lip = color_lip,
            color_mue = color_mue,
            color_gho = color_gho,
            color_veb = color_veb,
            color_eg = color_eg,
            violation_report = violation_report,
            viol_report_color = viol_report_color,
            isGraph = isGraph
        )
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)