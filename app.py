from flask import Flask, render_template, url_for, request
from flask_material import Material

import os
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from io import BytesIO

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.moleculeInputs import check_smiles, check_organic
from utils.features import getDescriptors, getFingerprints, getFeatures
from utils.prediction import scale_input, predict_activity
from utils.featureAnalysis import get_important_fingerprints, graph_important_descriptors

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
        
        molecule_img = Draw.MolToImage(mol)
        img_path = os.path.join('static', 'imgs', 'molecule.png')
        molecule_img.save(img_path)

        features, rdkbi = getFeatures(mol)
        activity_result = predict_activity(features)

        sub_file_names, substructure_numbers = get_important_fingerprints(mol, rdkbi)
        
        return render_template(
            'results.html', 
            smiles_input=smiles, 
            activity_result = activity_result, 
            sub_file_names = sub_file_names, 
            substructure_numbers = substructure_numbers
        )
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)