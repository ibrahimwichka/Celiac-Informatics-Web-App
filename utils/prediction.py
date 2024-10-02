import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from models.MLP_adaboost import MLP_Base_For_AdaBoost, intialize_model



def scale_input(features):

    df = pd.read_csv('models/data/clf_celiac_desc_fing_data.csv')
    

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)
    

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)  
    features = sc.transform(features)   
    
    return features


def predict_activity(features):
    df = pd.read_csv('models/data/clf_celiac_desc_fing_data.csv')
    

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)
    
    features = scale_input(features)
    
    model = intialize_model()

    model.fit(X_train, y_train)

    prediction = model.predict(features)
    

    activity_value = int(prediction[0])
    result = "Active" if activity_value == 1 else "Inactive"
    pred_color = "green" if result == "Active" else "red"
    
    return result, pred_color
