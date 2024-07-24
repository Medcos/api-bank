from flask import Flask, jsonify, send_file
import pandas as pd
import mlflow.lightgbm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import jinja2
import os
import sys


app = Flask(__name__)


## Importer les données
data_path = os.path.join(os.getcwd(), 'info_clients.csv')
data = pd.read_csv(data_path)

df_path = os.path.join(os.getcwd(), 'data.csv')
df = pd.read_csv(df_path)

## Charger le modèle enregistré
local_path = r".\modele"
model = mlflow.lightgbm.load_model(local_path)

## Création d'un explainer SHAP
explainer = shap.TreeExplainer(model)

## Chemin d'accès
folder = r".\image"

## Préparation des données
train_df = df[df['TARGET'].notnull()]
test_df = df[df['TARGET'].isnull()]
# Séparer les caractéristiques et la variable cible
feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_BUREAU','SK_ID_PREV','index']]    
X = train_df[feats]
# Remplacer les NaN par la moyenne
X = X.fillna(X.mean())  

sys.stdout.flush()

## Page d'accueil
@app.route('/', methods=['GET'])
def hello():
    return " Bienvenue à la société financière, nommée 'Prêt à dépenser'"

## Récupérer les ID des clients à partir de la colonne "id" de la DataFrame
@app.route('/clients', methods=['GET'])
def get_clients():
    client_ids = data['SK_ID_CURR'].tolist()
    return jsonify(client_ids)

## Afficher les infos importantes sur un client
@app.route('/client/<int:id>', methods=['GET'])
def get_client(id):
    if id in data['SK_ID_CURR'].tolist() :
        clientid = data.loc[data['SK_ID_CURR']== id, : ]
        client = clientid.to_dict('records')
        return jsonify(client)
    else:   
        return f"Le numéro n'existe pas dans la base de données" 
    
## Faire la prediction et afficher le resultat
@app.route('/predict/<int:id>', methods=['GET'])
def predict(id):
    client_info = X[X['SK_ID_CURR'] == id]
    client_info = client_info.drop('SK_ID_CURR', axis=1)
    client_info= client_info.replace([np.inf, -np.inf], 1e9)

    if client_info.empty:
        return jsonify({"error": "Client pas trouvé"}), 404
    prediction = model.predict_proba(client_info)
    prediction = prediction.tolist()[0]
    return jsonify({"prediction": (prediction[1])})


if __name__ == '__main__':
    app.run()