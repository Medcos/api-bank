from flask import Flask, jsonify, send_file, render_template
import pandas as pd
import mlflow.lightgbm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import lime
import jinja2
import os
import sys


app = Flask(__name__)


## Importer les données
df_path = os.path.join(os.getcwd(), 'info_clients.csv')
df = pd.read_csv(df_path)
print('df :', df)

data_path = os.path.join(os.getcwd(), 'data.csv')
data = pd.read_csv(data_path)
print('data :', data)

## Charger le modèle enregistré
local_path = os.path.join(os.getcwd(), 'modele')
model = mlflow.lightgbm.load_model(local_path)

## Charger le modèlepipeline de preprocessing enregistré
local_path_pre = os.path.join(os.getcwd(), 'preprocessing')
preprocessing = mlflow.sklearn.load_model(local_path_pre)

## Activer les visualisations interactives de SHAP
shap.initjs()

## Chemin d'accès
folder = os.path.join(os.getcwd(), 'image')

sys.stdout.flush()

## Page d'accueil
@app.route('/', methods=['GET'])
def hello():
    return " Bienvenue à la société financière, nommée 'Prêt à dépenser'"


## Récupérer les ID des clients à partir de la colonne "id" de la DataFrame
@app.route('/clients', methods=['GET'])
def get_clients():
    client_ids = df['SK_ID_CURR'].tolist()
    return jsonify(client_ids)


if __name__ == '__main__':
    app.run()