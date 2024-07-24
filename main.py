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

if __name__ == '__main__':
    app.run()