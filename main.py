from flask import Flask, jsonify, send_file, render_template
import pandas as pd
import mlflow.lightgbm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
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
local_path = os.path.join(os.getcwd(), 'modele')
model = mlflow.lightgbm.load_model(local_path)

## Activer les visualisations interactives de SHAP
shap.initjs()
# Création d'un explainer SHAP
#explainer = shap.TreeExplainer(model)

## Chemin d'accès
folder = os.path.join(os.getcwd(), 'image')

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
    return jsonify (round(prediction[1]*100, 2))


## Faire l'interpretation locale de la prédiction
@app.route('/interpretation/local/<int:id>', methods=['GET'])
def get_local_interpretation(id):
    client_data = X[X['SK_ID_CURR'] == id]
    client_data = client_data.drop('SK_ID_CURR', axis=1)
    client_data = client_data.replace([np.inf, -np.inf], 1e9)

    if client_data.empty:
        return jsonify({"error": "Client pas trouvé"}), 404
    
    # Initialiser LIME
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data = np.array(X.drop('SK_ID_CURR', axis=1)),
        feature_names=X.columns[1:], 
        mode='classification'
    )
    # Obtenir l'explication
    exp = explainer.explain_instance(data_row = client_data.values[0], predict_fn = model.predict_proba)
    
    # Extraire les importances
    importances = exp.as_map()[1]  # Récupérer les importances des caractéristiques
    feature_names =  X.columns[1:]  # Obtenir les noms des caractéristique
    
    # Trier les importances et obtenir les 10 plus importantes
    sorted_indices = np.argsort([val[1] for val in importances])[-10:]  # Indices des 10 plus importantes
    top_importances = [importances[i] for i in sorted_indices]
    top_feature_names = [feature_names[i] for i in sorted_indices]

    # Visualiser l'importance des fonctionnalités 
    #plt.style.use("ggplot")  # Utiliser le style ggplot
    plt.figure(figsize=(20, 10))
    
    # Création du graphique à barres horizontales
    plt.barh(range(10), [val[1] for val in top_importances],
         color=["red" if val[1] < 0 else "green" for val in top_importances])
    plt.yticks(range(10), top_feature_names)
    plt.title("10 Caractéristiques les plus importantes", fontsize=16)
    plt.xlabel("Importance", fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)  # Ajouter une grille
    
    # Sauvegarder l'image
    plt.savefig(f'{folder}/local_interpretation_{id}.jpg', dpi = 300)
    plt.close()

    return send_file(f'{folder}/local_interpretation_{id}.jpg', mimetype='image/jpg') 


## Faire l'interpretation Global du modèle
@app.route('/interpretation/global', methods=['GET'])
def get_global_interpretation():
    # Initialiser l'explainer SHAP
    explainer = shap.TreeExplainer(model)
    
    # Obtenir les valeurs SHAP pour l'ensemble des données
    sample_data = X
    sample_data = sample_data.drop('SK_ID_CURR', axis=1)
    sample_data = sample_data.replace([np.inf, -np.inf], 1e9)

    shap_values = explainer(sample_data)

     # Extraire les valeurs SHAP et les noms des caractéristiques
    feature_names = X.columns[1:]  # Obtenir les noms des caractéristiques
    shap_values_mean = np.mean(np.abs(shap_values.values), axis=0)

    # Trier et obtenir les 10 caractéristiques les plus importantes
    sorted_indices = np.argsort(shap_values_mean)[-10:]
    top_feature_names = feature_names[sorted_indices]
    top_shap_values = shap_values_mean[sorted_indices]

    # Création du graphique à barres horizontales
    plt.style.use("ggplot")  # Utiliser le style ggplot
    plt.figure(figsize=(20, 10))
    
    plt.barh(range(10), top_shap_values,
             color=[ "green" for val in top_shap_values])
    plt.yticks(range(10), top_feature_names)
    plt.title("10 Caractéristiques les plus importantes", fontsize=16)
    plt.xlabel("Importance (valeurs SHAP)", fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)  # Ajouter une grille

    # Sauvegarder l'image
    plt.savefig(f'{folder}/global_interpretation.png')
    plt.close()

    return send_file(f'{folder}/global_interpretation.png', mimetype='image/png')


## Afficher le fichier d'analyse drift
@app.route('/drift', methods=['GET'])
def drift():
    return render_template('drift.html')

if __name__ == '__main__':
    app.run()