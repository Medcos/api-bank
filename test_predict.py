import pytest
import pandas as pd
from main import app


def test_get_clients(client):
    response = client.get('/clients')
    assert response.status_code == 200
 
    
## Afficher les infos importantes sur un client
def test_get_client_existing(client):
    # Supposons que le SK_ID_CURR 100001 existe dans votre fichier info_clients.csv
    existing_client_id = 100001
    response = client.get(f'/client/{existing_client_id}')
    
    assert response.status_code == 200
    client_data = response.get_json()
    assert isinstance(client_data, list)  # Doit être une liste
    assert len(client_data) > 0  # Assurez-vous qu'il y a des données

def test_get_client_non_existing(client):
    # Utilisez un ID qui n'existe pas dans le fichier
    non_existing_client_id = 999999
    response = client.get(f'/client/{non_existing_client_id}')
    
    assert response.status_code == 200
    assert response.data.decode('utf-8') == "Le numéro n'existe pas dans la base de données"
    

## Faire la prediction et afficher le resultat 
def test_predict_existing_client(client):
    # Supposons que le SK_ID_CURR 100001 existe dans votre fichier info_clients.csv
    existing_client_id = 100001
    response = client.get(f'/predict/{existing_client_id}')
    
    # Vérifiez que la réponse est correcte
    assert response.status_code == 200
    prediction = response.get_json()
    assert 'error' not in prediction  # Assurez-vous qu'il n'y a pas d'erreur
    assert isinstance(prediction, float)  # Vérifiez que la prédiction est un nombre

def test_predict_non_existing_client(client):
    # Utilisez un ID qui n'existe pas dans le fichier
    non_existing_client_id = 999999
    response = client.get(f'/predict/{non_existing_client_id}')
    
    # Vérifiez que la réponse est une erreur 404
    assert response.status_code == 404
    error_message = response.get_json()
    assert error_message['error'] == "Client pas trouvé"