import pytest
import pandas as pd
from main import app


## Chargez les données de test
data_path = 'info_clients.csv'
data = pd.read_csv(data_path)

@pytest.fixture
## Configure l'application Flask pour le mode test. 
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_hello(client):
    response = client.get('/')
    assert response.data == " Bienvenue à la société financière, nommée 'Prêt à dépenser'"