"""
Tests automatisés de l'API de prédiction de no-show.

Lancer avec :
    pytest tests/ -v
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# ----------------------------------------------------------------------
# Un profil de référence valide, réutilisé dans plusieurs tests
# (celui déjà validé manuellement : notebook, Streamlit et API donnent 39.4%)
# ----------------------------------------------------------------------

PROFIL_VALIDE = {
    "age": 35,
    "tranche_age": 2,
    "sexe_encoded": 1,
    "hypertension": 0,
    "diabete": 0,
    "alcoolisme": 0,
    "handicap": 0,
    "boursier": 0,
    "sms_recu": 1,
    "delai_jours": 10,
    "jour_semaine_num": 0,
    "nb_rdv_precedents": 2,
    "taux_presence_historique": 0.8,
    "premier_rdv": 0,
}


# ----------------------------------------------------------------------
# Tests de l'endpoint /health
# ----------------------------------------------------------------------

def test_health_repond_ok():
    """L'endpoint /health doit répondre 200 et confirmer que le modèle est chargé."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["modele_charge"] is True
    assert data["nb_features"] > 0


# ----------------------------------------------------------------------
# Tests de l'endpoint /predict — cas valides
# ----------------------------------------------------------------------

def test_predict_renvoie_200_pour_un_profil_valide():
    response = client.post("/predict", json=PROFIL_VALIDE)
    assert response.status_code == 200


def test_predict_probabilites_coherentes():
    """Les deux probabilités doivent être entre 0 et 1, et sommer à 1."""
    response = client.post("/predict", json=PROFIL_VALIDE)
    data = response.json()

    assert 0.0 <= data["probabilite_presence"] <= 1.0
    assert 0.0 <= data["probabilite_absence"] <= 1.0
    assert abs(data["probabilite_presence"] + data["probabilite_absence"] - 1.0) < 1e-6


def test_predict_resultat_conforme_au_profil_de_reference():
    """
    Vérifie une non-régression : ce profil précis doit toujours renvoyer
    une probabilité de présence proche de 39.4%, comme validé manuellement
    dans le notebook et le dashboard Streamlit.
    """
    response = client.post("/predict", json=PROFIL_VALIDE)
    data = response.json()

    assert abs(data["probabilite_presence"] - 0.394) < 0.01
    assert data["risque_eleve"] is True  # cohérent avec ~60% d'absence


def test_predict_champ_risque_eleve_est_un_booleen():
    response = client.post("/predict", json=PROFIL_VALIDE)
    data = response.json()
    assert isinstance(data["risque_eleve"], bool)


# ----------------------------------------------------------------------
# Tests de l'endpoint /predict — cas invalides (validation Pydantic)
# ----------------------------------------------------------------------

def test_predict_rejette_age_negatif():
    profil_invalide = {**PROFIL_VALIDE, "age": -5}
    response = client.post("/predict", json=profil_invalide)
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_rejette_age_trop_eleve():
    profil_invalide = {**PROFIL_VALIDE, "age": 200}
    response = client.post("/predict", json=profil_invalide)
    assert response.status_code == 422


def test_predict_rejette_valeur_binaire_hors_limites():
    """sms_recu doit être 0 ou 1 — une valeur comme 3 doit être rejetée."""
    profil_invalide = {**PROFIL_VALIDE, "sms_recu": 3}
    response = client.post("/predict", json=profil_invalide)
    assert response.status_code == 422


def test_predict_rejette_taux_presence_hors_bornes():
    """taux_presence_historique doit être entre 0.0 et 1.0."""
    profil_invalide = {**PROFIL_VALIDE, "taux_presence_historique": 1.5}
    response = client.post("/predict", json=profil_invalide)
    assert response.status_code == 422


def test_predict_rejette_champ_obligatoire_manquant():
    """Si 'age' est absent de la requête, l'API doit refuser plutôt que deviner une valeur."""
    profil_incomplet = {k: v for k, v in PROFIL_VALIDE.items() if k != "age"}
    response = client.post("/predict", json=profil_incomplet)
    assert response.status_code == 422


def test_predict_rejette_type_incorrect():
    """Envoyer une chaîne de caractères là où un nombre est attendu doit être rejeté."""
    profil_invalide = {**PROFIL_VALIDE, "age": "trente-cinq"}
    response = client.post("/predict", json=profil_invalide)
    assert response.status_code == 422
