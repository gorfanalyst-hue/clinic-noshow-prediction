from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import json

app = FastAPI(
    title="Clinic No-Show Prediction API",
    description="Prédit la probabilité qu'un patient se présente à son rendez-vous médical.",
    version="1.0.0",
)

# ----------------------------------------------------------------------
# Chargement du modèle et des features au démarrage de l'API
# (une seule fois, pas à chaque requête)
# ----------------------------------------------------------------------

model = joblib.load("data/models/xgb_model.pkl")

with open("data/models/features.json") as f:
    FEATURES = json.load(f)


# ----------------------------------------------------------------------
# Schéma de la requête attendue (validation automatique par Pydantic)
# ----------------------------------------------------------------------

class Patient(BaseModel):
    age: int = Field(..., ge=0, le=115, description="Âge du patient")
    tranche_age: int = Field(..., ge=0, le=4, description="0=enfant, 1=ado, 2=jeune adulte, 3=adulte, 4=senior")
    sexe_encoded: int = Field(..., ge=0, le=1, description="1 = Femme, 0 = Homme")
    hypertension: int = Field(0, ge=0, le=1)
    diabete: int = Field(0, ge=0, le=1)
    alcoolisme: int = Field(0, ge=0, le=1)
    handicap: int = Field(0, ge=0, le=1)
    boursier: int = Field(0, ge=0, le=1)
    sms_recu: int = Field(..., ge=0, le=1)
    delai_jours: int = Field(..., ge=0, description="Nombre de jours entre la prise de RDV et la consultation")
    jour_semaine_num: int = Field(..., ge=0, le=6, description="0=Lundi ... 6=Dimanche")
    nb_rdv_precedents: int = Field(..., ge=0)
    taux_presence_historique: float = Field(..., ge=0.0, le=1.0)
    premier_rdv: int = Field(..., ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    probabilite_presence: float
    probabilite_absence: float
    risque_eleve: bool


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------

@app.get("/")
def racine():
    return {
        "message": "Clinic No-Show Prediction API",
        "documentation": "/docs",
        "endpoint_prediction": "/predict",
    }


@app.get("/health")
def sante():
    """Endpoint de vérification que l'API et le modèle sont bien chargés."""
    return {"status": "ok", "modele_charge": model is not None, "nb_features": len(FEATURES)}


@app.post("/predict", response_model=PredictionResponse)
def predire(patient: Patient):
    """Prédit la probabilité de présence/absence d'un patient à son rendez-vous."""
    try:
        donnees = patient.model_dump()
        vecteur = [[donnees[feature] for feature in FEATURES]]

        proba = model.predict_proba(vecteur)[0]
        proba_presence = float(proba[1])
        proba_absence = float(proba[0])

        return PredictionResponse(
            probabilite_presence=round(proba_presence, 4),
            probabilite_absence=round(proba_absence, 4),
            risque_eleve=proba_absence > 0.35,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
