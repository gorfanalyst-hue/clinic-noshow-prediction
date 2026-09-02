import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Clinic No-Show Dashboard", layout="wide")

# ----------------------------------------------------------------------
# Chargement des données et du modèle (mis en cache pour la performance)
# ----------------------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/dataset_dashboard.csv")

@st.cache_resource
def load_model():
    model = joblib.load("data/models/xgb_model.pkl")
    with open("data/models/features.json") as f:
        features = json.load(f)
    return model, features

@st.cache_resource
def load_explainer(_model):
    # Le underscore devant "_model" indique à Streamlit de ne pas essayer
    # de mettre le modèle lui-même en cache par valeur (trop coûteux),
    # seulement le résultat de cette fonction (l'explainer).
    return shap.TreeExplainer(_model)

df = load_data()
model, features_modele = load_model()
explainer = load_explainer(model)

# ----------------------------------------------------------------------
# En-tête
# ----------------------------------------------------------------------

st.title("🏥 Clinic No-Show Analytics")
st.caption("Analyse et prédiction des rendez-vous médicaux manqués — dataset Kaggle Medical Appointment No Shows")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Vue d'ensemble", "🔍 Exploration", "🧩 Segmentation patients", "🎯 Simulateur de risque"])

# ----------------------------------------------------------------------
# Onglet 1 — Vue d'ensemble (KPI)
# ----------------------------------------------------------------------

with tab1:
    st.subheader("Indicateurs clés")

    total_rdv = len(df)
    taux_absence = (1 - df["presence"].mean()) * 100
    delai_moyen = df["delai_jours"].mean()
    taux_sms = df["sms_recu"].mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total rendez-vous", f"{total_rdv:,}".replace(",", " "))
    col2.metric("Taux d'absence global", f"{taux_absence:.1f} %")
    col3.metric("Délai moyen avant RDV", f"{delai_moyen:.1f} jours")
    col4.metric("Taux de rappel SMS", f"{taux_sms:.1f} %")

    st.divider()
    st.subheader("Taux d'absence par quartier (top 10)")

    quartier_stats = (
        df.groupby("quartier")
        .agg(total=("presence", "size"), taux_absence=("presence", lambda x: (1 - x.mean()) * 100))
        .query("total >= 100")
        .sort_values("taux_absence", ascending=False)
        .head(10)
    )
    st.bar_chart(quartier_stats["taux_absence"])

# ----------------------------------------------------------------------
# Onglet 2 — Exploration
# ----------------------------------------------------------------------

with tab2:
    st.subheader("Facteurs associés à l'absentéisme")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Taux d'absence selon le délai avant consultation**")
        bins = [-1, 0, 7, 30, df["delai_jours"].max()]
        labels = ["Même jour", "1-7 jours", "8-30 jours", "30+ jours"]
        df["tranche_delai"] = pd.cut(df["delai_jours"], bins=bins, labels=labels)
        delai_stats = df.groupby("tranche_delai", observed=True).apply(
            lambda x: (1 - x["presence"].mean()) * 100
        )
        st.bar_chart(delai_stats)

    with col2:
        st.markdown("**Taux d'absence selon le jour de la semaine**")
        jours_labels = {0: "Lundi", 1: "Mardi", 2: "Mercredi", 3: "Jeudi", 4: "Vendredi", 5: "Samedi", 6: "Dimanche"}
        df["jour_label"] = df["jour_semaine_num"].map(jours_labels)
        jour_stats = df.groupby("jour_label", observed=True).apply(
            lambda x: (1 - x["presence"].mean()) * 100
        ).reindex(list(jours_labels.values())).dropna()
        st.bar_chart(jour_stats)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Taux d'absence selon la tranche d'âge**")
        age_stats = df.groupby("tranche_age", observed=True).apply(
            lambda x: (1 - x["presence"].mean()) * 100
        )
        st.bar_chart(age_stats)

    with col4:
        st.markdown("**Effet croisé : délai × SMS reçu**")
        croise = df.groupby(["tranche_delai", "sms_recu"], observed=True).apply(
            lambda x: (1 - x["presence"].mean()) * 100
        ).unstack()
        croise.columns = ["Sans SMS", "Avec SMS"]
        st.bar_chart(croise)

    st.info(
        "L'effet du SMS semble négatif si on l'analyse seul, mais s'inverse une fois le délai "
        "pris en compte : à délai égal, le SMS réduit bien le taux d'absence (paradoxe de Simpson)."
    )

# ----------------------------------------------------------------------
# Onglet 3 — Segmentation (clusters)
# ----------------------------------------------------------------------

with tab3:
    st.subheader("Profils de patients identifiés par clustering")

    noms_clusters = {
        0: "Délais longs, assiduité modérée",
        1: "Seniors réguliers",
        2: "Patients chroniques très suivis",
        3: "Profil à risque d'absentéisme",
        4: "Jeunes patients / RDV rapprochés",
    }
    df["profil"] = df["cluster"].map(noms_clusters)

    profil_stats = df.groupby("profil").agg(
        taille=("cluster", "size"),
        age_moyen=("age", "mean"),
        delai_moyen=("delai_jours", "mean"),
        taux_presence=("presence", lambda x: x.mean() * 100),
    ).round(1).sort_values("taux_presence", ascending=False)

    st.dataframe(
        profil_stats.style.format({
            "taille": "{:,.0f}",
            "age_moyen": "{:.1f}",
            "delai_moyen": "{:.1f}",
            "taux_presence": "{:.1f} %",
        }),
        use_container_width=True,
    )

    st.markdown("**Taux de présence réel par profil**")
    st.bar_chart(profil_stats["taux_presence"])

    st.caption(
        "Ces profils ont été construits sans utiliser la variable de présence — "
        "leur écart de taux de présence réel (jusqu'à 31 points) valide la pertinence de la segmentation."
    )

# ----------------------------------------------------------------------
# Onglet 4 — Simulateur de risque
# ----------------------------------------------------------------------

with tab4:
    st.subheader("Estimer le risque de no-show d'un patient")
    st.caption("Renseigne le profil d'un patient pour obtenir une probabilité de présence estimée par le modèle XGBoost.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Âge", 0, 100, 35)
        tranche_age_val = 0 if age < 12 else 1 if age < 18 else 2 if age < 40 else 3 if age < 65 else 4
        sexe = st.selectbox("Sexe", ["Femme", "Homme"])
        sexe_encoded = 1 if sexe == "Femme" else 0

    with col2:
        delai_jours = st.slider("Délai avant consultation (jours)", 0, 180, 10)
        sms_recu = st.checkbox("Rappel SMS envoyé", value=True)
        jour_semaine_num = st.selectbox(
            "Jour de la consultation",
            options=list(range(7)),
            format_func=lambda x: ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"][x]
        )

    with col3:
        nb_rdv_precedents = st.slider("Nombre de RDV précédents", 0, 50, 2)
        taux_presence_historique = st.slider("Taux de présence historique", 0.0, 1.0, 0.8)
        premier_rdv = 1 if nb_rdv_precedents == 0 else 0

    with st.expander("Comorbidités et statut socio-économique (optionnel)"):
        col4, col5, col6, col7, col8 = st.columns(5)
        hypertension = col4.checkbox("Hypertension")
        diabete = col5.checkbox("Diabète")
        alcoolisme = col6.checkbox("Alcoolisme")
        handicap = col7.checkbox("Handicap")
        boursier = col8.checkbox("Boursier")

    if st.button("Estimer le risque", type="primary"):
        patient = pd.DataFrame([{
            "age": age,
            "tranche_age": tranche_age_val,
            "sexe_encoded": sexe_encoded,
            "hypertension": int(hypertension),
            "diabete": int(diabete),
            "alcoolisme": int(alcoolisme),
            "handicap": int(handicap),
            "boursier": int(boursier),
            "sms_recu": int(sms_recu),
            "delai_jours": delai_jours,
            "jour_semaine_num": jour_semaine_num,
            "nb_rdv_precedents": nb_rdv_precedents,
            "taux_presence_historique": taux_presence_historique,
            "premier_rdv": premier_rdv,
        }])[features_modele]

        proba_presence = model.predict_proba(patient)[0][1]
        proba_absence = 1 - proba_presence

        col_a, col_b = st.columns(2)
        col_a.metric("Probabilité de présence", f"{proba_presence * 100:.1f} %")
        col_b.metric("Probabilité d'absence", f"{proba_absence * 100:.1f} %")

        if proba_absence > 0.35:
            st.warning("⚠️ Risque d'absence élevé — un rappel supplémentaire est recommandé pour ce patient.")
        else:
            st.success("✅ Risque d'absence faible à modéré.")

        # ---- Explication SHAP pour ce patient précis ----
        st.divider()
        st.markdown("**Pourquoi cette prédiction ?**")
        st.caption(
            "Contribution de chaque caractéristique à la prédiction pour ce patient précis "
            "(vers la droite = pousse vers la présence, vers la gauche = pousse vers l'absence)."
        )

        shap_values_patient = explainer.shap_values(patient)[0]

        contributions = pd.DataFrame({
            "feature": features_modele,
            "impact": shap_values_patient,
        }).sort_values("impact", key=abs, ascending=True)

        colors = ["#d62728" if v < 0 else "#2ca02c" for v in contributions["impact"]]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(contributions["feature"], contributions["impact"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Impact SHAP (négatif = pousse vers l'absence)")
        fig.tight_layout()
        st.pyplot(fig)

        top_facteur = contributions.iloc[-1]
        sens = "vers l'absence" if top_facteur["impact"] < 0 else "vers la présence"
        st.caption(
            f"Le facteur le plus déterminant pour ce patient est **{top_facteur['feature']}**, "
            f"qui pousse la prédiction {sens}."
        )
