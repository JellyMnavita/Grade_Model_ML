import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# Charger le modèle
model = joblib.load("rf_model_grade.joblib")

# Titre de l'application
st.title("Prédiction des Notes")
st.markdown("## Prédisez les notes des étudiants en fonction de plusieurs critères !")

# Champs de saisie
socio_score = st.slider("Score Socioéconomique", 0.0, 100.0, 50.0, 0.1)
study_hours = st.slider("Heures d'Étude", 0.0, 24.0, 5.0, 0.1)
sleep_hours = st.slider("Heures de Sommeil", 0.0, 24.0, 7.0, 0.1)
attendance = st.slider("Présence (%)", 0.0, 100.0, 90.0, 0.1)

# Bouton de prédiction
if st.button("Prédire la Note"):
    try:
        # Création de l'input pour le modèle
        user_input = np.array([[socio_score, study_hours, sleep_hours, attendance]])
        predicted_grade = model.predict(user_input)
        
        # Affichage du résultat
        st.success(f"Note Prédite : {predicted_grade[0]:.2f}")
        
        # Radar Chart (Diagramme en toile d'araignée)
        categories = ['Socioeconomic', 'Study Hours', 'Sleep Hours', 'Attendance']
        values = [socio_score, study_hours, sleep_hours, attendance]
        
        values += values[:1]  # Boucler le graphique
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='blue', alpha=0.3)
        ax.plot(angles, values, color='blue', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        st.pyplot(fig)
        
    except ValueError:
        st.error("Veuillez entrer des valeurs valides.")

# Personnalisation de la section About
st.sidebar.markdown("## À propos")
st.sidebar.info(
    "Cette application utilise un modèle de Machine Learning pour prédire les notes des étudiants en fonction de plusieurs facteurs tels que les heures d'étude, le sommeil, la présence et le statut socioéconomique.\n\n"+
    "Copyright CelestWeb | 2025")
