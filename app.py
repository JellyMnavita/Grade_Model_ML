import streamlit as st
import joblib
import numpy as np

# Charger le modèle avec joblib
model = joblib.load("rf_model_grade.joblib")  # Charger le modèle sauvegardé avec joblib

# Titre de l'application
st.title("Prédiction des Notes")

# Créer des champs de saisie pour les variables
socio_score = st.number_input("Socioeconomic Score:", min_value=0.0, max_value=100.0, step=0.1)
study_hours = st.number_input("Study Hours:", min_value=0.0, max_value=24.0, step=0.1)
sleep_hours = st.number_input("Sleep Hours:", min_value=0.0, max_value=24.0, step=0.1)
attendance = st.number_input("Attendance (%):", min_value=0.0, max_value=100.0, step=0.1)

# Bouton pour prédire les notes
if st.button("Predict Grade"):
    try:
        # Créer un tableau avec les valeurs saisies
        user_input = np.array([[socio_score, study_hours, sleep_hours, attendance]])

        # Prédire avec le modèle
        predicted_grade = model.predict(user_input)
        
        # Afficher la prédiction
        st.success(f"Predicted Grade: {predicted_grade[0]:.2f}")
    except ValueError:
        st.error("Please enter valid numbers for all fields.")