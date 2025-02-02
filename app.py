import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Charger le modèle
model = joblib.load("rf_model_grade.joblib", mmap_mode=None)


# Titre de l'application
st.title("Prédiction des Notes")
st.markdown("## Prédisez les notes des étudiants en fonction de plusieurs critères !")

# Initialisation des valeurs dans session_state pour éviter la réinitialisation
if "socio_score" not in st.session_state:
    st.session_state.socio_score = 0.500
if "study_hours" not in st.session_state:
    st.session_state.study_hours = 5.0
if "sleep_hours" not in st.session_state:
    st.session_state.sleep_hours = 7.0
if "attendance" not in st.session_state:
    st.session_state.attendance = 90.0

# Choix de la méthode d'entrée
manual_input = st.checkbox("Entrer les valeurs manuellement", value=False)

# Champs de saisie : soit sliders, soit inputs
if manual_input:
    st.session_state.socio_score = st.number_input("Score Socioéconomique", 
                                                   min_value=0.000, max_value=1.000, 
                                                   value=st.session_state.socio_score, 
                                                   step=0.001, format="%.3f")
    st.session_state.study_hours = st.number_input("Heures d'Étude", 
                                                   min_value=0.0, max_value=24.0, 
                                                   value=st.session_state.study_hours, 
                                                   step=0.1)
    st.session_state.sleep_hours = st.number_input("Heures de Sommeil", 
                                                   min_value=0.0, max_value=24.0, 
                                                   value=st.session_state.sleep_hours, 
                                                   step=0.1)
    st.session_state.attendance = st.number_input("Présence (%)", 
                                                  min_value=0.0, max_value=100.0, 
                                                  value=st.session_state.attendance, 
                                                  step=0.1)
else:
    st.session_state.socio_score = st.slider("Score Socioéconomique", 
                                             min_value=0.000, max_value=1.000, 
                                             value=st.session_state.socio_score, 
                                             step=0.001, format="%.3f")
    st.session_state.study_hours = st.slider("Heures d'Étude", 0.0, 24.0, 
                                             st.session_state.study_hours, 0.1)
    st.session_state.sleep_hours = st.slider("Heures de Sommeil", 0.0, 24.0, 
                                             st.session_state.sleep_hours, 0.1)
    st.session_state.attendance = st.slider("Présence (%)", 0.0, 100.0, 
                                            st.session_state.attendance, 0.1)

# Fonction de normalisation Min-Max
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0

# Normalisation des valeurs pour le graphique radar
norm_socio_score = normalize(st.session_state.socio_score, 0, 1)
norm_study_hours = normalize(st.session_state.study_hours, 0, 24)
norm_sleep_hours = normalize(st.session_state.sleep_hours, 0, 24)
norm_attendance = normalize(st.session_state.attendance, 0, 100)

# Bouton de prédiction
if st.button("Prédire la Note"):
    try:
        # Création de l'input pour le modèle
        user_input = np.array([[st.session_state.socio_score, 
                                st.session_state.study_hours, 
                                st.session_state.sleep_hours, 
                                st.session_state.attendance]])
        predicted_grade = model.predict(user_input)
        
        # Affichage du résultat
        st.success(f"Note Prédite : {predicted_grade[0]:.2f}")
        
        # Radar Chart (Diagramme en toile d'araignée)
        categories = ['Socioeconomic', 'Study Hours', 'Sleep Hours', 'Attendance']
        values = [norm_socio_score, norm_study_hours, norm_sleep_hours, norm_attendance]
        
        values += values[:1]  # Boucler le graphique
        angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='blue', alpha=0.3)
        ax.plot(angles, values, color='blue', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_yticklabels([])  # Cacher les valeurs de l'axe radial
        st.pyplot(fig)
        
    except ValueError:
        st.error("Veuillez entrer des valeurs valides.")

# Personnalisation de la section About
st.sidebar.markdown("## À propos")
st.sidebar.info(
    "Cette application utilise un modèle de Machine Learning pour prédire les notes des étudiants en fonction de plusieurs facteurs tels que les heures d'étude, le sommeil, la présence et le statut socioéconomique.\n\n"+
    "Copyright CelestWeb | 2025")
