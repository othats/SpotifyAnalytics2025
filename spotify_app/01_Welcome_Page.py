import streamlit as st
import os
import sys

logo = "https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg"

project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_root)

st.set_page_config(
    page_title = "Spotify Analytics App",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

st.logo(logo)
st.title("Spotify Analytics App")
st.markdown("## Análisis y Modelos Impulsados por Datos de Spotify")

st.markdown("---")
st.header("Introducción")
st.markdown(
    """
    Bienvenido a la aplicación web de nuestro proyecto final. Esta plataforma permite explorar los datos
    de millones de canciones de Spotify y utilizar modelos de Machine Learning
    para la recomendación y predicción musical.

    **Utiliza el menú de la izquierda para navegar entre las secciones:**

    * **EDA:** Explora las distribuciones, tendencias temporales y correlaciones de las características de audio.
    * **Similar Song Recommender:** Encuentra canciones similares a cualquier pista dada usando K-Nearest Neighbors (KNN).
    * **Popularity Predictor + XAI:** Predice la popularidad de una canción y usa SHAP para explicar qué características impulsan la predicción.
    * **Genre Predictor:** Un clasificador que predice el género de una canción basado en sus características de audio.
    """
)

st.markdown("---")
st.header("Tecnologías")
col1, col2, col3 = st.columns(3)
col1.metric("Análisis y ML", "Python, Pandas, Scikit-learn, SHAP")
col2.metric("Visualización", "Matplotlib, Seaborn, Streamlit, Tableau")
col3.metric("Datos", "Spotify API Data, Kaggle Datasets")

st.markdown("---")
st.write("Miembros: Marina Castellano Blanco & Júlia Othats-Dalès Gibert")