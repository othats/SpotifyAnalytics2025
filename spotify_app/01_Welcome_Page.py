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
st.title("A Visual Music Recommender System")
st.markdown(f"""
## Powered by **:primary[Spotify Data]** and Advanced Analytics
""")

st.markdown("""
This project integrates **Machine Learning** models with **Visual Analytics** tools (Streamlit, Tableau) to explore the musical landscape of thousands of tracks. 
""")
st.divider()

# --- Project Summary Section ---
col1, col2, col3 = st.columns(3)

with col1:
    st.header("1. Data Exploration")
    st.markdown("Dive into the raw statistics, temporal trends, and feature distributions that shape music popularity.")

with col2:
    st.header("2. Recommender System")
    st.markdown("Use a **K-Nearest Neighbors** model on audio features to find songs highly similar to a given track or artist.")
    
with col3:
    st.header("3. Predictive Analytics")
    st.markdown("Explore our **Random Forest Regressor** and the **SHAP** explanations to understand what drives a song's **:primary[Popularity]** score.")
    
st.divider()

# --- Team Info ---
st.info(f"""
**Project Members:** Marina Castellano Blanco & Júlia Othats-Dalès Gibert
""")