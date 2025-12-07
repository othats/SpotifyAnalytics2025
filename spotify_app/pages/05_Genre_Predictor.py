# pages/05_Genre_Predictor.py

import streamlit as st
import pandas as pd
import joblib
import os
import sys

# ------------------------------------------------
# Setup
# ------------------------------------------------
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_root)
from utils import load_data

st.set_page_config(layout="wide")
st.title("Spotify Genre Predictor")

logo = "https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg"
st.logo(logo)

# ------------------------------------------------
# Load Data & Model
# ------------------------------------------------
df_audiofeatures, _ = load_data()

# Load scaler and model
scaler = joblib.load("models/genre_scaler.pkl")
model = joblib.load("models/genre_classifier.pkl")

feature_cols = ['bpm', 'dnce', 'live', 'val', 'dur', 'acous', 'spch', 'nrgy', 'dB']

# ------------------------------------------------
# UI: Song Selection
# ------------------------------------------------
song_options = df_audiofeatures['title'] + "  -  " + df_audiofeatures['artist']
song_choice = st.selectbox(
    "Choose a Song to Predict Genre:",
    options=[""] + sorted(song_options.unique())
)

# ------------------------------------------------
# Predict Genre
# ------------------------------------------------
if song_choice:
    title_input, artist_input = song_choice.split("  -  ")
    song_row = df_audiofeatures[(df_audiofeatures['title'] == title_input) & 
                                (df_audiofeatures['artist'] == artist_input)]
    
    if song_row.empty:
        st.warning("Song not found in audio features dataset.")
    else:
        X_song = song_row[feature_cols]
        X_song_scaled = scaler.transform(X_song)
        genre_pred = model.predict(X_song_scaled)[0]
        true_genre = song_row['genre'].values[0]

        # ---------------------------
        # Layout: Two Columns
        # ---------------------------
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Song Info")
            st.markdown(f"**Title:** {title_input}")
            st.markdown(f"**Artist:** {artist_input}")
            st.markdown(f"**True Genre:** {true_genre}")

        with col2:
            st.subheader("Predicted Genre")
            st.metric("Predicted Genre", genre_pred)
