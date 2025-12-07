# pages/04_Popularity_Predictor.py

import streamlit as st
import pandas as pd
import joblib
import shap
import os
import sys
import numpy as np
import altair as alt

# ------------------------------------------------
# Setup
# ------------------------------------------------
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_root)
from utils import load_data

st.set_page_config(layout="wide")
st.title("Spotify Popularity Predictor")

logo = "https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg"
st.logo(logo)

# ------------------------------------------------
# Load Data & Model
# ------------------------------------------------
df_audiofeatures, df_tracks = load_data()

# Only for embedding
df_tracks_clean = df_tracks.rename(columns={'track_name': 'title', 'artist_name': 'artist'})
df_spotify = df_tracks_clean.merge(df_audiofeatures, on=['title', 'artist'], how='inner')

# Load pipeline & feature names
popularity_pipeline = joblib.load('models/popularity_pipeline.pkl')
feature_names = joblib.load('models/popularity_features.pkl')

# ------------------------------------------------
# Helper: Spotify Embed
# ------------------------------------------------
def embed_spotify_track(track_id):
    if pd.isna(track_id) or track_id == '':
        return "<p>Track ID not available for embed.</p>"
    return f"""
    <iframe style="border-radius:12px" 
        src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0" 
        width="100%" 
        height="152" 
        frameBorder="0" 
        allowfullscreen 
        allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
        loading="lazy">
    </iframe>
    """

# ------------------------------------------------
# UI: Song Selection
# ------------------------------------------------
song_options = df_spotify['title'] + " - " + df_spotify['artist']
song_choice = st.selectbox(
    "Choose a Song to Predict Popularity:",
    options=[""] + sorted(song_options.unique())
)

if song_choice:
    title_input, artist_input = song_choice.split(" - ")

    # ---------------------------
    # Prepare features from df_audiofeatures
    # ---------------------------
    song_row = df_audiofeatures[(df_audiofeatures['title'] == title_input) & (df_audiofeatures['artist'] == artist_input)]

    if song_row.empty:
        st.warning("Song not found in audio features dataset.")
    else:
        X_song = song_row.drop(columns=['pop', 'pop_bin', 'title', 'artist', 'genre'], errors='ignore')

        X_song = X_song.reindex(columns=feature_names, fill_value=0)
        X_song = X_song.astype(float)

        predicted_pop = popularity_pipeline.predict(X_song)[0]
        predicted_prob = popularity_pipeline.predict_proba(X_song)[0][1]


        col1, col2 = st.columns([1, 1])  # Equal width columns

        with col1:
            # Embed Spotify track
            song_embed_row = df_spotify[(df_spotify['title'] == title_input) & (df_spotify['artist'] == artist_input)]
            if not song_embed_row.empty:
                st.subheader(f"Song: {title_input} by {artist_input}")
                st.components.v1.html(embed_spotify_track(song_embed_row['track_id'].values[0]), height=160)

        with col2:
            st.subheader("Prediction Results")
            st.metric("Predicted Popularity (Binary)", "High" if predicted_pop == 1 else "Low")
            st.info(f"Predicted probability of being popular: {predicted_prob:.2%}")

        # ---------------------------
        # SHAP explainability
        # ---------------------------
        st.subheader("Feature Contribution (SHAP)")

        # Extract model and preprocessor
        model = popularity_pipeline.named_steps['model']
        preprocessor = popularity_pipeline.named_steps['preprocessor']

        # Transform input (numeric scaled, categorical passed through)
        X_transformed = preprocessor.transform(X_song)

        # Use a zero array as background for linear explainer
        background = np.zeros((1, X_transformed.shape[1]))

        # Initialize SHAP explainer for linear model
        explainer = shap.LinearExplainer(model, background, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_transformed)

        # Convert to DataFrame
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_value': shap_values[0]
        }).sort_values(by='SHAP_value', key=abs, ascending=False).head(10)

        # Add a color column: green for positive, red for negative
        shap_df['color'] = shap_df['SHAP_value'].apply(lambda x: 'green' if x > 0 else 'red')

        # Plot with Altair
        chart = alt.Chart(shap_df).mark_bar().encode(
            x=alt.X('SHAP_value:Q', title='SHAP Value'),
            y=alt.Y('Feature:N', sort='-x'),  # sort features in decreasing order by value
            color=alt.Color('color:N', scale=None)  # use predefined color column
        ).properties(
            width=700,
            height=400,
            title='Top 10 Feature Contributions'
        )

        st.altair_chart(chart)
