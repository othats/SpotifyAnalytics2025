import streamlit as st
import pandas as pd
from utils import load_data, load_models, recommend_songs, recommend_songs_by_artist

st.set_page_config(layout="wide")
st.title("KNN Song Recommender")

# Load data and models
df_audiofeatures, _ = load_data()
knn_model, knn_preprocessor, _, _, _ = load_models()

if df_audiofeatures is not None and knn_model is not None:
    st.header("Find Your Next Track", divider="green")

    # Recommendation Mode Selector
    mode = st.radio(
        "Select Recommendation Mode:", 
        ["By Similar Song", "By Similar Artist Tracks"],
        horizontal=True,
    )
    
    # Inputs
    n_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
    
    results = pd.DataFrame()

    if mode == "By Similar Song":
        song_list = df_audiofeatures['title'].unique()
        selected_song = st.selectbox(
            "Select a Song to find similar tracks:", 
            song_list
        )
        if st.button("Find Similar Songs", type="primary"):
            st.info(f"Searching for songs similar to **{selected_song}**...")
            results = recommend_songs(selected_song, df_audiofeatures, knn_model, knn_preprocessor, n_recommendations)

    elif mode == "By Similar Artist Tracks":
        artist_list = df_audiofeatures['artist'].unique()
        selected_artist = st.selectbox(
            "Select an Artist to find tracks similar to their style:",
            artist_list
        )
        if st.button("Find Artist Recommendations", type="primary"):
            st.info(f"Searching for tracks similar to the style of **{selected_artist}**...")
            results = recommend_songs_by_artist(selected_artist, df_audiofeatures, knn_model, knn_preprocessor, n_recommendations)

    # Output Results
    if not results.empty:
        st.subheader(f"Top {len(results)} Recommendations")
        # Rename columns for display
        results = results.rename(columns={'title': 'Track Name', 'artist': 'Artist Name', 'similarity': 'Similarity Score (0-1)'})
        
        # Display results with a progress bar for similarity score
        st.dataframe(
            results.sort_values('Similarity Score (0-1)', ascending=False),
            use_container_width=True,
            column_config={
                "Similarity Score (0-1)": st.column_config.ProgressColumn(
                    "Similarity Score",
                    help="Closer to 1.0 means higher similarity to the input track/artist.",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                )
            }
        )