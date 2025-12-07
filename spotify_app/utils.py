import streamlit as st
import pandas as pd
import joblib
import os
import sys

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(PROJECT_ROOT)

# global constants
ID_COLS = ['title', 'artist']
FEATURE_COLS = ['bpm','nrgy','dnce','dB','live','val','dur','acous','spch']

@st.cache_data
def load_data():
    try: 
        df_audiofeatures = pd.read_csv(
            os.path.join(PROJECT_ROOT, 'data', 'spotify_audio_features.csv')
        )
        df_tracks = pd.read_csv(
            os.path.join(PROJECT_ROOT, 'data', 'spotify_tracks_artists.csv')
        )
        return df_audiofeatures, df_tracks
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None
    
@st.cache_resource
def load_models():
    try: 
        knn = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'recommender_knn.pkl'))
        knn_preprocessor = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'recommender_preprocessor.pkl'))
        popularity_pipeline = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'popularity_pipeline.pkl'))
        genre_model = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'genre_classifier.pkl')) 
        genre_scaler = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'genre_scaler.pkl'))
        return knn, knn_preprocessor, popularity_pipeline, genre_model, genre_scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None
    
def recommend_songs_by_artist(artist_name, df, model, transformer, n=10):
    """Recommends songs by finding similar tracks to all tracks by a given artist."""

    # Search for songs by the artist (case-insensitive)
    artist_songs = df[df['artist'].str.lower() == artist_name.lower()]

    if artist_songs.empty:
        # Returning a DataFrame instead of a string handles the Streamlit page logic cleaner
        return pd.DataFrame() 

    recommendations = []

    for _, song in artist_songs.iterrows():

        # Get features and transform the song vector
        song_df = song.to_frame().T  
        song_vector = transformer.transform(song_df)

        # Find neighbors (n+1 to exclude the song itself)
        distances, indices = model.kneighbors(song_vector, n_neighbors=n+1)

        # Get similar songs
        recs = df.iloc[indices[0][1:]].copy()

        # Add similarity score (1 / (1 + distance))
        recs["similarity"] = 1 / (1 + distances[0][1:])

        # Save columns
        recommendations.append(recs[[*ID_COLS, "similarity"]])

    # Aggregate, deduplicate, and sort
    recommendations = (
        pd.concat(recommendations, ignore_index=True)
        .drop_duplicates(subset=ID_COLS)
        .sort_values("similarity", ascending=False)
        .head(n)
    )

    return recommendations

def recommend_songs(song_title, df, model, transformer, n=10):
    """Recommends songs similar to a specific track."""

    # Search for the exact song (case-insensitive)
    match = df[df['title'].str.lower() == song_title.lower()]

    if match.empty:
        return pd.DataFrame() 

    # Transform the song vector
    song_vector = transformer.transform(match)

    # Find nearest neighbors (n+1 to exclude the same song)
    distances, indices = model.kneighbors(song_vector, n_neighbors=n+1)

    results = df.loc[indices[0][1:]].copy() # Exclude the input song
    results["similarity"] = 1 / (1 + distances[0][1:]) # Convert distance to similarity

    return results[[*ID_COLS, "similarity"]]


# --- Prediction Function (Adapted from your code) ---

def predict_genre(song_title, df, model, scaler):
    """Predicts the root genre for a selected song."""
    
    # df is assumed to be df_audiofeatures, which contains 'genre_root'
    song_data = df[df['title'].str.lower() == song_title.lower()]
    
    if song_data.empty:
        return {'Predicted Genre': 'N/A', 'True Genre': 'N/A'}
    
    # Use the defined features
    song_features = song_data[FEATURE_COLS]
    
    # Scale and predict
    song_features_scaled = scaler.transform(song_features)
    genre_pred = model.predict(song_features_scaled)
    
    # Get the true genre (assuming 'genre_root' is the target column)
    # Use .iloc[0] for robustness if multiple songs have the same title
    try:
        true_genre = song_data['genre_root'].values[0]
    except KeyError:
        true_genre = "Column 'genre_root' missing"
    
    return {
        'Predicted Genre': genre_pred[0],
        'True Genre': true_genre
    }