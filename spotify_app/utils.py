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
    """
    Recommends songs similar to all songs by a given artist.
    Aggregates results, removes duplicates, and returns top N.
    """

    # Columns used during model training
    numeric_cols = ['bpm','nrgy','dnce','dB','live','val','dur','acous','spch','pop']
    categorical_cols = ['genre']
    id_cols = ['title', 'artist']
    feature_cols = numeric_cols + categorical_cols

    # Find all songs by the artist
    artist_songs = df[df['artist'].str.lower() == artist_name.lower()]

    if artist_songs.empty:
        return pd.DataFrame()

    recommendations = []

    # Loop through each song by the artist
    for _, song in artist_songs.iterrows():

        # Convert single row to DataFrame
        song_df = song[feature_cols].to_frame().T

        # Transform
        song_vector = transformer.transform(song_df)

        # Nearest neighbors
        distances, indices = model.kneighbors(song_vector, n_neighbors=n+1)

        # Get recommended songs (excluding the same one)
        recs = df.iloc[indices[0][1:]].copy()
        recs["similarity"] = 1 / (1 + distances[0][1:])

        # Keep consistent columns
        recs = recs[id_cols + ["similarity"]]

        recommendations.append(recs)

    # Combine all recommendation batches
    recommendations = pd.concat(recommendations, ignore_index=True)

    # Remove duplicates: keep highest similarity
    recommendations = (
        recommendations
        .drop_duplicates(subset=id_cols)
        .sort_values("similarity", ascending=False)
        .head(n)
    )

    return recommendations



def recommend_songs(song_title, df, model, transformer, n=10):

    # find exact song (case-insensitive)
    match = df[df['title'].str.lower() == song_title.lower()]

    if match.empty:
        return pd.DataFrame()

    # Columns used during training
    numeric_cols = ['bpm','nrgy','dnce','dB','live','val','dur','acous','spch','pop']
    categorical_cols = ['genre']
    id_cols = ['title', 'artist']

    feature_cols = numeric_cols + categorical_cols

    # Subset to training columns only
    X_song = match[feature_cols]

    # Transform
    song_vector = transformer.transform(X_song)

    # Neighbors
    distances, indices = model.kneighbors(song_vector, n_neighbors=n+1)

    # Results
    results = df.iloc[indices[0][1:]].copy()
    results["similarity"] = 1 / (1 + distances[0][1:])

    # Deduplicate
    results = (
        results
        .drop_duplicates(subset=id_cols)
        .sort_values("similarity", ascending=False)
        .head(n)
    )

    # Remove the searched song (safety)
    results = results[results["title"].str.lower() != song_title.lower()]

    return results[id_cols + ["similarity"]]

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