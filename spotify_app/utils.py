import streamlit as st
import pandas as pd
import joblib
import os
import sys

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(PROJECT_ROOT)

@st.cache_data
def load_data():
    try: 
        df_audiofeatures = pd.read_csv(
            os.path.join(PROJECT_ROOT, 'data', 'spotify_audiofeatures.csv')
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
        knn = joblib.load('models/recommender_knn.pkl')
        knn_preprocessor = joblib.load('models/recommender_preprocessor.pkl')
        popularity_pipeline = joblib.load('models/popularity_pipeline.pkl')
        return knn, knn_preprocessor, popularity_pipeline
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None
    
def recommend_songs(song_title, df, model, transformer, n=10):
    match = df[df['title'].str.lower() == song_title.lower()]

    if match.empty:
        return f"No encontré la canción '{song_title}'."

    song_vector = transformer.transform(match)

    distances, indices = model.kneighbors(song_vector, n_neighbors=n+1)

    results = df.loc[indices[0][1:]]
    results["similarity"] = 1 / (1 + distances[0][1:])
    return pd.DataFrame(results)

def recommend_songs_by_artist(artist_name, df, model, transformer, n=10):

    artist_songs = df[df['artist'].str.lower() == artist_name.lower()]

    if artist_songs.empty:
        return f"No encontré canciones del artista '{artist_name}'."

    recommendations = []

    for _, song in artist_songs.iterrows():

        song_df = song.to_frame().T  

        song_vector = transformer.transform(song_df)

        distances, indices = model.kneighbors(song_vector, n_neighbors=n+1)

        recs = df.iloc[indices[0][1:]].copy()

        recs["similarity"] = 1 / (1 + distances[0][1:])

        recommendations.append(recs[[*id_cols, "similarity"]])

    recommendations = pd.concat(recommendations, ignore_index=True)

    recommendations = (
        recommendations
        .drop_duplicates(subset=id_cols)
        .sort_values("similarity", ascending=False)
        .head(n)
    )

    return recommendations