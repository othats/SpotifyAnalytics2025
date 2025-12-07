import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px #type: ignore
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from utils import load_data

st.set_page_config(layout="wide")
st.title("Spotify Music Analytics – Data Exploration")

# Theme color shortcut
THEME_COLOR = st.get_option("theme.primaryColor")

# Load data
df_audiofeatures, df_tracks = load_data()

df_tracks_clean = df_tracks.rename(columns={'track_name': 'title', 'artist_name': 'artist'})

# Merge datasets
df_spotify = (
    df_tracks_clean
    .merge(df_audiofeatures, on=['title', 'artist'], how='inner', suffixes=('_meta', '_audio'))
)

# Numeric audio features used later
numeric_cols = ['bpm', 'nrgy', 'dnce', 'dB', 'live', 'val', 'dur', 'acous', 'spch', 'pop']

df_spotify[numeric_cols] = df_spotify[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Define helper for matplotlib figures
def show_pyplot(fig):
    st.pyplot(fig)
    plt.close(fig)

st.write("Navigate using the tabs below to explore the Spotify datasets.")

# Create Tabs
tab1, tab2, tab3 = st.tabs([
    "Dataset Overview",
    "Tracks & Artists - EDA",
    "Audio Features - EDA"
])

# ===================================================================
# TAB 1 — OVERVIEW
# ===================================================================
with tab1:
    st.header("Overview of the Datasets", divider="green")

    st.markdown("""
    Our analysis is based on **two complementary Spotify datasets**, each contributing a unique view of modern music:

    ### **1. Tracks & Artists Dataset (`df_tracks`)**
    Provides **contextual metadata** about each song:
    - **Track name, artist name**
    - **Track popularity** (Spotify's global score)
    - **Release year**
    - **Track number** in the album
    - **Artist followers**
    - **Artist popularity**
    - **Artist genres**

    This allows us to answer:
    - *Who are the most influential artists?*
    - *How does popularity vary by year or by album structure?*
    - *Which genres dominate the platform?*

    ### **2. Audio Features Dataset (`df_audiofeatures`)**
    Contains **algorithmically computed acoustic descriptors**, such as:
    - **BPM (tempo)**
    - **Danceability (`dnce`)**
    - **Energy (`nrgy`)**
    - **Loudness (`dB`)**
    - **Acousticness (`acous`)**
    - **Speechiness (`spch`)**
    - **Valence (`val`)**
    - **Instrumentalness (`live`)**
    - **Machine-learning target: `pop` (track popularity)**

    This dataset is key for:
    - Understanding the *sound* of popular music  
    - Building ML models (genre prediction, popularity regression)
    - Exploring correlations between audio characteristics
    """)

    st.subheader("Dataset Samples")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Tracks & Artists Sample**")
        st.dataframe(df_tracks.head(5), use_container_width=True)

    with col2:
        st.markdown("**Audio Features Sample**")
        st.dataframe(df_audiofeatures.head(5), use_container_width=True)

# ===================================================================
# TAB 2 — TRACKS & ARTISTS EDA
# ===================================================================
with tab2:
    st.header("Tracks & Artists: Exploratory Data Analysis", divider="green")
    st.markdown("""
    In this section we analyze **metadata-driven trends**: popularity, genres,
    release patterns, and artist impact.
    """)

    # --- Popularity Distributions ---
    st.subheader("Popularity Distributions")

    colA, colB = st.columns(2)

    with colA:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(df_spotify['track_popularity'], kde=True, bins=30, color=THEME_COLOR, ax=ax)
        ax.set_title("Distribution of Track Popularity")
        show_pyplot(fig)

    with colB:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(df_spotify['artist_popularity'], kde=True, bins=30, color=THEME_COLOR, ax=ax)
        ax.set_title("Distribution of Artist Popularity")
        show_pyplot(fig)

    # --- Artist vs Track Popularity ---
    st.subheader("Artist Popularity vs. Average Track Popularity")

    artist_avg = (
        df_spotify.groupby("artist")["track_popularity"]
        .mean()
        .reset_index()
        .rename(columns={'track_popularity': 'avg_track_popularity'})
    )

    artist_info = df_spotify[['artist','artist_popularity','artist_followers']].drop_duplicates()
    artist_df = artist_avg.merge(artist_info, on='artist')

    fig_scatter = px.scatter(
        artist_df,
        x="artist_popularity",
        y="avg_track_popularity",
        size="artist_followers",
        hover_data=['artist'],
        color_discrete_sequence=[THEME_COLOR],
        title="Artist Popularity vs Average Track Popularity",
        height=450
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Genre Frequency ---
    st.subheader("Most Common Genres")

    def clean_genre_list(s):
        if pd.isna(s):
            return []
        return [g.strip("[]'\" ") for g in s.split(",") if g.strip()]

    all_genres = df_spotify['artist_genres'].apply(clean_genre_list).explode()
    all_genres = all_genres[all_genres != ""]

    genre_counts = Counter(all_genres).most_common(20)
    df_genres = pd.DataFrame(genre_counts, columns=["Genre", "Count"])

    fig_gen = px.bar(
        df_genres, x="Count", y="Genre", orientation="h",
        title="Top 20 Most Frequent Genres",
        color_discrete_sequence=[THEME_COLOR]
    )
    fig_gen.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_gen, use_container_width=True)

    # --- Year Trends ---
    st.subheader("Tracks Released by Year")

    tracks_per_year = (
        df_spotify.groupby("release_year")["title"]
        .count().reset_index().rename(columns={'title':'num_tracks'})
    )

    fig_year = px.bar(
        tracks_per_year, x="release_year", y="num_tracks",
        color_discrete_sequence=[THEME_COLOR],
        title="Tracks Released per Year"
    )
    st.plotly_chart(fig_year, use_container_width=True)

    # --- Popularity by Track Number ---
    st.subheader("Popularity by Track Number (Album Structure)")

    df_filtered = df_spotify[df_spotify['track_number'] <= 25]

    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(data=df_filtered, x="track_number", y="track_popularity",
                color=THEME_COLOR, ax=ax)
    ax.set_title("Popularity by Track Number (First 25)")
    ax.tick_params(axis='x', rotation=45)
    show_pyplot(fig)

# ===================================================================
# TAB 3 — AUDIO FEATURES EDA
# ===================================================================
with tab3:
    st.header("Audio Features: Exploratory Data Analysis", divider="green")
    st.markdown("""
    In this section we analyze the **sound characteristics** of the songs:
    distributions, correlations, and how features relate to popularity.
    """)

    # --- Distributions of audio features ---
    st.subheader("Audio Feature Distributions")

    fig, axes = plt.subplots(3, 4, figsize=(15,10))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(df_spotify[col].dropna(), kde=True, bins=30,
                     color=THEME_COLOR, ax=axes[i])
        axes[i].set_title(col.upper())
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    show_pyplot(fig)

    # --- Correlation heatmap ---
    st.subheader("Correlation Between Audio Features")

    fig, ax = plt.subplots(figsize=(8,6))
    corr = df_spotify[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix of Audio Features")
    show_pyplot(fig)

    # --- Popularity vs Individual Features ---
    st.subheader("Popularity vs Individual Audio Features")

    cols_plot = st.columns(3)
    features = ['nrgy','dnce','val','dur','acous','spch','bpm','dB','live']

    for i, col in enumerate(features):
        with cols_plot[i % 3]:
            fig_feat = px.scatter(
                df_spotify, x=col, y="pop",
                opacity=0.4, title=f"Popularity vs {col.upper()}",
                color_discrete_sequence=[THEME_COLOR],
                height=350
            )
            st.plotly_chart(fig_feat, use_container_width=True)

    # --- Evolution of Features Over Time ---
    st.subheader("Evolution of Key Audio Features Over Time")

    avg_feat_year = df_spotify.groupby("release_year")[numeric_cols].mean().reset_index()

    fig_evol = px.line(
        avg_feat_year,
        x="release_year",
        y=["nrgy","dnce","val","acous"],
        title="Evolution of Key Features Over Time",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Set2,
        height=450
    )
    st.plotly_chart(fig_evol, use_container_width=True)
