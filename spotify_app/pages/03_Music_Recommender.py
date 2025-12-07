import streamlit as st
import pandas as pd
import plotly.express as px  # type: ignore
from utils import load_data, load_models, recommend_songs, recommend_songs_by_artist

st.set_page_config(layout="wide")
st.title("The Spotify Recommender")

logo = "https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg"
st.logo(logo)

# -----------------------
# LOAD DATA
# -----------------------
df_audiofeatures, df_tracks = load_data()

df_tracks_clean = df_tracks.rename(columns={'track_name': 'title', 'artist_name': 'artist'})

# Main merged DF used for Artist-based recommendations
df_recommender = (
    df_tracks_clean
    .merge(df_audiofeatures, on=['title', 'artist'], how='inner', suffixes=('_meta', '_audio'))
)

# Load models
knn, knn_preprocessor, _, _, _ = load_models()

# -----------------------
# HELPER: Spotify Embed
# -----------------------
def embed_spotify_track(track_id):
    """Embeds a Spotify track using its track ID."""
    if pd.isna(track_id) or track_id == '':
        return "<p>Track ID not available for embed.</p>"
    embed_code = f"""
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
    return embed_code


# -----------------------
# UI: TABS
# -----------------------
tab_song, tab_artist = st.tabs(["Find Similar Tracks", "Find Similar Artists' Music"])


# ===========================================================
# TAB 1: SONG-TO-SONG RECOMMENDATIONS
# ===========================================================
with tab_song:
    st.header("Similar Tracks by Song Title")
    
    col_input, col_slider = st.columns([3, 1])

    # Create selectbox choices: "Song - Artist"
    song_options = df_tracks_clean['title'] + "  -  " + df_tracks_clean['artist']

    with col_input:
        song_display = st.selectbox(
            "Choose a Song:",
            options=[""] + sorted(song_options.unique()),
            placeholder="e.g., Shallow - Lady Gaga, Shape of You - Ed Sheeran"
        )
    
    with col_slider:
        n_recs = st.slider("Number of Recommendations", 5, 20, 10, key='song_n_recs')

    if st.button("Get Song Recommendations", use_container_width=True):
        if song_display:
            # Split back into title + artist
            try:
                title_input, artist_input = song_display.split("  -  ")
            except:
                st.warning("Invalid song format.")
                st.stop()

            with st.spinner(f"Finding {n_recs} similar tracks to **{title_input}**..."):
                recommendations = recommend_songs(
                    title_input, df_audiofeatures, knn, knn_preprocessor, n_recs+1
                )

                if recommendations.empty:
                    st.warning(f"No match found for song: **{title_input}**. Check spelling or try another track.")
                else:
                    st.success(f"Top {len(recommendations)} tracks similar to **{title_input}**:")

                    # --- Visualization
                    recommendations['title_artist'] = recommendations['title'] + " - " + recommendations['artist']
                    fig_rec = px.bar(
                        recommendations.head(10),
                        x='similarity',
                        y='title_artist',
                        orientation='h',
                        title='Similarity Score of Recommended Tracks (KNN)',
                        labels={'similarity': 'Similarity Score (0-1)', 'title_artist': 'Track'},
                        color_discrete_sequence=[st.get_option("theme.primaryColor")]
                    )
                    fig_rec.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_rec, use_container_width=True)

                    # --- Details + Embeds (SONG TAB)
                    st.subheader("Detailed Recommendations")

                    # Safe merge using df_recommender, which includes genre + track_id
                    rec_details = recommendations.merge(
                        df_recommender[['title', 'artist', 'genre', 'track_id']],
                        on=['title', 'artist'],
                        how='left'
                    ).head(n_recs)
                    # drop duplicates just in case
                    rec_details = rec_details.drop_duplicates(subset=['title', 'artist']) 

                    for i, row in rec_details.iterrows():
                        col_rank, col_info, col_embed = st.columns([0.5, 3, 4])

                        with col_rank:
                            st.metric(label="Rank", value=i + 1)

                        with col_info:
                            st.markdown(f"**{row['title']}** by **{row['artist']}**")
                            st.markdown(f"Similarity: **{row['similarity']:.4f}**")
                            st.markdown(
                                f"Genre: *{row['genre']}*" 
                                if pd.notna(row.get("genre")) else 
                                "Genre: *Not Available*"
                            )

                        with col_embed:
                            if pd.isna(row.get("track_id")):
                                st.info("⚠️ No Spotify preview available for this track.")
                            else:
                                st.components.v1.html(embed_spotify_track(row['track_id']), height=160)


        else:
            st.warning("Please choose a song first.")

# ===========================================================
# TAB 2: ARTIST-TO-SONG RECOMMENDATIONS
# ===========================================================
with tab_artist:
    st.header("Similar Tracks by Artist")

    col_input, col_slider = st.columns([3, 1])

    # Artist list for dropdown
    artist_options = sorted(df_tracks_clean['artist'].unique())

    with col_input:
        artist_input = st.selectbox(
            "Choose an Artist:",
            options=[""] + artist_options,
            placeholder="e.g., Adele, Drake, Coldplay"
        )

    with col_slider:
        n_recs_artist = st.slider("Number of Recommendations", 5, 20, 10, key='artist_n_recs')

    if st.button("Get Artist Recommendations", use_container_width=True):
        if artist_input:

            with st.spinner(f"Finding {n_recs_artist} tracks similar to **{artist_input}**..."):
                recs_artist = recommend_songs_by_artist(
                    artist_input,
                    df_audiofeatures,
                    knn,
                    knn_preprocessor,
                    n_recs_artist
                )

                if recs_artist.empty:
                    st.warning(f"No songs found for artist **{artist_input}**.")
                else:
                    st.success(f"Top {len(recs_artist)} songs similar to **{artist_input}**:")

                    # Add combined label for chart
                    recs_artist['title_artist'] = recs_artist['title'] + " - " + recs_artist['artist']

                    # --- Visualization
                    fig_artist = px.bar(
                        recs_artist.head(10),
                        x='similarity',
                        y='title_artist',
                        orientation='h',
                        title=f"Similarity Score of Recommended Tracks to {artist_input}",
                        labels={'similarity': 'Similarity Score (0-1)', 'title_artist': 'Track'},
                        color_discrete_sequence=[st.get_option("theme.primaryColor")]
                    )
                    fig_artist.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_artist, use_container_width=True)

                    # --- Detailed Results with Embeds
                    st.subheader("Detailed Recommendations")

                    # Merge with df_recommender to get track_id + genre
                    artist_details = recs_artist.merge(
                        df_recommender[['title', 'artist', 'genre', 'track_id']],
                        on=['title', 'artist'],
                        how='left'
                    )

                    # Deduplicate after merge (very important)
                    artist_details = artist_details.drop_duplicates(subset=['title', 'artist'])

                    for i, row in artist_details.head(n_recs_artist).iterrows():
                        col_rank, col_info, col_embed = st.columns([0.5, 3, 4])

                        with col_rank:
                            st.metric(label="Rank", value=i + 1)

                        with col_info:
                            st.markdown(f"**{row['title']}** by **{row['artist']}**")
                            st.markdown(f"Similarity: **{row['similarity']:.4f}**")
                            st.markdown(
                                f"Genre: *{row['genre']}*"
                                if pd.notna(row.get("genre")) else
                                "Genre: *Not Available*"
                            )

                        with col_embed:
                            if pd.isna(row.get("track_id")):
                                st.info("⚠️ No Spotify preview available.")
                            else:
                                st.components.v1.html(embed_spotify_track(row['track_id']), height=160)

        else:
            st.warning("Please choose an artist first.")
