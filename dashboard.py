import os
import numpy as np
import pandas as pd
import streamlit as st
import umap
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Import the common data loading function from data_utils.py
from data_utils import load_and_preprocess_data

# ---- Set Up Spotify API ----



sp = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET
    )
)

# ---- Check and Fix NumPy Compatibility Issues ----
if np.__version__.startswith("2"):
    st.warning(
        "NumPy 2.x detected, which may cause compatibility issues. Consider downgrading with: pip install numpy<2>."
    )

# ---- Load Dataset via the Centralized Function ----
DATA_PATH = "Resource/SpotifyFeatures.csv"
df, scaler = load_and_preprocess_data(DATA_PATH, sample_size=5000)

# ---- Ensure Valid Indices After Sampling ----
df = df.reset_index(drop=True)

# ---- Handle Missing Values (NaNs) in Numeric Features ----
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())  # Fill NaNs with column means

# ---- Dashboard Header ----
st.title("🎶 Music Recommendation Dashboard")
st.write("Explore song similarities and discover new tracks!")

# ---- Find Similar Tracks ----
st.subheader("Find Similar Tracks")
try:
    # Ensure the 'track_display' column exists
    if "track_display" not in df.columns:
        raise ValueError("Column 'track_display' not found in the dataset.")

    track_options = sorted(df["track_display"].dropna().unique())
    selected_display = st.selectbox("Choose a track:", track_options)

    # Extract the track name and artist from "track_display" (format: "track_name - artist_name").
    parts = selected_display.split(" - ", 1)
    if len(parts) != 2:
        raise ValueError("Selected track does not follow the expected 'track_name - artist_name' format.")
    selected_track, selected_artist = parts[0].strip(), parts[1].strip()

    # Get numeric features for the selected track.
    track_features = df[
        (df["track_name"] == selected_track) & (df["artist_name"] == selected_artist)
    ]
    track_features = track_features.select_dtypes(include=["float64", "int64"])

    if track_features.empty:
        st.warning("Track features unavailable for similarity scoring.")
    else:
        # ---- ✅ Adjusted Cosine Similarity Calculation ----
        similarity_scores = cosine_similarity(
            track_features, df.select_dtypes(include=["float64", "int64"])
        )

        # Ensure sorted indices remain within bounds
        sorted_indices = np.argsort(-similarity_scores[0])
        sorted_indices = sorted_indices[sorted_indices < len(df)]  # Keeps indices valid

        # Retrieve similar tracks, exclude the selected track
        similar_tracks = df.iloc[sorted_indices].copy()
        similar_tracks = similar_tracks.reset_index(drop=True)  # Prevents index mismatches

        # Exclude exact match from recommendations
        similar_tracks = similar_tracks[similar_tracks["track_display"] != selected_display]
        similar_tracks = similar_tracks.head(5)

        # Ensure similarity scores have decimals and negative values where applicable
        similar_tracks["Similarity Score"] = np.round(similarity_scores[0][similar_tracks.index], 4)

        # ---- Helper Functions for Spotify Links ----
        def get_spotify_link(track_name, artist_name):
            query = f"{track_name} {artist_name}"
            result = sp.search(query, type="track", limit=1)
            if result["tracks"]["items"]:
                return result["tracks"]["items"][0]["external_urls"]["spotify"]
            return None

        def get_spotify_link_html(row):
            link = get_spotify_link(row["track_name"], row["artist_name"])
            return f'<a href="{link}" target="_blank">Listen on Spotify</a>' if link else "No Link"

        similar_tracks["Listen on Spotify"] = similar_tracks.apply(get_spotify_link_html, axis=1)

        # ---- Prepare DataFrame for Display ----
        if "genre" not in similar_tracks.columns:
            similar_tracks["genre"] = "N/A"

        display_df = similar_tracks[["track_name", "artist_name", "genre", "Listen on Spotify", "Similarity Score"]].copy()
        display_df.rename(
            columns={
                "track_name": "TRACK NAME",
                "artist_name": "ARTIST NAME",
                "genre": "GENRE",
                "Listen on Spotify": "PLAY SONG 🎵",
                "Similarity Score": "SIMILARITY SCORE",
            },
            inplace=True,
        )

        # ---- Display Recommendations ----
        st.subheader("Recommended Tracks")
        st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        st.subheader("Dataset Preview: Recommended Songs Only")
        st.write(similar_tracks)

except Exception as e:
    st.error(f"Error in generating recommendations: {e}")
