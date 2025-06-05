import os
import numpy as np
import pandas as pd
import streamlit as st
import umap
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import matplotlib.pyplot as plt

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

# ---- Ensure Valid Indices After Sampling and Fill NaNs ----
df = df.reset_index(drop=True)
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# ==========================================
# Define Modular Functions for Each Tab
# ==========================================

def show_home():
    # Header stays on one line
    st.markdown(
        "<h1 style='white-space: nowrap;'>MUSIC RECOMMENDATION DASHBOARD</h1>",
        unsafe_allow_html=True
    )
    
    # Welcome text stays on one line using inline CSS
    st.markdown(
        "<p style='white-space: nowrap;'>Welcome! Discover new music, explore genres, and generate playlists using the navigation sidebar on the left.</p>",
        unsafe_allow_html=True
    )
    
    # Specify the path for the banner image
    banner_path = "/Users/GURU/Desktop/Music_Recommendation/Images/birds-7717268_640.png"
    
    # Check if the file exists
    if os.path.isfile(banner_path):
        try:
            st.image(banner_path, use_container_width=True)
        except Exception as e:
            st.warning(f"An error occurred while opening the banner image. Error: {e}")
    else:
        st.warning(f"Dashboard banner image not found. Please ensure the file exists at:\n{banner_path}")



    
def show_similar_tracks():
    st.subheader("Find Similar Tracks")
    try:
        if "track_display" not in df.columns:
            raise ValueError("Column 'track_display' not found in the dataset.")

        # Build the dropdown from all available track_display entries
        track_options = sorted(df["track_display"].dropna().unique())
        selected_display = st.selectbox("Choose a track:", track_options)

        # Extract track and artist names (expected format: "track_name - artist_name")
        parts = selected_display.split(" - ", 1)
        if len(parts) != 2:
            raise ValueError("Selected track does not follow the expected 'track_name - artist_name' format.")
        selected_track = parts[0].strip()
        selected_artist = parts[1].strip()

        # Use case-insensitive matching to get the target track row
        mask = (df["track_name"].str.lower().str.strip() == selected_track.lower()) & \
               (df["artist_name"].str.lower().str.strip() == selected_artist.lower())
        track_features = df[mask].select_dtypes(include=["float64", "int64"])

        if track_features.empty:
            st.warning("Track features unavailable for similarity scoring.")
        else:
            # Compute cosine similarity between the selected track and all songs
            similarity_scores = cosine_similarity(
                track_features, df.select_dtypes(include=["float64", "int64"])
            )

            # Sort indices so that higher similarity scores come first
            sorted_indices = np.argsort(-similarity_scores[0])
            sorted_indices = sorted_indices[sorted_indices < len(df)]

            # Retrieve similar tracks; exclude the selected track (using case-insensitive comparison)
            similar_tracks = df.iloc[sorted_indices].copy().reset_index(drop=True)
            similar_tracks = similar_tracks[
                ~(
                    (similar_tracks["track_name"].str.lower().str.strip() == selected_track.lower()) &
                    (similar_tracks["artist_name"].str.lower().str.strip() == selected_artist.lower())
                )
            ]

            # Use a slider to let the user control how many recommended songs to display (1 to 10)
            num_recs = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
            similar_tracks = similar_tracks.head(num_recs)

            # Attach the corresponding similarity scores (rounded)
            similar_tracks["Similarity Score"] = np.round(similarity_scores[0][similar_tracks.index], 4)

            # Helper functions for generating Spotify links
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

            # Ensure that genre exists for display
            if "genre" not in similar_tracks.columns:
                similar_tracks["genre"] = "N/A"

            # Prepare the DataFrame for display
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

            st.subheader("Recommended Tracks (Limited to 10)")
            st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)

            st.subheader("Dataset Preview: Recommended Tracks")
            st.write(similar_tracks)
    except Exception as e:
        st.error(f"Error in generating recommendations: {e}")



def show_track_search():
    st.title("🔍 Track Search")
    query = st.text_input("Enter track name or artist:")
    if query:
        matched = df[df["track_display"].str.contains(query, case=False, na=False)]
        if not matched.empty:
            st.write(matched[["track_display", "genre"]])
        else:
            st.write("No matching tracks found.")

def show_personalized_recommendations():
    st.title("🎯 Personalized Recommendations")
    mood = st.selectbox("Choose your mood", ["Happy", "Sad", "Energetic", "Relaxed"])
    if "mood" in df.columns:
        recs = df[df["mood"] == mood].sample(10)
        st.write(recs)
    else:
        st.write("Mood data not available in the dataset.")

def show_genre_explorer():
    st.title("📊 Genre Explorer")
    if "genre" in df.columns:
        genre_counts = df["genre"].value_counts()
        fig, ax = plt.subplots()
        ax.bar(genre_counts.index, genre_counts.values)
        ax.set_title("Track Distribution by Genre")
        ax.set_xticklabels(genre_counts.index, rotation=45)
        st.pyplot(fig)
    else:
        st.write("Genre information not available.")

def show_playlist_generator():
    st.title("📀 Playlist Generator")
    if "genre" in df.columns:
        selected_genres = st.multiselect("Select Genres:", df["genre"].unique())
        if selected_genres:
            playlist = df[df["genre"].isin(selected_genres)].sample(15)
            st.write(playlist)
        else:
            st.write("Select genres to generate a playlist.")
    else:
        st.write("Genre information not available.")

def show_audio_feature_analysis():
    st.title("🎵 Audio Feature Analysis")
    features = ["tempo", "danceability", "energy"]
    if set(features).issubset(df.columns):
        st.write(df[["track_name", "artist_name"] + features].head(20))
    else:
        st.write("Audio feature data not available.")

def show_visualizations():
    st.title("📈 Visualization & Trends")
    if "release_year" in df.columns and "popularity" in df.columns:
        trend_df = df.groupby("release_year")["popularity"].mean().reset_index()
        st.line_chart(trend_df.set_index("release_year"))
    else:
        st.write("Trend data not available.")

def show_settings():
    st.title("⚙️ Settings & Customization")
    if "genre" in df.columns:
        default_genre = st.selectbox("Set your preferred default genre:", df["genre"].unique())
        st.write(f"Your default genre is now: {default_genre}")
    else:
        st.write("Genre information not available.")

# ==========================================
# Sidebar Navigation
# ==========================================
st.sidebar.title("Navigation")
tab = st.sidebar.radio(
    "Select a tab:",
    ["Home", "Similar Tracks", "Track Search", "Personalized Recommendations", "Genre Explorer",
     "Playlist Generator", "Audio Feature Analysis", "Visualization & Trends", "Settings & Customization"]
)

if tab == "Home":
    show_home()
elif tab == "Similar Tracks":
    show_similar_tracks()
elif tab == "Track Search":
    show_track_search()
elif tab == "Personalized Recommendations":
    show_personalized_recommendations()
elif tab == "Genre Explorer":
    show_genre_explorer()
elif tab == "Playlist Generator":
    show_playlist_generator()
elif tab == "Audio Feature Analysis":
    show_audio_feature_analysis()
elif tab == "Visualization & Trends":
    show_visualizations()
elif tab == "Settings & Customization":
    show_settings()
