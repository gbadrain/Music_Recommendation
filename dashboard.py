import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import load_and_preprocess_data
from dotenv import load_dotenv
# ---------------------------
# Load Environment Variables (Secure API Credentials)
# ---------------------------
load_dotenv()
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

sp = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
)

# ---------------------------
# Data Loading & Preprocessing
# ---------------------------
MAIN_DATA_PATH = "Resource/SpotifyFeatures.csv"
ADDITIONAL_DATA_PATH = "Output/umap_results.csv"

df, scaler = load_and_preprocess_data(MAIN_DATA_PATH, additional_filepath=ADDITIONAL_DATA_PATH)
df = df.reset_index(drop=True)
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# ---------------------------
# Dashboard Tab Functions
# ---------------------------

def show_home():
    st.markdown("<h1 style='white-space: nowrap;'>MUSIC RECOMMENDATION DASHBOARD</h1>", unsafe_allow_html=True)
    
    st.markdown("<p style='white-space: nowrap;'>Welcome! Discover new music, explore genres, and generate playlists using the navigation sidebar.</p>", unsafe_allow_html=True)
    
    banner_path = "/Users/GURU/Desktop/Music_Recommendation/Images/birds-7717268_640.png"
    
    if os.path.isfile(banner_path):
        try:
            st.image(banner_path, use_container_width=True)
        except Exception as e:
            st.warning(f"An error occurred while opening the banner image. Error: {e}")
    else:
        st.warning(f"Dashboard banner image not found. Please ensure the file exists at:\n{banner_path}")

def show_similar_tracks():
    try:
        st.subheader("Find Similar Tracks")
        if "track_display" not in df.columns:
            raise ValueError("Column 'track_display' not found in the dataset.")
        track_options = sorted(df["track_display"].dropna().unique())
        selected_display = st.selectbox("Choose a track or an artist:", track_options)

        mask = (df["track_display"].str.lower().str.strip() == selected_display.lower().strip())
        track_features = df.loc[mask, df.select_dtypes(include=["float64", "int64"]).columns]

        if track_features.empty:
            st.warning("Track features unavailable for similarity scoring.")
            return
        
        similarity_scores = cosine_similarity(track_features, df.select_dtypes(include=["float64", "int64"]))
        
        sorted_indices = np.argsort(-similarity_scores[0])
        sorted_indices = sorted_indices[sorted_indices < len(df)]
        
        similar_tracks = df.iloc[sorted_indices].copy().reset_index(drop=True)
        similar_tracks = similar_tracks[similar_tracks["track_display"] != selected_display]
        
        if "Cluster" in df.columns or "cluster" in df.columns:
            cluster_col = "Cluster" if "Cluster" in df.columns else "cluster"
            selected_cluster = df.loc[mask, cluster_col].iloc[0]
            similar_tracks = similar_tracks[similar_tracks[cluster_col] == selected_cluster]
        
        min_similarity = st.slider("Minimum Similarity Score", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        
        similar_tracks["Similarity Score"] = np.round(similarity_scores[0][similar_tracks.index], 4)
        similar_tracks = similar_tracks[similar_tracks["Similarity Score"] >= min_similarity]
        
        num_recs = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
        similar_tracks = similar_tracks.head(num_recs)
        
        import time

        # Define cache outside the function if not already defined
        global cache
        if 'cache' not in globals():
            cache = {}

        def get_spotify_link(track_display):
            query = track_display.replace("-", " ")
            if track_display in cache:
                return cache[track_display]
            try:
                result = sp.search(query, type="track", limit=1)
                if result["tracks"]["items"]:
                    link = result["tracks"]["items"][0]["external_urls"]["spotify"]
                    cache[track_display] = link  # Store result in cache
                    time.sleep(2)  # Increase delay to reduce API calls
                    return link
            except Exception as e:
                st.warning(f"Spotify API error: {str(e)}")
                return None

        def get_spotify_link_html(row):
            link = get_spotify_link(row["track_display"])
            return f'<a href="{link}" target="_blank">Listen on Spotify</a>' if link else "No Link"
        
        similar_tracks["Listen on Spotify"] = similar_tracks.apply(get_spotify_link_html, axis=1)
        
        if "genre" not in similar_tracks.columns:
            similar_tracks["genre"] = "N/A"
        
        display_df = similar_tracks[["track_display", "genre", "Listen on Spotify", "Similarity Score"]].copy()
        display_df.rename(
            columns={
                "track_display": "TRACK NAME - ARTIST NAME",
                "genre": "GENRE",
                "Listen on Spotify": "PLAY SONG 🎵",
                "Similarity Score": "SIMILARITY SCORE"
            },
            inplace=True,
        )
        
        st.subheader("Recommended Tracks (Filtered by Similarity Score)")
        st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.subheader("Dataset Preview: Recommended Tracks")
        st.write(similar_tracks)
    except Exception as e:
        st.error(f"Error in generating recommendations: {e}")

def show_personalized_recommendations():
    st.subheader("Personalized Recommendations")

    # Mood selection dropdown (Happy, Sad, Loving)
    mood_options = ["Happy", "Sad", "Loving"]
    selected_mood = st.selectbox("Select your mood:", mood_options)

    # Define mood-based filtering logic
    mood_mapping = {
        "Happy": df[df["valence"] >= 0.7],
        "Sad": df[df["valence"] <= 0.3],
        "Loving": df[(df["energy"].between(0.4, 0.7)) & (df["valence"].between(0.4, 0.7))]
    }

    # Filter songs based on selected mood
    mood_filtered_df = mood_mapping.get(selected_mood, df)

    # Ensure safe column selection
    available_columns = ["track_display"]
    if "genre" in mood_filtered_df.columns:
        available_columns.append("genre")
    if "artist_name" in mood_filtered_df.columns:
        available_columns.append("artist_name")

    # Function to get Spotify track link
    def get_spotify_link(track_display):
        query = track_display.replace("-", " ")
        result = sp.search(query, type="track", limit=1)
        if result["tracks"]["items"]:
            return result["tracks"]["items"][0]["external_urls"]["spotify"]
        return None

    # Function to generate Spotify play button as HTML
    def get_spotify_link_html(row):
        link = get_spotify_link(row["track_display"])
        return f'<a href="{link}" target="_blank">Listen on Spotify</a>' if link else "No Link"

    # Add play button column to mood-based recommendations
    mood_filtered_df["Play Song 🎵"] = mood_filtered_df.apply(get_spotify_link_html, axis=1)

    # Ensure varied genres appear
    if "genre" in mood_filtered_df.columns:
        genre_counts = mood_filtered_df["genre"].value_counts()
        if len(mood_filtered_df) > 0 and genre_counts.max() / len(mood_filtered_df) > 0.75:
            st.warning("Most songs belong to one genre. Try modifying mood filtering.")

    # Style table to expand Genre column width
    table_html = mood_filtered_df.to_html(escape=False, index=False)

    st.markdown(
        """
        <style>
            table { width: 100%; }
            th, td { padding: 10px; text-align: left; }
            td:nth-child(2) { min-width: 250px; } /* Expands Genre column */
        </style>
        """ + table_html,
        unsafe_allow_html=True
    )

def show_visualization_trends():
    st.subheader("🎨 Visualization & Trends")

    # Check if UMAP data exists
    if "UMAP1" in df.columns and "UMAP2" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(df["UMAP1"], df["UMAP2"], c=df["Cluster"], cmap="viridis", alpha=0.7)
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        st.pyplot(fig)
    else:
        st.warning("UMAP information is not available. Please check the dataset.")

    # Add an interactive genre distribution plot
    if "genre" in df.columns:
        st.subheader("🎶 Genre Distribution")
        genre_counts = df["genre"].value_counts().reset_index()
        genre_counts.columns = ["Genre", "Count"]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=genre_counts.head(10), x="Count", y="Genre", ax=ax, palette="coolwarm")
        ax.set_xlabel("Number of Tracks")
        ax.set_ylabel("Genre")
        st.pyplot(fig)
    else:
        st.warning("Genre data not found in the dataset.")




st.sidebar.title("Navigation")
tabs = ["Home", "Similar Tracks", "Personalized Recommendations", "Visualization & Trends"]
tab = st.sidebar.radio("Select a tab:", tabs)

if tab == "Home":
    show_home()
elif tab == "Similar Tracks":
    show_similar_tracks()
elif tab == "Personalized Recommendations":
    show_personalized_recommendations()
elif tab == "Visualization & Trends":
    show_visualization_trends()

