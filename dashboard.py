import os
import time
import re
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
from PIL import Image
# --- New imports for advanced visualizations ---
import plotly.express as px
from sklearn.cluster import KMeans

# ---------------------------
# Configuration
# ---------------------------
# Set paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DATA_PATH = os.path.join(SCRIPT_DIR, "Resource", "SpotifyFeatures.csv")
ADDITIONAL_DATA_PATH = os.path.join(SCRIPT_DIR, "Output", "umap_results.csv")
IMAGE_DIR = os.path.join(SCRIPT_DIR, "Images")
SLIDES_DIR = os.path.join(SCRIPT_DIR, "Slides")

# ---------------------------
# Load Environment Variables (Secure API Credentials)
# ---------------------------
load_dotenv()
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Initialize Spotify client with error handling
try:
    sp = spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        )
    )
except Exception as e:
    st.error(f"Failed to initialize Spotify client: {str(e)}")
    st.stop()

# ---------------------------
# Data Loading & Preprocessing
# ---------------------------
@st.cache_data
def load_data(main_path, additional_path):
    try:
        df, scaler = load_and_preprocess_data(main_path, additional_filepath=additional_path)
        df = df.reset_index(drop=True)
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        return df
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        st.stop()

try:
    df = load_data(MAIN_DATA_PATH, ADDITIONAL_DATA_PATH)
except Exception as e:
    st.error(f"Data loading failed: {str(e)}")
    st.stop()

# ---------------------------
# Helper Functions
# ---------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_spotify_link(track_display):
    query = track_display.replace("-", " ")
    max_attempts = 3
    attempt = 1
    delay = 2  # start with a 2-second delay
    
    while attempt <= max_attempts:
        try:
            result = sp.search(query, type="track", limit=1)
            time.sleep(delay)
            if result["tracks"]["items"]:
                return result["tracks"]["items"][0]["external_urls"]["spotify"]
            return None
        except Exception as e:
            st.warning(f"Spotify API error (attempt {attempt}): {str(e)}")
            attempt += 1
            delay *= 2  # exponential backoff
    return None

def get_spotify_link_html(row):
    link = get_spotify_link(row["track_display"])
    return f'<a href="{link}" target="_blank">Listen on Spotify</a>' if link else "No Link"

# ---------------------------
# Dashboard Tab Functions
# ---------------------------
def show_home():
    # Custom CSS for styling
    st.markdown("""
    <style>
        .welcome-header {
            color: #1DB954;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .welcome-text {
            color: #FFFFFF;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        .banner-container {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            margin-bottom: 2rem;
        }
        .feature-card {
            background: #282828;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        .feature-title {
            color: #1DB954;
            font-size: 1.3rem;
            margin-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header Section
    st.markdown('<div class="welcome-header">MUSIC RECOMMENDATION DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="welcome-text">Welcome! Discover new music, explore genres, and generate playlists using the navigation sidebar.</div>', 
                unsafe_allow_html=True)

    # Banner Section with Error Handling
    banner_path = "/Users/GURU/Desktop/Music_Recommendation/Images/birds-7717268_640.png"
    banner_container = st.container()
    
    with banner_container:
        st.markdown('<div class="banner-container">', unsafe_allow_html=True)
        
        if os.path.exists(banner_path):
            try:
                st.image(banner_path, 
                        use_container_width=True,
                        caption="Discover Music with Gurpreet Singh Badrain",)
            except Exception as e:
                st.warning(f"Error loading banner: {str(e)}")
                show_placeholder_banner()
        else:
            st.warning("Custom banner image not found at specified path")
            show_placeholder_banner()
            
        st.markdown('</div>', unsafe_allow_html=True)

    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🎵 Discover Music</div>
            <div style="color: #B3B3B3;">
                Find similar tracks based on your favorite songs using Spotify's audio analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📊 Genre Explorer</div>
            <div style="color: #B3B3B3;">
                Visualize music characteristics across 100+ genres and sub-genres.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🎨 Audio Features</div>
            <div style="color: #B3B3B3;">
                Explore how danceability, energy, valence and acousticness define your music taste.
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_placeholder_banner():
    """Display a music-themed placeholder banner"""
    st.image(
        "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80",
        use_container_width=True,
        caption="Music connects us all"
    )

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
        
    except Exception as e:
        st.error(f"Error in generating recommendations: {str(e)}")

# ---------------------------
# Visualization Functions
# ---------------------------
# Function to show audio feature distribution using violin plots
def show_feature_density_violin(df):
    st.header("Audio Feature Distribution by Genre")
    st.markdown("Inspect how audio features vary across genres using density-based violin plots.")

    features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness']
    features = [f for f in features if f in df.columns]
    
    selected_feature = st.selectbox("Choose an audio feature:", features)
    
    top_genres = df['genre'].value_counts().nlargest(10).index.tolist()
    filtered_df = df[df['genre'].isin(top_genres)]

    fig = px.violin(filtered_df, y=selected_feature, x='genre', color='genre', box=True, points='all',
                    color_discrete_sequence=px.colors.qualitative.Set3)
    
    fig.update_layout(height=600, xaxis_title="Genre", yaxis_title=selected_feature.capitalize())
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Genre Constellation Visualization
# ---------------------------
def show_genre_constellation(df):
    st.header("The Genre Constellation")
    st.markdown("""
    Explore your music collection as a 3D constellation. Songs with similar audio features cluster together in space.
    """)

    # Check for UMAP columns
    required_cols = ['UMAP1', 'UMAP2', 'UMAP3']
    if not all(col in df.columns for col in required_cols):
        st.error("Error: Required UMAP dimensions ('UMAP1', 'UMAP2', 'UMAP3') are missing. Please ensure UMAP preprocessing was done.")
        return

    # Confirm 'genre' exists
    if 'genre' not in df.columns:
        st.error("Error: Column 'genre' is required for coloring. Please check your dataset.")
        return

    # Sidebar genre filtering
    top_genres = df['genre'].value_counts().nlargest(15).index.tolist()
    selected_genres = st.sidebar.multiselect("Select genres to display (max 15):", sorted(df['genre'].unique()), default=top_genres)

    if not selected_genres:
        st.warning("Please select at least one genre to visualize.")
        return

    # Filter dataset
    filtered_df = df[df['genre'].isin(selected_genres)]

    fig = px.scatter_3d(
        filtered_df,
        x='UMAP1', y='UMAP2', z='UMAP3',
        color='genre',
        hover_name='track_display',
        hover_data={'genre': True, 'popularity': True, 'danceability': ':.2f'},
        color_discrete_sequence=px.colors.qualitative.Vivid
    )

    fig.update_layout(
        title='3D Genre-Based Music Constellation',
        margin=dict(l=0, r=0, b=0, t=40),
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Genre Signature Visualization
# ---------------------------

def show_genre_signature(df):
    st.header("The Genre Signature")
    st.markdown("""
    Deconstruct and compare the audio 'signature' of different genres. 
    """)

    #  Sidebar for genre selection
    all_genres = sorted(df['genre'].unique())
    selected_genres = st.sidebar.multiselect("Select genres:", all_genres, default=all_genres[:4])

    # Feature selection dropdown
    all_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'speechiness']
    available_features = [f for f in all_features if f in df.columns]
    selected_features = st.sidebar.multiselect("Select features:", available_features, default=available_features[:4])

    #  Dynamic color scaling selection
    color_options = available_features
    selected_color = st.sidebar.selectbox("Select color scale:", color_options, index=color_options.index("valence") if "valence" in color_options else 0)

    if not selected_genres or not selected_features:
        st.warning("Please select at least one genre and one feature.")
        return

    # Filter dataset based on selected genres
    df_filtered = df[df['genre'].isin(selected_genres)]
    if len(df_filtered) > 2000:
        df_filtered = df_filtered.sample(2000, random_state=42)

    #  Automatically select top songs based on color scale feature
    top_songs = df_filtered.sort_values(by=selected_color, ascending=False).head(5)

    #  Create interactive parallel coordinates plot
    fig = px.parallel_coordinates(
        df_filtered, dimensions=selected_features, color=selected_color,
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={feature: feature.capitalize() for feature in selected_features},
        title="Audio Feature Comparison Across Selected Genres"
    )

    # Display plot
    st.plotly_chart(fig, use_container_width=True)

    # Display selected songs dynamically
    st.subheader("Top Songs Based on Selected Criteria")
    for _, row in top_songs.iterrows():
        st.markdown(f"🎵 **{row['track_display']}** - {row['artist_name_x']} ({row['genre']})")
        st.markdown(f"🔹 **{selected_color.capitalize()}:** {row[selected_color]:.2f}")
        st.markdown(f"[Listen on Spotify](https://open.spotify.com/track/{row['track_id']}) 🎧")
        st.write("---")



# ---------------------------
# Visualization Trends Tab
# ---------------------------

def show_visualization_trends():
    st.title("Advanced Music Visualizations")
    st.markdown("""
    Use these interactive plots to explore your music collection in rich detail—grouped by genre, mood, and audio profiles. If you want feature, refer to Tab ' For Back-end Geeks'.
    """)

    plot_types = {
        "3D UMAP Constellation ": show_genre_constellation,
        "Genre Audio Signature ": show_genre_signature,
        "Audio Feature Distribution Map ": show_feature_density_violin
    }

    selected_plot = st.sidebar.radio("Choose Visualization:", list(plot_types.keys()))

    # Call selected function
    plot_func = plot_types[selected_plot]
    plot_func(df)



# ---------------------------
# Slideshow Function
# ---------------------------
def show_slideshow():
    # Custom CSS for full-screen images
    st.markdown("""
    <style>
        [data-testid="stImage"] {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            height: 100vh !important; /* Full viewport height */
            width: 100vw !important; /* Full viewport width */
        }
        
        [data-testid="stImage"] img {
            max-width: 100vw !important; /* Ensure it fills width */
            max-height: 100vh !important; /* Ensure it fills height */
            object-fit: contain !important; /* Maintain aspect ratio */
        }
    </style>
""", unsafe_allow_html=True)

    # Load images
    slides_dir = "/Users/GURU/Desktop/Music_Recommendation/Slides/"
    try:
        slides = sorted(
            [f for f in os.listdir(slides_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
        )
    except Exception:
        st.error("Image loading failed. Check:")
        st.write("1. Folder exists at:", slides_dir)
        st.write("2. Images are named like: 1.jpg, 2.jpg, etc.")
        return

    # Initialize session state
    if 'slide_index' not in st.session_state:
        st.session_state.slide_index = 0

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        st.button("◄", on_click=lambda: st.session_state.update(slide_index=max(0, st.session_state.slide_index-1)), 
                  disabled=(st.session_state.slide_index == 0))
    
    with col3:
        st.button("►", on_click=lambda: st.session_state.update(slide_index=min(len(slides)-1, st.session_state.slide_index+1)),
                  disabled=(st.session_state.slide_index == len(slides)-1))

    # Display image (modern Streamlit approach)
    with col2:
        if slides:
            current_slide = os.path.join(slides_dir, slides[st.session_state.slide_index])
            st.image(current_slide, width=1200, output_format="PNG")




# ---------------------------
# Backend Geeks Tab
# ---------------------------

def show_backend_geeks_tab():
    """Displays detailed track information and finds similar tracks using cosine similarity."""
    try:
        st.header("For Back-end Geeks")
        st.markdown("Detailed track information including danceability, valence, track ID, genre, artist name, and other features.")

        # Ensure the dataframe is not empty
        if df.empty:
            st.warning("No track data available.")
            return

        # Select relevant columns
        columns_to_display = ["track_display", "track_id", "artist_name_x", "genre", "danceability", "energy", "valence", "acousticness"]
        if not all(col in df.columns for col in columns_to_display):
            raise ValueError("Some required columns are missing in the dataset.")

        # Allow user to select a track or artist
        track_options = sorted(df["track_display"].dropna().unique())
        selected_display = st.selectbox("Choose a track or an artist:", track_options)

        # Filter data based on selection
        mask = (df["track_display"].str.lower().str.strip() == selected_display.lower().strip())
        track_features = df.loc[mask, df.select_dtypes(include=["float64", "int64"]).columns]

        if track_features.empty:
            st.warning("Track features unavailable for similarity scoring.")
            return

        # Compute similarity scores
        similarity_scores = cosine_similarity(track_features, df.select_dtypes(include=["float64", "int64"]))
        sorted_indices = np.argsort(-similarity_scores[0])
        sorted_indices = sorted_indices[sorted_indices < len(df)]

        similar_tracks = df.iloc[sorted_indices].copy().reset_index(drop=True)
        similar_tracks = similar_tracks[similar_tracks["track_display"] != selected_display]

        # Filter by cluster if available
        if "Cluster" in df.columns or "cluster" in df.columns:
            cluster_col = "Cluster" if "Cluster" in df.columns else "cluster"
            selected_cluster = df.loc[mask, cluster_col].iloc[0]
            similar_tracks = similar_tracks[similar_tracks[cluster_col] == selected_cluster]

        #  Removed similarity score slider but kept similarity-based filtering
        similar_tracks["Similarity Score"] = np.round(similarity_scores[0][similar_tracks.index], 4)

        # Slider for number of recommendations
        num_recs = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)
        similar_tracks = similar_tracks.head(num_recs)

        # Add Spotify links
        similar_tracks["Listen on Spotify"] = similar_tracks.apply(get_spotify_link_html, axis=1)

        # Ensure genre column exists
        if "genre" not in similar_tracks.columns:
            similar_tracks["genre"] = "N/A"

        # Show all relevant columns
        display_df = similar_tracks[columns_to_display + ["Listen on Spotify", "Similarity Score"]].copy()

        # Rename columns for better readability
        display_df.rename(
            columns={
                "track_display": "TRACK NAME - ARTIST NAME",
                "track_id": "TRACK ID",
                "artist_name_x": "ARTIST NAME",
                "genre": "GENRE",
                "danceability": "DANCEABILITY",
                "energy": "ENERGY",
                "valence": "VALENCE",
                "acousticness": "ACOUSTICNESS",
                "Listen on Spotify": "PLAY SONG 🎵",
                "Similarity Score": "SIMILARITY SCORE"
            },
            inplace=True,
        )

        # Display recommendations
        st.subheader("Recommended Tracks (Filtered by Similarity Score)")
        st.dataframe(display_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error in generating recommendations: {str(e)}")





# ---------------------------
# Main function to handle navigation and display content
# ---------------------------

def main():
    """Main function to handle navigation."""
    st.sidebar.title("Navigation")
    tabs = ["Home", "Slideshow", "Similar Tracks", "Visualizations", "For Back-end Geeks"]  # Added new tab
    
    if 'tab' not in st.session_state:
        st.session_state.tab = tabs[0]
    
    st.session_state.tab = st.sidebar.radio("Go to:", tabs)
    
    if st.session_state.tab == "Home":
        show_home()
    elif st.session_state.tab == "Slideshow":
        show_slideshow()
    elif st.session_state.tab == "Similar Tracks":
        show_similar_tracks()
    elif st.session_state.tab == "Visualizations":
        show_visualization_trends()
    elif st.session_state.tab == "For Back-end Geeks":  # New tab
        show_backend_geeks_tab()

if __name__ == "__main__":
    main()
    
    