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
                        caption="Discover Your Next Favorite Song")
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
                Explore how danceability, energy, and valence define your music taste.
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
def show_genre_constellation(df):
    st.header("The Genre Constellation")
    st.markdown("""
    Explore your music library as a 3D universe. Each point is a song, positioned by its UMAP coordinates. 
    Songs with similar audio features are closer together.
    """)

    umap_dims = [c for c in ['UMAP1', 'UMAP2', 'UMAP3'] if c in df.columns]
    if len(umap_dims) < 3:
        st.error("Error: Requires 3 UMAP dimensions (UMAP1, UMAP2, UMAP3).")
        return

    color_by = st.sidebar.radio("Color points by:", ["genre", "kmeans_cluster"], 
                              format_func=lambda x: "Genre" if x == "genre" else "K-Means Cluster")
    
    if color_by not in df.columns:
        if color_by == 'kmeans_cluster':
            st.warning("Run clustering in 'Sub-Genre DNA' first.")
            return
        st.error(f"Column '{color_by}' not found.")
        return
        
    top_n_cats = df[color_by].value_counts().nlargest(15).index
    selected_cats = st.sidebar.multiselect(f"Select {color_by.replace('_', ' ')} to display:",
                                    sorted(df[color_by].unique()),
                                    default=top_n_cats.tolist())

    if not selected_cats:
        st.warning("Please select at least one category.")
        return

    filtered_df = df[df[color_by].isin(selected_cats)]

    fig = px.scatter_3d(
        filtered_df, x='UMAP1', y='UMAP2', z='UMAP3', color=color_by,
        hover_name='track_display',
        hover_data={'genre': True, 'popularity': True, 'danceability': ':.2f'},
        color_discrete_sequence=px.colors.qualitative.Vivid
    )

    fig.update_layout(
        title=f'3D Music Universe (Colored by {color_by.replace("_", " ").title()})',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig, use_container_width=True)

def show_sunburst_dna(df):
    st.header("The Sub-Genre DNA")
    st.markdown("""
    Reveals the hidden 'DNA' of music genres. The inner ring is the primary genre. 
    The outer ring shows algorithmic 'moods' (K-Means clusters).
    """)

    if 'kmeans_cluster' not in df.columns:
        umap_dims = [c for c in df.columns if 'UMAP' in str(c)]
        if len(umap_dims) < 2:
            st.error("Error: UMAP dimensions required.")
            return
        with st.spinner("Calculating clusters..."):
            kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
            df['kmeans_cluster'] = kmeans.fit_predict(df[umap_dims])

    num_genres = st.sidebar.slider("Number of Top Genres:", min_value=3, max_value=20, value=10)
    top_genres = df['genre'].value_counts().nlargest(num_genres).index
    df_filtered = df[df['genre'].isin(top_genres)]
    
    fig = px.sunburst(
        df_filtered, path=['genre', 'kmeans_cluster'], values='popularity', color='genre',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig, use_container_width=True)

def show_genre_signature(df):
    st.header("The Genre Signature")
    st.markdown("""
    Deconstruct and compare the audio 'signature' of different genres. 
    """)

    all_genres = sorted(df['genre'].unique())
    selected_genres = st.sidebar.multiselect("Select genres:", all_genres, 
                                           default=all_genres[:4])
    dimensions = ['danceability', 'energy', 'valence', 'acousticness']
    dimensions = [d for d in dimensions if d in df.columns]

    if not selected_genres:
        st.warning("Please select genres.")
        return
        
    df_filtered = df[df['genre'].isin(selected_genres)]
    if len(df_filtered) > 2000:
        df_filtered = df_filtered.sample(2000, random_state=42)

    fig = px.parallel_coordinates(
        df_filtered, dimensions=dimensions, color="valence",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig, use_container_width=True)

def show_visualization_trends():
    st.title("Advanced Visualizations")
    plot_choice = st.selectbox(
        "Select:",
        [
            "The Genre Constellation (3D UMAP)",
            "The Sub-Genre DNA (Sunburst Chart)",
            "The Genre Signature (Parallel Coordinates)"
        ]
    )

    if plot_choice == "The Genre Constellation (3D UMAP)":
        show_genre_constellation(df)
    elif plot_choice == "The Sub-Genre DNA (Sunburst Chart)":
        show_sunburst_dna(df)
    elif plot_choice == "The Genre Signature (Parallel Coordinates)":
        show_genre_signature(df)


# ---------------------------
# Slideshow Function
# ---------------------------
def show_slideshow():
    # Custom CSS for full-screen images
    st.markdown("""
    <style>
        [data-testid="stImage"] {
            max-height: 85vh !important;
            width: auto !important;
            margin: 0 auto !important;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        [data-testid="stImage"] img {
            object-fit: contain !important;
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
        st.image(
            current_slide,
            use_container_width=True,  # Correct modern parameter
            output_format="PNG"
        )

# ---------------------------
# Main App
# ---------------------------
def main():
    st.sidebar.title("Navigation")
    tabs = ["Home", "Slideshow", "Similar Tracks", "Visualizations"]
    
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

if __name__ == "__main__":
    main()