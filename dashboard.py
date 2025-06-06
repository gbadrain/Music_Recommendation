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
# Data Loading & Preprocessing in Streamlit
# ---------------------------
# Use st.cache_data to load data only once
@st.cache_data
def load_data(main_path, additional_path):
    df, scaler = load_and_preprocess_data(main_path, additional_filepath=additional_path)
    df = df.reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df

MAIN_DATA_PATH = "Resource/SpotifyFeatures.csv"
ADDITIONAL_DATA_PATH = "Output/umap_results.csv"
df = load_data(MAIN_DATA_PATH, ADDITIONAL_DATA_PATH)


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

# Define cache and helper functions at the top level
global cache
if 'cache' not in globals():
    cache = {}

def get_spotify_link(track_display):
    query = track_display.replace("-", " ")
    if track_display in cache:
        return cache[track_display]
    
    max_attempts = 3
    attempt = 1
    delay = 2  # start with a 2-second delay
    
    while attempt <= max_attempts:
        try:
            result = sp.search(query, type="track", limit=1)
            time.sleep(delay)
            if result["tracks"]["items"]:
                link = result["tracks"]["items"][0]["external_urls"]["spotify"]
                cache[track_display] = link  # store in cache for efficiency
                return link
            else:
                return None
        except Exception as e:
            st.warning(f"Spotify API error (attempt {attempt}): {str(e)}")
            attempt += 1
            delay *= 2  # increase the delay exponentially
    return None

def get_spotify_link_html(row):
    link = get_spotify_link(row["track_display"])
    return f'<a href="{link}" target="_blank">Listen on Spotify</a>' if link else "No Link"

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
        st.subheader("Dataset Preview: Recommended Tracks")
        st.write(similar_tracks)
    except Exception as e:
        st.error(f"Error in generating recommendations: {e}")

# ---------------------------
# ADVANCED VISUALIZATION FUNCTIONS
# ---------------------------

def show_genre_constellation(df):
    st.header("The Genre Constellation")
    st.markdown("""
    Explore your music library as a 3D universe. Each point is a song, positioned by its UMAP coordinates. 
    Songs with similar audio features are closer together. Use your mouse to rotate, zoom, and hover over points to discover new music.
    """)

    umap_dims = [c for c in ['UMAP1', 'UMAP2', 'UMAP3'] if c in df.columns]
    if len(umap_dims) < 3:
        st.error("Error: This visualization requires at least 3 UMAP dimensions (UMAP1, UMAP2, UMAP3).")
        return

    st.sidebar.subheader("Constellation Controls")
    color_by = st.sidebar.radio("Color points by:", ["genre", "kmeans_cluster"], format_func=lambda x: "Genre" if x == "genre" else "K-Means Cluster")
    
    if color_by not in df.columns:
        if color_by == 'kmeans_cluster':
            st.warning("K-Means clusters not yet calculated. Please run clustering in the 'Sub-Genre DNA' plot first.")
            return
        st.error(f"Error: Column '{color_by}' not found.")
        return
        
    top_n_cats = df[color_by].value_counts().nlargest(15).index
    default_cats = top_n_cats.tolist()
    
    selected_cats = st.sidebar.multiselect(f"Select {color_by.replace('_', ' ')} to display:",
                                    sorted(df[color_by].unique()),
                                    default=default_cats)

    if not selected_cats:
        st.warning("Please select at least one category to display.")
        return

    filtered_df = df[df[color_by].isin(selected_cats)]

    fig = px.scatter_3d(
        filtered_df, x='UMAP1', y='UMAP2', z='UMAP3', color=color_by,
        hover_name='track_display',
        hover_data={'genre': True, 'popularity': True, 'danceability': ':.2f', 'energy': ':.2f', 'valence': ':.2f', 'UMAP1': False, 'UMAP2': False, 'UMAP3': False},
        color_discrete_sequence=px.colors.qualitative.Vivid, template='plotly_dark'
    )

    fig.update_layout(
        title=f'Interactive 3D Music Universe (Colored by {color_by.replace("_", " ").title()})',
        margin=dict(l=0, r=0, b=0, t=40), legend_title_text=color_by.replace("_", " ").title(),
        scene=dict(xaxis_title='UMAP 1', yaxis_title='UMAP 2', zaxis_title='UMAP 3')
    )
    fig.update_traces(marker=dict(size=3, opacity=0.8))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

def show_sunburst_dna(df):
    st.header("The Sub-Genre DNA")
    st.markdown("""
    This sunburst chart reveals the hidden 'DNA' of your music genres. The inner ring is the primary genre. 
    The outer ring shows the algorithmic 'moods' (K-Means clusters) that compose each genre. 
    Hover or click to explore the compositional breakdown.
    """)

    if 'kmeans_cluster' not in df.columns:
        umap_dims = [c for c in df.columns if 'UMAP' in str(c)]
        if len(umap_dims) < 2:
            st.error("Error: UMAP dimensions are required to generate K-Means clusters.")
            return
        with st.spinner("Calculating K-Means clusters..."):
            kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
            df['kmeans_cluster'] = kmeans.fit_predict(df[umap_dims])

    st.sidebar.subheader("Sunburst Controls")
    num_genres = st.sidebar.slider("Number of Top Genres to Display:", min_value=3, max_value=20, value=10)
    
    top_genres = df['genre'].value_counts().nlargest(num_genres).index
    df_filtered = df[df['genre'].isin(top_genres)]
    
    fig = px.sunburst(
        df_filtered, path=['genre', 'kmeans_cluster'], values='popularity', color='genre',
        color_discrete_sequence=px.colors.qualitative.Pastel, template='plotly_dark'
    )
    
    fig.update_layout(
        title=f'Hierarchical Breakdown of Top {num_genres} Genres into Sonic Clusters',
        margin=dict(l=10, r=10, b=10, t=50),
    )
    fig.update_traces(textinfo='label+percent parent')
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

def show_genre_signature(df):
    st.header("The Genre Signature")
    st.markdown("""
    Deconstruct and compare the audio 'signature' of different genres. Each line represents a single song, 
    charting its path across key audio features. This reveals the unique sonic fingerprint and internal diversity of each genre.
    """)

    st.sidebar.subheader("Signature Controls")
    all_genres = sorted(df['genre'].unique())
    default_genres = df['genre'].value_counts().nlargest(4).index.tolist()
    
    selected_genres = st.sidebar.multiselect("Select genres to compare:", all_genres, default=default_genres)
    dimensions = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'speechiness']
    dimensions = [d for d in dimensions if d in df.columns]

    if not selected_genres:
        st.warning("Please select at least one genre.")
        return
        
    df_filtered = df[df['genre'].isin(selected_genres)]

    if len(df_filtered) > 2000:
        st.sidebar.info(f"Displaying a random sample of 2000 songs out of {len(df_filtered)} for performance.")
        df_filtered = df_filtered.sample(2000, random_state=42)

    fig = px.parallel_coordinates(
        df_filtered, dimensions=dimensions, color="valence",
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={col: col.replace('_', ' ').title() for col in dimensions},
        title="Comparing the Audio Signatures of Genres"
    )
    fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def show_visualization_trends():
    st.title("Advanced Data Visualizations")
    st.markdown("---")
    st.markdown("Use the controls in the sidebar to customize the plots.")

    plot_choice = st.selectbox(
        "Select a Visualization:",
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
    st.subheader("I love Spotify")
    st.write("Enjoy a curated selection of slides that celebrate music!")
    image_dir = "/Users/GURU/Desktop/Music_Recommendation/Slides/"
    
    if not os.path.isdir(image_dir):
        st.error(f"Slideshow directory not found at: {image_dir}")
        return

    pattern = re.compile(r'^(\d+)\.png$', re.IGNORECASE)
    images = []
    try:
        for f in os.listdir(image_dir):
            if f.lower().endswith('.png'):
                match = pattern.match(f)
                if match:
                    num = int(match.group(1))
                    if 2 <= num <= 12:
                        images.append(os.path.join(image_dir, f))
    except FileNotFoundError:
        st.error(f"Slideshow directory not found at: {image_dir}")
        return

    if not images:
        st.warning("No slides named 2.png through 12.png found in the Slides folder.")
        return
    
    images = sorted(images, key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))
    
    if "slide_index" not in st.session_state:
        st.session_state.slide_index = 0
    
    st.image(
        images[st.session_state.slide_index],
        caption=f"Slide {st.session_state.slide_index + 1} of {len(images)}",
        use_container_width=True
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    if col1.button("◄ Previous", key="prev_btn"):
        st.session_state.slide_index = (st.session_state.slide_index - 1) % len(images)
        st.rerun()
    if col3.button("Next ►", key="next_btn"):
        st.session_state.slide_index = (st.session_state.slide_index + 1) % len(images)
        st.rerun()

# ---------------------------
# Main App Layout
# ---------------------------
st.sidebar.title("Navigation")
tabs = ["Home", "Slideshow Carousel", "Similar Tracks", "Visualization & Trends"]
# Use st.session_state to keep track of the current tab
if 'tab' not in st.session_state:
    st.session_state.tab = "Home"

st.session_state.tab = st.sidebar.radio("Select a tab:", tabs, index=tabs.index(st.session_state.tab))

# Display the selected tab content
if st.session_state.tab == "Home":
    show_home()
elif st.session_state.tab == "Slideshow Carousel":
    show_slideshow()
elif st.session_state.tab == "Similar Tracks":
    show_similar_tracks()
elif st.session_state.tab == "Visualization & Trends":
    show_visualization_trends()