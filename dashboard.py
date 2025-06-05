import os
import numpy as np

# ---- Handle Imports Gracefully ----
try:
    import streamlit as st
    import pandas as pd
    import umap
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    st.error(f"Missing dependency: {e}. Try installing required packages.")

# ---- Check and Fix NumPy Compatibility Issues ----
if np.__version__.startswith("2"):
    st.warning("NumPy 2.x detected, which may cause compatibility issues. Consider downgrading with: pip install numpy<2.")

# ---- Load Dataset ----
DATA_PATH = "Resource/SpotifyFeatures.csv"

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
else:
    st.error(f"Dataset not found at {DATA_PATH}. Ensure the path is correct.")

# ---- Dashboard Header ----
st.title("Music Recommendation Dashboard")
st.write("Explore song similarities and discover new tracks!")

# ---- Placeholder for Recommendations ----
st.subheader("Find Similar Tracks")
if "df" in locals():
    try:
        selected_track = st.selectbox("Choose a track:", df["track_name"].dropna().unique())
        track_features = df[df["track_name"] == selected_track].select_dtypes(include=["float64", "int64"])
        
        if not track_features.empty:
            similarity_scores = cosine_similarity(track_features, df.select_dtypes(include=["float64", "int64"]))
            similar_tracks = df.iloc[np.argsort(-similarity_scores[0])[:5]]  # Top 5 recommendations
            
            st.write("Recommended Tracks:", similar_tracks[["track_name", "artist_name"]])

            # ---- Updated Dataset Preview (Only Recommended Songs) ----
            st.subheader("Dataset Preview: Recommended Songs Only")
            st.write(similar_tracks)
        else:
            st.warning("Track features unavailable for similarity scoring.")
    except Exception as e:
        st.error(f"Error in generating recommendations: {e}")

# ---- Debugging Logs ----
st.subheader("System Info")
st.write(f"NumPy Version: {np.__version__}")
st.write(f"UMAP Version: {umap.__version__}")

st.write("If there are errors, check package versions or run: `pip install -r requirements.txt`")

# ---- Run App ----
if __name__ == "__main__":
    st.success("Dashboard is running successfully!")
