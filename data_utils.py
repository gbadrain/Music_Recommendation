"""
data_utils.py

This module provides functions for loading and preprocessing your Spotify dataset,
cleaning text, computing UMAP embeddings, matching songs (through exact, substring,
and fuzzy matching), and generating song recommendations based on cosine similarity.

Functions provided:
    - normalize_text(text): Normalize Unicode text to ASCII.
    - clean_text(text): Lowercase text and remove specific punctuation.
    - full_clean_text(text): Combine normalization and cleaning.
    - load_and_preprocess_data(data_path, sample_size=None): Loads and preprocesses the dataset.
    - find_all_matches(input_song, input_artist=None, cutoff=0.7, num_matches=5, df=None):
          Finds matching songs by exact, substring, and fuzzy matching.
    - recommend_song(input_song, cutoff=0.995, similarity_threshold=0.5, pre_matched=False,
                     df=None, features=["danceability", "energy", "valence", "tempo", "speechiness"]):
          Recommends songs similar to the input based on cosine similarity on a fixed set of features.
    - compute_umap(df, features, n_components=6, random_state=42, metric="cosine"):
          Computes UMAP embedding, cosine similarity matrix, and returns a DataFrame of UMAP components.
    - get_similar_tracks_umap(track_index, similarity_matrix, df, top_n=5):
          Retrieves similar tracks from the dataset based on the UMAP embedding.
    - plot_elbow_curve(X, k_range=range(1, 11)):
          (Optional) Plots the elbow curve to help find the optimal number of clusters for KMeans.

Usage Example:
    In a notebook or dashboard, you can do the following:
    
    from data_utils import load_and_preprocess_data, recommend_song, find_all_matches
    
    # Load and preprocess the dataset
    df, scaler = load_and_preprocess_data("Resource/SpotifyFeatures.csv", sample_size=3000)
    
    # Define a unified feature list for recommendations
    selected_features = ['danceability', 'energy', 'valence', 'tempo', 'speechiness']
    print("Selected features for recommendations:", selected_features)
    
    # Get recommendations (using pre_matched=True to bypass fuzzy matching)
    recommendations = recommend_song("tell me", pre_matched=True, df=df, features=selected_features)
    print("Recommended songs:")
    print(recommendations)
"""

import pandas as pd
import numpy as np
import unicodedata
import difflib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import umap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ================================================================
# Text Normalization and Cleaning Functions
# ================================================================
def normalize_text(text):
    """
    Normalize Unicode text to ASCII using NFKD normalization.
    """
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode()

def clean_text(text):
    """
    Lowercase text, strip whitespace, and remove specific punctuation.
    """
    return (
        text.lower()
        .strip()
        .replace('"', '')
        .replace("'", "")
        .replace("’", "'")
        .replace(".", "")
        .replace(":", "")
        .strip()
    )

def full_clean_text(text):
    """
    Combine Unicode normalization and text cleaning.
    """
    return clean_text(normalize_text(text))

# ================================================================
# Data Loading and Preprocessing Function
# ================================================================
def load_and_preprocess_data(data_path, sample_size=None):
    """
    Loads and preprocesses the Spotify dataset from a CSV file.
    
    Preprocessing includes:
      - Loading the CSV.
      - Cleaning 'track_name' and 'artist_name' with full_clean_text.
      - Creating a 'track_display' column.
      - Converting 'mode' (if present) into numeric (e.g., "major" -> 1, "minor" -> 0).
      - Label-encoding 'artist_name' into 'artist_encoded'.
      - Mapping 'key' to numeric codes (missing keys -> -1).
      - Processing 'time_signature' and computing 'song_length' (tempo * time_signature).
      - Optionally sampling for faster processing.
      - Handling genre: standardizing names, filtering out unwanted genres, and encoding.
      - Scaling numeric columns using StandardScaler.
    
    Returns:
      df (pd.DataFrame): The preprocessed DataFrame.
      scaler (StandardScaler): The scaler fitted on numeric columns.
    """
    df = pd.read_csv(data_path, encoding="utf-8")
    
    # Clean text columns
    df["track_name"] = df["track_name"].astype(str).apply(full_clean_text)
    df["artist_name"] = df["artist_name"].astype(str).str.lower().str.strip()
    df["track_display"] = df["track_name"] + " - " + df["artist_name"]
    
    # Convert mode to numeric if present
    if "mode" in df.columns:
        df["mode"] = df["mode"].astype(str).str.strip().str.lower().map({"major": 1, "minor": 0})
    
    # Label encode artist names
    artist_encoder = LabelEncoder()
    df["artist_encoded"] = artist_encoder.fit_transform(df["artist_name"])
    
    # Map musical keys to numbers
    key_mapping = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
                   "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
    if "key" in df.columns:
        df["key"] = df["key"].astype(str).str.strip().str.upper().map(key_mapping).fillna(-1)
    
    # Process time_signature and fill missing values with the mode
    if "time_signature" in df.columns:
        df["time_signature"] = df["time_signature"].astype(str).str.extract(r'(\d+)').astype(float)
        ts_mode = df["time_signature"].mode()[0]
        df["time_signature"] = df["time_signature"].fillna(ts_mode)
    
    # Compute song_length as tempo * time_signature
    if "tempo" in df.columns and "time_signature" in df.columns:
        df["song_length"] = df["tempo"] * df["time_signature"]
    
    # Optionally sample the data for faster processing
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    
    # Handle genre
    if "genre" in df.columns:
        df["genre"] = df["genre"].replace("Children’s Music", "Children's Music")
        df = df[df["genre"] != "A Capella"]
        genre_encoder = LabelEncoder()
        df["genre_encoded"] = genre_encoder.fit_transform(df["genre"])
        df["genre_encoded"] = df["genre_encoded"] - df["genre_encoded"].min()
    
    # Scale numeric columns
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    scaler = StandardScaler()
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, scaler

# ================================================================
# Matching Functions for Song Lookup
# ================================================================
def find_all_matches(input_song, input_artist=None, cutoff=0.7, num_matches=5, df=None):
    """
    Finds matching songs in the dataset using:
      1. Exact matching (using fullmatch on track_name and artist_name).
      2. Substring matching (using str.contains).
      3. Fuzzy matching (using difflib.get_close_matches).
    
    Parameters:
      input_song (str): Song title to search.
      input_artist (str, optional): Artist to help refine the search.
      cutoff (float): Fuzzy matching cutoff (0 to 1), default 0.7.
      num_matches (int): Maximum number of fuzzy matches to return.
      df (pd.DataFrame): The preprocessed DataFrame.
      
    Returns:
      Tuple of two lists: (matched_songs, matched_artists).
    """
    if df is None:
        raise ValueError("DataFrame 'df' must be provided for matching.")
    
    input_song_cleaned = full_clean_text(input_song)
    
    # 1. Attempt exact match on both song and artist if provided
    if input_artist:
        input_artist_cleaned = full_clean_text(input_artist)
        exact_matches_df = df[
            (df["track_name"].str.fullmatch(input_song_cleaned, case=False, na=False)) &
            (df["artist_name"].str.fullmatch(input_artist_cleaned, case=False, na=False))
        ]
        if not exact_matches_df.empty:
            unique_matches = list({(row["track_name"], row["artist_name"]) for _, row in exact_matches_df.iterrows()})
            for song, art in unique_matches:
                print(f"Exact match found: '{song}' by {art}")
            return [song for song, _ in unique_matches], [art for _, art in unique_matches]
        else:
            print("Exact match on both song and artist not found. Searching by song name only:")
    
    # 2. Attempt exact match on song name only
    exact_matches_song_df = df[df["track_name"].str.fullmatch(input_song_cleaned, case=False, na=False)]
    if not exact_matches_song_df.empty:
        unique_matches = list({(row["track_name"], row["artist_name"]) for _, row in exact_matches_song_df.iterrows()})
        for song, art in unique_matches:
            print(f"Exact match found: '{song}' by {art}")
        return [song for song, _ in unique_matches], [art for _, art in unique_matches]
    
    # 3. If not found, attempt substring matching
    substring_matches_df = df[df["track_name"].str.contains(input_song_cleaned, case=False, na=False)]
    if not substring_matches_df.empty:
        unique_matches = list({(row["track_name"], row["artist_name"]) for _, row in substring_matches_df.iterrows()})
        for song, art in unique_matches:
            print(f"Substring match found: '{song}' by {art}")
        return [song for song, _ in unique_matches], [art for _, art in unique_matches]
    
    # 4. Fallback to fuzzy matching using difflib
    all_songs = df["track_name"].dropna().tolist()
    closest_matches = difflib.get_close_matches(input_song_cleaned, all_songs, n=num_matches, cutoff=cutoff)
    if closest_matches:
        matched = []
        print("Possible fuzzy matches:")
        for song in closest_matches:
            song_rows = df[df["track_name"].str.fullmatch(song, case=False, na=False)]
            for _, row in song_rows.iterrows():
                matched.append((row["track_name"], row["artist_name"]))
        unique_matches = []
        for pair in matched:
            if pair not in unique_matches:
                unique_matches.append(pair)
        for song, art in unique_matches:
            print(f"Fuzzy match found: '{song}' by {art}")
        return [song for song, _ in unique_matches], [art for _, art in unique_matches]
    else:
        print(f"No matches found for '{input_song}'" + (f" by '{input_artist}'." if input_artist else "."))
        return [], []

# ================================================================
# Recommendation Function Using Cosine Similarity
# ================================================================
def recommend_song(input_song, cutoff=0.995, similarity_threshold=0.5, pre_matched=False, df=None,
                   features=["danceability", "energy", "valence", "tempo", "speechiness"]):
    """
    Recommends songs similar to the input song based on cosine similarity computed on a fixed set of numeric features.
    This function should be used consistently by your notebook and dashboard.
    
    Parameters:
      input_song (str): Song title to use for recommendation.
      cutoff (float): Fuzzy matching cutoff.
      similarity_threshold (float): Minimum normalized similarity score to consider.
      pre_matched (bool): If True, use the input_song directly without matching.
      df (pd.DataFrame): Preprocessed DataFrame.
      features (list): List of numeric feature names used for cosine similarity.
      
    Returns:
      pd.DataFrame containing recommended songs (columns: 'track_name' and 'artist_name'),
      or an error message string.
    """
    if df is None:
        raise ValueError("DataFrame 'df' must be provided for recommendations.")
    
    # Identify the song to use for recommendations.
    if not pre_matched:
        matched_song, _ = find_all_matches(input_song, df=df)
        if isinstance(matched_song, list) and len(matched_song) > 0:
            matched_song = matched_song[0]
        else:
            return f"Song '{input_song}' not found. Try adjusting filters or verifying spelling."
    else:
        matched_song = input_song
    
    # Retrieve numeric features for the matched song.
    numeric_features = features
    matched_idx = df[df["track_name"] == matched_song].index
    if len(matched_idx) == 0:
        return f"Matched song '{matched_song}' not found in dataset."
    
    song_features = df.loc[matched_idx[0], numeric_features].values.reshape(1, -1)
    valid_rows = df.dropna(subset=numeric_features).index
    df_valid = df.loc[valid_rows, numeric_features]
    
    similarities = cosine_similarity(song_features, df_valid)
    similarity_scores = similarities[0] / similarities[0].max()  # Normalize scores
    
    df_filtered = df.loc[valid_rows].copy()
    df_filtered["Similarity"] = similarity_scores
    input_artist_val = df_filtered.loc[df_filtered["track_name"] == matched_song, "artist_name"].values[0]
    
    dynamic_threshold = max(similarity_threshold, df_filtered["Similarity"].mean())
    
    recommendations = df_filtered[
        (df_filtered["artist_name"] != input_artist_val) & (df_filtered["Similarity"] >= dynamic_threshold)
    ]
    recommendations = recommendations.drop_duplicates(subset=["track_name", "artist_name"])
    
    # Optionally filter by genre if available in the dataset.
    if "genre" in df.columns:
        input_genre = df.loc[df["track_name"] == matched_song, "genre"].values[0]
        recommendations = recommendations[recommendations["genre"] == input_genre]
    
    recommendations = recommendations.sort_values(
        by=["Similarity", "danceability", "energy"],
        ascending=[False, False, False]
    ).head(10)
    
    if recommendations.empty:
        return f"No recommendations found for '{matched_song}'. Try adjusting filters."
    
    return recommendations[["track_name", "artist_name"]]

# ================================================================
# UMAP Embedding and Similarity Functions
# ================================================================
def compute_umap(df, features, n_components=6, random_state=42, metric="cosine"):
    """
    Computes a UMAP embedding using the specified features.
    
    Returns:
      umap_embedding (np.array): The UMAP embedding.
      similarity_matrix (np.array): Cosine similarity matrix derived from the embedding.
      umap_df (pd.DataFrame): DataFrame with UMAP component columns (UMAP1, UMAP2, ..., UMAP{n_components}).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    umap_model = umap.UMAP(n_components=n_components, random_state=random_state, metric=metric)
    umap_embedding = umap_model.fit_transform(X_scaled)
    similarity_matrix = cosine_similarity(umap_embedding)
    umap_columns = [f"UMAP{i+1}" for i in range(n_components)]
    umap_df = pd.DataFrame(umap_embedding, columns=umap_columns)
    return umap_embedding, similarity_matrix, umap_df

def get_similar_tracks_umap(track_index, similarity_matrix, df, top_n=5):
    """
    Retrieves top_n similar tracks based on the precomputed UMAP cosine similarity matrix.
    
    Parameters:
      track_index (int): The index of the target track.
      similarity_matrix (np.array): Cosine similarity matrix obtained from UMAP embeddings.
      df (pd.DataFrame): DataFrame containing track details.
      top_n (int): Number of similar tracks to return.
      
    Returns:
      pd.DataFrame containing columns: 'track_name', 'artist_name', 'genre' and 'Similarity Score'.
    """
    similarity_scores = similarity_matrix[track_index]
    similar_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]
    recommended_songs = df.iloc[similar_indices][["track_name", "artist_name", "genre"]].copy()
    recommended_songs["Similarity Score"] = similarity_scores[similar_indices].round(2)
    return recommended_songs

# ================================================================
# Utility: Plot Elbow Curve for KMeans Clustering (Optional)
# ================================================================
def plot_elbow_curve(X, k_range=range(1, 11)):
    """
    Computes and plots the elbow curve to help determine the optimal number of clusters.
    
    Parameters:
      X (np.array or pd.DataFrame): The data for clustering.
      k_range (iterable): Range of cluster values to test.
      
    Returns:
      inertia_values (list): Inertia (sum of squared distances) for each k.
    """
    inertia_values = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia_values.append(kmeans.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertia_values, marker="o", linestyle="-", color="blue")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal Clusters")
    plt.show()
    return inertia_values

# ================================================================
# Usage Examples (Executed only when running this module directly)
# ================================================================
if __name__ == "__main__":
    # Load and preprocess data (adjust sample_size if needed)
    data_path = "Resource/SpotifyFeatures.csv"
    df, scaler = load_and_preprocess_data(data_path, sample_size=3000)
    print("Data loaded and preprocessed. Sample:")
    print(df.head())
    
    # Test matching and recommendation functions
    test_song = "TEll Me"
    test_artist = "karine costa"
    matched_songs, matched_artists = find_all_matches(test_song, test_artist, df=df)
    if matched_songs:
        chosen_song = matched_songs[0]
        print(f"\nUsing song: '{chosen_song}' by {matched_artists[0]} for recommendations.\n")
    else:
        print(f"\nNo matching songs found for '{test_song}' by '{test_artist}'.\n")
        chosen_song = test_song  # fallback
        
    # Unified feature selection for recommendations
    selected_features = ['danceability', 'energy', 'valence', 'tempo', 'speechiness']
    print("Selected features for recommendations:", selected_features)
    recommendations = recommend_song(chosen_song, pre_matched=True, df=df, features=selected_features)
    print("Recommended songs based on cosine similarity:")
    print(recommendations)
    
    # Test UMAP embedding and retrieving similar tracks
    umap_features = ["danceability", "energy", "valence", "tempo", "speechiness",
                     "acousticness", "loudness", "instrumentalness"]
    umap_embedding, similarity_matrix, umap_df = compute_umap(df, umap_features, n_components=6)
    target_index = 42  # Example track index; change as necessary
    similar_tracks = get_similar_tracks_umap(target_index, similarity_matrix, df, top_n=5)
    print("Similar tracks based on UMAP embedding:")
    print(similar_tracks)
