import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def full_clean_text(text):
    """Basic text cleaning function: lowercases and removes extra spaces."""
    return text.strip().lower()

def load_and_preprocess_data(data_path, additional_filepath=None, sample_size=None):
    """
    Loads and preprocesses the Spotify dataset, merging additional features if provided.
    """
    # Load main dataset
    df = pd.read_csv(data_path, encoding="utf-8")

    # Optionally sample dataset
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Clean track and artist names
    df["track_name"] = df["track_name"].astype(str).apply(full_clean_text)
    df["artist_name"] = df["artist_name"].astype(str).apply(full_clean_text)
    df["track_display"] = df["track_name"] + " - " + df["artist_name"]

    # Encode 'mode' (major/minor)
    if "mode" in df.columns:
        df["mode"] = df["mode"].astype(str).str.strip().str.lower().map({"major": 1, "minor": 0})

    # Encode artist names
    artist_encoder = LabelEncoder()
    df["artist_encoded"] = artist_encoder.fit_transform(df["artist_name"])

    # Map musical keys to numeric values
    key_mapping = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
                   "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
    if "key" in df.columns:
        df["key"] = df["key"].astype(str).str.strip().str.upper().map(key_mapping).fillna(-1)

    # Process 'time_signature'
    if "time_signature" in df.columns:
        df["time_signature"] = df["time_signature"].astype(str).str.extract(r'(\d+)').astype(float)
        df["time_signature"] = df["time_signature"].fillna(df["time_signature"].mode()[0])

    # Compute 'song_length'
    if "tempo" in df.columns and "time_signature" in df.columns:
        df["song_length"] = df["tempo"] * df["time_signature"]

    # Encode genre
    if "genre" in df.columns:
        df["genre"] = df["genre"].replace("Children’s Music", "Children's Music")
        df = df[df["genre"] != "A Capella"]
        genre_encoder = LabelEncoder()
        df["genre_encoded"] = genre_encoder.fit_transform(df["genre"])
        df["genre_encoded"] = df["genre_encoded"] - df["genre_encoded"].min()

    # Merge additional features if provided
    if additional_filepath:
        additional_df = pd.read_csv(additional_filepath, encoding="utf-8")

        # Clean column names
        additional_df.columns = additional_df.columns.str.strip()

        # Ensure 'track_display' exists
        if "track_display" not in additional_df.columns:
            if "track_name" in additional_df.columns and "artist_name" in additional_df.columns:
                additional_df["track_name"] = additional_df["track_name"].astype(str).apply(full_clean_text)
                additional_df["artist_name"] = additional_df["artist_name"].astype(str).apply(full_clean_text)
                additional_df["track_display"] = additional_df["track_name"] + " - " + additional_df["artist_name"]
            else:
                raise ValueError("The additional file must contain 'track_name' and 'artist_name' columns.")

        # Merge datasets on 'track_display'
        df = pd.merge(df, additional_df, on="track_display", how="left")

    # Scale numeric columns
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    scaler = StandardScaler()
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, scaler

# ================================================================
# UMAP Embedding and Similarity Functions
# ================================================================
def compute_umap(df, features, n_components=6, random_state=42, metric="cosine"):
    """
    Computes a UMAP embedding using the specified features.
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
    """
    similarity_scores = similarity_matrix[track_index]
    similar_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]
    recommended_songs = df.iloc[similar_indices][["track_name", "artist_name", "genre"]].copy()
    recommended_songs["Similarity Score"] = similarity_scores[similar_indices].round(2)
    return recommended_songs

# ================================================================
# Recommendation Function Using Cosine Similarity
# ================================================================
def recommend_song(input_song, cutoff=0.995, similarity_threshold=0.5, pre_matched=False, df=None,
                   features=["danceability", "energy", "valence", "tempo", "speechiness"]):
    """
    Recommends songs similar to the input song based on cosine similarity computed on a fixed set of numeric features.
    """
    if df is None:
        raise ValueError("DataFrame 'df' must be provided for recommendations.")
    
    # Identify the song to use for recommendations.
    if not pre_matched:
        matched_song = df[df["track_name"].str.lower().str.strip() == input_song.lower().strip()]["track_name"].values
        if len(matched_song) == 0:
            return f"Song '{input_song}' not found. Try adjusting filters or verifying spelling."
        matched_song = matched_song[0]
    else:
        matched_song = input_song
    
    # Retrieve numeric features for the matched song.
    matched_idx = df[df["track_name"] == matched_song].index
    if len(matched_idx) == 0:
        return f"Matched song '{matched_song}' not found in dataset."
    
    song_features = df.loc[matched_idx[0], features].values.reshape(1, -1)
    valid_rows = df.dropna(subset=features).index
    df_valid = df.loc[valid_rows, features]
    
    similarities = cosine_similarity(song_features, df_valid)
    similarity_scores = similarities[0] / similarities[0].max()  # Normalize scores
    
    df_filtered = df.loc[valid_rows].copy()
    df_filtered["Similarity"] = similarity_scores
    input_artist = df_filtered.loc[df_filtered["track_name"] == matched_song, "artist_name"].values[0]
    
    dynamic_threshold = max(similarity_threshold, df_filtered["Similarity"].mean())
    
    recommendations = df_filtered[
        (df_filtered["artist_name"] != input_artist) & (df_filtered["Similarity"] >= dynamic_threshold)
    ].drop_duplicates(subset=["track_name", "artist_name"])
    
    recommendations = recommendations.sort_values(by=["Similarity", "danceability", "energy"],
                                                  ascending=[False, False, False]).head(10)
    
    return recommendations[["track_name", "artist_name"]]
