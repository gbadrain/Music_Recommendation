import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Import centralized data loading function
from data_utils import load_and_preprocess_data

# --- Load and Preprocess Dataset ---
df, scaler = load_and_preprocess_data("Resource/SpotifyFeatures.csv")

# --- Additional Preprocessing for Modeling ---
# Encode Categorical Features
df["mode"] = df["mode"].map({"major": 1, "minor": 0})
artist_encoder = LabelEncoder()
df["artist_encoded"] = artist_encoder.fit_transform(df["artist_name"])

# Define important features for recommendation
important_features = ["danceability", "energy", "tempo", "acousticness",
                      "instrumentalness", "valence", "loudness", "speechiness",
                      "artist_encoded"]

# --- Scale Features for Modeling (if needed, here you can reuse or refit a scaler) ---
scaler_model = StandardScaler()
X_scaled_df = pd.DataFrame(scaler_model.fit_transform(df[important_features]), columns=important_features)

# --- Compute UMAP Embeddings ---
umap_model = umap.UMAP(n_components=6, random_state=42, metric="cosine")
umap_embedding = umap_model.fit_transform(X_scaled_df)

# Create DataFrame for UMAP results
umap_columns = [f"UMAP{i+1}" for i in range(umap_embedding.shape[1])]
umap_df = pd.DataFrame(umap_embedding, columns=umap_columns)
umap_df["track_name"] = df["track_name"]
umap_df["artist_name"] = df["artist_name"]
umap_df["Cluster"] = df["genre"].astype('category').cat.codes  # clusters based on genre

# Compute similarity matrix using UMAP embeddings
similarity_matrix_umap = cosine_similarity(umap_embedding)

# Function to retrieve similar tracks using UMAP embeddings
def get_similar_tracks_umap(track_index, top_n=5, within_cluster=True):
    if track_index >= len(umap_df):
        raise ValueError("Track index out of range.")
    
    track_cluster = umap_df.iloc[track_index]["Cluster"]
    if within_cluster:
        cluster_indices = umap_df[umap_df["Cluster"] == track_cluster].index.to_numpy()
    else:
        cluster_indices = umap_df.index.to_numpy()

    if len(cluster_indices) < top_n:
        top_n = len(cluster_indices) - 1

    relative_index = np.where(cluster_indices == track_index)[0]
    if len(relative_index) == 0:
        raise ValueError("Track not found in filtered cluster.")

    similarity_matrix_cluster = cosine_similarity(umap_embedding[cluster_indices])
    similarity_scores = similarity_matrix_cluster[relative_index[0]]
    similar_indices = cluster_indices[np.argsort(similarity_scores)[::-1][1:top_n+1]]
    
    recommended_songs = umap_df.iloc[similar_indices][["track_name", "artist_name"]].copy()
    recommended_songs["Similarity Score"] = similarity_scores[np.argsort(similarity_scores)[::-1][1:top_n+1]].round(2)
    return recommended_songs

# Save computed UMAP results & recommendations
umap_df.to_csv("Output/umap_results.csv", index=False)

def save_recommendations(track_index):
    recommended_songs_umap = get_similar_tracks_umap(track_index)
    if isinstance(recommended_songs_umap, pd.DataFrame) and not recommended_songs_umap.empty:
        recommended_songs_umap.to_csv("Output/recommended_songs_with_similarity_umap.csv", index=False)
        print(f"Saved recommendations for track index {track_index}!")
    else:
        print(f"No recommendations found for track index {track_index}.")
