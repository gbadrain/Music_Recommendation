# Backend Analysis of the 'Music Recommendation System'

## Overview

The backend of Music Recommendation System is designed to efficiently process, analyze, and recommend songs based on audio features, clustering, and similarity computations. It integrates data preprocessing, dimensionality reduction, clustering, and recommendation logic to provide accurate and personalized music suggestions. This analysis covers the key components of the backend, including data handling, feature engineering, clustering, similarity computations, and data storage.

## 1. Data Preprocessing & Feature Engineering

### Text Normalization & Cleaning

To ensure consistency in song and artist names, the system applies text normalization and cleaning functions:

- **Unicode Normalization (normalize_text)**: Removes inconsistencies in text encoding.
- **Text Cleaning (clean_text)**: Standardizes formatting by removing punctuation and converting text to lowercase.
- **Full Cleaning (full_clean_text)**: Combines both functions for complete normalization.

These functions are applied to the dataset to prevent mismatches during song selection and recommendation.

### Feature Scaling

The system uses StandardScaler to normalize numerical features such as:
- Danceability, Energy, Valence, Tempo, Speechiness
- Instrumentalness, Acousticness, Loudness

Scaling ensures that all features are on a uniform scale, preventing bias in similarity computations.

### Feature Selection
<img width="1172" alt="Screenshot 2025-06-06 at 1 02 31 PM" src="https://github.com/user-attachments/assets/45223010-ecce-4fe2-a47d-76627caa3d79" />


The system selects important features for clustering and recommendation:
- **Audio Features**: Danceability, Energy, Tempo, Acousticness, Instrumentalness, Valence, Loudness, Speechiness.
- **Computed Features**: Song Length, Key.
- **Categorical Encoding**: Artist and Genre labels are converted into numerical values using LabelEncoder.
  
### Classification Report For Random Forest model
<img width="488" alt="Screenshot 2025-06-06 at 1 03 32 PM" src="https://github.com/user-attachments/assets/e620d91b-3531-4261-8480-cceaa1dfebab" />


## 2. Dimensionality Reduction Using PCA & UMAP

### Principal Component Analysis (PCA)

To reduce the high-dimensional feature space, the system applies PCA (Principal Component Analysis):
- Extracts 7 principal components (PC1-PC7) while retaining most of the variance.
- Improves computational efficiency by reducing redundant information.
- Facilitates clustering and visualization of song relationships.
<img width="1504" alt="Screenshot 2025-06-06 at 1 05 14 PM" src="https://github.com/user-attachments/assets/d68c49e0-dfd2-48d8-a2f7-10336fbafc6e" />

The explained variance ratio ensures that PCA retains sufficient information for accurate recommendations.

### UMAP Embeddings

UMAP (Uniform Manifold Approximation and Projection) is used to:
- Generate low-dimensional embeddings for similarity computations.
- Improve clustering quality by grouping similar songs.
- Provide an alternative similarity measure beyond cosine similarity.

UMAP embeddings are stored for faster retrieval and visualization.

## 3. Clustering & Evaluation

### K-Means Clustering

The system applies K-Means clustering to group songs based on their audio features.
- Clusters songs into meaningful groups for better recommendations.
- Uses PCA-reduced features (PC1-PC7) for clustering.
- Ensures well-separated clusters using Davies-Bouldin Index (DBI).

### Clustering Evaluation Using Davies-Bouldin Index (DBI)

To determine the optimal number of clusters (K), the system evaluates DBI scores for different values of K:
- Lower DBI values indicate better clustering quality.
- Optimal K is selected based on the lowest DBI score.
- Cluster labels are stored for filtering recommendations.

## 4. Song Matching & Recommendation Logic

### Exact & Fuzzy Matching for Song Selection

The system applies exact and fuzzy matching to find songs:

1. **Exact Match on Song & Artist**
- If both song and artist match exactly, return results immediately.

2. **Exact Match on Song Only**
- If no artist match is found, search by song name only.

3. **Fuzzy Matching Using difflib.get_close_matches()**
- If no exact match is found, retrieve closest matches based on similarity score (cutoff=0.9).

### Cosine Similarity-Based Recommendations

Once a song is selected, the system computes cosine similarity between its features and all other songs:
- Uses numerical features (danceability, energy, valence, tempo, speechiness).
- Normalizes similarity scores to range [0, 1] for better ranking.
- Filters recommendations based on similarity threshold.

### UMAP-Based Recommendations

UMAP embeddings provide an alternative similarity measure:
- Finds nearest neighbors in UMAP space.
- Sorts similarity scores to retrieve top matches.
- Allows hybrid recommendations combining UMAP and cosine similarity.
<img width="1542" alt="Screenshot 2025-06-06 at 1 05 33 PM" src="https://github.com/user-attachments/assets/eb3a4604-7439-4d2d-ad18-f22db7cd356b" />

## 5. Data Storage & Export

### Why Save Processed Data?

To ensure fast retrieval, reproducibility, and integration, the system saves processed data:
- **PCA Results (pca_results.csv)**: Stores dimensionality-reduced features.
- **Scaled Features (scaled_features.csv)**: Saves normalized numerical features.
- **Clustered Data (original_with_clusters.csv)**: Preserves cluster assignments.
- **UMAP Embeddings (umap_results.csv)**: Stores low-dimensional song representations.
- **UMAP-Based Recommendations (recommended_songs_with_similarity_umap.csv)**: Saves precomputed song recommendations.

### Dashboard Integration

- Loads precomputed recommendations instead of recalculating similarity scores.
- Uses PCA and UMAP embeddings for interactive song exploration.
- Allows filtering recommendations based on clusters or similarity scores.

## 6. Final Recommendations for Optimization

- Tune UMAP Parameters (n_neighbors, min_dist) to improve similarity granularity.
- Integrate Cluster-Based Filtering for more refined recommendations.
- Allow dynamic similarity threshold adjustments in the dashboard.
- Combine UMAP and cosine similarity for hybrid recommendations.
- Enable interactive PCA and UMAP visualizations in the dashboard.

## 7. Implementation Challenges & Solutions

### Dashboard-Backend Consistency Issues

**Challenge**: Discrepancies observed between recommendations generated by the dashboard interface and direct execution of `data_utils.py`, resulting in different song suggestions for identical inputs.

**Root Causes Identified**:
- **Code Path Variations**: Dashboard implementation may use different similarity computation methods compared to the centralized recommendation function
- **Input Processing Differences**: Text normalization and preprocessing inconsistencies between dashboard and backend utilities
- **Random Sampling Variability**: Different random states in dataset sampling affecting available tracks for recommendation
- **Sorting Logic Disparities**: Variations in handling cosine similarity ties and threshold applications

**Solutions Implemented**:
- Unified function calls ensuring both dashboard and backend use identical recommendation logic
- Standardized preprocessing pipeline with consistent text cleaning and normalization
- Fixed random seeds for reproducible sampling across environments
- Centralized similarity computation through `data_utils.py` functions

### UMAP Integration Challenges

**Challenge**: UMAP similarity scores consistently returning 1.0, indicating potential parameter optimization issues and lack of recommendation granularity.

**Technical Analysis**:
- UMAP parameters (`n_neighbors`, `min_dist`) require fine-tuning for optimal embedding quality
- High-dimensional feature space may need better dimensionality reduction before UMAP application
- Distance metrics in UMAP space may need recalibration for music feature similarity

### Multi-Module Architecture Coordination

**Challenge**: Ensuring seamless integration between `data_utils.py`, `model_script.py`, and dashboard components while maintaining consistency across preprocessing, modeling, and recommendation workflows.

**Architecture Considerations**:
- **data_utils.py**: Serves as the central utility library for data loading, preprocessing, and similarity computations
- **model_script.py**: Handles machine learning workflows including clustering, UMAP embeddings, and model training
- **Dashboard Integration**: Requires coordination between preprocessed data and model outputs for comprehensive recommendations

**Integration Strategy**:
- Standardized data pipeline ensuring all modules use consistent preprocessing functions
- Model artifacts (cluster labels, UMAP coordinates) integrated into main dataset for dashboard access
- Unified feature engineering approach across training and inference workflows

## Summary

The 'Music Recommendation System' efficiently integrates data preprocessing, dimensionality reduction, clustering, and similarity computations to provide personalized song recommendations. Despite implementation challenges related to dashboard consistency and UMAP optimization, the system demonstrates robust architecture with effective solutions for multi-module coordination. By addressing these challenges and optimizing similarity measures, refining clustering, and enhancing dashboard interactivity, you can further improve recommendation accuracy and user experience.
