"""
Regression tests for data_utils.py.

Run with:  pytest tests/ -v
(from the project root /Users/.../Music_Recommendation)

These tests use a tiny in-memory CSV so they never touch the 33 MB
SpotifyFeatures.csv.  They document the expected behaviour of:
  - full_clean_text              (NaN / blank / normal strings)
  - load_and_preprocess_data     (blank rows dropped, track_display format)
  - cosine similarity scoring    (sorted descending, scores tied to right rows)
  - get_recommendations          (top-n returned, sorted, scores correct, dupes removed)
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity

# Make data_utils importable when pytest is run from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import full_clean_text, load_and_preprocess_data, get_recommendations, build_feature_matrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINI_CSV = """\
genre,track_name,artist_name,danceability,energy,valence
Pop,Song A,Artist 1,0.8,0.7,0.6
Pop,Song B,Artist 2,0.9,0.8,0.5
Pop,Song C,Artist 3,0.7,0.9,0.4
Hip-Hop,,Artist 4,0.6,0.5,0.3
Hip-Hop,Song E,Artist 5,0.5,0.4,0.2
"""


@pytest.fixture
def mini_csv(tmp_path):
    p = tmp_path / "mini.csv"
    p.write_text(MINI_CSV)
    return str(p)


# ---------------------------------------------------------------------------
# full_clean_text
# ---------------------------------------------------------------------------

def test_full_clean_text_nan():
    """NaN (float) must not crash and must return empty string."""
    assert full_clean_text(np.nan) == ""


def test_full_clean_text_blank():
    """Whitespace-only input returns empty string."""
    assert full_clean_text("   ") == ""


def test_full_clean_text_normal():
    """Normal string is lowercased and stripped of leading/trailing spaces."""
    assert full_clean_text("  Hello World  ") == "hello world"


# ---------------------------------------------------------------------------
# load_and_preprocess_data — blank rows must be dropped
# ---------------------------------------------------------------------------

def test_blank_rows_dropped(mini_csv):
    """Rows whose track_name is blank after cleaning must be removed."""
    df, _ = load_and_preprocess_data(mini_csv)
    assert "" not in df["track_name"].values, "Blank track_name rows leaked into output"
    assert len(df) == 4, f"Expected 4 rows (1 blank dropped), got {len(df)}"


# ---------------------------------------------------------------------------
# track_display format
# ---------------------------------------------------------------------------

def test_track_display_format(mini_csv):
    """track_display must be '<track_name> - <artist_name>' (both lowercased)."""
    df, _ = load_and_preprocess_data(mini_csv)
    for _, row in df.iterrows():
        expected = row["track_name"] + " - " + row["artist_name"]
        assert row["track_display"] == expected, (
            f"track_display mismatch: got '{row['track_display']}', "
            f"expected '{expected}'"
        )


# ---------------------------------------------------------------------------
# Similarity scoring logic (pure numpy — does not import dashboard.py)
# ---------------------------------------------------------------------------

def test_similarity_scores_sorted_descending():
    """cosine_similarity + argsort produces scores in non-increasing order."""
    rng = np.random.default_rng(0)
    matrix = rng.random((20, 5))
    query = matrix[3:4]
    scores = cosine_similarity(query, matrix)[0]
    sorted_idx = np.argsort(-scores)
    for i in range(len(sorted_idx) - 1):
        assert scores[sorted_idx[i]] >= scores[sorted_idx[i + 1]], (
            f"Score at position {i} < score at position {i+1}"
        )


def test_similarity_scores_match_rows():
    """Each displayed row's score equals the cosine similarity of *that* track, not a neighbour's."""
    rng = np.random.default_rng(1)
    n = 15
    matrix = rng.random((n, 4))
    df = pd.DataFrame(matrix, columns=["f1", "f2", "f3", "f4"])
    df["track_display"] = [f"song {i}" for i in range(n)]

    query_display = "song 7"
    feat_cols = ["f1", "f2", "f3", "f4"]
    mask = df["track_display"] == query_display
    scores = cosine_similarity(df.loc[mask, feat_cols], df[feat_cols])[0]
    sorted_idx = np.argsort(-scores)

    # Correct pattern: assign scores using sorted_idx BEFORE reset_index
    result = df.iloc[sorted_idx].copy()
    result["Similarity Score"] = np.round(scores[sorted_idx], 4)
    result = result[result["track_display"] != query_display].reset_index(drop=True)

    for _, row in result.iterrows():
        original_pos = df.index[df["track_display"] == row["track_display"]][0]
        expected_score = round(scores[original_pos], 4)
        assert row["Similarity Score"] == pytest.approx(expected_score, abs=1e-4), (
            f"Score mismatch for {row['track_display']}: "
            f"got {row['Similarity Score']}, expected {expected_score}"
        )


# ---------------------------------------------------------------------------
# get_recommendations helper
# ---------------------------------------------------------------------------

@pytest.fixture
def rec_df():
    """Small DataFrame with known audio features and track_display."""
    rng = np.random.default_rng(42)
    n = 30
    data = {
        "f1": rng.random(n),
        "f2": rng.random(n),
        "f3": rng.random(n),
        "track_display": [f"song {i} - artist {i}" for i in range(n)],
        "genre": (["Pop"] * 15) + (["Hip-Hop"] * 15),
    }
    return pd.DataFrame(data)


def _feature_matrix(df):
    return df.select_dtypes(include=["float64", "int64"]).values


def test_get_recommendations_returns_n(rec_df):
    """Returns exactly n rows when enough candidates exist."""
    fm = _feature_matrix(rec_df)
    result = get_recommendations(rec_df, fm, "song 0 - artist 0", n=5)
    assert len(result) == 5


def test_get_recommendations_excludes_query(rec_df):
    """The query track itself is never in the results."""
    fm = _feature_matrix(rec_df)
    query = "song 3 - artist 3"
    result = get_recommendations(rec_df, fm, query, n=5)
    assert query not in result["track_display"].values


def test_get_recommendations_sorted_descending(rec_df):
    """Results are ordered from highest to lowest similarity score."""
    fm = _feature_matrix(rec_df)
    result = get_recommendations(rec_df, fm, "song 1 - artist 1", n=8)
    scores = result["Similarity Score"].tolist()
    assert scores == sorted(scores, reverse=True), "Scores not in descending order"


def test_get_recommendations_scores_match_rows(rec_df):
    """Each row's Similarity Score equals the actual cosine similarity for that track."""
    fm = _feature_matrix(rec_df)
    query = "song 2 - artist 2"
    result = get_recommendations(rec_df, fm, query, n=5)

    mask = rec_df["track_display"] == query
    true_scores = cosine_similarity(fm[mask.values], fm)[0]

    for _, row in result.iterrows():
        pos = rec_df.index[rec_df["track_display"] == row["track_display"]][0]
        assert row["Similarity Score"] == pytest.approx(round(true_scores[pos], 4), abs=1e-4)


def test_get_recommendations_min_score_filter(rec_df):
    """No result has a Similarity Score below min_score."""
    fm = _feature_matrix(rec_df)
    threshold = 0.98
    result = get_recommendations(rec_df, fm, "song 0 - artist 0", n=10, min_score=threshold)
    assert (result["Similarity Score"] >= threshold).all()


def test_get_recommendations_empty_on_unknown_track(rec_df):
    """Returns an empty DataFrame when the query track is not in the dataset."""
    fm = _feature_matrix(rec_df)
    result = get_recommendations(rec_df, fm, "nonexistent track - nobody", n=5)
    assert result.empty


def test_get_recommendations_cluster_fallback():
    """Phase 3: Cluster column is ignored — highest-similarity track wins regardless of cluster.

    Setup: 200 rows, query in cluster 0, 198 high-similarity tracks in cluster 1,
    1 low-similarity track in cluster 0.  With no cluster filter, the top result
    must come from cluster 1 (highest audio similarity), not cluster 0.
    """
    rng = np.random.default_rng(7)
    total = 200
    base = rng.random(3)

    data = {
        "f1": rng.random(total),
        "f2": rng.random(total),
        "f3": rng.random(total),
        "track_display": [f"song {i} - artist {i}" for i in range(total)],
        "Cluster": [0, 0] + [1] * (total - 2),
    }
    df = pd.DataFrame(data)

    df.loc[0, ["f1", "f2", "f3"]] = base
    df.loc[2:, "f1"] = base[0] + rng.random(total - 2) * 0.001
    df.loc[2:, "f2"] = base[1] + rng.random(total - 2) * 0.001
    df.loc[2:, "f3"] = base[2] + rng.random(total - 2) * 0.001
    df.loc[1, ["f1", "f2", "f3"]] = 1.0 - base  # low-similarity, cluster 0

    fm = df[["f1", "f2", "f3"]].values
    result = get_recommendations(df, fm, "song 0 - artist 0", n=1)

    assert not result.empty, "Expected a result when Cluster column is present but ignored"
    result_cluster = df.loc[
        df["track_display"] == result.iloc[0]["track_display"], "Cluster"
    ].iloc[0]
    assert result_cluster == 1, (
        f"Expected cluster-1 track (highest similarity); got cluster {result_cluster}"
    )


# ---------------------------------------------------------------------------
# Phase 3 — new tests (6 total)
# ---------------------------------------------------------------------------

def test_feature_matrix_excludes_bad_cols():
    """build_feature_matrix drops the 12 non-sonic columns from the numeric feature set."""
    bad_cols = ["artist_encoded", "genre_encoded", "song_length",
                "UMAP1", "UMAP2", "UMAP3", "UMAP4", "UMAP5", "UMAP6",
                "Cluster", "popularity", "key"]
    good_cols = ["danceability", "energy", "valence", "acousticness", "tempo"]
    n = 5
    data = {col: np.ones(n) for col in good_cols + bad_cols}
    data["track_display"] = [f"song {i}" for i in range(n)]
    df = pd.DataFrame(data)

    matrix = build_feature_matrix(df)
    assert matrix.shape[1] == len(good_cols), (
        f"Expected {len(good_cols)} feature columns, got {matrix.shape[1]}"
    )


def test_no_duplicate_track_display():
    """Results never contain two rows with the same track_display (dedup keeps highest score)."""
    rng = np.random.default_rng(99)
    base = np.array([0.9, 0.8, 0.7])
    rows = []
    for i in range(15):
        rows.append({
            "f1": rng.random(), "f2": rng.random(), "f3": rng.random(),
            "track_display": f"song {i} - artist {i}",
        })
    rows[0].update({"f1": base[0], "f2": base[1], "f3": base[2]})
    # Same track_display appearing 3 times (simulates genre-row duplication)
    for j in range(3):
        rows.append({
            "f1": base[0] + 0.001 * (j + 1),
            "f2": base[1] + 0.001 * (j + 1),
            "f3": base[2] + 0.001 * (j + 1),
            "track_display": "dupe song - dupe artist",
        })
    df = pd.DataFrame(rows)
    fm = df[["f1", "f2", "f3"]].values

    result = get_recommendations(df, fm, "song 0 - artist 0", n=5)
    assert result["track_display"].nunique() == len(result), "Duplicate track_display in results"
    dupe_count = (result["track_display"] == "dupe song - dupe artist").sum()
    assert dupe_count <= 1, f"Dupe track appeared {dupe_count} times"


def test_cluster_col_ignored():
    """A Cluster column in df does not filter results — all 5 come from cluster 99."""
    rng = np.random.default_rng(42)
    n = 30
    data = {
        "f1": rng.random(n),
        "f2": rng.random(n),
        "f3": rng.random(n),
        "track_display": [f"song {i} - artist {i}" for i in range(n)],
        "Cluster": [0] + [99] * (n - 1),  # query in cluster 0, all candidates in cluster 99
    }
    df = pd.DataFrame(data)
    fm = df[["f1", "f2", "f3"]].values

    result = get_recommendations(df, fm, "song 0 - artist 0", n=5)
    assert len(result) == 5, f"Expected 5 results ignoring Cluster, got {len(result)}"


def test_same_genre_filters_correctly():
    """With same_genre=True, all results share a genre with the query track."""
    rng = np.random.default_rng(10)
    base = np.array([0.9, 0.9, 0.9])
    n = 40
    data = {
        "f1": rng.random(n),
        "f2": rng.random(n),
        "f3": rng.random(n),
        "track_display": [f"song {i} - artist {i}" for i in range(n)],
        "genre": ["Rock"] * 10 + ["Pop"] * 10 + ["Jazz"] * 10 + ["Hip-Hop"] * 10,
    }
    df = pd.DataFrame(data)
    df.loc[0, ["f1", "f2", "f3"]] = base  # query is Rock (index 0)
    # Make Hip-Hop songs highly similar to the query so they'd appear without filtering
    for i in range(30, 40):
        df.loc[i, "f1"] = base[0] + rng.random() * 0.001
        df.loc[i, "f2"] = base[1] + rng.random() * 0.001
        df.loc[i, "f3"] = base[2] + rng.random() * 0.001
    fm = df[["f1", "f2", "f3"]].values

    result = get_recommendations(df, fm, "song 0 - artist 0", n=5, same_genre=True)
    assert not result.empty
    assert (result["genre"] == "Rock").all(), (
        f"Non-Rock results returned with same_genre=True: {result['genre'].tolist()}"
    )


def test_same_genre_false_allows_cross_genre():
    """With same_genre=False, songs from other genres can appear in results."""
    rng = np.random.default_rng(11)
    base = np.array([0.9, 0.9, 0.9])
    n = 30
    data = {
        "f1": rng.random(n),
        "f2": rng.random(n),
        "f3": rng.random(n),
        "track_display": [f"song {i} - artist {i}" for i in range(n)],
        "genre": ["Rock"] * 5 + ["Pop"] * 25,  # query is Rock, most tracks are Pop
    }
    df = pd.DataFrame(data)
    df.loc[0, ["f1", "f2", "f3"]] = base
    # Make Pop songs very similar to the query
    for i in range(5, 15):
        df.loc[i, "f1"] = base[0] + rng.random() * 0.001
        df.loc[i, "f2"] = base[1] + rng.random() * 0.001
        df.loc[i, "f3"] = base[2] + rng.random() * 0.001
    fm = df[["f1", "f2", "f3"]].values

    result = get_recommendations(df, fm, "song 0 - artist 0", n=5, same_genre=False)
    assert "Pop" in result["genre"].values, (
        "Expected Pop (cross-genre) songs when same_genre=False"
    )


def test_same_genre_dedup_returns_n():
    """same_genre=True + duplicate track_display rows still yields n distinct results."""
    base = np.array([0.9, 0.8, 0.7])
    rows = [{"f1": base[0], "f2": base[1], "f3": base[2],
              "track_display": "song 0 - artist 0", "genre": "Pop"}]
    # 10 unique Pop songs with descending similarity
    for i in range(1, 11):
        rows.append({
            "f1": base[0] - i * 0.01,
            "f2": base[1] - i * 0.01,
            "f3": base[2] - i * 0.01,
            "track_display": f"pop song {i} - artist {i}",
            "genre": "Pop",
        })
    # Same 5 track_displays again under a different genre (filtered by genre, then deduped)
    for i in range(1, 6):
        rows.append({
            "f1": base[0] - i * 0.01,
            "f2": base[1] - i * 0.01,
            "f3": base[2] - i * 0.01,
            "track_display": f"pop song {i} - artist {i}",
            "genre": "Dance",
        })
    df = pd.DataFrame(rows)
    fm = df[["f1", "f2", "f3"]].values

    result = get_recommendations(df, fm, "song 0 - artist 0", n=5, same_genre=True)
    assert len(result) == 5, f"Expected 5 results, got {len(result)}"
    assert result["track_display"].nunique() == 5, "Duplicate track_display in results"
    assert (result["genre"] == "Pop").all(), "Non-Pop result with same_genre=True"
