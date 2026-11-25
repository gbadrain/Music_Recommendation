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
from data_utils import full_clean_text, load_and_preprocess_data, get_recommendations


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
