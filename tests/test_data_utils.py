"""
Regression tests for data_utils.py.

Run with:  pytest tests/ -v
(from the project root /Users/.../Music_Recommendation)

These tests use a tiny in-memory CSV so they never touch the 33 MB
SpotifyFeatures.csv.  They document the expected behaviour of:
  - full_clean_text   (NaN / blank / normal strings)
  - load_and_preprocess_data   (blank rows dropped, track_display format)
  - cosine similarity scoring   (sorted descending, scores tied to right rows)
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity

# Make data_utils importable when pytest is run from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import full_clean_text, load_and_preprocess_data


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
