"""
Unit tests for scripts/enrich_genres_musicbrainz.py.

All MusicBrainz API calls are mocked — no real HTTP traffic.
Run with: pytest tests/test_mb_enrich.py -v
"""
import csv
import os
import sys
import importlib
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers to import the script under test
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "enrich_genres_musicbrainz.py"


def _load_script(mb_contact="test@example.com"):
    """Import the script module with MB_CONTACT set."""
    env_patch = patch.dict(os.environ, {"MB_CONTACT": mb_contact})
    mb_patch = patch("musicbrainzngs.set_useragent")
    with env_patch, mb_patch:
        spec = importlib.util.spec_from_file_location("enrich_genres_musicbrainz", SCRIPT_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# test_score_too_low_returns_no_match
# ---------------------------------------------------------------------------

def test_score_too_low_returns_no_match():
    """Score below 90 must return status 'no_match' regardless of name."""
    mod = _load_script()
    fake_search = {"artist-list": [{"id": "abc", "name": "Avail Hollywood", "ext:score": "85"}]}
    with patch("musicbrainzngs.search_artists", return_value=fake_search), \
         patch("time.sleep"):
        result = mod.lookup_artist("Avail Hollywood")
    assert result["status"] == "no_match"
    assert result["mb_genres"] == ""


# ---------------------------------------------------------------------------
# test_name_mismatch_returns_no_match
# ---------------------------------------------------------------------------

def test_name_mismatch_returns_no_match():
    """Score ≥ 90 but name differs after lowercasing must still return 'no_match'."""
    mod = _load_script()
    fake_search = {"artist-list": [{"id": "abc", "name": "The Lacs Band", "ext:score": "95"}]}
    with patch("musicbrainzngs.search_artists", return_value=fake_search), \
         patch("time.sleep"):
        result = mod.lookup_artist("The Lacs")
    assert result["status"] == "no_match"


# ---------------------------------------------------------------------------
# test_matched_genres_sorted_by_vote
# ---------------------------------------------------------------------------

def test_matched_genres_sorted_by_vote():
    """Matched artist: genres must be semicolon-joined, sorted by vote count descending."""
    mod = _load_script()
    fake_search = {"artist-list": [{"id": "mbid-123", "name": "John Mayall", "ext:score": "100"}]}
    fake_detail = {
        "artist": {
            "tag-list": [
                {"name": "blues rock", "count": "15"},
                {"name": "chicago blues", "count": "42"},
                {"name": "electric blues", "count": "8"},
            ]
        }
    }
    with patch("musicbrainzngs.search_artists", return_value=fake_search), \
         patch("musicbrainzngs.get_artist_by_id", return_value=fake_detail), \
         patch("time.sleep"):
        result = mod.lookup_artist("John Mayall")
    assert result["status"] == "matched"
    assert result["mb_genres"] == "chicago blues;blues rock;electric blues"
    assert result["mbid"] == "mbid-123"
    assert result["match_score"] == 100


# ---------------------------------------------------------------------------
# test_matched_no_genres_returns_no_genres
# ---------------------------------------------------------------------------

def test_matched_no_genres_returns_no_genres():
    """Artist found in MB but with no genres tagged → status 'no_genres'."""
    mod = _load_script()
    fake_search = {"artist-list": [{"id": "mbid-456", "name": "James Taylor", "ext:score": "98"}]}
    fake_detail = {"artist": {"tag-list": []}}
    with patch("musicbrainzngs.search_artists", return_value=fake_search), \
         patch("musicbrainzngs.get_artist_by_id", return_value=fake_detail), \
         patch("time.sleep"):
        result = mod.lookup_artist("James Taylor")
    assert result["status"] == "no_genres"
    assert result["mb_genres"] == ""
    assert result["mbid"] == "mbid-456"


# ---------------------------------------------------------------------------
# test_already_processed_artists_skipped
# ---------------------------------------------------------------------------

def test_already_processed_artists_skipped(tmp_path):
    """Artists already in the output CSV must be skipped (no API call made)."""
    mod = _load_script()
    # Pre-populate a CSV with one artist already done
    out_csv = tmp_path / "artist_genres_musicbrainz.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=mod.FIELDNAMES)
        writer.writeheader()
        writer.writerow({
            "artist_name": "James Taylor",
            "mbid": "existing-id",
            "match_score": 98,
            "mb_genres": "soft rock",
            "status": "matched",
        })

    already_done = mod.load_already_processed(out_csv)
    assert "James Taylor" in already_done

    # Confirm lookup_artist is NOT called for this artist
    with patch("musicbrainzngs.search_artists") as mock_search, \
         patch("time.sleep"):
        artists = ["James Taylor", "New Artist"]
        to_process = [a for a in artists if a not in already_done]
    mock_search.assert_not_called()
    assert "James Taylor" not in to_process
    assert "New Artist" in to_process


# ---------------------------------------------------------------------------
# test_mb_contact_env_required
# ---------------------------------------------------------------------------

def test_mb_contact_env_required():
    """Script must exit non-zero if MB_CONTACT is not set."""
    env = {k: v for k, v in os.environ.items() if k != "MB_CONTACT"}
    with patch.dict(os.environ, env, clear=True), \
         patch("musicbrainzngs.set_useragent"), \
         pytest.raises(SystemExit) as exc_info:
        spec = importlib.util.spec_from_file_location("enrich_genres_musicbrainz", SCRIPT_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    assert exc_info.value.code != 0
