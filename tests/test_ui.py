import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui import spotify_track_url, match_percent, song_card_html, badges_html


def test_spotify_track_url_built_from_id():
    assert spotify_track_url("0BRjO6ga9RKCKjfDqeFgWV") == "https://open.spotify.com/track/0BRjO6ga9RKCKjfDqeFgWV"


def test_spotify_track_url_rejects_blank_or_odd_ids():
    assert spotify_track_url("") == ""
    assert spotify_track_url(None) == ""
    assert spotify_track_url("abc/../evil") == ""


def test_match_percent_rounds_and_clamps():
    assert match_percent(0.9682) == 97
    assert match_percent(1.4) == 100
    assert match_percent(-0.2) == 0
    assert match_percent("not a number") == 0


def test_song_card_escapes_dataset_text():
    html = song_card_html('<script>alert(1)</script>', 'AC/DC & "Friends"', ["Rock"], 0.9, "abc123")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&amp;" in html and "&quot;" in html


def test_song_card_shows_match_and_listen_link():
    html = song_card_html("Adictiva", "Daddy Yankee", ["Reggaeton", "Pop"], 0.9682, "0BRjO6ga9RKCKjfDqeFgWV")
    assert "97% match" in html
    assert "https://open.spotify.com/track/0BRjO6ga9RKCKjfDqeFgWV" in html
    assert "Reggaeton" in html and "Pop" in html


def test_song_card_without_track_id_has_no_listen_link():
    html = song_card_html("Song", "Artist", [], 0.5, "")
    assert "Listen" not in html


def test_badges_skip_empty_values():
    assert badges_html(["Pop", "", None]).count("sm-badge") == 1


def test_spotify_embed_url():
    from ui import spotify_embed_url
    assert spotify_embed_url("0BRjO6ga9RKCKjfDqeFgWV").startswith("https://open.spotify.com/embed/track/0BRjO6ga9RKCKjfDqeFgWV")
    assert spotify_embed_url("") == ""


def test_song_card_can_hide_listen_link():
    html = song_card_html("Song", "Artist", ["Pop"], 0.9, "abc123", show_listen=False)
    assert "Listen" not in html and "90% match" in html
