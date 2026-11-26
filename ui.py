"""SoundMatch look and feel: theme CSS and small HTML building blocks.

Kept separate from dashboard.py so the page logic stays readable and the
HTML helpers can be unit-tested. Every piece of text that comes from the
dataset is HTML-escaped before it is placed in markup.
"""
from html import escape

APP_NAME = "SoundMatch"
AUTHOR = "Gurpreet Singh Badrain"
AUTHOR_TITLE = "Data Analyst"
GITHUB_URL = "https://github.com/gbadrain/Music_Recommendation"
PORTFOLIO_URL = "https://www.datascienceportfol.io/gbadrain"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', sans-serif; }
.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(124, 92, 255, 0.22), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(0, 200, 180, 0.14), transparent 60%),
    #0b0f1a;
}
[data-testid="stSidebar"] { background: rgba(14, 18, 32, 0.92); border-right: 1px solid rgba(255,255,255,0.06); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -0.02em; }

.sm-brand { font-size: 1.45rem; font-weight: 700; margin: 0 0 2px; display: flex; align-items: center; gap: 10px; }
.sm-brand .sm-icon { font-size: 2.3rem; line-height: 1; -webkit-text-fill-color: initial; }
.sm-brand span, .sm-grad {
  background: linear-gradient(90deg, #a78bfa, #38bdf8 55%, #2dd4bf);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.sm-tagline { color: #9aa3b8; font-size: 0.85rem; margin: 0 0 1rem; }

.sm-hero {
  border-radius: 22px; padding: 2.4rem 2.2rem; margin-bottom: 1.6rem;
  background: linear-gradient(135deg, rgba(124,92,255,0.28), rgba(56,189,248,0.12) 60%, rgba(45,212,191,0.10));
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 24px 60px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.08);
}
.sm-hero h1 { font-size: 2.6rem; font-weight: 700; margin: 0 0 0.4rem; line-height: 1.1; }
.sm-hero p { color: #c9cfe0; font-size: 1.08rem; max-width: 720px; margin: 0; }

.sm-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.025));
  border: 1px solid rgba(255,255,255,0.09); border-radius: 18px; padding: 1.2rem 1.3rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
  height: 100%;
}
.sm-card:hover { transform: translateY(-4px); box-shadow: 0 18px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(167,139,250,0.35); }
.sm-card h4 { margin: 0.2rem 0 0.35rem; font-size: 1.05rem; font-weight: 600; }
.sm-card p { color: #a9b1c6; margin: 0; font-size: 0.93rem; line-height: 1.55; }
.sm-step { display: inline-flex; width: 30px; height: 30px; border-radius: 10px; align-items: center; justify-content: center;
  background: rgba(167,139,250,0.18); color: #c4b5fd; font-weight: 700; font-size: 0.9rem; }

.sm-stat { text-align: left; }
.sm-stat .v { font-size: 1.9rem; font-weight: 700; }
.sm-stat .l { color: #9aa3b8; font-size: 0.85rem; }

.sm-card.sm-song { height: auto; margin-bottom: 8px; }
.sm-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 16px; margin-top: 0.6rem; }
.sm-song { display: flex; flex-direction: column; gap: 8px; }
.sm-art {
  height: 92px; border-radius: 14px; display: flex; align-items: center; justify-content: center;
  font-size: 2rem; font-weight: 700; color: rgba(255,255,255,0.92);
  box-shadow: inset 0 -20px 40px rgba(0,0,0,0.25);
}
.sm-title { font-weight: 600; font-size: 0.98rem; line-height: 1.3; }
.sm-artist { color: #a9b1c6; font-size: 0.86rem; }
.sm-row { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-top: 2px; }
.sm-match { font-weight: 700; font-size: 0.9rem; color: #5eead4; }
.sm-bar { height: 6px; border-radius: 6px; background: rgba(255,255,255,0.08); overflow: hidden; }
.sm-bar > div { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #a78bfa, #2dd4bf); }
.sm-badge { display: inline-block; font-size: 0.74rem; padding: 3px 9px; border-radius: 999px; margin: 0 4px 4px 0;
  background: rgba(167,139,250,0.16); color: #d8ccff; border: 1px solid rgba(167,139,250,0.28); }
a.sm-listen { font-size: 0.85rem; font-weight: 600; color: #e9e6ff !important; text-decoration: none;
  padding: 5px 11px; border-radius: 999px; background: rgba(124,92,255,0.35); border: 1px solid rgba(167,139,250,0.45); }
a.sm-listen:hover { background: rgba(124,92,255,0.55); }

.sm-footer { color: #7f879c; font-size: 0.8rem; line-height: 1.6; margin-top: 1rem; }
.sm-footer a { color: #b9a8ff !important; text-decoration: none; }
.sm-note { color: #8790a6; font-size: 0.8rem; margin-top: 1.4rem; }
</style>
"""

# Album-art stand-in colors (no external images needed); picked per genre.
_PALETTE = [
    ("#7c5cff", "#38bdf8"), ("#f472b6", "#a78bfa"), ("#2dd4bf", "#3b82f6"),
    ("#f59e0b", "#ef4444"), ("#22c55e", "#14b8a6"), ("#6366f1", "#ec4899"),
]


def spotify_track_url(track_id) -> str:
    """Direct Spotify link built from the dataset's track ID (no API call)."""
    track_id = str(track_id or "").strip()
    if not track_id or not track_id.isalnum():
        return ""
    return f"https://open.spotify.com/track/{track_id}"


def match_percent(score) -> int:
    """Cosine similarity (0–1) shown as a whole-number match percentage."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0
    return int(round(max(0.0, min(1.0, value)) * 100))


def _art_colors(key: str):
    return _PALETTE[sum(map(ord, key or "")) % len(_PALETTE)]


def badges_html(genres) -> str:
    return "".join(f'<span class="sm-badge">{escape(str(g))}</span>' for g in genres if g)


def spotify_embed_url(track_id) -> str:
    """Spotify's embeddable player for one track (compact when shown at 80px high)."""
    url = spotify_track_url(track_id)
    return url.replace("/track/", "/embed/track/") + "?utm_source=generator&theme=0" if url else ""


def song_card_html(title, artist, genres, score, track_id, show_listen=True) -> str:
    """One recommendation card. All dataset text is escaped."""
    title = str(title or "").strip() or "Untitled"
    artist = str(artist or "").strip()
    pct = match_percent(score)
    c1, c2 = _art_colors(", ".join(map(str, genres)) if genres else title)
    initial = escape(title[0].upper())
    url = spotify_track_url(track_id)
    listen = (f'<a class="sm-listen" href="{escape(url)}" target="_blank" rel="noopener">Listen ↗</a>'
              if url and show_listen else "")
    return (
        '<div class="sm-card sm-song">'
        f'<div class="sm-art" style="background: linear-gradient(135deg, {c1}, {c2});">{initial}</div>'
        f'<div><div class="sm-title">{escape(title)}</div><div class="sm-artist">{escape(artist)}</div></div>'
        f'<div>{badges_html(genres)}</div>'
        f'<div class="sm-bar"><div style="width:{pct}%"></div></div>'
        f'<div class="sm-row"><span class="sm-match">{pct}% match</span>{listen}</div>'
        '</div>'
    )


def card_html(title, body, step=None) -> str:
    step_html = f'<span class="sm-step">{escape(str(step))}</span>' if step is not None else ""
    return f'<div class="sm-card">{step_html}<h4>{escape(title)}</h4><p>{escape(body)}</p></div>'


def stat_html(value, label) -> str:
    return f'<div class="sm-card sm-stat"><div class="v sm-grad">{escape(str(value))}</div><div class="l">{escape(label)}</div></div>'


def footer_html() -> str:
    return (
        '<div class="sm-footer">'
        f'Built by <b>{escape(AUTHOR)}</b><br>{escape(AUTHOR_TITLE)}<br>'
        f'<a href="{GITHUB_URL}" target="_blank" rel="noopener">GitHub</a> · '
        f'<a href="{PORTFOLIO_URL}" target="_blank" rel="noopener">Portfolio</a><br>'
        '<span>Data: Spotify audio-features dataset (Kaggle, 2019). Not affiliated with Spotify.</span>'
        '</div>'
    )
