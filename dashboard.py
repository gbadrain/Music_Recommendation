import os
import re
import logging

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

from data_utils import load_and_preprocess_data, get_recommendations, build_feature_matrix, feature_columns, search_tracks
import ui

logger = logging.getLogger(__name__)

st.set_page_config(page_title=f"{ui.APP_NAME} · Music that sounds alike", page_icon="🎧", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

# ---------------------------
# Configuration
# ---------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_DATA_PATH = os.path.join(SCRIPT_DIR, "Resource", "SpotifyFeatures.csv")
ADDITIONAL_DATA_PATH = os.path.join(SCRIPT_DIR, "Output", "umap_results.csv")
SLIDES_DIR = os.path.join(SCRIPT_DIR, "Slides")

PAGES = ["Overview", "Discover", "Explore the data", "Project story", "Under the hood"]
PAGE_ICONS = {"Overview": "◆", "Discover": "✦", "Explore the data": "◎", "Project story": "▤", "Under the hood": "⚙"}


# ---------------------------
# Data loading (cached once per server)
# ---------------------------
@st.cache_data(show_spinner="Loading the song library…")
def load_data(main_path, additional_path):
    df, _scaler = load_and_preprocess_data(main_path, additional_filepath=additional_path)
    df = df.reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df


try:
    df = load_data(MAIN_DATA_PATH, ADDITIONAL_DATA_PATH)
except Exception:
    logger.exception("Data loading failed")
    st.error("The music data couldn't be loaded. Refresh the page in a minute.")
    st.stop()


@st.cache_data
def get_feature_matrix(_df):
    """12-column audio feature matrix (no ID-like columns, UMAP, Cluster, popularity, or key)."""
    return build_feature_matrix(_df)


@st.cache_data
def get_track_options(track_display):
    return sorted(track_display.dropna().unique())


@st.cache_data
def get_track_info(_df):
    """track_display -> original title, artist, track_id, and all dataset genres."""
    first = _df.drop_duplicates("track_display").set_index("track_display")
    genres = _df.groupby("track_display")["genre"].agg(lambda g: sorted(set(g.dropna())))
    info = first[["song_title", "artist_display", "track_id"]].copy()
    info["genres"] = genres
    return info


TRACK_INFO = get_track_info(df)

RADAR_FEATURES = ["danceability", "energy", "valence", "acousticness", "instrumentalness",
                  "speechiness", "liveness", "loudness", "tempo"]


@st.cache_data
def get_percentiles(_df):
    """Each song's percentile (0–100) on the radar features, across all songs."""
    cols = [c for c in RADAR_FEATURES if c in _df.columns]
    return (_df[cols].rank(pct=True) * 100).round(1)


def pretty(track_display):
    """'adictiva - daddy yankee' -> 'Adictiva — Daddy Yankee' (original capitalization)."""
    if track_display in TRACK_INFO.index:
        row = TRACK_INFO.loc[track_display]
        return f"{row['song_title']} — {row['artist_display']}"
    return track_display


# ---------------------------
# Shared pieces
# ---------------------------
def page_header(title, subtitle):
    st.markdown(f"<h1 style='margin-bottom:0.2rem'>{title}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#a9b1c6; margin-top:0; font-size:1.02rem'>{subtitle}</p>", unsafe_allow_html=True)


def pick_track(key):
    """Search box + short pick list. Returns the chosen track_display or None."""
    query = st.text_input(
        "Search for a song or artist",
        placeholder="Try “adictiva”, “country road”, or “daddy yankee”",
        key=f"{key}_search",
    )
    if not query.strip():
        st.info("Type part of a song or artist name to begin.")
        return None
    matches = search_tracks(get_track_options(df["track_display"]), query)
    if not matches:
        st.warning("No songs match that search. Try fewer letters or a different spelling.")
        return None
    return st.selectbox(f"Pick a song ({len(matches)} shown)", matches, key=f"{key}_pick", format_func=pretty)


def selected_song_panel(track_display):
    """Shows the chosen song with its genres and a Spotify preview player."""
    row = TRACK_INFO.loc[track_display]
    left, right = st.columns([3, 2], vertical_alignment="center")
    with left:
        st.markdown(
            f"<div class='sm-card'><div class='sm-artist'>Your pick</div>"
            f"<div class='sm-title' style='font-size:1.25rem'>{ui.escape(row['song_title'])}</div>"
            f"<div class='sm-artist' style='margin-bottom:8px'>{ui.escape(row['artist_display'])}</div>"
            f"{ui.badges_html(row['genres'])}"
            f"<div class='sm-note' style='margin-top:6px'>Genre labels come from the dataset.</div></div>",
            unsafe_allow_html=True,
        )
    with right:
        embed = ui.spotify_embed_url(row["track_id"])
        if embed:
            st.iframe(embed, height=152)


def render_song_grid(results, per_row=3):
    """Song cards in rows of three, each with its own small Spotify player."""
    rows = list(results.iterrows())
    for i in range(0, len(rows), per_row):
        cols = st.columns(per_row)
        for col, (_, r) in zip(cols, rows[i:i + per_row]):
            info = TRACK_INFO.loc[r["track_display"]] if r["track_display"] in TRACK_INFO.index else None
            title = info["song_title"] if info is not None else r["track_display"]
            artist = info["artist_display"] if info is not None else ""
            genres = info["genres"] if info is not None else [r.get("genre", "")]
            track_id = r.get("track_id", "")
            with col:
                st.markdown(ui.song_card_html(title, artist, genres, r["Similarity Score"], track_id,
                                              show_listen=False), unsafe_allow_html=True)
                embed = ui.spotify_embed_url(track_id)
                if embed:
                    st.iframe(embed, height=80)
                else:
                    st.caption("No preview available for this song.")


# ---------------------------
# Pages
# ---------------------------
def show_overview():
    st.markdown(
        f"<div class='sm-hero'><h1>Find music that <span class='sm-grad'>sounds like</span> the songs you love</h1>"
        f"<p>{ui.APP_NAME} compares more than 175,000 songs on how they actually sound — energy, mood, tempo, "
        "danceability and more — and suggests songs that match. Pick a song, and get your matches in a second.</p></div>",
        unsafe_allow_html=True,
    )
    n_genres = df["genre"].nunique() if "genre" in df.columns else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(ui.stat_html(f"{df['track_display'].nunique():,}", "unique songs"), unsafe_allow_html=True)
    c2.markdown(ui.stat_html(f"{n_genres}", "genres"), unsafe_allow_html=True)
    c3.markdown(ui.stat_html("12", "audio features compared"), unsafe_allow_html=True)
    c4.markdown(ui.stat_html("< 1 s", "to get matches"), unsafe_allow_html=True)

    st.markdown("<h3 style='margin-top:1.8rem'>How it works</h3>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.markdown(ui.card_html("Search a song", "Type any part of a song or artist name and pick it from the list.", 1), unsafe_allow_html=True)
    s2.markdown(ui.card_html("We compare its sound", "Its 12 audio features are matched against every other track using cosine similarity.", 2), unsafe_allow_html=True)
    s3.markdown(ui.card_html("Get your matches", "See the closest songs with a match score, stay in the same genre or explore beyond it.", 3), unsafe_allow_html=True)

    st.markdown("<h3 style='margin-top:1.8rem'>What's inside</h3>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    f1.markdown(ui.card_html("Discover", "Song recommendations with match scores and a one-click Spotify link."), unsafe_allow_html=True)
    f2.markdown(ui.card_html("Explore the data", "Interactive 3D genre map, genre fingerprints, and how features vary by genre."), unsafe_allow_html=True)
    f3.markdown(ui.card_html("Under the hood", "The method, the features used, and the full numbers behind each match."), unsafe_allow_html=True)
    st.markdown("<p class='sm-note'>Open <b>Discover</b> in the menu on the left to try it.</p>", unsafe_allow_html=True)


def show_discover():
    page_header("Discover", "Pick a song you like. We'll find songs that sound like it.")
    selected = pick_track("discover")
    if selected is None:
        return
    selected_song_panel(selected)

    c1, c2, c3 = st.columns([2, 2, 1.4], vertical_alignment="bottom")
    with c1:
        min_similarity = st.slider("How close a match?", 0.0, 1.0, 0.5, 0.01,
                                   help="Only show songs at least this similar (0 = anything, 1 = identical).")
    with c2:
        num_recs = st.slider("How many songs?", 3, 12, 6, step=3)
    with c3:
        same_genre = st.toggle("Same genre only", value=True,
                               help="On: stay within the song's genres. Off: pure 'sounds alike', any genre.")

    try:
        results = get_recommendations(df, get_feature_matrix(df), selected,
                                      n=num_recs, min_score=min_similarity, same_genre=same_genre)
    except Exception:
        logger.exception("Recommendation failed")
        st.error("Couldn't get recommendations for that song. Try another one.")
        return

    if results.empty:
        st.warning("No songs are that close a match. Lower “How close a match?” or turn off “Same genre only”.")
        return
    st.markdown("<h3 style='margin-top:1.2rem'>Songs you might like</h3>"
                "<p class='sm-note' style='margin-top:-6px'>Press play to hear a preview right here. If you're signed in to Spotify in this browser, you may hear the full song.</p>", unsafe_allow_html=True)
    render_song_grid(results)


def _style_fig(fig, height):
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif"),
                      margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    return fig


def show_feature_spread(data):
    st.subheader("Feature spread by genre")
    st.caption("How one audio feature varies across the 10 biggest genres (sample of 5,000 songs).")
    features = [f for f in ["danceability", "energy", "valence", "acousticness", "instrumentalness"] if f in data.columns]
    feature = st.selectbox("Audio feature", features, format_func=str.capitalize)
    top = data["genre"].value_counts().nlargest(10).index
    sample = data[data["genre"].isin(top)]
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)
    fig = px.violin(sample, y=feature, x="genre", color="genre", box=True, points="all",
                    color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(xaxis_title="", yaxis_title=feature.capitalize(), showlegend=False)
    st.plotly_chart(_style_fig(fig, 560), width="stretch")


def show_genre_map(data):
    st.subheader("3D genre map")
    st.caption("Songs with similar sound sit close together. Drag to rotate, scroll to zoom.")
    if not all(c in data.columns for c in ["UMAP1", "UMAP2", "UMAP3"]):
        st.info("The 3D map isn't available for this dataset.")
        return
    mapped = data[data["Cluster"].round(6) != round(float(data["Cluster"].mean()), 6)] if "Cluster" in data.columns else data
    st.caption(f"Built from the {mapped['track_display'].nunique():,} songs analyzed with UMAP in the original study.")
    top = mapped["genre"].value_counts().nlargest(15).index.tolist()
    chosen = st.multiselect("Genres to show", sorted(mapped["genre"].unique()), default=top)
    if not chosen:
        st.info("Pick at least one genre.")
        return
    sample = mapped[mapped["genre"].isin(chosen)]
    if len(sample) > 10000:
        sample = sample.sample(10000, random_state=42)
    fig = px.scatter_3d(sample, x="UMAP1", y="UMAP2", z="UMAP3", color="genre", hover_name="track_display",
                        hover_data={"genre": True, "UMAP1": False, "UMAP2": False, "UMAP3": False},
                        color_discrete_sequence=px.colors.qualitative.Vivid)
    fig.update_traces(marker=dict(size=3, opacity=0.85))
    st.plotly_chart(_style_fig(fig, 680), width="stretch")


def show_genre_fingerprints(data):
    st.subheader("Genre fingerprints")
    st.caption("Each line is a song. Compare how genres differ across several audio features at once.")
    genres = sorted(data["genre"].unique())
    features_all = [f for f in ["danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness"] if f in data.columns]
    c1, c2 = st.columns(2)
    chosen = c1.multiselect("Genres", genres, default=genres[:4])
    features = c2.multiselect("Features", features_all, default=features_all[:4], format_func=str.capitalize)
    if not chosen or not features:
        st.info("Pick at least one genre and one feature.")
        return
    color_by = st.selectbox("Color lines by", features_all, index=features_all.index("valence") if "valence" in features_all else 0,
                            format_func=str.capitalize)
    sample = data[data["genre"].isin(chosen)]
    if len(sample) > 2000:
        sample = sample.sample(2000, random_state=42)
    fig = px.parallel_coordinates(sample, dimensions=features, color=color_by,
                                  color_continuous_scale=px.colors.sequential.Plasma,
                                  labels={f: f.capitalize() for f in features})
    fig = _style_fig(fig, 520)
    fig.update_layout(margin=dict(l=80, r=40, t=60, b=20))
    st.plotly_chart(fig, width="stretch")


def show_explore():
    page_header("Explore the data", "Interactive views of what makes each genre sound the way it does.")
    views = {"3D genre map": show_genre_map, "Genre fingerprints": show_genre_fingerprints,
             "Feature spread by genre": show_feature_spread}
    choice = st.radio("View", list(views), horizontal=True, label_visibility="collapsed")
    views[choice](df)


def show_story():
    page_header("Project story", "The slides from the original analysis: data, method, and findings.")
    try:
        slides = sorted(
            [f for f in os.listdir(SLIDES_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
            key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 0,
        )
    except Exception:
        logger.exception("Slides folder missing")
        st.info("The slides aren't available right now.")
        return
    if not slides:
        st.info("The slides aren't available right now.")
        return
    st.session_state.setdefault("slide_index", 0)
    i = min(st.session_state.slide_index, len(slides) - 1)
    prev_col, mid, next_col = st.columns([1, 4, 1], vertical_alignment="center")
    if prev_col.button("← Previous", disabled=(i == 0), width="stretch"):
        st.session_state.slide_index = i - 1
        st.rerun()
    mid.markdown(f"<p style='text-align:center; color:#a9b1c6; margin:0'>Slide {i + 1} of {len(slides)}</p>",
                 unsafe_allow_html=True)
    if next_col.button("Next →", disabled=(i == len(slides) - 1), width="stretch"):
        st.session_state.slide_index = i + 1
        st.rerun()
    st.image(os.path.join(SLIDES_DIR, slides[i]), width="stretch")


def show_under_the_hood():
    page_header("Under the hood", "How the matches are made, with the full numbers.")
    m1, m2, m3 = st.columns(3)
    m1.markdown(ui.card_html("12 audio features", "Acousticness, danceability, duration, energy, instrumentalness, liveness, "
                             "loudness, mode, speechiness, tempo, time signature, valence — each standardized."), unsafe_allow_html=True)
    m2.markdown(ui.card_html("Cosine similarity", "Every song becomes a 12-number vector. The match score is the cosine of "
                             "the angle between two songs' vectors."), unsafe_allow_html=True)
    m3.markdown(ui.card_html("Clean results", "ID-like columns are excluded, duplicate listings of the same song are removed, "
                             "and the genre filter is optional."), unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:1.4rem'>Try it with the numbers</h3>", unsafe_allow_html=True)
    selected = pick_track("geeks")
    if selected is None:
        return
    selected_song_panel(selected)
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    num_recs = c1.slider("How many songs?", 1, 12, 6, key="geeks_n")
    same_genre = c2.toggle("Same genre only", value=True, key="geeks_genre")
    try:
        results = get_recommendations(df, get_feature_matrix(df), selected, n=num_recs, same_genre=same_genre)
    except Exception:
        logger.exception("Recommendation failed")
        st.error("Couldn't get recommendations for that song. Try another one.")
        return
    if results.empty:
        st.warning("No similar songs found for this selection.")
        return
    table = pd.DataFrame({
        "Song": [pretty(t) for t in results["track_display"]],
        "Genre": results["genre"],
        "Match": [ui.match_percent(s) for s in results["Similarity Score"]],
        "Similarity": results["Similarity Score"].round(4),
        "Danceability": results["danceability"].round(2),
        "Energy": results["energy"].round(2),
        "Valence": results["valence"].round(2),
        "Acousticness": results["acousticness"].round(2),
        "Listen": [ui.spotify_track_url(t) for t in results["track_id"]],
    })
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Match": st.column_config.ProgressColumn("Match", format="%d%%", min_value=0, max_value=100, width="small"),
            "Listen": st.column_config.LinkColumn("Listen", display_text="Open ↗", width="small"),
            "Song": st.column_config.TextColumn("Song", width="large"),
        },
    )
    st.caption("Feature values are standardized (0 = average across all songs).")

    show_match_radar(selected, results)
    show_similarity_histogram(selected, results)


def show_match_radar(selected, results):
    """Radar chart: the picked song's sound profile vs its top 3 matches (percentiles)."""
    pct = get_percentiles(df)
    cols = list(pct.columns)
    pick_idx = df.index[df["track_display"] == selected][0]
    st.markdown("<h3 style='margin-top:1.6rem'>Why these songs match</h3>", unsafe_allow_html=True)
    st.caption("Each axis shows where a song ranks among all songs (0 = lowest, 100 = highest). "
               "Matching songs have a similar shape.")
    fig = go.Figure()
    palette = ["#a78bfa", "#2dd4bf", "#38bdf8", "#f472b6"]
    labels = [c.capitalize() for c in cols] + [cols[0].capitalize()]
    traces = [(pretty(selected) + " (your pick)", pick_idx)]
    for t in results["track_display"].head(3):
        traces.append((pretty(t), df.index[df["track_display"] == t][0]))
    for i, (name, idx) in enumerate(traces):
        vals = list(pct.loc[idx, cols]) + [pct.loc[idx, cols[0]]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=labels, name=name, fill="toself" if i == 0 else None,
                                      line=dict(color=palette[i % len(palette)], width=3 if i == 0 else 2),
                                      opacity=0.95 if i == 0 else 0.8))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.12)"),
                                 angularaxis=dict(gridcolor="rgba(255,255,255,0.12)")),
                      legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(_style_fig(fig, 520), width="stretch")


def show_similarity_histogram(selected, results):
    """Histogram of similarity between the pick and every song, with matches marked."""
    X = get_feature_matrix(df)
    mask = (df["track_display"] == selected).values
    scores = cosine_similarity(X[mask][:1], X)[0]
    scores = scores[~mask]
    top = float(results["Similarity Score"].min())
    share = (scores >= top).mean() * 100
    st.markdown("<h3 style='margin-top:1.6rem'>How rare is a close match?</h3>", unsafe_allow_html=True)
    st.caption(f"Similarity between your pick and every other song in the library ({len(scores):,} listings). "
               f"Only {share:.2f}% score as high as the songs recommended above.")
    sample = pd.Series(scores).sample(min(len(scores), 60000), random_state=42)
    fig = px.histogram(sample, nbins=80, color_discrete_sequence=["#7c5cff"])
    fig.add_vrect(x0=top, x1=1.0, fillcolor="#2dd4bf", opacity=0.25, line_width=0,
                  annotation_text="Recommended", annotation_position="top left")
    fig.update_layout(showlegend=False, xaxis_title="Cosine similarity to your pick", yaxis_title="Songs (sample)",
                      bargap=0.05)
    st.plotly_chart(_style_fig(fig, 380), width="stretch")


# ---------------------------
# Navigation
# ---------------------------
def main():
    with st.sidebar:
        st.markdown(f"<p class='sm-brand'><span class='sm-icon'>🎧</span><span>{ui.APP_NAME}</span></p>"
                    "<p class='sm-tagline'>Music that sounds alike</p>", unsafe_allow_html=True)
        page = st.radio("Go to", PAGES, format_func=lambda p: f"{PAGE_ICONS[p]}  {p}", label_visibility="collapsed")
        st.markdown(ui.footer_html(), unsafe_allow_html=True)

    {"Overview": show_overview, "Discover": show_discover, "Explore the data": show_explore,
     "Project story": show_story, "Under the hood": show_under_the_hood}[page]()


if __name__ == "__main__":
    main()
