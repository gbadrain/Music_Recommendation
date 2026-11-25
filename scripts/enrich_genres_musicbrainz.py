#!/usr/bin/env python3
"""
MusicBrainz genre enrichment script.

Fetches artist-level genres from MusicBrainz for every unique artist in
SpotifyFeatures.csv and writes results to Resource/artist_genres_musicbrainz.csv.

Usage:
    python scripts/enrich_genres_musicbrainz.py --pilot   # 100 artists, then report
    python scripts/enrich_genres_musicbrainz.py --full    # all ~14k artists
    python scripts/enrich_genres_musicbrainz.py --report  # print stats from existing CSV

Requirements:
    export MB_CONTACT="your@email.com"   # required; never hard-coded

MusicBrainz rules:
    - 1 request per second max (SLEEP_SEC >= 1.1)
    - User-Agent built from MB_CONTACT env var
    - Genres are CC BY-NC-SA → non-commercial use only
"""
import csv
import os
import random
import sys
import time
import argparse
from pathlib import Path

import musicbrainzngs
import pandas as pd

# ---------------------------------------------------------------------------
# Guard: MB_CONTACT must be set before anything else runs
# ---------------------------------------------------------------------------
_CONTACT = os.environ.get("MB_CONTACT", "")
if not _CONTACT:
    print(
        "ERROR: MB_CONTACT environment variable is not set.\n"
        "Please run:  export MB_CONTACT=\"your@email.com\"\n"
        "then re-run this script.",
        file=sys.stderr,
    )
    sys.exit(1)

musicbrainzngs.set_useragent("MusicRecommendationApp", "1.0", _CONTACT)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).parent.parent
SOURCE_CSV = _SCRIPT_DIR / "Resource" / "SpotifyFeatures.csv"
OUTPUT_CSV = _SCRIPT_DIR / "Resource" / "artist_genres_musicbrainz.csv"

FIELDNAMES = ["artist_name", "mbid", "match_score", "mb_genres", "status"]
SLEEP_SEC = 1.1
SAVE_EVERY = 50
PILOT_TOP_N = 50
PILOT_RAND_N = 50
PILOT_SEED = 42


# ---------------------------------------------------------------------------
# Artist selection
# ---------------------------------------------------------------------------

def select_artists(pilot: bool) -> list[str]:
    df = pd.read_csv(SOURCE_CSV, usecols=["artist_name"])
    counts = df["artist_name"].value_counts()
    all_artists = counts.index.tolist()

    if not pilot:
        return all_artists

    top = counts.head(PILOT_TOP_N).index.tolist()
    top_set = set(top)
    rest = [a for a in all_artists if a not in top_set]
    rng = random.Random(PILOT_SEED)
    rand_pick = rng.sample(rest, min(PILOT_RAND_N, len(rest)))
    return top + rand_pick


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------

def load_already_processed(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    with open(output_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["artist_name"] for row in reader}


# ---------------------------------------------------------------------------
# Single artist lookup
# ---------------------------------------------------------------------------

def lookup_artist(artist_name: str) -> dict:
    """Query MusicBrainz for one artist. Returns a row dict (without artist_name key)."""
    try:
        result = musicbrainzngs.search_artists(artist=artist_name, limit=1)
        time.sleep(SLEEP_SEC)

        artists = result.get("artist-list", [])
        if not artists:
            return {"mbid": "", "match_score": 0, "mb_genres": "", "status": "no_match"}

        top = artists[0]
        score = int(top.get("ext:score", 0))
        mb_name = top.get("name", "")

        if score < 90 or mb_name.lower() != artist_name.lower():
            return {"mbid": "", "match_score": score, "mb_genres": "", "status": "no_match"}

        mbid = top["id"]

        detail = musicbrainzngs.get_artist_by_id(mbid, includes=["genres"])
        time.sleep(SLEEP_SEC)

        genres_data = detail.get("artist", {}).get("genre-list", [])
        if not genres_data:
            return {"mbid": mbid, "match_score": score, "mb_genres": "", "status": "no_genres"}

        genres_sorted = sorted(genres_data, key=lambda g: int(g.get("count", 0)), reverse=True)
        mb_genres = ";".join(g["name"] for g in genres_sorted)

        return {"mbid": mbid, "match_score": score, "mb_genres": mb_genres, "status": "matched"}

    except Exception as exc:
        time.sleep(SLEEP_SEC)
        return {
            "mbid": "",
            "match_score": 0,
            "mb_genres": "",
            "status": f"error:{str(exc)[:60]}",
        }


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process(artists: list[str], already_done: set[str]) -> None:
    to_do = [a for a in artists if a not in already_done]
    total = len(to_do)
    if total == 0:
        print("All artists already processed.")
        return

    mode = "a" if OUTPUT_CSV.exists() else "w"
    print(f"Processing {total} artists → {OUTPUT_CSV}")
    print(f"(Resumable: saves every {SAVE_EVERY} artists, safe to Ctrl-C between saves)\n")

    with open(OUTPUT_CSV, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if mode == "w":
            writer.writeheader()

        batch: list[dict] = []
        for i, artist in enumerate(to_do, 1):
            row_data = lookup_artist(artist)
            row = {"artist_name": artist, **row_data}
            batch.append(row)
            already_done.add(artist)

            status_icon = {"matched": "✓", "no_match": "✗", "no_genres": "○"}.get(
                row_data["status"], "!"
            )
            print(
                f"[{i}/{total}] {status_icon} {artist} "
                f"(score={row_data['match_score']}, status={row_data['status']})"
            )

            if len(batch) >= SAVE_EVERY:
                for r in batch:
                    writer.writerow(r)
                f.flush()
                print(f"  → saved {SAVE_EVERY} artists (total done: {i})")
                batch.clear()

        # flush remainder
        for r in batch:
            writer.writerow(r)
        f.flush()

    print(f"\nDone. {total} artists processed.")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(source_csv: Path | None = None) -> None:
    if not OUTPUT_CSV.exists():
        print("No output CSV found yet. Run --pilot first.")
        return

    rows = list(csv.DictReader(open(OUTPUT_CSV, encoding="utf-8")))
    total = len(rows)
    if total == 0:
        print("Output CSV is empty.")
        return

    matched = sum(1 for r in rows if r["status"] == "matched")
    no_genres = sum(1 for r in rows if r["status"] == "no_genres")
    no_match = sum(1 for r in rows if r["status"] == "no_match")
    with_genres = matched  # only "matched" have mb_genres

    print("\n" + "=" * 60)
    print("MusicBrainz Pilot Report")
    print("=" * 60)
    print(f"Total artists processed : {total}")
    print(f"Matched (score≥90)       : {matched}  ({matched/total*100:.1f}%)")
    print(f"  — with genres          : {with_genres}  ({with_genres/total*100:.1f}%)")
    print(f"  — no genres tagged     : {no_genres}  ({no_genres/total*100:.1f}%)")
    print(f"No match                 : {no_match}  ({no_match/total*100:.1f}%)")

    # Build lookup for dataset genres if source CSV available
    ds_genre: dict[str, str] = {}
    if source_csv and source_csv.exists():
        ds_df = pd.read_csv(source_csv, usecols=["artist_name", "genre"])
        ds_genre = (
            ds_df.groupby("artist_name")["genre"]
            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "")
            .to_dict()
        )

    # Priority artists (show first if present)
    priority = {"james taylor", "avail hollywood", "john mayall", "the lacs"}
    rows_by_name = {r["artist_name"].lower(): r for r in rows}

    table_rows: list[dict] = []
    for name in priority:
        if name in rows_by_name:
            table_rows.append(rows_by_name[name])

    # Fill up to 20 with other rows (prefer matched-with-genres, then others)
    others = [r for r in rows if r["artist_name"].lower() not in priority]
    with_g = [r for r in others if r["mb_genres"]]
    without_g = [r for r in others if not r["mb_genres"]]
    rng = random.Random(42)
    rng.shuffle(with_g)
    rng.shuffle(without_g)
    for r in with_g + without_g:
        if len(table_rows) >= 20:
            break
        table_rows.append(r)

    print(f"\n{'Artist':<30} {'Dataset Genre':<20} {'MB Genres':<40} Status")
    print("-" * 118)
    for r in table_rows:
        a = r["artist_name"]
        dg = ds_genre.get(a, ds_genre.get(a.lower(), "—"))
        mb = r["mb_genres"][:38] + ".." if len(r["mb_genres"]) > 40 else r["mb_genres"] or "—"
        print(f"{a:<30} {dg:<20} {mb:<40} {r['status']}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich artist genres from MusicBrainz")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true", help="Run on 100 artists (50 top + 50 random)")
    group.add_argument("--full", action="store_true", help="Run on all artists")
    group.add_argument("--report", action="store_true", help="Print report from existing CSV, no API calls")
    args = parser.parse_args()

    if args.report:
        report(SOURCE_CSV)
        return

    artists = select_artists(pilot=args.pilot)
    already_done = load_already_processed(OUTPUT_CSV)
    process(artists, already_done)

    if args.pilot:
        report(SOURCE_CSV)
        print("PILOT COMPLETE — review the report above, then approve Step B (--full run).")


if __name__ == "__main__":
    main()
