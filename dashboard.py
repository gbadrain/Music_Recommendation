#!/usr/bin/env python
# coding: utf-8

import sys
import os
import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# In[3]:

print("Python executable being used:", sys.executable)
print("Current working directory:", os.getcwd())
print("Python sys.path:", sys.path)

# ------- Spotify API Authentication -------
CLIENT_ID = ""       # Replace with your Spotify client ID
CLIENT_SECRET = ""   # Replace with your Spotify client secret
REDIRECT_URI = "http://127.0.0.1:7777/callback"
SCOPE = "user-read-private user-modify-playback-state user-read-playback-state"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE
))

# ------- Streamlit Dashboard -------
st.title("Spotify Song Recommendations Dashboard (Static Data)")

# Load the recommended songs from the CSV file
df_recs = pd.DataFrame()  # Ensure df_recs is always defined
try:
    df_recs = pd.read_csv("Output/recommended_songs.csv")
except Exception as e:
    st.error("Error loading recommended_songs.csv. Make sure the file exists and re-run your notebook.")
    st.stop()

if df_recs.empty:
    st.write("No recommendations found. Please run the notebook to generate new recommendations.")
else:
    st.subheader("Recommended Songs")
    st.dataframe(df_recs)

    st.write("Play one of these songs:")
    # For each recommended song, display a Play button
    for idx, row in df_recs.iterrows():
        if st.button(f"Play '{row['track_name']}' by {row['artist_name']}", key=idx):
            # Use Spotify search to get the track URI
            track_results = sp.search(q=row["track_name"], type="track", limit=1)
            if track_results['tracks']['items']:
                track_uri = track_results['tracks']['items'][0]['uri']
                sp.start_playback(uris=[track_uri])
            else:
                st.error("Track not found on Spotify.")

    

