import spotipy
import json
from spotipy.oauth2 import SpotifyClientCredentials
import pprint
import requests
from spotipy.oauth2 import SpotifyOAuth
scope = "user-library-read"
spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope, client_id="13d6abc247f5427599cf6e02af93cb9b", client_secret="580e1111ea114f5e92ca0c9e61743254", redirect_uri="http://127.0.0.1:8080/login"))
# pprint.pp(, indent=4)
track = spotify.playlist('1kX96U66J2VgnUE3eSB4R2')['tracks']['items']


ids = [x["track"].get("id") for x in track]
result = ",".join(ids)

url = f"https://api.reccobeats.com/v1/audio-features"

payload = {}
headers = {
  'Accept': 'application/json'
}

all_audio_features = []
chunk_size = 40
for i in range(0, len(ids), chunk_size):
    chunk = ids[i:i+chunk_size]
    result = ",".join(chunk)
    response = requests.get(url, headers=headers, params={"ids": result})
    if response.status_code == 200:
        data = response.json()

        # ReccoBeats typically returns a dictionary with a 'content' list
        if isinstance(data, dict) and "content" in data:
            all_audio_features.extend(data["content"])
        elif isinstance(data, list):
            all_audio_features.extend(data)
    else:
        print(f"Batch {i // chunk_size + 1} failed with status {response.status_code}: {response.text}")

with open('track_data.json', 'w') as f:
    json.dump({"content": all_audio_features}, f, indent=2)
tracks = [
    {
        "title" : x["track"]["name"],
        "artist" : ", ".join(artist["name"] for artist in x["track"]["artists"]),
        "duration" :x["track"]["duration_ms"],
        "isrc" : x["track"].get("external_ids", {}).get("isrc"),
        "track_id" : x["track"].get("id")
    }
    for x in track if x.get("track")
]

tracks_json = json.dumps(tracks, indent=2, ensure_ascii=False)
with open('tracks.json', 'w') as f:
    f.write(tracks_json)