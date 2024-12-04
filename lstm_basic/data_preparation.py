import pandas as pd
import spotipy
import os
from spotipy.oauth2 import SpotifyOAuth

def data_preparation(sp, recently_played):

    track_metadata = []
    track_ids = []

    for obj in recently_played['items']:
        track = obj['track']
        track_ids.append(track['id'])
        audio_features = sp.audio_features([track['id']])[0]
        track_metadata.append({
            'track_id': track['id'],
            'track_name': track['name'],
            'artist': track['artists'][0]['name'], # just get the main artist
            'album': track['album']['name'],
            'popularity': track['popularity'],
            'played_at': obj['played_at'],
            'danceability': audio_features['danceability'],
            'energy': audio_features['energy'],
            'key': audio_features['key'],
            'loudness': audio_features['loudness'],
            'mode': audio_features['mode'],
            'speechiness': audio_features['speechiness'],
            'acousticness': audio_features['acousticness'],
            'instrumentalness': audio_features['instrumentalness'],
            'liveness': audio_features['liveness'],
            'valence': audio_features['valence'],
            'tempo': audio_features['tempo'],
            'time_signature': audio_features['time_signature']
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(track_metadata)
    print(df) # visualize
    df.to_csv('data.csv', index=False)
    return df

def main():
    sp_client = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id="[redacted]",
        client_secret="[redacted]",
        redirect_uri="http://localhost:8888/callback",
        scope=["user-read-recently-played user-top-read"]
    ))

    try:
        recently_played = sp_client.current_user_recently_played(limit=10)
        data_preparation(sp_client, recently_played)
    except Exception as e:
        print("Error:", e)


if __name__ == '__main__':
    main()
