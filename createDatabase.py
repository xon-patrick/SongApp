import sqlite3
import os
import numpy as np
from scipy.io import wavfile


def create_database(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS songs (
            id INTEGER PRIMARY KEY,
            file_path TEXT UNIQUE,
            song_name TEXT,
            artist TEXT,
            features BLOB,
            image BLOB
        )
    ''')
    conn.commit()
    conn.close()


def extract_features(audio_data, samplerate, feature_length=1024):
    feature = np.abs(np.fft.rfft(audio_data)) / len(audio_data)
    if len(feature) > feature_length:
        feature = feature[:feature_length]
    elif len(feature) < feature_length:
        feature = np.pad(feature, (0, feature_length - len(feature)), 'constant')
    return feature


def load_image(image_path):
    with open(image_path, 'rb') as f:
        return f.read()


def insert_song_data(db_file, directory):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    for file_name in os.listdir(directory):
        if file_name.endswith('.wav'):
            base_name = os.path.splitext(file_name)[0]
            file_path = os.path.join(directory, file_name)
            image_path = os.path.join(directory, f"{base_name}.png")

            if not os.path.exists(image_path):
                print(f"Image for {file_name} not found, skipping.")
                continue

            cursor.execute('SELECT 1 FROM songs WHERE file_path = ?', (file_path,))
            if cursor.fetchone():
                print(f"Song {file_name} already exists in the database, skipping.")
                continue

            # Extract song name and artist name if present
            if '-' in base_name:
                song_name, artist = base_name.split('-', 1)
                song_name = song_name.strip()
                artist = artist.strip()
            else:
                song_name = base_name
                artist = None

            samplerate, audio_data = wavfile.read(file_path)
            if len(audio_data.shape) == 2:
                audio_data = np.mean(audio_data, axis=1)
            features = extract_features(audio_data, samplerate)
            image = load_image(image_path)

            cursor.execute('INSERT INTO songs (file_path, song_name, artist, features, image) VALUES (?, ?, ?, ?, ?)',
                           (file_path, song_name, artist, features.tobytes(), image))

    conn.commit()
    conn.close()


# Example usage
create_database('songs.db')
insert_song_data('songs.db', 'test/Songs')
