import sqlite3
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.io import wavfile
import joblib
import os


def load_data_from_db(db_file, feature_length=1024, chunk_duration=10, samplerate=44100):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('SELECT file_path, song_name FROM songs')
    data = cursor.fetchall()
    conn.close()

    labels = []
    features = []
    chunk_size = chunk_duration * samplerate

    for row in data:
        file_path, song_name = row
        samplerate, audio_data = wavfile.read(file_path)
        if len(audio_data.shape) == 2:
            audio_data = np.mean(audio_data, axis=1)

        num_chunks = len(audio_data) // chunk_size
        for i in range(num_chunks):
            chunk = audio_data[i * chunk_size:(i + 1) * chunk_size]
            feature = np.abs(np.fft.rfft(chunk)) / len(chunk)
            if len(feature) > feature_length:
                feature = feature[:feature_length]
            elif len(feature) < feature_length:
                feature = np.pad(feature, (0, feature_length - len(feature)), 'constant')
            features.append(feature)
            labels.append(song_name)

    return np.array(features), np.array(labels)


def train_model(features, labels):
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    return model, scaler


# Load from database
features, labels = load_data_from_db('songs.db')

# Train model
model, scaler = train_model(features, labels)

# Save trained model and scaler
joblib.dump(model, 'trainedModel.pkl')
joblib.dump(scaler, 'scaler.pkl')