import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import joblib
import numpy as np
from scipy.io import wavfile
import sqlite3
import io

def extract_features_from_wav(file_path, feature_length=1024, chunk_duration=10, samplerate=44100):
    samplerate, audio_data = wavfile.read(file_path)
    if len(audio_data.shape) == 2:
        audio_data = np.mean(audio_data, axis=1)

    chunk_size = chunk_duration * samplerate
    chunk = audio_data[:chunk_size]
    feature = np.abs(np.fft.rfft(chunk)) / len(chunk)
    if len(feature) > feature_length:
        feature = feature[:feature_length]
    elif len(feature) < feature_length:
        feature = np.pad(feature, (0, feature_length - len(feature)), 'constant')

    return feature

def get_song_details_from_db(song_name):
    conn = sqlite3.connect('songs.db')
    cursor = conn.cursor()
    cursor.execute("SELECT artist, image FROM songs WHERE song_name = ?", (song_name,))
    result = cursor.fetchone()
    conn.close()
    if result:
        artist_name, image_data = result
        image = Image.open(io.BytesIO(image_data))
        return artist_name, image
    return None, None

# Load the trained model and scaler
model = joblib.load('trainedModel.pkl')
scaler = joblib.load('scaler.pkl')

def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        try:
            features = extract_features_from_wav(file_path)
            features = scaler.transform([features])
            predicted_song = model.predict(features)
            song_name = predicted_song[0]

            # Get artist name and image from the database
            artist_name, image = get_song_details_from_db(song_name)
            if artist_name and image:
                result_label.config(text=f"Song: {song_name}\nArtist: {artist_name}")

                # Load and display the image
                image = image.resize((200, 200), Image.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                image_label.config(image=photo)
                image_label.image = photo
            else:
                result_label.config(text=f"Song: {song_name}\nArtist: Not found")
                image_label.config(image='')
                image_label.image = None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process the file: {e}")

# Main window
root = tk.Tk()
root.title("Audio Recognizer")
root.geometry("600x600")
root.configure(bg='black')

# Title
label = tk.Label(root, text="Audio Recognizer", font=("Helvetica", 32, "bold"), fg="white", bg="black")
label.pack(pady=20)

# Load file button
load_button = tk.Button(root, text="Load WAV File", font=("Helvetica", 20, "bold"), bg="white", fg="black", command=load_file)
load_button.pack(pady=20)

# Result label
result_label = tk.Label(root, text="", font=("Helvetica", 20), fg="white", bg="black")
result_label.pack(pady=20)

# Image label
image_label = tk.Label(root, bg="black")
image_label.pack(pady=20)

# Run the application
root.mainloop()