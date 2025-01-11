# Audio processing
import pickle
import sqlite3
import io
import threading
import numpy as np
from scipy.io import wavfile
# UX / UI
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
# Real time audio animation
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the trained model
with open('trainedModel.pkl', 'rb') as f:
    model = pickle.load(f)

### Constants for real-time audio visualization ###
CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 44100

stream = None
p = None
recording_active = True


### Functions for audio processing and recognition ###
def preprocess_audio(file_path, feature_length=1024):
    samplerate, data = wavfile.read(file_path)
    if len(data.shape) == 2:
        data = np.mean(data, axis=1)
    n = len(data)
    fft_features = np.abs(np.fft.rfft(data)) / n
    if len(fft_features) > feature_length:
        fft_features = fft_features[:feature_length]
    elif len(fft_features) < feature_length:
        fft_features = np.pad(fft_features, (0, feature_length - len(fft_features)), 'constant')
    return fft_features


def get_song_details(prediction):
    conn = sqlite3.connect('songs.db')
    cursor = conn.cursor()
    cursor.execute('SELECT song_name, artist, image FROM songs WHERE song_name = ?', (prediction,))
    result = cursor.fetchone()
    conn.close()
    return result


def process_file(file_path):
    try:
        features = preprocess_audio(file_path)
        prediction = model.predict([features])[0]
        song_details = get_song_details(prediction)
        if song_details:
            song_name, artist, image_data = song_details
            image = Image.open(io.BytesIO(image_data))
            image = image.resize((300, 300), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            cover_label.config(image=photo)
            cover_label.image = photo
            result_label.config(text=f"♫ {song_name}\n♪ by {artist}", font=("Arial", 20, "bold"), fg="#FFD700")
        else:
            result_label.config(text="\u26A0 Song not found in the database.")
            cover_label.config(image='')
            cover_label.image = None
    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        stop_loading_animation()


def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        start_loading_animation()
        threading.Thread(target=process_file, args=(file_path,)).start()


### Loading "Animation" ###
def start_loading_animation():
    loading_label.config(text="Processing...", font=("Arial", 18, "italic"), fg="#FFCC00")
    loading_label.pack(pady=10)


def stop_loading_animation():
    loading_label.pack_forget()


### Credit screen ###
def toggle_credits():
    if credits_frame.winfo_ismapped():
        credits_frame.pack_forget()
        load_button.pack(pady=20)
        cover_label.pack(pady=20)
        result_label.pack(pady=10)
        credits_button.config(text="Credits")
        visualizer_canvas.get_tk_widget().lower(root)  # Ensure visualizer is behind after returning
    else:
        for widget in root.winfo_children():
            if widget not in {credits_frame, credits_button, visualizer_canvas.get_tk_widget()}:
                widget.pack_forget()
        credits_button.config(text="Back")
        credits_frame.pack(expand=True, fill="both")


### Real-time audio visualizer ###
def create_visualizer():
    global p, stream, fill, visualizer_canvas
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    fig, ax = plt.subplots(figsize=(15, 1))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margins
    fig.patch.set_alpha(0)  # Transparent background
    x = np.linspace(0, SAMPLE_RATE // 2, CHUNK // 2)
    line, = ax.plot(x, np.zeros(CHUNK // 2), '-', lw=2, color='#FFCC00')
    fill = ax.fill_between(x, np.zeros(CHUNK // 2), color='#FF5733', alpha=0.6)
    ax.set_xlim(0, SAMPLE_RATE // 2)
    ax.set_ylim(0, 50)
    ax.axis('off')

    visualizer_canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = visualizer_canvas.get_tk_widget()
    canvas_widget.configure(bg='#1F1F1F')
    canvas_widget.pack(side="bottom", fill="x")
    canvas_widget.lower(root)

    def update_visualizer():
        global fill
        if not recording_active or not root.winfo_exists():
            return
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        fft_data = np.fft.rfft(data)
        fft_magnitude = np.abs(fft_data[:CHUNK // 2]) / (CHUNK // 2)
        fft_magnitude = 20 * np.log10(np.maximum(fft_magnitude, 1e-10))
        fft_magnitude[fft_magnitude < 0] = 0

        line.set_ydata(fft_magnitude)
        fill.remove()
        fill = ax.fill_between(x, fft_magnitude, color='#FF5733', alpha=0.6)
        visualizer_canvas.draw()
        root.after(10, update_visualizer)

    update_visualizer()


def on_closing():
    global recording_active, stream, p
    recording_active = False
    if stream is not None:
        stream.stop_stream()
        stream.close()
    if p is not None:
        p.terminate()
    root.destroy()


### Front Panel ###
root = tk.Tk()
root.title("\u266B Fourier Transform Audio Recognizer \u266B")
root.geometry("700x700")
root.configure(bg="#1F1F1F")

load_button = tk.Button(root, text="Select WAV File\u266A", font=("Arial", 20, "bold"), bg="#FF5733", fg="#FFFFFF", activebackground="#C70039", activeforeground="#FFFFFF", relief="flat", command=load_file)
load_button.pack(pady=20)

cover_label = tk.Label(root, bg="#1F1F1F")
cover_label.pack(pady=20)

result_label = tk.Label(root, text="", font=("Arial", 22, "italic"), fg="#F5F5F5", bg="#1F1F1F", justify="center")
result_label.pack(pady=10)

loading_label = tk.Label(root, text="", font=("Arial", 16), fg="#F5F5F5", bg="#1F1F1F")


credits_frame = tk.Frame(root, bg="#1F1F1F")
credits_title = tk.Label(credits_frame, text="Credits", font=("Arial", 28, "bold"), fg="#FFD700", bg="#1F1F1F")
credits_title.pack(pady=30)

credits_container = tk.Frame(credits_frame, bg="#1F1F1F")
credits_container.pack(pady=20)

credits_details = [
    ("Andrei Patrick", "Database, Trained the MLM, UX and UI"),
    ("Caldararu Denisa", "Presentation"),
    ("Deacu Octavian", "Audio Processing"),
    ("Dumitru Claudia", "Documentation"),
]

for i, (name, role) in enumerate(credits_details):
    name_label = tk.Label(credits_container, text=name, font=("Arial", 18, "bold"), fg="#FFFFFF", bg="#1F1F1F")
    role_label = tk.Label(credits_container, text=role, font=("Arial", 16), fg="#B0B0B0", bg="#1F1F1F")
    name_label.grid(row=i, column=0, sticky="w", padx=40, pady=5)
    role_label.grid(row=i, column=1, sticky="w", padx=20, pady=5)

credits_button = tk.Button(root, text="Credits", font=("Arial", 14), bg="#333333", fg="#FFFFFF", activebackground="#444444", activeforeground="#FFFFFF", command=toggle_credits)
credits_button.place(relx=0.95, rely=0.05, anchor="ne")

create_visualizer()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
