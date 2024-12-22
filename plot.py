import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from tkinter import messagebox


# Constants
CHUNK = 4096
FORMAT = pyaudio.paInt16  # Audio format (16-bit per sample)
CHANNELS = 1  # Mono audio
SAMPLE_RATE = 44100  # (Hz)

# Global variables
stream = None
p = None
recording_active = False


def run_rt_freq(label, play_button, stop_button, root, line, canvas):
    global stream, p, recording_active, fill
    try:
        label.config(text="recording")
        play_button.pack_forget()
        stop_button.pack(pady=20)

        p = pyaudio.PyAudio()


        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        recording_active = True

        fill = line.axes.fill_between(line.get_xdata(), np.zeros(CHUNK // 2), color='cyan', alpha=0.5)

        def update_plot_wrapper():
            global fill
            if not recording_active or not root.winfo_exists():
                return
            fill = update_plot(stream, line, fill, canvas)
            root.after(10, update_plot_wrapper)

        update_plot_wrapper()

    except Exception as e:
        import traceback
        error_message = traceback.format_exc()
        messagebox.showerror("Error", f"Failed to run RT-freq.py: {e}\n\n{error_message}")

    pass


def stop_recording(label, play_button, stop_button, line, canvas):
    global recording_active
    recording_active = False
    reset_ui(label, play_button, stop_button, line, canvas)


def reset_ui(label, play_button, stop_button, line, canvas):
    global stream, p, fill
    # Reset UI elements
    label.config(text="recognize")
    play_button.pack(pady=20)
    stop_button.pack_forget()

    # Cleanup
    if stream is not None:
        stream.stop_stream()
        stream.close()
        stream = None
    if p is not None:
        p.terminate()
        p = None

    # Clear plot
    line.set_ydata(np.zeros(CHUNK // 2))
    fill.remove()
    fill = line.axes.fill_between(line.get_xdata(), np.zeros(CHUNK // 2), color='cyan', alpha=0.5)
    canvas.draw()


def create_plot():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    x = np.linspace(0, 22050, CHUNK // 2)  # Frequency range
    line, = ax.plot(x, np.zeros(CHUNK // 2), '-', lw=1, color='cyan')
    fill = ax.fill_between(x, np.zeros(CHUNK // 2), color='cyan', alpha=0.5)
    ax.set_xlim(0, 22000)  # Set lower limit to 0
    ax.set_ylim(-50, 50)

    # remove plot frame
    ax.set_title("Real-Time Audio Spectrum", color='black')
    ax.set_xlabel("Frequency (Hz)", color='black')
    ax.set_ylabel("Amplitude (dB)", color='black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    return fig, ax, line, fill


def update_plot(stream, line, fill, canvas,):
    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    fft_data = np.fft.rfft(data)
    fft_magnitude = np.abs(fft_data[:CHUNK // 2]) / (CHUNK // 2)
    fft_magnitude = 20 * np.log10(np.maximum(fft_magnitude, 1e-10))  # Convert to dB

    # Filter out negative values
    fft_magnitude[fft_magnitude < 0] = 0

    line.set_ydata(fft_magnitude)
    fill.remove()
    fill = line.axes.fill_between(line.get_xdata(), fft_magnitude, color='cyan', alpha=0.5)
    canvas.draw()

    return fill
