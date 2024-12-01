import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


def wav_to_frequency_image(wav_file):

    base_name = os.path.splitext(os.path.basename(wav_file))[0]
    output_image = f"frequency_spectrum_{base_name}.png"

    # read WAV file
    samplerate, data = wavfile.read(wav_file)

    # convert to mono if stereo
    if len(data.shape) == 2:  # Stereo
        data = np.mean(data, axis=1)

    # FFT - i feel like i messed up here?
    n = len(data)
    freqs = np.fft.rfftfreq(n, 1 / samplerate)  # positive frequencies
    fft_magnitude = np.abs(np.fft.rfft(data)) / n  # normalize -> amplitude

    # plot frequency spectrum
    plt.figure(figsize=(10, 10))
    plt.plot(freqs, fft_magnitude, color='blue')
    plt.title(f"Frequency Spectrum of {base_name}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # image saving
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Frequency spectrum image saved to {output_image}")


def wav_to_spectrogram_image(wav_file):
    base_name = os.path.splitext(os.path.basename(wav_file))[0]
    output_image = f"spectrogram_{base_name}.png"

    samplerate, data = wavfile.read(wav_file)

    if len(data.shape) == 2:
        data = np.mean(data, axis=1)

    # spectrogram
    freqs, times, sxx = spectrogram(data, fs=samplerate, nperseg=1024)

    # Add a small epsilon to avoid log10 issues
    sxx = np.maximum(sxx, 1e-10)  # Ensures no zero values in Sxx

    plt.figure(figsize=(10, 10))
    plt.pcolormesh(times, freqs, 10 * np.log10(sxx), shading='gouraud', cmap='viridis')
    # 10 * np.log10(sxx) transform to dB
    plt.title(f"Spectrogram of {base_name}")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Intensity (dB)")

    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Spectrogram image saved to {output_image}")


# Example usage
wav_to_frequency_image("No Hero - NCS - Copyright Free Music.wav")
wav_to_spectrogram_image("No Hero - NCS - Copyright Free Music.wav")

