import joblib
import numpy as np
from scipy.io import wavfile


def preprocess_audio(file_path, feature_length=1024):
    samplerate, data = wavfile.read(file_path)
    if len(data.shape) == 2:  # Convert to mono if stereo
        data = np.mean(data, axis=1)
    n = len(data)
    fft_features = np.abs(np.fft.rfft(data)) / n
    if len(fft_features) > feature_length:
        fft_features = fft_features[:feature_length]
    elif len(fft_features) < feature_length:
        fft_features = np.pad(fft_features, (0, feature_length - len(fft_features)), 'constant')
    return fft_features


# Load the trained model
model = joblib.load('trainedModel.pkl')

# Preprocess the new audio data
new_audio_features = preprocess_audio('test.wav')

# Make a prediction
prediction = model.predict([new_audio_features])
print(f"Recognized as: {prediction[0]}")
