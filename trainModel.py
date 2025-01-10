import sqlite3
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.ndimage import gaussian_filter1d
import joblib


def load_data_from_db(db_file, feature_length=1024):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('SELECT song_name, features FROM songs')
    data = cursor.fetchall()
    conn.close()

    labels = []
    features = []
    for row in data:
        labels.append(row[0])
        feature = np.frombuffer(row[1], dtype=np.float64)
        if len(feature) > feature_length:
            feature = feature[:feature_length]
        elif len(feature) < feature_length:
            feature = np.pad(feature, (0, feature_length - len(feature)), 'constant')
        features.append(feature)

    return np.array(features), np.array(labels)


def augment_data(features):
    augmented_features = []
    for feature in features:
        # Add Gaussian noise
        noise = np.random.normal(0, 0.01, feature.shape)
        augmented_features.append(feature + noise)

        # Apply Gaussian filter
        smoothed = gaussian_filter1d(feature, sigma=1)
        augmented_features.append(smoothed)

        # Original feature
        augmented_features.append(feature)

    return np.array(augmented_features)


def train_model(features, labels):
    augmented_features = augment_data(features)
    augmented_labels = np.repeat(labels, 3)  # Repeat labels to match augmented data
    X_train, X_test, y_train, y_test = train_test_split(augmented_features, augmented_labels, test_size=0.2,
                                                        random_state=42)
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    return model


# Load from database
features, labels = load_data_from_db('songs.db')

# Train model
model = train_model(features, labels)

# Save trained model
joblib.dump(model, 'trainedModel.pkl')