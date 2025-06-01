"""
This script demonstrates a workflow for training a neural network on the Iris dataset
(loaded from a CSV) that mirrors the structure of the “bottleneck features + top model” 
example used for image classification.

Directory structure expected:
    data/
        iris1.csv

The script has two main functions:
1. save_preprocessed_features(): loads “data/iris1.csv”, encodes labels, splits into
   train/validation sets, scales features, and saves NumPy arrays to disk.
2. train_iris_model(): loads those saved .npy files, builds a small Keras MLP, trains it,
   logs metrics to “metrics.csv”, and saves model weights to “iris_model.weights.h5”.
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger
from tqdm.keras import TqdmCallback

# Path setup
pathname = os.path.dirname(sys.argv[0])
base_path = os.path.abspath(pathname)

# Filenames and parameters
iris_csv_path = os.path.join(base_path, "data", "iris1.csv")
# Number of validation samples will be determined by split ratio
validation_split = 0.20  # 20% for validation
random_state = 42

# Where to save preprocessed feature arrays
train_features_path      = "iris_features_train.npy"
train_labels_path        = "iris_labels_train.npy"
validation_features_path = "iris_features_validation.npy"
validation_labels_path   = "iris_labels_validation.npy"

# Training parameters
epochs = 20
batch_size = 16

# Where to save final model weights
iris_model_weights_path = "iris_model.weights.h5"


def save_preprocessed_features():
    """
    1. Load iris1.csv from data/ directory
    2. Separate features (X) and target (y)
    3. Encode string labels to integers 0,1,2
    4. Split into train and validation sets (stratified)
    5. Scale features with StandardScaler
    6. Save NumPy arrays for features and encoded labels to disk
    """
    # 1. Load the CSV
    df = pd.read_csv(iris_csv_path)
    
    # 2. Separate features and target
    #    Assuming columns: sepal_length, sepal_width, petal_length, petal_width, species
    X = df.iloc[:, :-1].values   # shape (n_samples, 4)
    y = df.iloc[:, -1].values    # shape (n_samples,)
    
    # 3. Encode string labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # e.g. 'Iris-setosa' -> 0, etc.
    
    # 4. Stratified train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_encoded,
        test_size=validation_split,
        random_state=random_state,
        stratify=y_encoded
    )
    
    # 5. Scale features (fit scaler on training, apply to both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    
    # 6. Save to disk as .npy
    np.save(open(train_features_path, "wb"), X_train_scaled)
    np.save(open(train_labels_path,   "wb"), y_train)
    np.save(open(validation_features_path, "wb"), X_val_scaled)
    np.save(open(validation_labels_path,   "wb"), y_val)
    
    print(f"Saved training features ({X_train_scaled.shape}) to '{train_features_path}'")
    print(f"Saved training labels   ({y_train.shape}) to '{train_labels_path}'")
    print(f"Saved validation features ({X_val_scaled.shape}) to '{validation_features_path}'")
    print(f"Saved validation labels   ({y_val.shape}) to '{validation_labels_path}'")


def train_iris_model():
    """
    1. Load preprocessed train/validation features and labels from .npy
    2. Convert integer labels to one-hot vectors (3 classes)
    3. Build a simple MLP with Keras:
       - Input layer matching 4 features
       - Dense(64) + Dropout
       - Dense(32) + Dropout
       - Dense(3, softmax)
    4. Compile with categorical_crossentropy, Adam optimizer
    5. Fit model, logging metrics to metrics.csv and showing progress with TqdmCallback
    6. Save trained weights to 'iris_model.weights.h5'
    """
    # 1. Load features and labels
    X_train = np.load(open(train_features_path, "rb"))
    y_train_int = np.load(open(train_labels_path, "rb"))
    X_val   = np.load(open(validation_features_path, "rb"))
    y_val_int   = np.load(open(validation_labels_path,   "rb"))
    
    # 2. One-hot encode integer labels (0,1,2 → [1,0,0], [0,1,0], [0,0,1])
    num_classes = 3
    y_train = to_categorical(y_train_int, num_classes)
    y_val   = to_categorical(y_val_int,   num_classes)
    
    # 3. Build MLP model
    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    
    # 4. Compile
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # 5. Fit model with CSVLogger and TqdmCallback
    csv_logger = CSVLogger("metrics.csv")
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=0,
        callbacks=[TqdmCallback(), csv_logger]
    )
    
    # 6. Save weights
    model.save_weights(iris_model_weights_path)
    print(f"Model trained for {epochs} epochs. Weights saved to '{iris_model_weights_path}'.")
    print("Training history logged to 'metrics.csv'.")


# Execute steps in sequence
save_preprocessed_features()
train_iris_model()
