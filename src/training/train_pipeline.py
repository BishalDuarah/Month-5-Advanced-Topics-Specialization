import numpy as np
from src.models.lstm_model import build_model
import os

def load_data():
    X_train = np.load("data/processed/X_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")
    return X_train, X_test, y_train, y_test

def train():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    print("Building model...")
    model = build_model()

    print("Training model...")
    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_data=(X_test, y_test)
    )

    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")

    os.makedirs("models", exist_ok=True)
    model.save("models/sentiment_model.keras")

    print("Model saved successfully!")

if __name__ == "__main__":
    train()