import numpy as np
from src.data_processing.preprocess import clean_text
from src.data_processing.tokenizer import TextTokenizer
from src.inference.model_loader import load_trained_model

# Load once
model = load_trained_model()
tokenizer = TextTokenizer.load("models/tokenizer.pkl")

def predict_sentiment(text: str):
    cleaned = clean_text(text)

    encoded = tokenizer.transform([text])

    prediction = model.predict(encoded)[0]

    sentiment = np.argmax(prediction)
    confidence = float(np.max(prediction))

    label_map = {
        0: "Negative",
        1: "Positive"
    }

    return {
        "sentiment": label_map[sentiment],
        "confidence": round(confidence, 3)
    }