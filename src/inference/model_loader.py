from tensorflow.keras.models import load_model

def load_trained_model(path="models/sentiment_model.h5"):
    model = load_model(path)
    return model