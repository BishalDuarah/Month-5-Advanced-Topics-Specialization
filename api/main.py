from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import time

from src.inference.predict import predict_sentiment

app = FastAPI(title="Sentiment Analysis API")

# Monitoring variables
request_count = 0
total_latency = 0


class ReviewInput(BaseModel):
    text: str


class BatchInput(BaseModel):
    texts: List[str]


@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running"}


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/predict")
def predict(data: ReviewInput):
    global request_count, total_latency

    start_time = time.time()

    result = predict_sentiment(data.text)

    latency = time.time() - start_time

    request_count += 1
    total_latency += latency

    return {
        "input": data.text,
        "prediction": result,
        "latency": round(latency, 4)
    }


@app.post("/batch_predict")
def batch_predict(data: BatchInput):
    results = [predict_sentiment(text) for text in data.texts]

    return {
        "predictions": results
    }


@app.get("/metrics")
def metrics():
    avg_latency = total_latency / request_count if request_count else 0

    return {
        "requests_served": request_count,
        "average_latency": round(avg_latency, 4)
    }