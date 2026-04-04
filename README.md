# 🚀 Production-Ready NLP Sentiment Analysis System

## 📌 Overview

This project is a **production-grade Natural Language Processing (NLP) system** that performs **sentiment analysis** on text data using a deep learning model.

It is built with a complete **end-to-end ML pipeline**, including:

* Data preprocessing
* Model training (BiLSTM)
* Inference pipeline
* REST API deployment
* Docker containerization
* Monitoring & metrics tracking

---

## 🎯 Key Features

* 🧠 Deep Learning Model (Bidirectional LSTM)
* ⚡ FastAPI-based REST API
* 📦 Dockerized deployment
* 📊 Real-time prediction with confidence score
* 🔁 Batch prediction support
* 📈 Monitoring (latency, request count)
* 🧩 Modular & scalable architecture

---

## 🧠 Model Architecture

* Embedding Layer (10,000 vocab, 64-dim)
* Bidirectional LSTM (64 units)
* Dropout Layer (0.5)
* LSTM Layer (32 units)
* Dense Layer (ReLU)
* Output Layer (Softmax)

---

## 📊 Performance Metrics

| Metric    | Value |
| --------- | ----- |
| Accuracy  | ~87%  |
| Precision | ~0.87 |
| Recall    | ~0.86 |
| F1 Score  | ~0.86 |

---

## 📁 Project Structure

```
advanced-nlp-sentiment-system/

├── data/
├── notebooks/
├── src/
│   ├── data_processing/
│   ├── models/
│   ├── training/
│   ├── inference/
│   ├── monitoring/
│
├── api/
├── tests/
├── scripts/
├── docs/
│
├── models/
├── Dockerfile
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/advanced-nlp-sentiment-system.git
cd advanced-nlp-sentiment-system
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run API Locally

```bash
uvicorn api.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 🧪 API Endpoints

### 🔹 POST `/predict`

Single text prediction

```json
{
  "text": "This movie was amazing!"
}
```

---

### 🔹 POST `/batch_predict`

Batch predictions

```json
{
  "texts": ["Great product!", "Worst experience ever"]
}
```

---

### 🔹 GET `/health`

Check system status

---

### 🔹 GET `/metrics`

View performance metrics

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t sentiment-api .
```

### Run Container

```bash
docker run -p 8000:8000 sentiment-api
```

---

## 📊 Monitoring

Tracks:

* Request count
* Average latency
* Prediction performance

---

## 💡 Business Use Cases

* Customer feedback analysis
* Product review classification
* Social media sentiment tracking
* Support ticket prioritization

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* FastAPI
* Docker
* Scikit-learn
* NLTK

---

## 🚀 Future Improvements

* Add Transformer models (BERT)
* Deploy on cloud (AWS/GCP)
* Add real-time dashboard (Grafana)
* CI/CD pipeline integration

---

## 👨‍💻 Author

**Bishal Duarah**
B.Tech CSE | Data Science & AI Enthusiast

---

## ⭐ Show Your Support

If you like this project, please ⭐ the repository!
