"""API HTTP do classificador. Inicie com uvicorn src.api:app --reload."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .predict import predict_text

app = FastAPI(title="Review Sentiment BR", version="0.1.0")

class TextRequest(BaseModel):
    text: str = Field(min_length=3, max_length=5000)

@lru_cache(maxsize=1)
def get_model():
    path = Path(os.environ.get("MODEL_PATH", "models/review_sentiment.joblib"))
    if not path.is_file():
        raise FileNotFoundError(f"Modelo ausente: {path}. Execute o treinamento primeiro.")
    return joblib.load(path)

@app.get("/health")
def health():
    path = Path(os.environ.get("MODEL_PATH", "models/review_sentiment.joblib"))
    return {"status": "ok", "model_available": path.is_file()}

@app.post("/predict")
def predict(request: TextRequest):
    try:
        return predict_text(get_model(), request.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
