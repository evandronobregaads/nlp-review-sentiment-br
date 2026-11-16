"""Inferência local: carregue arquivos joblib apenas de origem confiável."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib

DEFAULT_MODEL_PATH = Path("models/review_sentiment.joblib")

def predict_text(model, text: str) -> dict:
    if not text.strip():
        raise ValueError("O texto não pode estar vazio.")
    label = str(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]
    scores = {str(name): round(float(prob), 4) for name, prob in zip(model.classes_, probabilities)}
    return {"text": text, "sentiment": label, "scores": scores}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Modelo ausente: {args.model}. Execute o treinamento primeiro.")
    print(predict_text(joblib.load(args.model), args.text))
