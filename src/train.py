"""Treinamento reproduzível para comentários de avaliações Olist.
Notas 1-2 = negativo; notas 4-5 = positivo; 3 excluída.
Rótulos derivados de estrelas são aproximações de sentimento.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def load_reviews(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}. Baixe o CSV da Olist no Kaggle.")
    df = pd.read_csv(path, usecols=["review_score", "review_comment_message"])
    df = df.dropna(subset=["review_score", "review_comment_message"]).copy()
    df["text"] = df["review_comment_message"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 8].copy()
    df = df[df["review_score"].isin([1, 2, 4, 5])].copy()
    df["label"] = df["review_score"].map(lambda x: "negativo" if x <= 2 else "positivo")
    df["dedup_key"] = df["text"].str.casefold().str.replace(r"\s+", " ", regex=True)
    df = df.drop_duplicates(subset=["dedup_key"])
    if len(df) < 20 or df["label"].nunique() < 2 or df["label"].value_counts().min() < 2:
        raise ValueError("Dataset insuficiente para divisão estratificada. Verifique os dados.")
    return df[["text", "label"]]

def train(data_path: Path, model_path: Path, report_dir: Path) -> dict:
    df = load_reviews(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train.to_frame(), y_train)
    base_pred = baseline.predict(x_test.to_frame())
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = {
        "dataset": "Olist Brazilian E-Commerce Public Dataset",
        "source": "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
        "split": "80/20 estratificado, seed 42",
        "samples_total": int(len(df)),
        "samples_train": int(len(x_train)),
        "samples_test": int(len(x_test)),
        "class_distribution": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "baseline_macro_f1": round(float(f1_score(y_test, base_pred, average="macro")), 4),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=["negativo", "positivo"]).tolist(),
        "labels": ["negativo", "positivo"],
        "limitations": "Estrelas são rótulos aproximados; não mede delivery ou dados do iFood."
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    (report_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "classification_report.txt").write_text(
        classification_report(y_test, predictions, zero_division=0), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw/olist_order_reviews_dataset.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/review_sentiment.joblib"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()
    train(args.data, args.model, args.reports)
