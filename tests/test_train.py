"""Pipeline integration tests using synthetic text; not performance claims."""
import json
from pathlib import Path
import pandas as pd
import joblib
from src.train import load_reviews, train

def _make_csv(path: Path):
    positive = [f"produto otimo entrega excelente {i} gostei muito" for i in range(25)]
    negative = [f"produto pessimo entrega ruim {i} nao gostei" for i in range(25)]
    pd.DataFrame({
        "review_score": [5] * len(positive) + [1] * len(negative),
        "review_comment_message": positive + negative,
    }).to_csv(path, index=False)

def test_preparation_and_training(tmp_path):
    csv = tmp_path / "reviews.csv"
    _make_csv(csv)
    data = load_reviews(csv)
    assert len(data) == 50
    assert set(data.label) == {"positivo", "negativo"}
    model_path = tmp_path / "model.joblib"
    report_dir = tmp_path / "reports"
    metrics = train(csv, model_path, report_dir)
    assert metrics["samples_total"] == 50
    assert metrics["samples_train"] == 40
    assert metrics["samples_test"] == 10
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert (report_dir / "metrics.json").exists()
    assert "positivo" in joblib.load(model_path).classes_
    stored = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    assert stored["macro_f1"] == metrics["macro_f1"]

def test_deduplicates_comments(tmp_path):
    csv = tmp_path / "dedup.csv"
    positive = [f"excelente produto numero {i}" for i in range(20)]
    negative = [f"pessimo produto numero {i}" for i in range(20)]
    pd.DataFrame({
        "review_score": [5] * 20 + [1] * 20 + [5, 1],
        "review_comment_message": positive + negative + [positive[0].upper(), negative[0]],
    }).to_csv(csv, index=False)
    assert len(load_reviews(csv)) == 40
