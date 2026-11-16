import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.predict import predict_text

def tiny_model():
    texts = [
        "Produto excelente, chegou perfeito", "Compra ótima e entrega rápida",
        "Adorei a qualidade, recomendo", "Tudo certo, fiquei satisfeito",
        "Produto ruim, veio quebrado", "Não gostei, compra péssima",
        "Demorou e chegou com defeito", "Muito ruim, não recomendo",
    ]
    labels = ["positivo"] * 4 + ["negativo"] * 4
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ]).fit(texts, labels)

def test_predict_structure():
    response = predict_text(tiny_model(), "Entrega excelente, gostei muito")
    assert response["sentiment"] in {"positivo", "negativo"}
    assert set(response["scores"]) == {"positivo", "negativo"}
    assert abs(sum(response["scores"].values()) - 1.0) < 0.001

def test_empty_text():
    with pytest.raises(ValueError):
        predict_text(tiny_model(), "  ")
