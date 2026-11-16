"""Publicação opcional em Hugging Face após treinamento e revisão da licença."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi

def publish(repo_id: str, model: Path, metrics: Path) -> str:
    if not model.is_file() or not metrics.is_file():
        raise FileNotFoundError("Treine e avalie o modelo antes de publicar.")
    stats = json.loads(metrics.read_text(encoding="utf-8"))
    card = f"""---
language: pt
library_name: scikit-learn
pipeline_tag: text-classification
license: cc-by-nc-sa-4.0
tags:
- tf-idf
- portuguese
- sentiment-analysis
---
# Review Sentiment BR

Classificador TF-IDF + Regressão Logística treinado em avaliações Olist.
**Não é um LLM e não foi treinado com dados do iFood.**

## Avaliação em holdout
- F1 macro: {stats['macro_f1']}
- Baseline F1 macro: {stats['baseline_macro_f1']}
- Accuracy: {stats['accuracy']}
- Amostras de teste: {stats['samples_test']}

## Dados e licença
[Olist — Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), CC BY-NC-SA 4.0. Verifique restrições antes de redistribuir artefatos derivados.

## Limitações
Estrelas são aproximações de sentimento. Domínio de e-commerce, sem validação para delivery. Arquivos joblib não devem ser carregados de fontes não confiáveis.

## Reproduzir
Consulte o repositório de código: https://github.com/evandronobregaads/nlp-review-sentiment-br
"""
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=model, path_in_repo="review_sentiment.joblib", repo_id=repo_id, repo_type="model")
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=repo_id, repo_type="model")
    return f"https://huggingface.co/{repo_id}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="usuario/nlp-review-sentiment-br")
    parser.add_argument("--model", type=Path, default=Path("models/review_sentiment.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics.json"))
    args = parser.parse_args()
    print(publish(args.repo, args.model, args.metrics))
