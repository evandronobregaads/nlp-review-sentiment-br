"""Obtém apenas a tabela pública de avaliações da Olist para treino local.

Origem canônica (CC BY-NC-SA 4.0):
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Espelho GitHub usado para download sem login:
https://github.com/infobarbosa/dataset-olist
Não versionar o CSV no repositório.
"""
from __future__ import annotations
import argparse
import gzip
import shutil
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://raw.githubusercontent.com/infobarbosa/dataset-olist/main/data/olist_order_reviews_dataset.csv.gz"
OUTPUT = Path("data/raw/olist_order_reviews_dataset.csv")


def download(output: Path = OUTPUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        compressed = Path(temporary) / "reviews.csv.gz"
        request = Request(URL, headers={"User-Agent": "ReviewSentimentBR/0.1 (educational)"})
        with urlopen(request, timeout=90) as response, compressed.open("wb") as target:
            shutil.copyfileobj(response, target)
        with gzip.open(compressed, "rb") as source, output.open("wb") as target:
            shutil.copyfileobj(source, target)
    with output.open("rb") as handle:
        header = handle.readline().decode("utf-8-sig")
    if "review_score" not in header or "review_comment_message" not in header:
        output.unlink(missing_ok=True)
        raise ValueError("O arquivo baixado não contém as colunas Olist esperadas.")
    print(f"CSV pronto em {output} ({output.stat().st_size:,} bytes)")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    download(args.output)
