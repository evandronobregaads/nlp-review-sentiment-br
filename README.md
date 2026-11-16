# Review Sentiment BR — NLP e Machine Learning

[![Python Tests](https://github.com/evandronobregaads/nlp-review-sentiment-br/actions/workflows/ci.yml/badge.svg)](https://github.com/evandronobregaads/nlp-review-sentiment-br/actions/workflows/ci.yml)

Classificador educacional de **sentimento em avaliações brasileiras** com **Python, pandas, scikit-learn, TF-IDF, regressão logística, FastAPI e testes automatizados**.

> **Status:** código-base publicado e testes locais aprovados. **Treinamento e métricas sobre dados reais ainda pendentes.** Não é um modelo LLM nem utiliza dados privados ou internos do iFood.

## Problema e arquitetura

O objetivo é prever se uma avaliação textual é positiva ou negativa a partir de um conjunto público de avaliações de e-commerce brasileiro.

```text
CSV Olist → limpeza e deduplicação → split estratificado (80/20)
          → TF-IDF (1-2 gramas) → regressão logística → métricas
                                                    → joblib → FastAPI
```

## Dados

- Fonte: [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- Arquivo necessário: `olist_order_reviews_dataset.csv`
- Licença dos dados: **CC BY-NC-SA 4.0**, com atribuição e restrições de uso não comercial.
- `review_score` 1–2: **negativo**; 4–5: **positivo**; 3: excluído.
- Os rótulos são **aproximações derivadas das estrelas**, e não anotações independentes de sentimento.
- Textos ausentes, curtos e duplicados são removidos antes de separar treino e teste.
- Não faça upload da base bruta nem de dados pessoais no GitHub.

## Rodar localmente (Windows PowerShell)

```powershell
git clone https://github.com/evandronobregaads/nlp-review-sentiment-br.git
cd nlp-review-sentiment-br
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

Obtenha o CSV público usando `python -m scripts.download_olist` (espelho da Olist); se preferir, baixe diretamente no Kaggle e coloque em `data/raw/olist_order_reviews_dataset.csv`. Depois:

```powershell
python -m src.train
python -m src.predict "Produto excelente, chegou muito rápido!"
uvicorn src.api:app --reload
```

Abra `http://127.0.0.1:8000/docs` para testar o endpoint `POST /predict` com:

```json
{"text":"Produto ótimo, gostei muito da entrega"}
```

Antes de treinar, a API retorna HTTP 503 para inferência, pois o artefato ainda não existe.

## Avaliação e resultados

O script salva `reports/metrics.json` com tamanhos da amostra, distribuição das classes, **acurácia, F1 macro, F1 macro do baseline de classe majoritária e matriz de confusão**. Nenhum valor de desempenho em dados reais foi calculado ou publicado ainda.

| Experimento | F1 macro | Status |
| --- | --- | --- |
| Baseline de classe majoritária | — | A executar |
| TF-IDF + regressão logística | — | A executar |

## Arquivos importantes

| Caminho | Conteúdo |
| --- | --- |
| `src/train.py` | Preparação, treino e avaliação |
| `src/predict.py` | Inferência |
| `src/api.py` | API HTTP |
| `src/publish_hf.py` | Publicação opcional após treino |
| `tests/test_predict.py` | Testes com dados sintéticos |
| `article/relatorio-tecnico.md` | Relato metodológico |
| `.github/workflows/ci.yml` | Testes no GitHub Actions |

## Treinamento automático no GitHub Actions

Abra [Train Olist sentiment model](https://github.com/evandronobregaads/nlp-review-sentiment-br/actions/workflows/train-model.yml) para visualizar a execução. O fluxo baixa os comentários públicos, treina o classificador e publica **artefatos da execução** (modelo e relatórios) no Actions. Arquivos gerados não são enviados automaticamente ao código-fonte. Confira a execução e baixe os artefatos para auditoria.\n\n## Hugging Face (depois de treinar)

Após criar o modelo e os resultados, revise a licença dos artefatos e autentique-se com `hf auth login`. Use:

```powershell
python -m src.publish_hf --repo SEU_USUARIO/nlp-review-sentiment-br
```

Não publique tokens, senhas ou arquivos `.env` no repositório.

## Limitações

O score de estrelas nem sempre corresponde ao sentimento expresso no texto; exemplos podem incluir ironia, negação e avaliações mistas. A divisão aleatória não mede drift temporal. Dados da Olist são de e-commerce e não representam automaticamente delivery. Carregue artefatos `.joblib` apenas de fontes confiáveis.

## Autoria e transparência

Projeto de estudo e portfólio de **Evandro Nóbrega**, estudante de Análise e Desenvolvimento de Sistemas. Estrutura e implementação inicial elaboradas com assistência de IA; treinamento, avaliação em dados reais e decisões de melhoria devem ser documentados conforme realizados. O código não é afiliado ao iFood.
