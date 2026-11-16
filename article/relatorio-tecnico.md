# Review Sentiment BR — relatório de experimento de ML

**Autor:** Evandro Nóbrega  
**Data da execução:** 16/11/2026  
**Execução auditável:** https://github.com/evandronobregaads/nlp-review-sentiment-br/actions/runs/43939951435

## Pergunta de pesquisa aplicada

Uma pipeline simples de TF-IDF com regressão logística consegue superar um classificador trivial (classe majoritária) na polaridade de avaliações textuais em português?

## Dados e metodologia

Foi utilizado o **Olist Brazilian E-Commerce Public Dataset**, atribuído à Olist e publicado no Kaggle sob licença **CC BY-NC-SA 4.0**. Foram usadas as colunas `review_score` e `review_comment_message`. Mensagens sem texto, muito curtas e repetidas foram eliminadas antes da divisão. Notas 1–2 foram mapeadas como **negativo**; 4–5 como **positivo**. A nota 3 foi descartada por ambiguidade. São **rótulos fracos**, inferidos das notas, e não rotulagem manual independente.

O conjunto resultante tinha **31.610 avaliações**: 21.345 positivas e 10.265 negativas. Treino/teste: **25.288/6.322**, divisão estratificada de 80/20 com `random_state=42`.

Pipeline: TF-IDF com unigramas e bigramas (`max_features=20000`, `min_df=2`, `sublinear_tf=True`) seguido de `LogisticRegression` (`class_weight='balanced'`, `max_iter=1000`). Referência: `DummyClassifier(strategy='most_frequent')`.

## Resultados observados no holdout

| Indicador | Resultado |
| --- | ---: |
| F1 macro — baseline | **0,4031** |
| F1 macro — TF-IDF + regressão logística | **0,9076** |
| Acurácia — modelo | **91,63%** |
| Avaliações de teste | **6.322** |

Matriz de confusão, ordenada como negativo/positivo (linhas reais; colunas previstas):

```text
                  Prev. negativo   Prev. positivo
Real negativo           1927              126
Real positivo            403             3866
```

A melhoria do F1 macro frente à classe majoritária demonstra ganho substancial sobre essa referência simples. **Não significa generalização comprovada** para dados de delivery, para outros períodos ou para avaliações anotadas manualmente.

## Reprodutibilidade

```bash
python -m pip install -r requirements.txt
python -m scripts.download_olist
python -m src.train
python -m pytest -q
```

As métricas em JSON foram preservadas em `reports/metrics.json` com link para a execução que as originou. O artefato `review_sentiment.joblib` foi salvo no GitHub Actions como `sentiment-model-and-metrics` (retenção de 30 dias). A API FastAPI realiza inferência após carregar o arquivo local.

## Limitações e próximos experimentos

1. **Rótulos aproximados:** nota de estrelas pode não corresponder ao sentimento do texto.
2. **Possível otimismo:** divisão aleatória não simula mudança temporal nem outro domínio.
3. **E-commerce não é delivery:** não há dados internos do iFood e não houve validação sobre o domínio de entregas.
4. **Futuro:** análise de erros, avaliação temporal, curva de precisão/recall e comparação com modelos transformer em português.
5. **Segurança:** modelos serializados por joblib não devem ser carregados de fontes desconhecidas.

## Transparência da autoria

Projeto de estudo com assistência de IA na implementação inicial e na documentação. O treinamento e os valores de desempenho foram produzidos por uma execução verificável em CI. Este material não é afiliado ao iFood e não alega participação em competição Kaggle ou publicação no Hugging Face.
