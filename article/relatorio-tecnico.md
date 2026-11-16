# Relatório técnico — Review Sentiment BR

## Questão
É possível prever a polaridade de avaliações textuais em português usando um classificador linear e vetorização TF-IDF?

## Dataset e pré-processamento
Dados públicos Olist (licença CC BY-NC-SA 4.0). Usamos `review_comment_message` e `review_score`, descartamos mensagens ausentes, notas 3 e textos curtos. Notas 1–2 são mapeadas para negativo e 4–5 para positivo. Deduplicação do texto antes de dividir os conjuntos.

## Método
Divisão estratificada 80/20, `random_state=42`. Baseline de classe majoritária versus TF-IDF com unigramas e bigramas e regressão logística balanceada.

## Métricas e resultados
**Ainda não medidos em dados reais.** Após treinar, consulte `reports/metrics.json` para acurácia, F1 macro, baseline e matriz de confusão.

## Limitações e próximos experimentos
Os rótulos são inferidos de notas (weak supervision), não anotados separadamente. O sistema não foi validado em pedidos de delivery, textos irônicos, mudanças temporais ou ambientes produtivos. Futuro: análise de erros, separação temporal e comparação com modelos transformer em português.

## Transparência
Repositório educacional, sem vínculo com o iFood; não apresenta métricas estimadas como observadas.
