# Sistema Analítico de E-commerce — UNIVESP PI

Projeto Integrador (UNIVESP) que entrega um pipeline completo de **ciência de dados aplicada a um e-commerce**:
ingestão e limpeza dos dados transacionais, EDA, KPIs de negócio, segmentação de clientes via **RFM + K-Means**,
**recomendação híbrida** (filtragem colaborativa item-item + fallback de popularidade e recência) e
**avaliação técnica** com split temporal holdout — tudo acompanhado por um **frontend Streamlit** para
demonstração e exploração interativa dos resultados.

## Visão geral

| Camada | Tecnologias | Função |
|--------|-------------|--------|
| Pipeline | Python, pandas, scikit-learn, matplotlib, seaborn | Ingestão, EDA, modelagem, avaliação |
| Persistência | CSV, JSON, joblib | Dados intermediários/processados, relatórios e modelos |
| Frontend | Streamlit, Plotly | Dashboard navegável (KPIs, gráficos, clusters, recomendações) |
| Testes | pytest | Cobertura mínima de unidade + end-to-end |

## Estrutura do repositório

```text
project_root/
  app/                            <- Frontend Streamlit
    Home.py                       <- Visão geral, status, KPIs principais
    pages/
      1_Dataset.py                <- Preview, tipos, faltantes, estatísticas
      2_EDA.py                    <- KPIs, tendências, tops, retenção, RFM
      3_Segmentacao.py            <- Clusters RFM + K-Means, diagnóstico de k
      4_Recomendacoes.py          <- Recomendações por cliente, histórico
      5_Avaliacao.py              <- Métricas técnicas e de negócio
      6_Artefatos.py              <- Explorador de arquivos gerados
    utils.py                      <- Helpers compartilhados
  data/
    raw/transactions.csv          <- Dataset transacional (origem)
    interim/                      <- Limpeza intermediária
    processed/                    <- Dataset final com features
    external/                     <- (opcional) dados externos
  src/                            <- Backend analítico
    config/                       <- paths e settings
    ingestion/                    <- loader CSV/parquet
    preprocessing/                <- validation, cleaning, features
    analytics/                    <- EDA, KPIs, funil/retenção
    segmentation/                 <- RFM, K-Means, rotulagem
    recommendation/               <- matriz user-item, CF, hybrid
    evaluation/                   <- métricas técnicas e de negócio
    visualization/                <- plots matplotlib/seaborn
    reporting/                    <- exportadores CSV/JSON
    pipelines/                    <- run_preprocessing / eda / segmentation / recommendation / evaluation / all
    utils/                        <- logging e IO
  tests/                          <- pytest (unit + e2e)
  models/                         <- modelos persistidos (joblib)
  outputs/
    figures/                      <- PNGs
    tables/                       <- CSVs analíticos
    reports/                      <- JSON com sumários e métricas
    predictions/                  <- recomendações por cliente
  main.py                         <- CLI principal
  requirements.txt
  install.bat / run_pipeline.bat / run_frontend.bat / run_all.bat / run_tests.bat
```

## Pré-requisitos

- **Python 3.10+** (testado em 3.11.9 no Windows 11).
- **pip** com acesso à PyPI.
- ~150 MB livres em disco para o ambiente + dados processados.

## Dataset

O projeto espera o arquivo `data/raw/transactions.csv` com as seguintes colunas:

| Coluna | Tipo | Obrigatória | Descrição |
|--------|------|-------------|-----------|
| `customer_id` | string | sim | Identificador único do cliente |
| `order_id` | string | sim | Identificador do pedido |
| `product_id` | string | sim | Identificador do produto |
| `order_date` | datetime | sim | Data/hora do pedido |
| `quantity` | int | sim | Quantidade comprada (>0) |
| `unit_price` | float | sim | Preço unitário (>=0) |
| `total_value` | float | opcional | Se ausente, será calculado como `quantity × unit_price` |

### Dataset incluído

O repositório já vem com um **dataset sintético reproduzível** (`data/raw/transactions.csv`,
~5.897 linhas, 528 clientes, 211 produtos, jan/2024 → dez/2025) projetado para que EDA, clustering
e recomendação produzam resultados ricos. Ele também está versionado em
`data/raw/transactions_sinteticas_univesp.csv`. Veja [`data/README.md`](data/README.md) para
detalhes da geração.

Para usar **seus próprios dados**, basta substituir `data/raw/transactions.csv` mantendo o schema
acima (ou apontar para um arquivo customizado via JSON de configuração — ver final do README).

## Instalação

### Opção rápida (Windows)

```bat
install.bat
```

### Manual

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Em Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Como rodar

### Caminho mais simples (Windows, tudo de uma vez)

```bat
run_all.bat
```

Esse script roda o pipeline completo e em seguida abre o frontend.

### Pipeline analítico apenas

```bat
run_pipeline.bat
```

Equivalente a:

```bash
python main.py --pipeline all
```

Pipelines individuais:

```bash
python main.py --pipeline preprocessing
python main.py --pipeline eda
python main.py --pipeline segmentation
python main.py --pipeline recommendation
python main.py --pipeline evaluation
```

### Frontend Streamlit

```bat
run_frontend.bat
```

Equivalente a:

```bash
python -m streamlit run app/Home.py
```

A URL padrão é http://localhost:8501. O dashboard tem 7 páginas:

| Página | Conteúdo |
|--------|----------|
| **Home** | Status do pipeline + KPIs principais + botão para disparar o pipeline |
| **Dataset** | Preview do CSV, tipos, faltantes e estatísticas descritivas |
| **EDA** | KPIs, receita mensal, ticket médio, top produtos/clientes, funil de retenção, distribuição RFM, outliers |
| **Segmentação** | Diagnóstico de k (silhouette/inércia), perfis de cluster, visualização 3D RFM |
| **Recomendações** | Seletor por cliente, top-N com scores, histórico de compras, fallback de popularidade |
| **Avaliação** | Métricas de clustering, precision/recall/MAP@k, cobertura, personalização |
| **Artefatos** | Explorador dos arquivos gerados, com download direto |

### Testes

```bat
run_tests.bat
```

Equivalente a:

```bash
python -m pytest -q
```

## Saídas geradas

Após `python main.py --pipeline all`, os seguintes diretórios são populados:

- `data/interim/transactions_clean.csv` — dataset limpo
- `data/processed/transactions_processed.csv` — dataset com features e `total_value` consolidado
- `outputs/tables/*.csv` — eda_describe, revenue_by_month, avg_ticket_by_month, top_products, top_customers, rfm_distribution, rfm_clusters, cluster_summary, item_similarity, product_popularity, outlier_summary, repeat_purchase_funnel, purchase_frequency_distribution, orders_per_customer, k_selection_diagnostics
- `outputs/figures/*.png` — revenue_over_time, cluster_distribution
- `outputs/predictions/recommendations.csv|json` — top-N por cliente
- `outputs/reports/*.json` — eda_summary, kpis, retention_proxy_metrics, revenue_concentration, k_selection_summary, evaluation_report, holdout_recommendation_eval, run_all_report
- `models/kmeans_rfm.joblib` e `models/rfm_scaler.joblib` — artefatos do K-Means treinado

## Interpretação dos principais resultados

- **Clustering** usa **Silhouette + Davies-Bouldin + Calinski-Harabasz**. A seleção de **k**
  é automática via maior silhouette no range `[2, 8]`.
- **Recomendação** é híbrida: combinação ponderada (`alpha = 0.7`) entre similaridade
  colaborativa item-item (cosseno) e fallback (popularidade × recência).
- **Avaliação** do recomendador usa **holdout temporal** — o último pedido de cada cliente
  com 2+ pedidos vira ground-truth; o sistema é treinado nos pedidos anteriores e
  precision/recall/MAP@k são medidos contra esse holdout.
- **Rotulagem semântica dos clusters** segue regras explícitas baseadas em quantis de
  recência/frequência/monetário (`loyal_high_value`, `at_risk`, `frequent_buyers`,
  `big_spenders`, `occasional_low_value`, `regular`).

## Configuração avançada (opcional)

É possível sobrescrever colunas, parâmetros do modelo, nome do arquivo de entrada e nível de log
por meio de um JSON:

```bash
python main.py --pipeline all --config path/to/config.json
```

Exemplo:

```json
{
  "data": {"source_filename": "meu_dataset.csv"},
  "model": {"n_clusters": 5, "auto_select_k": false, "recommendation_top_n": 10},
  "log_level": "DEBUG"
}
```

## Licença / créditos

Trabalho acadêmico — Projeto Integrador UNIVESP.
