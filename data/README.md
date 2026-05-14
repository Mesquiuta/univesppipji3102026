# Data

Diretório de dados locais do projeto.

## Subpastas

- `raw/` — entrada bruta original (CSV/parquet). O arquivo `transactions.csv` é o **input do pipeline**.
- `interim/` — saídas intermediárias da etapa de limpeza (`transactions_clean.csv`).
- `processed/` — dados prontos para modelagem (`transactions_processed.csv`) com features adicionais.
- `external/` — espaço opcional para dados externos (ex.: tabelas de produto/categoria).

## Dataset incluído

Este repositório vem com um **dataset sintético reproduzível** em `data/raw/transactions.csv`
(cópia de `data/raw/transactions_sinteticas_univesp.csv`) com as seguintes características:

| Atributo | Valor |
|----------|-------|
| Período | 2024-01-01 → 2025-12-31 |
| Linhas | ~5.897 transações |
| Clientes únicos | 528 |
| Pedidos únicos | 3.600 |
| Produtos únicos | 211 |
| Receita bruta total | ~R$ 263.945 |

### Por que sintético?

- Não há acesso a dados reais de e-commerce no escopo acadêmico do Projeto Integrador.
- O dataset sintético foi desenhado para gerar resultados **realistas** em EDA,
  clustering e recomendação:
  - múltiplos comportamentos de cliente (clientes esporádicos, recorrentes, "VIPs"),
  - múltiplos níveis de preço,
  - sazonalidade leve por mês,
  - cauda longa de produtos (poucos best-sellers, muitos com baixa frequência).

### Schema esperado

| Coluna | Tipo | Obrigatória | Descrição |
|--------|------|-------------|-----------|
| `customer_id` | string | sim | Identificador único do cliente |
| `order_id` | string | sim | Identificador do pedido |
| `product_id` | string | sim | Identificador do produto |
| `order_date` | datetime | sim | Data/hora do pedido |
| `quantity` | int | sim | Quantidade comprada (>0) |
| `unit_price` | float | sim | Preço unitário (>=0) |
| `total_value` | float | opcional | Se ausente, recalculado como `quantity × unit_price` |

### Substituindo pelo seu dataset

Basta sobrescrever `data/raw/transactions.csv` mantendo o schema acima, ou apontar
para outro nome de arquivo via JSON de configuração:

```bash
python main.py --pipeline all --config minha_config.json
```

```json
{ "data": { "source_filename": "meu_dataset.csv" } }
```

## Limitações

- Como os dados são sintéticos, as métricas refletem a distribuição amostrada
  e não devem ser generalizadas para um e-commerce real.
- O recomendador pode apresentar baixa precision@k pelo perfil quase uniforme da
  cauda de produtos — comportamento esperado em catálogos com alta diversidade.
