# Credit Risk Intelligence Platform

## Objetivo

Plataforma analitica para avaliacao e monitoramento de risco de credito baseada em Machine Learning. A aplicacao consome tabelas analytics pre-computadas em Databricks, sem realizar treinamento, inferencia ou modificacao de dados. O objetivo e fornecer visualizacao interativa e consumo analitico dos resultados do pipeline de Credit Risk.

## Arquitetura

```text
Bronze (dados brutos)
  |
  v
Silver (qualidade e limpeza)
  |
  v
Gold (features para ML)
  |
  v
ML Dataset (treino e validacao)
  |
  v
EDA (analise exploratoria)
  |
  v
Training (6 modelos)
  |
  v
Evaluation (holdout estratificado 20%)
  |
  v
MLflow / Unity Catalog (registro de modelos)
  |
  v
Prediction (inferencia sobre 48.744 clientes)
  |
  v
Streamlit (esta aplicacao)
```

## Tecnologias

* Databricks
* Delta Lake
* Apache Spark / PySpark
* Python
* SQL
* Machine Learning (LightGBM, XGBoost, Scikit-learn)
* MLflow
* Unity Catalog
* Streamlit
* Plotly
* Databricks SQL Warehouse (conexao via databricks-sql-connector)

## Fontes de dados

A aplicacao consome exclusivamente as seguintes tabelas (apenas leitura):

| Tabela | Linhas | Conteudo |
| --- | --- | --- |
| `credit_risk.analytics.predictions` | 292.464 | Previsoes de 6 modelos sobre 48.744 clientes |
| `credit_risk.analytics.mlflow_model_registry` | 6 | Metadados dos modelos registrados no Unity Catalog |
| `credit_risk.analytics.evaluation_results` | 6 | Metricas de avaliacao (ROC-AUC, PR-AUC, Recall, etc.) |
| `credit_risk.analytics.audit_prediction` | 1 | Log de auditoria da execucao de predicao |

Nenhuma tabela e modificada, criada ou deletada pela aplicacao.

## Machine Learning

Seis modelos foram treinados e avaliados no pipeline:

1. Dummy (most_frequent) - baseline
2. Dummy (stratified) - baseline
3. Logistic Regression
4. Random Forest
5. XGBoost
6. LightGBM (campeao: ROC-AUC 0.7852)

O dataset de treino possui 307.511 registros com 229 features. A variavel TARGET tem 8.07% de classe minoritaria (ratio 1:11.39).

## MLflow

Todos os 6 modelos foram registrados no MLflow Model Registry e versionados no Unity Catalog sob o schema `credit_risk.analytics`. Cada modelo tem URI, Run ID e metricas documentadas.

## Databricks App

Esta aplicacao roda como um Databricks App utilizando Streamlit. A conexao com os dados e feita via Databricks SQL Warehouse usando `databricks-sql-connector`. A aplicacao nao utiliza `spark`, `dbutils` ou credenciais hardcoded.

### Variaveis de ambiente necessarias

| Variavel | Descricao |
| --- | --- |
| `DATABRICKS_HOST` | Hostname do workspace Databricks |
| `DATABRICKS_SQL_WAREHOUSE_PATH` | HTTP path do SQL Warehouse |
| `DATABRICKS_TOKEN` | Token de acesso (opcional em OAuth) |

Nao incluir PAT, senha, client_secret ou qualquer credencial no codigo.

## Como executar

### Databricks App (recomendado)

1. Criar um Databricks App com:
   - **Nome:** `credit-risk-intelligence`
   - **Entry point:** `app.py`
   - **Framework:** Streamlit
2. Configurar as variaveis de ambiente listadas acima
3. Configurar acesso ao SQL Warehouse
4. Fazer deploy
5. Acessar via navegador

### Execucao local

```bash
pip install -r requirements.txt
export DATABRICKS_HOST=<seu-workspace>
export DATABRICKS_SQL_WAREHOUSE_PATH=<sql-warehouse-path>
export DATABRICKS_TOKEN=<seu-token>
streamlit run app.py
```

## Estrutura dos arquivos

```text
credit-risk-intelligence/
  app.py            # Aplicacao Streamlit principal
  requirements.txt  # Dependencias Python
  README.md         # Esta documentacao
```

## Classificacao de risco

| Faixa | Condicao |
| --- | --- |
| LOW | probabilidade < 10% |
| MEDIUM | 10% <= probabilidade < 30% |
| HIGH | probabilidade >= 30% |

Essas faixas sao classificacoes operacionais utilizadas para visualizacao analitica e nao representam, isoladamente, uma decisao de concessao ou recusa de credito.

## Limitacoes

* A aplicacao nao treina modelos nem gera novas previsoes.
* A aplicacao nao modifica dados nas tabelas fonte.
* As classificacoes LOW/MEDIUM/HIGH sao operacionais e nao constituiem decisao de credito.
* A aplicacao depende da disponibilidade do SQL Warehouse.
* Os dados exibidos refletem o estado da ultima execucao do pipeline de predicao.