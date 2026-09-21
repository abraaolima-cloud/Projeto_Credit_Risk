# Projeto_Credit_Risk
PROJETO DE CLASSIFICAÇÃO DE RISCO DE CRÉDITO
# 🏦 Credit Risk Intelligence Platform

> Plataforma end-to-end de **Engenharia de Dados, Machine Learning e MLOps** para classificação e previsão de risco de inadimplência, desenvolvida com **Databricks, Delta Lake, PySpark, MLflow e Python**.

## 📌 Sobre o projeto

O **Credit Risk Intelligence Platform** tem como objetivo construir um pipeline completo para processamento de dados financeiros e desenvolvimento de um modelo de **classificação de risco de crédito**.

O projeto simula um cenário real de dados, desde a ingestão de fontes brutas até a disponibilização das previsões de Machine Learning para análise e tomada de decisão.

A solução combina:

* Engenharia de Dados;
* Arquitetura Medalhão;
* Data Quality;
* Feature Engineering;
* Machine Learning;
* MLOps;
* CI/CD;
* Monitoramento;
* Business Intelligence;
* Explainable AI.

---

## 🎯 Objetivo

Desenvolver uma solução capaz de classificar clientes de acordo com seu risco de inadimplência e, ao mesmo tempo, demonstrar uma arquitetura moderna de dados e Machine Learning executada no **Databricks**.

O pipeline será responsável por:

1. Ingerir dados de diferentes fontes;
2. Armazenar os dados na camada Bronze;
3. Realizar tratamento e validação na camada Silver;
4. Construir datasets analíticos e features na camada Gold;
5. Treinar diferentes modelos de classificação;
6. Registrar experimentos e métricas;
7. Selecionar e versionar modelos;
8. Disponibilizar previsões;
9. Monitorar qualidade dos dados e desempenho do modelo;
10. Disponibilizar informações para análise através de dashboards e aplicação web.

---

# 🏗️ Arquitetura

```text
                         FONTES DE DADOS
                ┌────────────┼────────────┐
                │            │            │
              CSV          API       SQL Server
                │            │            │
                └────────────┼────────────┘
                             ↓
                     ┌───────────────┐
                     │    BRONZE     │
                     │   Raw Data    │
                     │   Delta Lake  │
                     └───────┬───────┘
                             ↓
                     Data Quality
                             ↓
                     ┌───────────────┐
                     │    SILVER     │
                     │ Cleaned Data  │
                     │  Validated    │
                     └───────┬───────┘
                             ↓
                    Feature Engineering
                             ↓
                     ┌───────────────┐
                     │     GOLD      │
                     │ ML Features   │
                     │  Aggregations │
                     └───────┬───────┘
                             ↓
                    ┌────────────────┐
                    │   ML Pipeline  │
                    └───────┬────────┘
                            ↓
                ┌──────────────────────┐
                │       MLflow         │
                │ Experiment Tracking  │
                └──────────┬───────────┘
                           ↓
                Model Evaluation
                           ↓
                Model Registry
                           ↓
             ┌─────────────┴─────────────┐
             ↓                           ↓
       Batch Prediction            API Prediction
             ↓                           ↓
             └─────────────┬─────────────┘
                           ↓
                  Monitoring & Logs
                           ↓
              ┌────────────┴────────────┐
              ↓                         ↓
          Power BI                  Streamlit
```

---

# 🥉 Arquitetura Medalhão

## Bronze

Responsável pelo armazenamento dos dados em seu formato bruto.

Principais características:

* Dados provenientes das fontes originais;
* Preservação dos dados recebidos;
* Controle de ingestão;
* Timestamp de processamento;
* Identificação da origem;
* Armazenamento em Delta Lake.

Exemplos de metadados:

```text
source
ingestion_timestamp
batch_id
processing_date
```

---

## 🥈 Silver

Camada responsável pela preparação e qualidade dos dados.

Processos:

* Tratamento de valores nulos;
* Remoção de duplicidades;
* Padronização de tipos;
* Validação de dados;
* Tratamento de inconsistências;
* Tratamento de outliers;
* Regras de negócio;
* Normalização de categorias.

Exemplo de dados:

```text
customer_id
age
income
loan_amount
loan_term
credit_score
credit_utilization
previous_loans
payment_delays
default
```

---

## 🥇 Gold

Camada preparada para consumo analítico e Machine Learning.

Nesta etapa serão criadas as principais features utilizadas pelos modelos.

Exemplos:

```text
debt_to_income
loan_to_income
credit_utilization
payment_delay_rate
average_payment_delay
number_previous_loans
number_delinquencies
```

Dataset principal:

```text
gold_credit_risk_features
```

Variável alvo:

```text
default
```

Onde:

```text
0 = Não inadimplente
1 = Inadimplente
```

---

# 🤖 Machine Learning

Serão desenvolvidos e comparados diferentes modelos de classificação.

### Modelos

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM

### Métricas

A avaliação não será baseada apenas em Accuracy.

Serão analisadas:

* Precision;
* Recall;
* F1-score;
* ROC-AUC;
* PR-AUC;
* Confusion Matrix.

A escolha do modelo será baseada nas métricas e nos objetivos definidos para o problema.

---

# 🔬 MLflow

O **MLflow** será utilizado para gerenciamento dos experimentos de Machine Learning.

Serão registrados:

* parâmetros;
* métricas;
* versões dos modelos;
* artefatos;
* resultados dos experimentos.

Exemplo:

```text
Experiment
│
├── Logistic Regression
├── Random Forest
├── XGBoost
└── LightGBM
```

O modelo selecionado será posteriormente disponibilizado através do **Model Registry**.

---

# 🧪 Data Quality

O pipeline contará com validações para evitar que dados inconsistentes avancem entre as camadas.

Exemplos:

```text
✓ Schema validation
✓ NULL validation
✓ Duplicate detection
✓ Data type validation
✓ Range validation
✓ Referential integrity
✓ Business rules
✓ Class distribution monitoring
```

Indicadores de qualidade serão acompanhados durante o processamento.

---

# ⚙️ Lakeflow

Os pipelines de dados serão estruturados utilizando **Lakeflow Declarative Pipelines**, organizando o fluxo entre:

```text
Bronze
   ↓
Silver
   ↓
Gold
```

As regras de qualidade serão incorporadas ao pipeline para permitir validação contínua dos dados.

---

# 🔍 Explainable AI

Para interpretação das previsões será utilizado **SHAP**.

O objetivo é identificar quais características tiveram maior influência na previsão do modelo.

Exemplo:

```text
Principais fatores associados à previsão:

↑ Histórico de atrasos
↑ Utilização de crédito
↑ Relação dívida/renda
↓ Score de crédito
```

Isso permite analisar o comportamento do modelo além de simplesmente apresentar sua classificação.

---

# 📊 Dashboard

Será desenvolvido um dashboard para análise dos resultados.

Indicadores planejados:

* Taxa de inadimplência;
* Distribuição de risco;
* Risco por faixa de renda;
* Risco por faixa etária;
* Evolução temporal;
* Distribuição dos scores;
* Principais características associadas ao risco;
* Performance do modelo.

**Tecnologia:** Power BI

---

# 🖥️ Aplicação Web

Também será desenvolvida uma aplicação utilizando **Streamlit** para simulação de risco.

O usuário poderá informar características de um cliente:

```text
Renda: R$ 5.500
Valor do empréstimo: R$ 20.000
Prazo: 24 meses
Score: 620
Atrasos anteriores: 2
```

A aplicação retornará:

```text
Probabilidade estimada: 73%

Classificação:
ALTO RISCO
```

Além da classificação, serão apresentados os principais fatores que contribuíram para a previsão.

---

# 📡 Model Monitoring

Após o treinamento, serão simulados novos dados para acompanhamento do modelo.

Serão monitorados:

* Data Drift;
* Feature Drift;
* Prediction Drift;
* Mudança na distribuição das classes;
* Performance do modelo;
* Qualidade dos dados.

O objetivo é identificar mudanças no comportamento dos dados que possam afetar a qualidade das previsões.

---

# 🔄 CI/CD

O projeto utilizará **GitHub Actions** para automatizar etapas do ciclo de desenvolvimento.

Pipeline planejado:

```text
Git Push
   ↓
Pull Request
   ↓
Tests
   ↓
Data Quality Tests
   ↓
ML Tests
   ↓
Build
   ↓
Deploy
   ↓
Databricks
```

---

# 📁 Estrutura do projeto

```text
credit-risk-intelligence/
│
├── data/
│   ├── raw/
│   └── sample/
│
├── notebooks/
│   ├── 01_ingestion/
│   ├── 02_bronze/
│   ├── 03_silver/
│   ├── 04_gold/
│   ├── 05_eda/
│   ├── 06_feature_engineering/
│   ├── 07_training/
│   ├── 08_evaluation/
│   └── 09_prediction/
│
├── src/
│   ├── ingestion/
│   ├── transformations/
│   ├── features/
│   ├── models/
│   ├── monitoring/
│   └── utils/
│
├── tests/
│   ├── data/
│   ├── features/
│   └── models/
│
├── pipelines/
│   ├── bronze/
│   ├── silver/
│   └── gold/
│
├── dashboard/
│
├── streamlit/
│
├── .github/
│   └── workflows/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🛠️ Tecnologias

### Data Engineering

* Databricks
* Delta Lake
* PySpark
* SQL
* Lakeflow

### Data Science

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM

### Machine Learning / MLOps

* MLflow
* Model Registry
* SHAP
* Model Monitoring

### Analytics

* Power BI
* Streamlit

### DevOps

* Git
* GitHub
* GitHub Actions

### Infraestrutura

* Terraform *(planejado)*

---

# 🚀 Roadmap

* [ ] Definição do dataset
* [ ] Ingestão dos dados
* [ ] Implementação da camada Bronze
* [ ] Implementação da camada Silver
* [ ] Implementação da camada Gold
* [ ] Data Quality
* [ ] Feature Engineering
* [ ] EDA
* [ ] Treinamento dos modelos
* [ ] Comparação dos modelos
* [ ] MLflow
* [ ] Model Registry
* [ ] Explainable AI com SHAP
* [ ] Pipeline de previsão
* [ ] Model Monitoring
* [ ] Dashboard Power BI
* [ ] Aplicação Streamlit
* [ ] Testes automatizados
* [ ] CI/CD com GitHub Actions
* [ ] Deploy no Databricks
* [ ] Infraestrutura com Terraform

---

# 📚 Conceitos demonstrados

Este projeto demonstra conhecimentos em:

**Data Engineering → Data Quality → Data Modeling → Feature Engineering → Machine Learning → Experiment Tracking → MLOps → Model Monitoring → BI → CI/CD**

---

# 👨‍💻 Autor

**Abraão J.**

Analista de Software | Data & AI

Interesses: **Data Engineering, Data Science, Machine Learning, IA e automação.**
