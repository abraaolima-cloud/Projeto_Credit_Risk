# ============================================================
# CREDIT RISK INTELLIGENCE - DATABRICKS APP
# ============================================================
# Aplicacao Streamlit independente para visualizacao e consumo
# analitico do projeto Credit Risk Intelligence Platform.
# NAO utiliza spark, dbutils ou credenciais hardcoded.
# Acesso a dados via Databricks SQL Warehouse.
# ============================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from databricks import sql

# ============================================================
# CONFIGURACAO DA PAGINA
# ============================================================

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONSTANTES - TABELAS FONTE (APENAS LEITURA)
# ============================================================

PREDICTIONS_TABLE = "credit_risk.analytics.predictions"
REGISTRY_TABLE = "credit_risk.analytics.mlflow_model_registry"
EVALUATION_TABLE = "credit_risk.analytics.evaluation_results"
AUDIT_PREDICTION_TABLE = "credit_risk.analytics.audit_prediction"

# ============================================================
# CONEXAO COM DATABRICKS SQL WAREHOUSE
# ============================================================

@st.cache_resource
def get_connection():
    """
    Conecta ao Databricks SQL Warehouse usando variaveis de ambiente.
    Nao utiliza credenciais hardcoded.

    Variaveis necessarias:
      DATABRICKS_HOST              - hostname do workspace
      DATABRICKS_SQL_WAREHOUSE_PATH - HTTP path do SQL Warehouse
      DATABRICKS_TOKEN              - token de acesso (opcional em OAuth)
    """
    host = os.environ.get("DATABRICKS_HOST", "").replace("https://", "").replace("http://", "")
    http_path = os.environ.get("DATABRICKS_SQL_WAREHOUSE_PATH", "/sql/1.0/warehouses/3910e69a48ceb3f8")
    token = os.environ.get("DATABRICKS_TOKEN", "")

    if not host:
        st.error("DATABRICKS_HOST nao configurado. Defina a variavel de ambiente no app.")
        return None
    if not http_path:
        st.error("DATABRICKS_SQL_WAREHOUSE_PATH nao configurado. Defina a variavel de ambiente no app.")
        return None

    try:
        if token:
            return sql.connect(
                server_hostname=host,
                http_path=http_path,
                access_token=token
            )
        else:
            return sql.connect(
                server_hostname=host,
                http_path=http_path
            )
    except Exception as e:
        st.error(f"Erro ao conectar ao SQL Warehouse: {str(e)}")
        return None


def execute_query(query, params=None):
    """Executa consulta SQL e retorna DataFrame pandas."""
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        with conn.cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            result = cursor.fetchall_arrow()
            return result.to_pandas()
    except Exception as e:
        st.error(f"Erro na consulta: {str(e)}")
        return pd.DataFrame()


# ============================================================
# FUNCOES DE CARREGAMENTO DE DADOS (COM CACHE)
# ============================================================

@st.cache_data(ttl=3600, show_spinner="Carregando KPIs...")
def load_kpis():
    """Carrega KPIs principais via agregacao SQL (nao carrega 292K linhas)."""
    query = f"""
    SELECT
        COUNT(DISTINCT SK_ID_CURR) as unique_customers,
        COUNT(DISTINCT MODEL_TYPE) as unique_models,
        COUNT(*) as total_predictions,
        AVG(RISK_PROBABILITY) as avg_probability
    FROM {PREDICTIONS_TABLE}
    """
    return execute_query(query)


@st.cache_data(ttl=3600, show_spinner="Carregando distribuicao...")
def load_risk_distribution():
    """Carrega distribuicao de risco via SQL GROUP BY."""
    query = f"""
    SELECT RISK_CATEGORY, COUNT(*) as count
    FROM {PREDICTIONS_TABLE}
    GROUP BY RISK_CATEGORY
    ORDER BY
        CASE RISK_CATEGORY
            WHEN 'LOW' THEN 0
            WHEN 'MEDIUM' THEN 1
            WHEN 'HIGH' THEN 2
            ELSE 3
        END
    """
    return execute_query(query)


@st.cache_data(ttl=3600, show_spinner="Carregando modelos...")
def load_model_list():
    """Carrega lista de modelos disponiveis."""
    query = f"""
    SELECT DISTINCT MODEL_TYPE
    FROM {PREDICTIONS_TABLE}
    ORDER BY MODEL_TYPE
    """
    df = execute_query(query)
    if not df.empty:
        return df["MODEL_TYPE"].tolist()
    return []


@st.cache_data(ttl=3600, show_spinner="Carregando model registry...")
def load_model_registry():
    """Carrega metadados do MLflow Model Registry (6 linhas)."""
    query = f"""
    SELECT
        model_name as MODEL_NAME,
        model_type as MODEL_TYPE,
        model_version as MODEL_VERSION,
        run_id as RUN_ID,
        model_uri as MODEL_URI,
        roc_auc as ROC_AUC,
        pr_auc as PR_AUC,
        status as STATUS,
        experiment_id as EXPERIMENT_ID
    FROM {REGISTRY_TABLE}
    ORDER BY model_type
    """
    return execute_query(query)


@st.cache_data(ttl=3600, show_spinner="Carregando avaliacao...")
def load_evaluation():
    """Carrega metricas de avaliacao dos modelos (6 linhas)."""
    query = f"""
    SELECT
        MODEL, ROC_AUC, PR_AUC, PRECISION, RECALL, F1,
        SPECIFICITY, BALANCED_ACCURACY, BRIER_SCORE
    FROM {EVALUATION_TABLE}
    ORDER BY MODEL
    """
    return execute_query(query)


@st.cache_data(ttl=3600, show_spinner="Carregando auditoria...")
def load_audit():
    """Carrega log de auditoria de predicao (1 linha)."""
    query = f"""
    SELECT
        execution_id, execution_timestamp, models_processed,
        models_failed, total_predictions, test_rows, feature_count,
        status, execution_time_seconds, notes
    FROM {AUDIT_PREDICTION_TABLE}
    ORDER BY execution_timestamp DESC
    """
    return execute_query(query)


@st.cache_data(ttl=3600, show_spinner="Carregando distribuicao por modelo...")
def load_risk_distribution_by_model():
    """Carrega distribuicao de risco por modelo via SQL GROUP BY."""
    query = f"""
    SELECT MODEL_TYPE, RISK_CATEGORY, COUNT(*) as count,
           AVG(RISK_PROBABILITY) as avg_prob,
           COUNT(DISTINCT SK_ID_CURR) as unique_customers
    FROM {PREDICTIONS_TABLE}
    GROUP BY MODEL_TYPE, RISK_CATEGORY
    ORDER BY MODEL_TYPE,
        CASE RISK_CATEGORY
            WHEN 'LOW' THEN 0
            WHEN 'MEDIUM' THEN 1
            WHEN 'HIGH' THEN 2
            ELSE 3
        END
    """
    return execute_query(query)


@st.cache_data(ttl=3600, show_spinner="Carregando estatisticas...")
def load_probability_stats(model_filter="Todos"):
    """Carrega estatisticas de probabilidade via SQL aggregation."""
    where_clause = ""
    params = None
    if model_filter != "Todos":
        where_clause = "WHERE MODEL_TYPE = ?"
        params = (model_filter,)

    query = f"""
    SELECT
        COUNT(DISTINCT SK_ID_CURR) as total_customers,
        COUNT(*) as total_predictions,
        MIN(RISK_PROBABILITY) as min_prob,
        AVG(RISK_PROBABILITY) as mean_prob,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY RISK_PROBABILITY) as median_prob,
        MAX(RISK_PROBABILITY) as max_prob
    FROM {PREDICTIONS_TABLE}
    {where_clause}
    """
    return execute_query(query, params)


@st.cache_data(ttl=3600, show_spinner="Carregando histograma...")
def load_histogram_data(model_filter="Todos", num_bins=50):
    """Carrega dados de histograma via SQL binning (evita carregar 292K linhas)."""
    where_clause = ""
    params = None
    if model_filter != "Todos":
        where_clause = "WHERE MODEL_TYPE = ?"
        params = (model_filter,)

    query = f"""
    SELECT
        FLOOR(RISK_PROBABILITY * {num_bins}) / CAST({num_bins} AS DOUBLE) as bin_start,
        COUNT(*) as count
    FROM {PREDICTIONS_TABLE}
    {where_clause}
    GROUP BY FLOOR(RISK_PROBABILITY * {num_bins}) / CAST({num_bins} AS DOUBLE)
    ORDER BY bin_start
    """
    return execute_query(query, params)


def get_customer_predictions(sk_id_curr):
    """Busca previsoes de um cliente especifico via consulta parametrizada (6 linhas)."""
    query = f"""
    SELECT
        SK_ID_CURR, MODEL_NAME, MODEL_TYPE, MODEL_VERSION, RUN_ID,
        PREDICTION, RISK_PROBABILITY, RISK_CATEGORY, RISK_SCORE,
        PREDICTION_TIMESTAMP, EXECUTION_ID
    FROM {PREDICTIONS_TABLE}
    WHERE SK_ID_CURR = ?
    ORDER BY MODEL_TYPE
    """
    return execute_query(query, (int(sk_id_curr),))


@st.cache_data(ttl=3600, show_spinner="Carregando limites...")
def load_customer_id_range():
    """Carrega min, max e total de SK_ID_CURR via SQL."""
    query = f"""
    SELECT MIN(SK_ID_CURR) as min_id, MAX(SK_ID_CURR) as max_id,
           COUNT(DISTINCT SK_ID_CURR) as total_customers
    FROM {PREDICTIONS_TABLE}
    """
    return execute_query(query)


# ============================================================
# VALIDACAO DE QUALIDADE DE DADOS
# ============================================================

def validate_predictions_data(predictions_df):
    """Valida integridade dos dados de previsoes."""
    if predictions_df is None or predictions_df.empty:
        return {"valid": False, "warnings": ["Tabela de previsoes vazia ou indisponivel."]}

    warnings = []

    prob_min = float(predictions_df["RISK_PROBABILITY"].min())
    prob_max = float(predictions_df["RISK_PROBABILITY"].max())
    if prob_min < 0:
        warnings.append(f"RISK_PROBABILITY contem valor minimo invalido: {prob_min:.4f} (esperado >= 0)")
    if prob_max > 1:
        warnings.append(f"RISK_PROBABILITY contem valor maximo invalido: {prob_max:.4f} (esperado <= 1)")

    score_min = float(predictions_df["RISK_SCORE"].min())
    score_max = float(predictions_df["RISK_SCORE"].max())
    if score_min < 0:
        warnings.append(f"RISK_SCORE contem valor minimo invalido: {score_min:.2f} (esperado >= 0)")
    if score_max > 100:
        warnings.append(f"RISK_SCORE contem valor maximo invalido: {score_max:.2f} (esperado <= 100)")

    unique_preds = set(predictions_df["PREDICTION"].unique())
    if not unique_preds.issubset({0, 1}):
        warnings.append(f"PREDICTION contem valores invalidos: {unique_preds - {0, 1}}")

    valid_cats = {"LOW", "MEDIUM", "HIGH"}
    actual_cats = set(predictions_df["RISK_CATEGORY"].unique())
    if not actual_cats.issubset(valid_cats):
        warnings.append(f"RISK_CATEGORY contem valores invalidos: {actual_cats - valid_cats}")

    return {"valid": len(warnings) == 0, "warnings": warnings}


def show_dq_warning(validation_result):
    """Exibe alerta de data quality se houver inconsistencias."""
    if not validation_result["valid"]:
        st.warning("Data quality: Foram detectadas inconsistencias nos dados.")
        with st.expander("Detalhes da validacao"):
            for w in validation_result["warnings"]:
                st.write(f"- {w}")


# ============================================================
# SIDEBAR - NAVEGACAO E FILTROS
# ============================================================

st.sidebar.title("Navegacao")

PAGES = [
    "Visao Executiva",
    "Analise de Cliente",
    "Comparacao de Modelos",
    "Distribuicao de Risco",
    "Auditoria",
    "Sobre o Projeto"
]

page = st.sidebar.selectbox("Selecione a pagina", PAGES)

st.sidebar.divider()
st.sidebar.subheader("Filtros globais")

model_list = load_model_list()
selected_model = st.sidebar.selectbox("Modelo", ["Todos"] + model_list)
selected_risk = st.sidebar.selectbox("Categoria de risco", ["Todas", "LOW", "MEDIUM", "HIGH"])
selected_prediction = st.sidebar.selectbox("Predicao", ["Todas", "0 (Baixo risco)", "1 (Alto risco)"])

st.sidebar.divider()
if st.sidebar.button("Atualizar dados"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Ultima atualizacao:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================
# TITULO DA APLICACAO
# ============================================================

st.title("Credit Risk Intelligence Platform")
st.markdown("Plataforma analitica para avaliacao e monitoramento de risco de credito baseada em Machine Learning")
st.markdown("Pipeline completo desenvolvido em Databricks utilizando arquitetura Medallion, Delta Lake, Machine Learning, MLflow e Streamlit.")
st.divider()


# ============================================================
# PAGINA: VISAO EXECUTIVA
# ============================================================

if page == "Visao Executiva":
    st.header("Visao Executiva")

    kpi_df = load_kpis()

    if kpi_df.empty:
        st.warning("Nenhum dado disponivel.")
    else:
        row = kpi_df.iloc[0]
        unique_customers = int(row["unique_customers"])
        unique_models = int(row["unique_models"])
        total_predictions = int(row["total_predictions"])
        avg_probability = float(row["avg_probability"])

        risk_df = load_risk_distribution()

        if not risk_df.empty:
            dq_check_df = pd.DataFrame({
                "RISK_PROBABILITY": [0.5],
                "RISK_SCORE": [50.0],
                "PREDICTION": [0],
                "RISK_CATEGORY": risk_df["RISK_CATEGORY"].tolist()
            })
            dq_result = validate_predictions_data(dq_check_df)
            show_dq_warning(dq_result)

        high_count = int(risk_df[risk_df["RISK_CATEGORY"] == "HIGH"]["count"].sum()) if not risk_df.empty else 0
        medium_count = int(risk_df[risk_df["RISK_CATEGORY"] == "MEDIUM"]["count"].sum()) if not risk_df.empty else 0
        low_count = int(risk_df[risk_df["RISK_CATEGORY"] == "LOW"]["count"].sum()) if not risk_df.empty else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Clientes analisados", f"{unique_customers:,}")
        col2.metric("Modelos registrados", f"{unique_models}")
        col3.metric("Total de previsoes", f"{total_predictions:,}")
        col4.metric("Probabilidade media", f"{avg_probability:.4f}")

        col5, col6, col7 = st.columns(3)
        col5.metric("Previsoes LOW", f"{low_count:,}")
        col6.metric("Previsoes MEDIUM", f"{medium_count:,}")
        col7.metric("Previsoes HIGH", f"{high_count:,}")

        st.divider()

        st.subheader("Distribuicao de risco")
        if not risk_df.empty:
            total = risk_df["count"].sum()
            risk_df_display = risk_df.copy()
            risk_df_display["percent"] = (risk_df_display["count"] / total * 100).round(2)

            col_chart, col_data = st.columns([3, 2])

            with col_chart:
                fig = px.bar(
                    risk_df_display,
                    x="RISK_CATEGORY",
                    y="count",
                    text="count",
                    labels={"RISK_CATEGORY": "Categoria de risco", "count": "Quantidade"},
                    title="Distribuicao por categoria de risco"
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)

            with col_data:
                display_df = risk_df_display.rename(columns={
                    "RISK_CATEGORY": "Categoria",
                    "count": "Quantidade",
                    "percent": "Percentual (%)"
                })
                st.dataframe(display_df[["Categoria", "Quantidade", "Percentual (%)"]],
                             use_container_width=True, hide_index=True)

        st.divider()

        with st.expander("Classificacao operacional utilizada"):
            st.markdown("""
            **LOW:** probabilidade < 10%

            **MEDIUM:** 10% <= probabilidade < 30%

            **HIGH:** probabilidade >= 30%

            As faixas apresentadas sao classificacoes operacionais para analise.
            Nao representam, isoladamente, uma decisao de concessao ou recusa de credito.
            """)


# ============================================================
# PAGINA: ANALISE DE CLIENTE
# ============================================================

if page == "Analise de Cliente":
    st.header("Analise de Cliente")

    range_df = load_customer_id_range()

    if range_df.empty:
        st.warning("Dados indisponiveis.")
    else:
        min_id = int(range_df.iloc[0]["min_id"])
        max_id = int(range_df.iloc[0]["max_id"])
        total_customers = int(range_df.iloc[0]["total_customers"])

        col_input, col_hint = st.columns([1, 2])
        with col_input:
            selected_customer = st.number_input(
                "Digite o SK_ID_CURR do cliente",
                min_value=min_id,
                max_value=max_id,
                value=min_id,
                step=1
            )
        with col_hint:
            st.caption(f"Intervalo disponivel: {min_id} a {max_id} | Total de clientes: {total_customers:,}")

        customer_preds = get_customer_predictions(selected_customer)

        if customer_preds.empty:
            st.warning(f"Cliente {selected_customer} nao encontrado na tabela de previsoes.")
        else:
            dq_result = validate_predictions_data(customer_preds)
            show_dq_warning(dq_result)

            lgbm_pred = customer_preds[customer_preds["MODEL_TYPE"] == "LightGBM"]
            ref_pred = lgbm_pred.iloc[0] if not lgbm_pred.empty else customer_preds.iloc[0]

            st.subheader(f"Cliente {int(selected_customer)}")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Cliente", f"{int(selected_customer)}")
            col2.metric("Modelo", f"{ref_pred['MODEL_TYPE']}")
            col3.metric("Versao", f"{ref_pred['MODEL_VERSION']}")
            col4.metric("Probabilidade", f"{ref_pred['RISK_PROBABILITY']*100:.2f}%")
            col5.metric("Categoria", f"{ref_pred['RISK_CATEGORY']}")

            col6, col7 = st.columns(2)
            col6.metric("Score", f"{ref_pred['RISK_SCORE']:.2f}")
            col7.metric("Predicao", f"{int(ref_pred['PREDICTION'])}")

            st.markdown("---")
            st.markdown(f"**Probabilidade estimada pelo modelo:** {ref_pred['RISK_PROBABILITY']:.4f} ({ref_pred['RISK_PROBABILITY']*100:.2f}%)")
            st.markdown(f"**Classificacao operacional do modelo:** {ref_pred['RISK_CATEGORY']}")

            st.divider()

            st.subheader("Previsoes por modelo")
            display_cols = ["MODEL_TYPE", "MODEL_VERSION", "PREDICTION",
                            "RISK_PROBABILITY", "RISK_SCORE", "RISK_CATEGORY"]
            display_df = customer_preds[display_cols].rename(columns={
                "MODEL_TYPE": "Modelo",
                "MODEL_VERSION": "Versao",
                "PREDICTION": "Predicao",
                "RISK_PROBABILITY": "Probabilidade",
                "RISK_SCORE": "Score",
                "RISK_CATEGORY": "Categoria"
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.divider()

            st.subheader("Comparacao de modelos")

            fig = px.bar(
                customer_preds,
                x="MODEL_TYPE",
                y="RISK_PROBABILITY",
                text=customer_preds["RISK_PROBABILITY"].apply(lambda x: f"{x:.4f}"),
                labels={"MODEL_TYPE": "Modelo", "RISK_PROBABILITY": "Probabilidade de risco"},
                title="Probabilidade de inadimplencia por modelo"
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            prob_mean = float(customer_preds["RISK_PROBABILITY"].mean())
            prob_min = float(customer_preds["RISK_PROBABILITY"].min())
            prob_max = float(customer_preds["RISK_PROBABILITY"].max())
            prob_std = float(customer_preds["RISK_PROBABILITY"].std())

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Probabilidade media", f"{prob_mean:.4f}")
            col2.metric("Probabilidade minima", f"{prob_min:.4f}")
            col3.metric("Probabilidade maxima", f"{prob_max:.4f}")
            col4.metric("Desvio padrao", f"{prob_std:.4f}")

            st.divider()

            with st.expander("Classificacao operacional utilizada"):
                st.markdown("""
                **LOW:** probabilidade < 10%

                **MEDIUM:** 10% <= probabilidade < 30%

                **HIGH:** probabilidade >= 30%

                As faixas apresentadas sao classificacoes operacionais para analise.
                Nao representam, isoladamente, uma decisao de concessao ou recusa de credito.
                """)


# ============================================================
# PAGINA: COMPARACAO DE MODELOS
# ============================================================

if page == "Comparacao de Modelos":
    st.header("Comparacao de Modelos")

    evaluation_df = load_evaluation()

    if evaluation_df.empty:
        st.warning("Dados de avaliacao indisponiveis.")
    else:
        st.markdown("Metricas de avaliacao dos modelos em conjunto de validacao holdout estratificado (20% de ml_train).")
        st.divider()

        st.subheader("Tabela de metricas")
        st.dataframe(evaluation_df, use_container_width=True, hide_index=True)

        st.divider()

        st.subheader("Graficos de comparacao")

        metrics_to_plot = [
            ("ROC_AUC", "ROC-AUC"),
            ("PR_AUC", "PR-AUC"),
            ("RECALL", "Recall"),
            ("PRECISION", "Precision"),
            ("F1", "F1 Score")
        ]

        for col, title in metrics_to_plot:
            if col in evaluation_df.columns:
                fig = px.bar(
                    evaluation_df,
                    x="MODEL",
                    y=col,
                    text=evaluation_df[col].apply(lambda x: f"{x:.4f}"),
                    labels={"MODEL": "Modelo", col: title},
                    title=title
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGINA: DISTRIBUICAO DE RISCO
# ============================================================

if page == "Distribuicao de Risco":
    st.header("Distribuicao de Risco")

    dist_df = load_risk_distribution_by_model()

    if dist_df.empty:
        st.warning("Dados indisponiveis.")
    else:
        available_models = dist_df["MODEL_TYPE"].unique().tolist()

        col_sel, col_stats = st.columns([1, 3])
        with col_sel:
            dist_model = st.selectbox("Selecione o modelo", ["Todos"] + available_models)

        stats_df = load_probability_stats(dist_model)

        if not stats_df.empty:
            s = stats_df.iloc[0]
            total_cust = int(s["total_customers"])
            total_preds = int(s["total_predictions"])
            min_p = float(s["min_prob"])
            mean_p = float(s["mean_prob"])
            median_p = float(s["median_prob"])
            max_p = float(s["max_prob"])

            with col_stats:
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Total de clientes", f"{total_cust:,}")
                c2.metric("Prob. media", f"{mean_p:.4f}")
                c3.metric("Prob. mediana", f"{median_p:.4f}")
                c4.metric("Prob. minima", f"{min_p:.4f}")
                c5.metric("Prob. maxima", f"{max_p:.4f}")
                c6.metric("Total previsoes", f"{total_preds:,}")

        st.divider()

        # Distribuicao por modelo
        if dist_model == "Todos":
            st.subheader("Distribuicao por modelo")
            fig = px.bar(
                dist_df,
                x="MODEL_TYPE",
                y="count",
                color="RISK_CATEGORY",
                barmode="group",
                labels={"MODEL_TYPE": "Modelo", "count": "Quantidade", "RISK_CATEGORY": "Categoria"},
                title="Distribuicao de risco por modelo"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Percentuais
            st.subheader("Percentuais por modelo")
            pivot_df = dist_df.pivot_table(index="MODEL_TYPE", columns="RISK_CATEGORY", values="count", fill_value=0)
            for col in ["LOW", "MEDIUM", "HIGH"]:
                if col not in pivot_df.columns:
                    pivot_df[col] = 0
            pivot_df["TOTAL"] = pivot_df.sum(axis=1)
            for col in ["LOW", "MEDIUM", "HIGH"]:
                pivot_df[f"{col} (%)"] = (pivot_df[col] / pivot_df["TOTAL"] * 100).round(2)
            display_pct = pivot_df[["LOW (%)", "MEDIUM (%)", "HIGH (%)"]].reset_index()
            st.dataframe(display_pct, use_container_width=True, hide_index=True)
        else:
            model_dist = dist_df[dist_df["MODEL_TYPE"] == dist_model]
            st.subheader(f"Distribuicao - {dist_model}")

            total_m = model_dist["count"].sum()
            model_dist_display = model_dist.copy()
            model_dist_display["percent"] = (model_dist_display["count"] / total_m * 100).round(2) if total_m > 0 else 0.0

            col_chart, col_data = st.columns([3, 2])
            with col_chart:
                fig = px.bar(
                    model_dist_display,
                    x="RISK_CATEGORY",
                    y="count",
                    text="count",
                    labels={"RISK_CATEGORY": "Categoria", "count": "Quantidade"},
                    title=f"Distribuicao - {dist_model}"
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, use_container_width=True)
            with col_data:
                st.dataframe(
                    model_dist_display[["RISK_CATEGORY", "count", "percent"]].rename(columns={
                        "RISK_CATEGORY": "Categoria",
                        "count": "Quantidade",
                        "percent": "Percentual (%)"
                    }),
                    use_container_width=True, hide_index=True
                )

        st.divider()

        # Histograma via SQL binning
        st.subheader("Histograma de probabilidade")
        hist_df = load_histogram_data(dist_model, num_bins=50)

        if not hist_df.empty:
            fig = px.bar(
                hist_df,
                x="bin_start",
                y="count",
                labels={"bin_start": "Probabilidade", "count": "Frequencia"},
                title="Distribuicao de RISK_PROBABILITY"
            )
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGINA: AUDITORIA
# ============================================================

if page == "Auditoria":
    st.header("Auditoria")

    st.subheader("Registro de execucao de predicao")
    audit_df = load_audit()

    if audit_df.empty:
        st.warning("Dados de auditoria indisponiveis.")
    else:
        audit_display = audit_df.rename(columns={
            "execution_id": "Execution ID",
            "execution_timestamp": "Timestamp",
            "models_processed": "Modelos processados",
            "models_failed": "Modelos falhados",
            "total_predictions": "Total previsoes",
            "test_rows": "Linhas de teste",
            "feature_count": "Features",
            "status": "Status",
            "execution_time_seconds": "Tempo (s)",
            "notes": "Notas"
        })
        st.dataframe(audit_display, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Model Registry (MLflow / Unity Catalog)")
    registry_df = load_model_registry()

    if registry_df.empty:
        st.warning("Model registry indisponivel.")
    else:
        registry_display = registry_df.rename(columns={
            "MODEL_NAME": "Nome do modelo",
            "MODEL_TYPE": "Tipo",
            "MODEL_VERSION": "Versao",
            "RUN_ID": "Run ID",
            "MODEL_URI": "URI",
            "ROC_AUC": "ROC-AUC",
            "PR_AUC": "PR-AUC",
            "STATUS": "Status",
            "EXPERIMENT_ID": "Experiment ID"
        })
        st.dataframe(registry_display, use_container_width=True, hide_index=True)

        st.divider()

        with st.expander("Detalhes tecnicos do Model Registry"):
            for _, row in registry_df.iterrows():
                st.markdown(f"**{row['MODEL_NAME']}** (v{row['MODEL_VERSION']})")
                st.markdown(f"- Tipo: {row['MODEL_TYPE']}")
                st.markdown(f"- Run ID: {row['RUN_ID']}")
                st.markdown(f"- URI: {row['MODEL_URI']}")
                st.markdown(f"- ROC-AUC: {row['ROC_AUC']}")
                st.markdown(f"- PR-AUC: {row['PR_AUC']}")
                st.markdown(f"- Status: {row['STATUS']}")
                st.markdown("")


# ============================================================
# PAGINA: SOBRE O PROJETO
# ============================================================

if page == "Sobre o Projeto":
    st.header("Sobre o Projeto")

    st.markdown("O **Credit Risk Intelligence Platform** e uma plataforma analitica completa para avaliacao e monitoramento de risco de credito, baseada em Machine Learning e desenvolvida em Databricks.")

    st.divider()

    st.subheader("Arquitetura do Pipeline")

    st.markdown("""
    ```
    Bronze
      |
      v
    Silver
      |
      v
    Gold
      |
      v
    ML Dataset
      |
      v
    EDA
      |
      v
    Training
      |
      v
    Evaluation
      |
      v
    MLflow / Unity Catalog
      |
      v
    Prediction
      |
      v
    Streamlit
    ```
    """)

    st.divider()

    st.subheader("Arquitetura Medallion")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**BRONZE**")
        st.markdown("Dados brutos e auditoria")
        st.caption("8 tabelas originais do dataset Home Credit Default Risk")

    with col2:
        st.markdown("**SILVER**")
        st.markdown("Qualidade, limpeza e padronizacao")
        st.caption("8 tabelas com colunas de DQ e controle")

    with col3:
        st.markdown("**GOLD**")
        st.markdown("Features e dataset para Machine Learning")
        st.caption("229 features + SK_ID_CURR + TARGET")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("**MLFLOW**")
        st.markdown("Experimentacao e Model Registry")
        st.caption("6 modelos registrados no Unity Catalog")

    with col5:
        st.markdown("**PREDICTION**")
        st.markdown("Inferencia e classificacao de risco")
        st.caption("292.464 previsoes sobre 48.744 clientes")

    with col6:
        st.markdown("**STREAMLIT**")
        st.markdown("Visualizacao e consumo analitico")
        st.caption("Dashboard interativo de risco de credito")

    st.divider()

    st.subheader("Tecnologias utilizadas")

    techs = [
        "Databricks", "Delta Lake", "Apache Spark", "PySpark", "Python", "SQL",
        "Machine Learning", "LightGBM", "XGBoost", "Scikit-learn",
        "MLflow", "Unity Catalog", "Streamlit", "Plotly"
    ]

    tech_cols = st.columns(7)
    for i, tech in enumerate(techs):
        tech_cols[i % 7].markdown(f"- {tech}")

    st.divider()

    st.subheader("Classificacao operacional utilizada")

    st.markdown("""
    | Faixa | Condicao |
    | --- | --- |
    | LOW | probabilidade < 10% |
    | MEDIUM | 10% <= probabilidade < 30% |
    | HIGH | probabilidade >= 30% |
    """)

    st.markdown("As faixas apresentadas sao classificacoes operacionais para analise. Nao representam, isoladamente, uma decisao de concessao ou recusa de credito.")

    st.divider()

    st.subheader("Informacoes do dataset")

    st.markdown("""
    | Item | Valor |
    | --- | --- |
    | Dataset | Home Credit Default Risk |
    | Clientes analisados | 48.744 |
    | Features | 229 |
    | Modelos | 6 |
    | Total de previsoes | 292.464 |
    | Registry | credit_risk.analytics.mlflow_model_registry |
    | Tabela de previsoes | credit_risk.analytics.predictions |
    """)

    st.divider()

    st.subheader("Resumo da Aplicacao")
    st.markdown("""
    **Application:** Credit Risk Intelligence Platform
    **Data source:** credit_risk.analytics.predictions
    **Customers:** 48,744
    **Models:** 6
    **Features:** 229
    **Predictions:** 292,464
    **MLflow:** Unity Catalog
    **Registry:** credit_risk.analytics.mlflow_model_registry
    **Application status:** READY
    """)