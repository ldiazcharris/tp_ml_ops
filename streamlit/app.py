import os
from datetime import datetime, timezone

import mlflow
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://api:8800")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "stroke_prediction")


def build_prediction_payload(
    gender: str,
    age: float,
    hypertension: int,
    heart_disease: int,
    ever_married: str,
    work_type: str,
    residence_type: str,
    avg_glucose_level: float,
    bmi: float,
    smoking_status: str,
) -> dict:
    return {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status,
    }


def request_prediction(payload: dict):
    return requests.post(f"{API_URL}/predict", json=payload, timeout=10)


def format_run_timestamp(start_time_ms: int) -> str:
    start_dt = datetime.fromtimestamp(start_time_ms / 1000, tz=timezone.utc)
    return start_dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def get_latest_run():
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if not experiment:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def render_prediction_result(result: dict):
    probability = result["stroke_probability"]
    prediction_label = result["prediction_label"]

    if prediction_label == "stroke":
        st.error(
            f"Riesgo estimado de stroke: **{probability:.2%}** "
            f"(threshold {result['threshold']:.2f})"
        )
    else:
        st.success(
            f"Riesgo estimado de stroke: **{probability:.2%}** "
            f"(threshold {result['threshold']:.2f})"
        )

    st.caption(f"Version del modelo: {result['model_version']}")
    if result.get("cached"):
        st.info("Resultado obtenido desde cache (Redis)")

    st.progress(probability, text=f"Probabilidad de stroke: {probability:.2%}")


def render_prediction_tab():
    st.header("Ingrese la información clínica del paciente")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=50.0)
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=str)
        heart_disease = st.selectbox("Heart disease", [0, 1], format_func=str)
        ever_married = st.selectbox("Ever married", ["Yes", "No"])

    with col2:
        work_type = st.selectbox(
            "Work type",
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        )
        residence_type = st.selectbox("Residence type", ["Urban", "Rural"])
        avg_glucose_level = st.number_input(
            "Average glucose level",
            min_value=0.0,
            value=100.0,
            step=0.1,
        )
        bmi = st.number_input("BMI", min_value=0.0, value=25.0, step=0.1)
        smoking_status = st.selectbox(
            "Smoking status",
            ["never smoked", "formerly smoked", "smokes", "Unknown"],
        )

    if st.button("Predecir Riesgo", type="primary"):
        payload = build_prediction_payload(
            gender=gender,
            age=age,
            hypertension=hypertension,
            heart_disease=heart_disease,
            ever_married=ever_married,
            work_type=work_type,
            residence_type=residence_type,
            avg_glucose_level=avg_glucose_level,
            bmi=bmi,
            smoking_status=smoking_status,
        )

        try:
            response = request_prediction(payload)
            if response.status_code == 200:
                render_prediction_result(response.json())
            elif response.status_code == 503:
                st.error(
                    "El modelo no está cargado. "
                    "Ejecute el DAG de entrenamiento en Airflow primero."
                )
            else:
                st.error(f"Error: {response.text}")
        except requests.ConnectionError:
            st.error("No se pudo conectar con la API. Verifique que esté corriendo.")


def render_metrics_tab():
    st.header("Métricas del último entrenamiento")

    try:
        run = get_latest_run()
        if not run:
            st.info("No hay runs registrados. Ejecute el DAG de entrenamiento.")
            return

        metrics = run.data.metrics
        params = run.data.params

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.subheader("Métricas")
            for metric_name in [
                "test_recall",
                "test_precision",
                "test_f1",
                "test_f2",
                "test_roc_auc",
                "test_pr_auc",
                "test_brier_score",
            ]:
                if metric_name in metrics:
                    st.metric(metric_name, f"{metrics[metric_name]:.4f}")

        with col_m2:
            st.subheader("Parámetros")
            for name in [
                "model_type",
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "decision_threshold",
                "train_rows",
            ]:
                if name in params:
                    st.text(f"{name}: {params[name]}")

        st.caption(f"Run ID: {run.info.run_id}")
        st.caption(f"Fecha: {format_run_timestamp(run.info.start_time)}")
    except Exception as exc:
        st.error(f"Error al conectar con MLflow: {exc}")


st.set_page_config(page_title="Stroke Risk Predictor", layout="wide")
st.title("Stroke Risk Predictor")
st.write(
    "Predicción clínica del riesgo de accidente cerebrovascular a partir de "
    "variables demográficas y antecedentes de salud."
)

tab_predict, tab_metrics = st.tabs(["Predicción", "Métricas del Modelo"])

with tab_predict:
    render_prediction_tab()

with tab_metrics:
    render_metrics_tab()
