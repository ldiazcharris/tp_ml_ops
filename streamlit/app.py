import os
from datetime import datetime, timezone

import mlflow
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8800")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

st.set_page_config(page_title="Wine Quality Predictor", layout="wide")
st.title("Wine Quality Predictor")

tab_predict, tab_metrics = st.tabs(["Prediccion", "Metricas del Modelo"])

# Prediccion
with tab_predict:
    st.header("Ingrese las caracteristicas del vino")

    col1, col2, col3 = st.columns(3)

    with col1:
        alcohol = st.number_input("Alcohol", value=13.0, step=0.1)
        malic_acid = st.number_input("Malic Acid", value=2.0, step=0.1)
        ash = st.number_input("Ash", value=2.3, step=0.1)
        alcalinity_of_ash = st.number_input("Alcalinity of Ash", value=19.0, step=0.5)
        magnesium = st.number_input("Magnesium", value=100.0, step=1.0)

    with col2:
        total_phenols = st.number_input("Total Phenols", value=2.5, step=0.1)
        flavanoids = st.number_input("Flavanoids", value=2.5, step=0.1)
        nonflavanoid_phenols = st.number_input(
            "Nonflavanoid Phenols", value=0.3, step=0.05
        )
        proanthocyanins = st.number_input("Proanthocyanins", value=1.5, step=0.1)

    with col3:
        color_intensity = st.number_input("Color Intensity", value=5.0, step=0.1)
        hue = st.number_input("Hue", value=1.0, step=0.05)
        od280 = st.number_input("OD280/OD315", value=3.0, step=0.1)
        proline = st.number_input("Proline", value=1000.0, step=10.0)

    if st.button("Predecir", type="primary"):
        payload = {
            "alcohol": alcohol,
            "malic_acid": malic_acid,
            "ash": ash,
            "alcalinity_of_ash": alcalinity_of_ash,
            "magnesium": magnesium,
            "total_phenols": total_phenols,
            "flavanoids": flavanoids,
            "nonflavanoid_phenols": nonflavanoid_phenols,
            "proanthocyanins": proanthocyanins,
            "color_intensity": color_intensity,
            "hue": hue,
            "od280_od315_of_diluted_wines": od280,
            "proline": proline,
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                st.success(
                    f"Clase predicha: **{result['prediction_label']}** "
                    f"(clase {result['prediction']})"
                )
                if result.get("cached"):
                    st.info("Resultado obtenido desde cache (Redis)")

                st.subheader("Probabilidades por clase")
                probs = result["probabilities"]
                for cls, prob in probs.items():
                    st.progress(prob, text=f"{cls}: {prob:.2%}")
            elif response.status_code == 503:
                st.error(
                    "El modelo no esta cargado. "
                    "Ejecute el DAG de entrenamiento en Airflow primero."
                )
            else:
                st.error(f"Error: {response.text}")
        except requests.ConnectionError:
            st.error("No se pudo conectar con la API. Verifique que este corriendo.")

# Metricas
with tab_metrics:
    st.header("Metricas del ultimo entrenamiento")

    try:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()

        experiment = client.get_experiment_by_name("wine_quality")

        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )

            if runs:
                run = runs[0]
                metrics = run.data.metrics
                params = run.data.params

                col_m1, col_m2 = st.columns(2)

                with col_m1:
                    st.subheader("Metricas")
                    for name, value in sorted(metrics.items()):
                        st.metric(label=name, value=f"{value:.4f}")

                with col_m2:
                    st.subheader("Parametros")
                    for name, value in sorted(params.items()):
                        st.text(f"{name}: {value}")

                start_time_ms = run.info.start_time
                start_dt = datetime.fromtimestamp(
                    start_time_ms / 1000, tz=timezone.utc
                ).astimezone()

                st.caption(f"Run ID: {run.info.run_id}")
                st.caption(f"Fecha: {start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            else:
                st.info("No hay runs registrados. Ejecute el DAG de entrenamiento.")
        else:
            st.info(
                "El experimento 'wine_quality' no existe. "
                "Ejecute el DAG de entrenamiento."
            )
    except Exception as e:
        st.error(f"Error al conectar con MLflow: {e}")
