from airflow.decorators import task

from stroke_pipeline.config import RELOAD_RETRIES, RELOAD_RETRY_DELAY


def build_reload_prediction_api_task():
    """Devuelve la tarea que fuerza la recarga del modelo en FastAPI."""

    @task.virtualenv(
        task_id="reload_prediction_api",
        requirements=["requests"],
        retries=RELOAD_RETRIES,
        retry_delay=RELOAD_RETRY_DELAY,
    )
    def reload_prediction_api(training_run_id: str, evaluation_metrics: dict) -> dict:
        import os

        import requests

        reload_url = os.getenv("API_RELOAD_URL", "http://api:8800/reload-model")
        response = requests.post(reload_url, timeout=30)
        response.raise_for_status()

        return {
            "training_run_id": training_run_id,
            "reload_status": response.json().get("status", "unknown"),
            "metrics_logged": sorted(evaluation_metrics.keys()),
        }

    return reload_prediction_api
