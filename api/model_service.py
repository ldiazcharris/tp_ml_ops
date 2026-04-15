import hashlib
import json
import logging
import os

import mlflow
import pandas as pd
import redis

from api_config import (
    CLASS_LABELS,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    PREDICTION_CACHE_PREFIX,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_TTL,
)


logger = logging.getLogger(__name__)

redis_client = None
model = None
model_version = None
decision_threshold = 0.5

REQUEST_TO_MODEL_COLUMNS = {
    "gender": "gender",
    "age": "age",
    "hypertension": "hypertension",
    "heart_disease": "heart_disease",
    "ever_married": "ever_married",
    "work_type": "work_type",
    "residence_type": "Residence_type",
    "avg_glucose_level": "avg_glucose_level",
    "bmi": "bmi",
    "smoking_status": "smoking_status",
}


def get_redis():
    global redis_client

    if redis_client is None:
        try:
            redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=0,
                decode_responses=False,
            )
            redis_client.ping()
            logger.info("Connected to Redis")
        except redis.ConnectionError:
            logger.warning("Redis not available, running without cache")
            redis_client = None

    return redis_client


def load_model():
    global model, model_version, decision_threshold

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
        "MLFLOW_S3_ENDPOINT_URL", "http://s3:9000"
    )
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["None", "Production"])
    if not versions:
        raise ValueError(f"No versions found for model '{MODEL_NAME}'")

    latest = versions[-1]
    run = client.get_run(latest.run_id)

    model_uri = f"models:/{MODEL_NAME}/{latest.version}"
    model = mlflow.sklearn.load_model(model_uri)
    model_version = str(latest.version)
    decision_threshold = float(run.data.params.get("decision_threshold", 0.5))

    logger.info("Model loaded: %s v%s", MODEL_NAME, model_version)


def build_cache_key(features_dict: dict) -> str:
    return hashlib.md5(json.dumps(features_dict, sort_keys=True).encode()).hexdigest()


def get_cached_prediction(cache_key: str):
    cache = get_redis()
    if not cache:
        return None

    cached = cache.get(f"{PREDICTION_CACHE_PREFIX}{cache_key}")
    return json.loads(cached) if cached else None


def store_prediction(cache_key: str, result: dict) -> None:
    cache = get_redis()
    if not cache:
        return

    cache.set(
        f"{PREDICTION_CACHE_PREFIX}{cache_key}",
        json.dumps(result),
        ex=REDIS_TTL,
    )


def build_feature_frame(features_dict: dict) -> pd.DataFrame:
    mapped = {
        model_column: features_dict[request_field]
        for request_field, model_column in REQUEST_TO_MODEL_COLUMNS.items()
    }
    return pd.DataFrame([mapped])


def predict_from_features(features_dict: dict) -> dict:
    if model is None:
        raise RuntimeError("Model not loaded")

    feature_frame = build_feature_frame(features_dict)
    stroke_probability = float(model.predict_proba(feature_frame)[0][1])
    prediction = int(stroke_probability >= decision_threshold)

    return {
        "stroke_probability": round(stroke_probability, 4),
        "prediction": prediction,
        "prediction_label": CLASS_LABELS[prediction],
        "threshold": round(decision_threshold, 4),
        "model_version": model_version or "unknown",
        "cached": False,
    }


def get_health_payload() -> dict:
    cache = get_redis()
    return {
        "status": "healthy",
        "model": "loaded" if model is not None else "not_loaded",
        "model_version": model_version,
        "threshold": decision_threshold,
        "cache": "connected" if cache else "unavailable",
    }
