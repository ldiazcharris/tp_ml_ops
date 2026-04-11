import hashlib
import json
import logging
import os
import pickle

import mlflow
import numpy as np
import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Wine Quality Prediction API",
    description="API para predicción de calidad de vino usando modelo registrado en MLflow",
    version="1.0.0",
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_TTL = int(os.getenv("REDIS_TTL", "3600"))
MODEL_NAME = os.getenv("MODEL_NAME", "wine_quality_model")

redis_client = None
model = None
scaler = None
feature_names = None


class WineFeatures(BaseModel):
    alcohol: float = Field(..., description="Alcohol content")
    malic_acid: float = Field(..., description="Malic acid")
    ash: float = Field(..., description="Ash")
    alcalinity_of_ash: float = Field(..., description="Alcalinity of ash")
    magnesium: float = Field(..., description="Magnesium")
    total_phenols: float = Field(..., description="Total phenols")
    flavanoids: float = Field(..., description="Flavanoids")
    nonflavanoid_phenols: float = Field(..., description="Nonflavanoid phenols")
    proanthocyanins: float = Field(..., description="Proanthocyanins")
    color_intensity: float = Field(..., description="Color intensity")
    hue: float = Field(..., description="Hue")
    od280_od315_of_diluted_wines: float = Field(..., description="OD280/OD315 of diluted wines")
    proline: float = Field(..., description="Proline")


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probabilities: dict
    cached: bool = False


WINE_CLASSES = {0: "class_0", 1: "class_1", 2: "class_2"}


def normalize_feature_name(name: str):
    return name.replace("/", "_").replace(" ", "_")


def get_redis():
    global redis_client
    if redis_client is None:
        try:
            redis_client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False
            )
            redis_client.ping()
            logger.info("Connected to Redis")
        except redis.ConnectionError:
            logger.warning("Redis not available, running without cache")
            redis_client = None
    return redis_client


def load_model():
    global model, scaler, feature_names

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=["None", "Production"])
        if not versions:
            raise ValueError(f"No versions found for model '{MODEL_NAME}'")

        latest = versions[-1]
        run_id = latest.run_id

        model_uri = f"models:/{MODEL_NAME}/{latest.version}"
        model = mlflow.sklearn.load_model(model_uri)

        scaler_path = client.download_artifacts(run_id, "preprocessing/scaler.pkl")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        names_path = client.download_artifacts(run_id, "preprocessing/feature_names.txt")
        with open(names_path, "r") as f:
            feature_names = f.read().strip().split("\n")

        logger.info(f"Model loaded: {MODEL_NAME} v{latest.version}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.on_event("startup")
async def startup():
    get_redis()
    try:
        load_model()
    except Exception as e:
        logger.warning(f"Model not available at startup: {e}")


@app.get("/health")
def health():
    model_status = "loaded" if model is not None else "not_loaded"
    cache = get_redis()
    cache_status = "connected" if cache else "unavailable"
    return {
        "status": "healthy",
        "model": model_status,
        "cache": cache_status,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: WineFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features_dict = features.model_dump()
    features_key = hashlib.md5(
        json.dumps(features_dict, sort_keys=True).encode()
    ).hexdigest()

    cache = get_redis()
    if cache:
        try:
            cached = cache.get(f"pred:{features_key}")
            if cached:
                result = json.loads(cached)
                result["cached"] = True
                return PredictionResponse(**result)
        except Exception:
            pass

    values = [features_dict[normalize_feature_name(name)] for name in feature_names]
    X = np.array(values).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = int(model.predict(X_scaled)[0])
    probabilities = model.predict_proba(X_scaled)[0]
    prob_dict = {WINE_CLASSES[i]: round(float(p), 4) for i, p in enumerate(probabilities)}

    result = {
        "prediction": prediction,
        "prediction_label": WINE_CLASSES[prediction],
        "probabilities": prob_dict,
        "cached": False,
    }

    if cache:
        try:
            cache.set(f"pred:{features_key}", json.dumps(result), ex=REDIS_TTL)
        except Exception:
            pass

    return PredictionResponse(**result)


@app.post("/reload-model")
def reload_model():
    try:
        load_model()
        return {"status": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
