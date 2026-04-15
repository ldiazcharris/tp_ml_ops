import logging

from fastapi import FastAPI, HTTPException

from model_service import (
    build_cache_key,
    get_cached_prediction,
    get_health_payload,
    get_redis,
    load_model,
    predict_from_features,
    store_prediction,
)
from schemas import PredictionResponse, StrokeFeatures


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stroke Risk Prediction API",
    description="API para inferencia clínica basada en el modelo seleccionado del TP de Aprendizaje Automático",
    version="2.0.0",
)


@app.on_event("startup")
async def startup():
    get_redis()
    try:
        load_model()
    except Exception as exc:
        logger.warning("Model not available at startup: %s", exc)


@app.get("/health")
def health():
    return get_health_payload()


@app.post("/predict", response_model=PredictionResponse)
def predict(features: StrokeFeatures):
    features_dict = features.model_dump()
    cache_key = build_cache_key(features_dict)

    try:
        cached = get_cached_prediction(cache_key)
        if cached:
            cached["cached"] = True
            return PredictionResponse(**cached)

        result = predict_from_features(features_dict)
        store_prediction(cache_key, result)
        return PredictionResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reload-model")
def reload_model():
    try:
        load_model()
        return {"status": "Model reloaded successfully"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
