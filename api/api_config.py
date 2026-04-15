import os


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_TTL = int(os.getenv("REDIS_TTL", "3600"))
MODEL_NAME = os.getenv("MODEL_NAME", "stroke_prediction_model")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "stroke_prediction")

PREDICTION_CACHE_PREFIX = "stroke_pred:"
CLASS_LABELS = {0: "no_stroke", 1: "stroke"}
