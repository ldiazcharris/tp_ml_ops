from datetime import datetime, timedelta


PIPELINE_ID = "stroke_prediction_pipeline"
PIPELINE_DESCRIPTION = "Pipeline de entrenamiento y serving para stroke prediction"
PIPELINE_START_DATE = datetime(2024, 1, 1)
PIPELINE_TAGS = ["ml", "stroke", "healthcare", "serving"]

DEFAULT_DATASET_URL = (
    "https://gist.githubusercontent.com/aishwarya8615/"
    "d2107f828d3f904839cbcb7eaa85bd04/raw/healthcare-dataset-stroke-data.csv"
)
DEFAULT_DATASET_LOCAL_PATH = "/opt/project/data/healthcare-dataset-stroke-data.csv"

DATA_REQUIREMENTS = ["pandas", "boto3"]
MODEL_REQUIREMENTS = [
    "boto3",
    "imbalanced-learn==0.14.0",
    "mlflow==2.12.2",
    "numpy",
    "pandas",
    "scikit-learn==1.7.2",
]

RELOAD_RETRIES = 3
RELOAD_RETRY_DELAY = timedelta(seconds=10)
