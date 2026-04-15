from airflow.decorators import task

from stroke_pipeline.config import (
    DATA_REQUIREMENTS,
    DEFAULT_DATASET_LOCAL_PATH,
    DEFAULT_DATASET_URL,
    MODEL_REQUIREMENTS,
)


def build_ensure_artifact_bucket_task():
    """Devuelve la tarea que asegura la existencia del bucket de artefactos."""

    @task.virtualenv(task_id="ensure_artifact_bucket", requirements=["boto3"])
    def ensure_artifact_bucket() -> str:
        import os

        import boto3

        bucket_name = os.getenv("MLFLOW_BUCKET_NAME", "mlflow-artifacts")
        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
        )

        existing_buckets = {
            bucket["Name"] for bucket in s3.list_buckets().get("Buckets", [])
        }
        if bucket_name not in existing_buckets:
            s3.create_bucket(Bucket=bucket_name)

        return bucket_name

    return ensure_artifact_bucket


def build_get_data_task():
    """Devuelve la tarea que carga el dataset base desde disco o URL."""

    @task.virtualenv(task_id="get_data", requirements=DATA_REQUIREMENTS)
    def get_data(
        bucket_name: str,
        default_dataset_path: str = DEFAULT_DATASET_LOCAL_PATH,
        default_dataset_url: str = DEFAULT_DATASET_URL,
    ) -> str:
        import os

        import boto3
        import pandas as pd

        dataset_path = os.getenv("STROKE_DATASET_LOCAL_PATH", default_dataset_path)
        dataset_url = os.getenv("STROKE_DATASET_URL", default_dataset_url)

        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
        else:
            df = pd.read_csv(dataset_url)

        local_path = "/tmp/stroke_raw.csv"
        df.to_csv(local_path, index=False)

        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
        )
        s3_key = "data/stroke_raw.csv"
        s3.upload_file(local_path, bucket_name, s3_key)

        return s3_key

    return get_data


def build_process_data_task():
    """Devuelve la tarea que limpia y normaliza el dataset."""

    @task.virtualenv(task_id="process_data", requirements=DATA_REQUIREMENTS)
    def process_data(s3_key: str) -> dict:
        import os

        import boto3
        import pandas as pd

        bucket_name = os.getenv("MLFLOW_BUCKET_NAME", "mlflow-artifacts")
        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
        )

        local_path = "/tmp/stroke_raw.csv"
        s3.download_file(bucket_name, s3_key, local_path)
        df = pd.read_csv(local_path)

        initial_rows = len(df)
        df = df.drop_duplicates().copy()
        if "id" in df.columns:
            df["id"] = df["id"].astype(str)
        if "stroke" in df.columns:
            df["stroke"] = df["stroke"].astype(int)

        processed_path = "/tmp/stroke_processed.csv"
        df.to_csv(processed_path, index=False)

        s3_key_out = "data/stroke_processed.csv"
        s3.upload_file(processed_path, bucket_name, s3_key_out)

        return {
            "s3_key": s3_key_out,
            "initial_rows": initial_rows,
            "final_rows": len(df),
            "columns": list(df.columns),
            "target_rate": float(df["stroke"].mean()),
        }

    return process_data


def build_split_dataset_task():
    """Devuelve la tarea que genera los splits de train, validation y test."""

    @task.virtualenv(
        task_id="split_dataset",
        requirements=MODEL_REQUIREMENTS,
        multiple_outputs=True,
    )
    def split_dataset(process_result: dict) -> dict:
        import os

        import boto3
        import pandas as pd
        from sklearn.model_selection import train_test_split

        bucket_name = os.getenv("MLFLOW_BUCKET_NAME", "mlflow-artifacts")
        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
        )

        local_path = "/tmp/stroke_processed.csv"
        s3.download_file(bucket_name, process_result["s3_key"], local_path)
        df = pd.read_csv(local_path)

        train_df, temp_df = train_test_split(
            df,
            test_size=0.4,
            random_state=42,
            stratify=df["stroke"],
        )
        validation_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df["stroke"],
        )

        train_path = "/tmp/stroke_train.csv"
        validation_path = "/tmp/stroke_validation.csv"
        test_path = "/tmp/stroke_test.csv"

        train_df.to_csv(train_path, index=False)
        validation_df.to_csv(validation_path, index=False)
        test_df.to_csv(test_path, index=False)

        train_key = "data/stroke_train.csv"
        validation_key = "data/stroke_validation.csv"
        test_key = "data/stroke_test.csv"

        s3.upload_file(train_path, bucket_name, train_key)
        s3.upload_file(validation_path, bucket_name, validation_key)
        s3.upload_file(test_path, bucket_name, test_key)

        return {
            "train_key": train_key,
            "validation_key": validation_key,
            "test_key": test_key,
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "test_rows": len(test_df),
            "train_positive_rate": float(train_df["stroke"].mean()),
            "validation_positive_rate": float(validation_df["stroke"].mean()),
            "test_positive_rate": float(test_df["stroke"].mean()),
        }

    return split_dataset
