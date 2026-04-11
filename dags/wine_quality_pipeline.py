import os
from datetime import datetime

from airflow.decorators import dag, task


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT_NAME = "wine_quality"
S3_BUCKET = "s3://mlflow-artifacts"
RANDOM_STATE = 42


@dag(
    dag_id="wine_quality_pipeline",
    description="Pipeline de entrenamiento del modelo de calidad de vino",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["ml", "wine", "training"],
)
def wine_quality_pipeline():

    @task.virtualenv(requirements=["pandas", "scikit-learn", "boto3"])
    def get_data():
        """Descarga el dataset Wine Quality y lo sube a MinIO."""
        import os
        import pandas as pd
        from sklearn.datasets import load_wine

        data = load_wine(as_frame=True)
        df = data.frame
        df["target"] = data.target

        local_path = "/tmp/wine_raw.csv"
        df.to_csv(local_path, index=False)

        import boto3

        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
        )
        s3_key = "data/wine_raw.csv"
        s3.upload_file(local_path, "mlflow-artifacts", s3_key)

        return s3_key

    @task.virtualenv(requirements=["pandas", "scikit-learn", "boto3"])
    def process_data(s3_key: str):
        #Limpia el dataset
        import os
        import pandas as pd
        import boto3

        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
        )

        local_path = "/tmp/wine_raw.csv"
        s3.download_file("mlflow-artifacts", s3_key, local_path)
        df = pd.read_csv(local_path)

        initial_rows = len(df)
        df = df.drop_duplicates()
        df = df.dropna()
        final_rows = len(df)

        processed_path = "/tmp/wine_processed.csv"
        df.to_csv(processed_path, index=False)

        s3_key_out = "data/wine_processed.csv"
        s3.upload_file(processed_path, "mlflow-artifacts", s3_key_out)

        return {
            "s3_key": s3_key_out,
            "initial_rows": initial_rows,
            "final_rows": final_rows,
            "columns": list(df.columns),
        }

    @task.virtualenv(
        requirements=["pandas", "scikit-learn", "boto3"],
        multiple_outputs=True)
    def split_dataset(process_result: dict):
        import os
        import pandas as pd
        import boto3
        from sklearn.model_selection import train_test_split

        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
        )

        local_path = "/tmp/wine_processed.csv"
        s3.download_file("mlflow-artifacts", process_result["s3_key"], local_path)
        df = pd.read_csv(local_path)

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        train_path = "/tmp/wine_train.csv"
        test_path = "/tmp/wine_test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        train_key = "data/wine_train.csv"
        test_key = "data/wine_test.csv"
        s3.upload_file(train_path, "mlflow-artifacts", train_key)
        s3.upload_file(test_path, "mlflow-artifacts", test_key)

        return {
            "train_key": train_key,
            "test_key": test_key,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
        }

    @task.virtualenv(requirements=["pandas", "scikit-learn", "mlflow==2.12.2", "boto3"],)
    def train_model(split_result: dict):
        import os
        import pandas as pd
        import mlflow
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import pickle
        import boto3

        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
            "MLFLOW_S3_ENDPOINT_URL", "http://s3:9000"
        )

        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
        )

        train_path = "/tmp/wine_train.csv"
        s3.download_file("mlflow-artifacts", split_result["train_key"], train_path)
        train_df = pd.read_csv(train_path)

        X_train = train_df.drop("target", axis=1)
        y_train = train_df["target"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment("wine_quality")

        with mlflow.start_run(run_name="random_forest_training") as run:
            params = {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
                "n_features": X_train.shape[1],
                "n_samples": X_train.shape[0],
            }
            mlflow.log_params(params)

            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=params["random_state"],
            )
            model.fit(X_train_scaled, y_train)

            train_accuracy = model.score(X_train_scaled, y_train)
            mlflow.log_metric("train_accuracy", train_accuracy)

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name="wine_quality_model",
            )

            scaler_path = "/tmp/scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact(scaler_path, artifact_path="preprocessing")

            feature_names_path = "/tmp/feature_names.txt"
            with open(feature_names_path, "w") as f:
                f.write("\n".join(X_train.columns.tolist()))
            mlflow.log_artifact(feature_names_path, artifact_path="preprocessing")

            run_id = run.info.run_id

        return run_id

    @task.virtualenv(requirements=["pandas", "scikit-learn", "mlflow==2.12.2", "boto3"])
    def evaluate_model(training_run_id: str, split_result: dict):
        import os
        import pandas as pd
        import mlflow
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            classification_report,
        )
        import pickle
        import boto3
        import json

        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
            "MLFLOW_S3_ENDPOINT_URL", "http://s3:9000"
        )

        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
        )

        test_path = "/tmp/wine_test.csv"
        s3.download_file("mlflow-artifacts", split_result["test_key"], test_path)
        test_df = pd.read_csv(test_path)

        X_test = test_df.drop("target", axis=1)
        y_test = test_df["target"]

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

        client = mlflow.tracking.MlflowClient()
        scaler_path = client.download_artifacts(training_run_id, "preprocessing/scaler.pkl")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        X_test_scaled = scaler.transform(X_test)

        model_uri = f"runs:/{training_run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)

        y_pred = model.predict(X_test_scaled)

        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision_weighted": precision_score(
                y_test, y_pred, average="weighted"
            ),
            "test_recall_weighted": recall_score(y_test, y_pred, average="weighted"),
            "test_f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        }

        with mlflow.start_run(run_id=training_run_id):
            mlflow.log_metrics(metrics)

            report = classification_report(y_test, y_pred, output_dict=True)
            report_path = "/tmp/classification_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(report_path, artifact_path="evaluation")

        return metrics

    # DAG flow
    raw_key = get_data()
    processed = process_data(raw_key)
    splits = split_dataset(processed)
    run_id = train_model(splits)
    evaluate_model(run_id, splits)


wine_quality_pipeline()
