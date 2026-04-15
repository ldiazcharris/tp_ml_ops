from airflow.decorators import task

from stroke_pipeline.config import MODEL_REQUIREMENTS


def build_evaluate_model_task():
    """Devuelve la tarea que ajusta threshold y registra métricas finales."""

    @task.virtualenv(task_id="evaluate_model", requirements=MODEL_REQUIREMENTS)
    def evaluate_model(training_run_id: str, split_result: dict) -> dict:
        import json
        import os

        import boto3
        import mlflow
        import numpy as np
        import pandas as pd
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            brier_score_loss,
            classification_report,
            confusion_matrix,
            fbeta_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        feature_names = [
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "avg_glucose_level",
            "bmi",
            "smoking_status",
        ]

        def find_best_threshold(y_true, probabilities, beta=2.0):
            thresholds = np.linspace(0.05, 0.95, 181)
            scores = [
                fbeta_score(y_true, (probabilities >= threshold).astype(int), beta=beta)
                for threshold in thresholds
            ]
            best_index = int(np.argmax(scores))
            return float(thresholds[best_index]), float(scores[best_index])

        bucket_name = os.getenv("MLFLOW_BUCKET_NAME", "mlflow-artifacts")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
            "MLFLOW_S3_ENDPOINT_URL", "http://s3:9000"
        )
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
        )

        validation_path = "/tmp/stroke_validation.csv"
        test_path = "/tmp/stroke_test.csv"
        s3.download_file(bucket_name, split_result["validation_key"], validation_path)
        s3.download_file(bucket_name, split_result["test_key"], test_path)

        validation_df = pd.read_csv(validation_path)
        test_df = pd.read_csv(test_path)

        X_validation = validation_df[feature_names]
        y_validation = validation_df["stroke"]
        X_test = test_df[feature_names]
        y_test = test_df["stroke"]

        model_uri = f"runs:/{training_run_id}/model"
        model_pipeline = mlflow.sklearn.load_model(model_uri)

        validation_probabilities = model_pipeline.predict_proba(X_validation)[:, 1]
        decision_threshold, validation_f2 = find_best_threshold(
            y_validation, validation_probabilities
        )

        test_probabilities = model_pipeline.predict_proba(X_test)[:, 1]
        test_predictions = (test_probabilities >= decision_threshold).astype(int)

        metrics = {
            "validation_f2": validation_f2,
            "decision_threshold": decision_threshold,
            "test_accuracy": accuracy_score(y_test, test_predictions),
            "test_precision": precision_score(y_test, test_predictions, zero_division=0),
            "test_recall": recall_score(y_test, test_predictions, zero_division=0),
            "test_f1": fbeta_score(y_test, test_predictions, beta=1, zero_division=0),
            "test_f2": fbeta_score(y_test, test_predictions, beta=2, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, test_probabilities),
            "test_pr_auc": average_precision_score(y_test, test_probabilities),
            "test_brier_score": brier_score_loss(y_test, test_probabilities),
        }

        report = classification_report(
            y_test,
            test_predictions,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(y_test, test_predictions).tolist()

        report_path = "/tmp/stroke_classification_report.json"
        confusion_path = "/tmp/stroke_confusion_matrix.json"
        with open(report_path, "w") as report_file:
            json.dump(report, report_file, indent=2)
        with open(confusion_path, "w") as confusion_file:
            json.dump({"confusion_matrix": matrix}, confusion_file, indent=2)

        with mlflow.start_run(run_id=training_run_id):
            mlflow.log_param("decision_threshold", decision_threshold)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(report_path, artifact_path="evaluation")
            mlflow.log_artifact(confusion_path, artifact_path="evaluation")

        return metrics

    return evaluate_model
