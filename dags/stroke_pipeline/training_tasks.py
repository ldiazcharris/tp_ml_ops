from airflow.decorators import task

from stroke_pipeline.config import MODEL_REQUIREMENTS


def build_train_model_task():

    @task.virtualenv(task_id="train_model", requirements=MODEL_REQUIREMENTS)
    def train_model(split_result: dict) -> str:
        import json
        import os

        import boto3
        import mlflow
        import pandas as pd
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        numeric_features = ["age", "avg_glucose_level", "bmi"]
        binary_features = ["hypertension", "heart_disease"]
        categorical_features = [
            "gender",
            "ever_married",
            "work_type",
            "Residence_type",
            "smoking_status",
        ]
        feature_names = numeric_features + binary_features + categorical_features

        bucket_name = os.getenv("MLFLOW_BUCKET_NAME", "mlflow-artifacts")
        model_name = os.getenv("MODEL_NAME", "stroke_prediction_model")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "stroke_prediction")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
            "MLFLOW_S3_ENDPOINT_URL", "http://s3:9000"
        )

        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
        )

        train_path = "/tmp/stroke_train.csv"
        s3.download_file(bucket_name, split_result["train_key"], train_path)
        train_df = pd.read_csv(train_path)

        X_train = train_df[feature_names]
        y_train = train_df["stroke"]

        numeric_pipeline = SklearnPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        binary_pipeline = SklearnPipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
        )
        categorical_pipeline = SklearnPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_features),
                ("binary", binary_pipeline, binary_features),
                ("categorical", categorical_pipeline, categorical_features),
            ]
        )

        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="stroke_random_forest_training") as run:
            mlflow.log_params(
                {
                    "model_name": model_name,
                    "model_type": "RandomForestClassifier",
                    "n_estimators": 300,
                    "max_depth": 10,
                    "min_samples_leaf": 2,
                    "smote": True,
                    "feature_count": len(feature_names),
                    "train_rows": len(train_df),
                }
            )

            model_pipeline.fit(X_train, y_train)

            train_probabilities = model_pipeline.predict_proba(X_train)[:, 1]
            train_predictions = (train_probabilities >= 0.5).astype(int)
            train_accuracy = float((train_predictions == y_train).mean())
            mlflow.log_metric("train_accuracy_at_0_5", train_accuracy)

            feature_spec_path = "/tmp/stroke_feature_spec.json"
            with open(feature_spec_path, "w") as feature_spec_file:
                json.dump(
                    {
                        "numeric_features": numeric_features,
                        "binary_features": binary_features,
                        "categorical_features": categorical_features,
                    },
                    feature_spec_file,
                    indent=2,
                )

            mlflow.log_artifact(feature_spec_path, artifact_path="preprocessing")
            mlflow.sklearn.log_model(
                model_pipeline,
                artifact_path="model",
                registered_model_name=model_name,
            )

            return run.info.run_id

    return train_model
