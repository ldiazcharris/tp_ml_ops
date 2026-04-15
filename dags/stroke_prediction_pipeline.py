from airflow.decorators import dag

from stroke_pipeline.config import (
    PIPELINE_DESCRIPTION,
    PIPELINE_ID,
    PIPELINE_START_DATE,
    PIPELINE_TAGS,
)
from stroke_pipeline.data_tasks import (
    build_ensure_artifact_bucket_task,
    build_get_data_task,
    build_process_data_task,
    build_split_dataset_task,
)
from stroke_pipeline.evaluation_tasks import build_evaluate_model_task
from stroke_pipeline.serving_tasks import build_reload_prediction_api_task
from stroke_pipeline.training_tasks import build_train_model_task


@dag(
    dag_id=PIPELINE_ID,
    description=PIPELINE_DESCRIPTION,
    start_date=PIPELINE_START_DATE,
    schedule=None,
    catchup=False,
    tags=PIPELINE_TAGS,
)
def stroke_prediction_pipeline():

    ensure_artifact_bucket = build_ensure_artifact_bucket_task()
    get_data = build_get_data_task()
    process_data = build_process_data_task()
    split_dataset = build_split_dataset_task()
    train_model = build_train_model_task()
    evaluate_model = build_evaluate_model_task()
    reload_prediction_api = build_reload_prediction_api_task()

    bucket_name = ensure_artifact_bucket()
    raw_key = get_data(bucket_name)
    processed = process_data(raw_key)
    splits = split_dataset(processed)
    training_run_id = train_model(splits)
    evaluation_metrics = evaluate_model(training_run_id, splits)
    reload_prediction_api(training_run_id, evaluation_metrics)


stroke_prediction_pipeline()
