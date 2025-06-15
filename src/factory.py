from dataset_pipeline import DatasetPipelineBuilder
from models.xgboost_model import XGBoostModel
from models_pipeline import ModelsPipeline
from utils.data_utils import DatasetCleaner
from utils.filesystem_utils import DatasetLoader


def create_default_data_pipeline():
    return DatasetPipelineBuilder.default()


def create_data_pipeline_from_path_without_balancer(path: str):
    return (
        DatasetPipelineBuilder(DatasetLoader, DatasetCleaner)
        .with_path(path)
        .with_features(DatasetPipelineBuilder.get_standards_features())
        .build()
    )


def create_data_pipeline_from_path_with_oversampling_balancer(path: str):
    return (
        DatasetPipelineBuilder(DatasetLoader, DatasetCleaner)
        .with_path(path)
        .with_features(DatasetPipelineBuilder.get_standards_features())
        .build()
    )


def create_models_pipeline():
    return ModelsPipeline([XGBoostModel], create_default_data_pipeline())
