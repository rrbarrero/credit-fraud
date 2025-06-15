from pipeline import PipelineBuilder
from utils.data_utils import DatasetCleaner
from utils.filesystem_utils import DatasetLoader
from balancers.oversampling_balancer import OversamplingBalancer


def create_default_data_pipeline():
    return PipelineBuilder.default()


def create_data_pipeline_from_path_without_balancer(path: str):
    return (
        PipelineBuilder(DatasetLoader, DatasetCleaner)
        .with_path(path)
        .with_features(PipelineBuilder.get_standards_features())
        .build()
    )


def create_data_pipeline_from_path_with_oversampling_balancer(path: str):
    return (
        PipelineBuilder(DatasetLoader, DatasetCleaner)
        .with_path(path)
        .with_features(PipelineBuilder.get_standards_features())
        .with_balancer(OversamplingBalancer)
        .build()
    )
