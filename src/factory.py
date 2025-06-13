from pipeline import PipelineBuilder
from utils.data_utils import DatasetCleaner
from utils.filesystem_utils import DatasetLoader
from features.hour_of_day_feature import HourOfDayFeature
from features.time_hours_feature import TimeHoursFeature
from features.time_since_previous_feature import TimeSincePreviousFeature
from balancers.oversampling_balancer import OversamplingBalancer


def create_default_data_pipeline():
    return PipelineBuilder.default()


def create_data_pipeline_from_path_without_balancer(path: str):
    return (
        PipelineBuilder(DatasetLoader, DatasetCleaner)
        .with_path(path)
        .with_features([TimeHoursFeature, HourOfDayFeature, TimeSincePreviousFeature])
        .build()
    )


def create_data_pipeline_from_path_with_oversampling_balancer(path: str):
    return (
        PipelineBuilder(DatasetLoader, DatasetCleaner)
        .with_path(path)
        .with_features([TimeHoursFeature, HourOfDayFeature, TimeSincePreviousFeature])
        .with_balancer(OversamplingBalancer)
        .build()
    )
