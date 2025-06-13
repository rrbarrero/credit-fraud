from pipeline import PipelineBuilder
from utils.data_utils import DatasetCleaner
from utils.filesystem_utils import DatasetLoader
from config import settings
from features.hour_of_day_feature import HourOfDayFeature
from features.time_hours_feature import TimeHoursFeature
from features.time_since_previous_feature import TimeSincePreviousFeature
from balancers.oversampling_balancer import OversamplingBalancer


def pipeline_with_oversampling_balancer():
    dataset_path = str(settings.data_path / "creditcard.csv.zip")
    features = [TimeHoursFeature, HourOfDayFeature, TimeSincePreviousFeature]
    return (
        PipelineBuilder(DatasetLoader, DatasetCleaner)
        .with_balancer(OversamplingBalancer)
        .with_features(features)
        .with_path(dataset_path)
        .build()
    )
