from dataset_pipeline import DatasetPipeline
from models.model_protocol import EvaluationResult, ModelProtocol


class ModelsPipeline:
    def __init__(
        self, models: list[type[ModelProtocol]], dataset_pipeline: DatasetPipeline
    ):
        self.models = models
        self.dataset_pipeline = dataset_pipeline

    def evaluate(self) -> list[EvaluationResult]:
        return [model().run(self.dataset_pipeline) for model in self.models]
