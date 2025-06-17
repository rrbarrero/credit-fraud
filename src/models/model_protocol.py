from typing import Protocol

from dataset_pipeline import DatasetPipeline
from domain.evaluation_result import EvaluationResult


class ModelProtocol(Protocol):
    def run(self, dataset_pipeline: DatasetPipeline) -> EvaluationResult: ...
