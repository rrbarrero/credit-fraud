import pandas as pd
from unittest.mock import MagicMock
from models.lightgbm_model import LightGBMModel
from domain.evaluation_result import EvaluationResult


def test_lightgbm_model_run():
    mock_dataset_pipeline = MagicMock()
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y_train = pd.Series([0, 1, 0])
    X_test = pd.DataFrame({"feature1": [7, 8, 9], "feature2": [10, 11, 12]})
    y_test = pd.Series([1, 0, 1])
    mock_dataset_pipeline.split.return_value = (X_train, X_test, y_train, y_test)

    model = LightGBMModel(random_state=42)

    result = model.run(mock_dataset_pipeline)

    assert isinstance(result, EvaluationResult)
    assert result.model_name == "lightgbm_v1"
    assert result.pr_auc > 0.0
    assert result.accuracy > 0.0
    assert len(result.per_class) > 0
    mock_dataset_pipeline.split.assert_called_once_with(test_size=0.3, use_cache=True)
