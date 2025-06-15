from typing import Protocol, Tuple
import pandas as pd


class BalancerProtocol(Protocol):
    def fit_resample(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]: ...
