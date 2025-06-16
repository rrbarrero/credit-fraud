from typing import Tuple
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from config import settings


class OversamplingBalancer:

    def __init__(self, k_max: int = 5, random_state: int | None = None) -> None:
        self.k_max = k_max
        self.random_state = random_state or settings.random_state

    def _adaptive_k(self, y: pd.Series) -> int:
        n_min = min(Counter(y).values())
        return max(1, min(self.k_max, n_min - 1))

    def fit_resample(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:

        k = self._adaptive_k(y_train)

        preproc = ColumnTransformer(
            transformers=[
                (
                    "ohe",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ["amountBin"],
                )
            ],
            remainder="passthrough",
        )

        smt = SMOTETomek(
            random_state=self.random_state,
            smote=SMOTE(k_neighbors=k, random_state=self.random_state),
        )

        pipe = ImbPipeline(
            [
                ("encode", preproc),
                ("balance", smt),
            ]
        )

        X_train_bal, y_train_bal = pipe.fit_resample(X_train, y_train)  # type: ignore

        X_test_enc = pipe.named_steps["encode"].transform(X_test)
        cols = pipe.named_steps["encode"].get_feature_names_out(X_train.columns)
        X_test_bal = pd.DataFrame(X_test_enc, columns=cols, index=X_test.index)

        return (
            pd.DataFrame(X_train_bal, columns=cols),
            y_train_bal,
            X_test_bal,
        )
