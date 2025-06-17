from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_recall_curve, auc

from models.model_protocol import EvaluationResult
from dataset_pipeline import DatasetPipeline
from config import settings


class XGBoostModel:
    def __init__(self, random_state: int | None = None):

        self.random_state = random_state or settings.random_state

    def run(self, dataset_pipeline: DatasetPipeline) -> EvaluationResult:

        X_train, X_test, y_train, y_test = dataset_pipeline.split(
            test_size=0.3, use_cache=True
        )

        neg, pos = (len(y_train) - sum(y_train)), sum(y_train)
        scale_pos_weight = neg / pos

        model = Pipeline(
            [
                (
                    "clf",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        random_state=self.random_state,
                        n_jobs=-1,
                        colsample_bytree=0.7,
                        gamma=0,
                        learning_rate=0.05,
                        max_depth=7,
                        n_estimators=300,
                        subsample=0.7,
                    ),
                )
            ]
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)

        return EvaluationResult.from_report(
            model_name="xgboost_v1",
            report_dict=classification_report(y_test, y_pred, output_dict=True),  # type: ignore
            pr_auc=pr_auc,  # type: ignore
        )
