from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_recall_curve, auc

from pipeline import PipelineBuilder

data_pipeline = PipelineBuilder.default()
X_train, X_test, y_train, y_test = data_pipeline.split(test_size=0.3, use_cache=True)

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
                random_state=42,
                n_jobs=-1,
            ),
        )
    ]
)


model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)
print(f"PR AUC: {pr_auc:.4f}")
