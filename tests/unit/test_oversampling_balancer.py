import pandas as pd
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from balancers.oversampling_balancer import OversamplingBalancer


@pytest.fixture
def toy_dataset():

    X = pd.DataFrame(
        {
            "amountBin": ["A"] * 6 + ["B"] * 6,
            "feature1": np.arange(12),
        }
    )
    y = pd.Series([0] * 6 + [1] * 6, name="Class")
    return train_test_split(X, y, stratify=y, test_size=0.25, random_state=0)


def test_return_types(toy_dataset):
    X_tr, X_te, y_tr, _ = toy_dataset
    balancer = OversamplingBalancer()
    X_tr_bal, y_tr_bal, X_te_bal = balancer.fit_resample(X_tr, y_tr, X_te)

    assert isinstance(X_tr_bal, pd.DataFrame)
    assert isinstance(X_te_bal, pd.DataFrame)
    assert isinstance(y_tr_bal, pd.Series)


def test_train_is_perfectly_balanced(toy_dataset):
    X_tr, X_te, y_tr, _ = toy_dataset
    X_tr_bal, y_tr_bal, _ = OversamplingBalancer().fit_resample(X_tr, y_tr, X_te)

    counts = y_tr_bal.value_counts().to_dict()
    assert counts[0] == counts[1]


def test_one_hot_and_passthrough_columns(toy_dataset):
    X_tr, X_te, y_tr, _ = toy_dataset
    X_tr_bal, _, X_te_bal = OversamplingBalancer().fit_resample(X_tr, y_tr, X_te)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(X_tr[["amountBin"]])
    prefixed_ohe_cols = ["ohe__" + c for c in ohe.get_feature_names_out(["amountBin"])]

    expected_cols = set(prefixed_ohe_cols + ["remainder__feature1"])
    assert set(X_tr_bal.columns) == expected_cols
    assert set(X_te_bal.columns) == expected_cols


def test_column_order_identical(toy_dataset):
    X_tr, X_te, y_tr, _ = toy_dataset
    X_tr_bal, _, X_te_bal = OversamplingBalancer().fit_resample(X_tr, y_tr, X_te)

    assert list(X_tr_bal.columns) == list(X_te_bal.columns)


def test_unknown_category_is_handled():

    X_train = pd.DataFrame(
        {"amountBin": ["A", "A", "B", "B"], "feature1": [1, 2, 3, 4]}
    )
    y_train = pd.Series([0, 0, 1, 1], name="Class")

    X_test = pd.DataFrame({"amountBin": ["A", "C"], "feature1": [5, 6]}, index=[10, 11])

    balancer = OversamplingBalancer()
    X_tr_bal, y_tr_bal, X_te_bal = balancer.fit_resample(X_train, y_train, X_test)

    assert y_tr_bal.value_counts().to_dict()[0] == y_tr_bal.value_counts().to_dict()[1]

    assert list(X_tr_bal.columns) == list(X_te_bal.columns)
    assert X_te_bal.select_dtypes(include="object").empty
