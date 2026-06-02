import numpy as np
from src.preprocessing import load_and_preprocess


def test_output_shapes():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
    total = len(y_train) + len(y_test)

    # 80/20 split — check sizes are in the right ballpark
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert abs(len(X_train) / total - 0.8) < 0.01


def test_feature_count():
    # 5 original numeric/binary cols + 3 one-hot region cols = 8 features
    X_train, X_test, *_ = load_and_preprocess()
    assert X_train.shape[1] == 8
    assert X_test.shape[1] == 8


def test_scaling():
    # StandardScaler fitted on train should produce ~zero mean on train set
    X_train, *_ = load_and_preprocess()
    assert np.abs(X_train.mean()) < 0.01


def test_no_missing_values():
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess()
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    assert not np.isnan(y_train).any()
    assert not np.isnan(y_test).any()
