import numpy as np
import pytest
from src.model import LinearRegressionGD


@pytest.fixture
def simple_data():
    # A trivial dataset where y = 2*x — the model should learn weights ≈ [2]
    np.random.seed(0)
    X = np.random.randn(100, 1)
    y = 2 * X.ravel()
    return X, y


def test_loss_decreases(simple_data):
    X, y = simple_data
    model = LinearRegressionGD(learning_rate=0.1, n_iterations=100)
    model.fit(X, y)

    # Loss at the end should be lower than loss at the start
    assert model.loss_history[-1] < model.loss_history[0]


def test_loss_history_length(simple_data):
    X, y = simple_data
    n = 50
    model = LinearRegressionGD(learning_rate=0.1, n_iterations=n)
    model.fit(X, y)

    assert len(model.loss_history) == n


def test_predict_shape(simple_data):
    X, y = simple_data
    model = LinearRegressionGD(learning_rate=0.1, n_iterations=100)
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == y.shape


def test_learns_simple_relationship(simple_data):
    # On y = 2x with no noise the model should recover weight ≈ 2.0
    X, y = simple_data
    model = LinearRegressionGD(learning_rate=0.1, n_iterations=500)
    model.fit(X, y)

    assert abs(model.weights[0] - 2.0) < 0.01
