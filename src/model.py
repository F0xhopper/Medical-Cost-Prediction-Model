import numpy as np


class LinearRegressionGD:
    """
    Linear Regression trained with Gradient Descent (from scratch).

    The model learns a line (or hyperplane in multiple dimensions):
        y_pred = X @ weights + bias

    It does this by repeatedly nudging weights and bias in the direction
    that reduces the error — this process is called gradient descent.
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        """
        Parameters
        ----------
        learning_rate : how big a step we take each iteration.
            - Too large → the loss bounces around or explodes (overshooting).
            - Too small → training takes forever to converge.
        n_iterations : how many times we loop over the full dataset.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        self.weights = None
        self.bias = None

        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Run gradient descent to learn weights and bias from training data.

        The math in plain English:
          1. Make a prediction with current weights.
          2. Measure how wrong we were (MSE loss).
          3. Ask: "which direction should I nudge each weight to reduce the loss?"
             That direction is the negative gradient.
          4. Take a small step in that direction (scaled by learning_rate).
          5. Repeat for n_iterations.

        Parameters
        ----------
        X : training features, shape (n_samples, n_features)
        y : true target values, shape (n_samples,)
        """
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias

            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)

            errors = y - y_pred
            dw = -(2 / n_samples) * (X.T @ errors)

            db = -(2 / n_samples) * np.sum(errors)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the learned weights to new data.

        Parameters
        ----------
        X : feature matrix, shape (n_samples, n_features)

        Returns
        -------
        Predicted values, shape (n_samples,)
        """
        return X @ self.weights + self.bias
