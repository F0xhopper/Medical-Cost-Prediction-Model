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

        # These get set properly inside fit() once we know how many features X has.
        # We initialise to None so it's clear they don't exist yet.
        self.weights = None
        self.bias = None

        # We'll append the loss after every iteration so we can plot it later
        # and visually confirm the model is actually learning (loss should decrease).
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

        # Start weights at zero — gradient descent will move them from here.
        # Shape (n_features,) so matrix multiplication X @ weights works correctly.
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            # ── Forward pass: make predictions with current weights ──────────
            # X @ weights is a dot product: for each sample, multiply each
            # feature value by its corresponding weight, then sum them up.
            y_pred = X @ self.weights + self.bias

            # ── Loss: Mean Squared Error ─────────────────────────────────────
            # MSE = average of (actual - predicted)^2
            # Squaring means large errors are penalised more than small ones,
            # and the loss is always positive regardless of sign of the error.
            loss = np.mean((y - y_pred) ** 2)
            self.loss_history.append(loss)

            # ── Gradients: calculus tells us the slope of the loss surface ───
            # These come from differentiating MSE with respect to weights/bias.
            #
            # dL/dw = -(2/n) * X^T @ (y - y_pred)
            # In words: how much does the loss change if we nudge each weight?
            # X.T has shape (n_features, n_samples), so X.T @ errors gives
            # a (n_features,) vector — one gradient value per weight.
            errors = y - y_pred
            dw = -(2 / n_samples) * (X.T @ errors)

            # dL/db = -(2/n) * sum(y - y_pred)
            # The bias is a single number, so its gradient is a scalar.
            db = -(2 / n_samples) * np.sum(errors)

            # ── Update: move weights opposite to the gradient ────────────────
            # If the gradient is positive, the loss increases as we increase the
            # weight, so we subtract. If negative, we add.
            # learning_rate controls the step size.
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self  # allows chaining: model.fit(X, y).predict(X_test)

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
