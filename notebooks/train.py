# %% [markdown]
# # Linear Regression from Scratch — Training Notebook
#
# This notebook wires together all the pieces:
#   1. Preprocess the data  (src/preprocessing.py)
#   2. Train our custom model  (src/model.py)
#   3. Evaluate on the test set
#   4. Plot the loss curve
#   5. Tune hyperparameters and compare against sklearn

# %% Imports
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Notebooks live one level below the project root, so we add the root to the
# path. This is notebook-only boilerplate — train.py at the root doesn't need it.
sys.path.append("..")

from src.preprocessing import load_and_preprocess
from src.model import LinearRegressionGD
from src.evaluate import evaluate, print_metrics

# %% [markdown]
# ## Step 1 — Load and preprocess the data
#
# load_and_preprocess() handles all the encoding, splitting, and scaling.
# After this call:
#   - X_train / X_test are numpy arrays with scaled numeric values
#   - y_train / y_test are the raw charge amounts (in dollars)

# %% Load data
X_train, X_test, y_train, y_test = load_and_preprocess()

print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")

# %% [markdown]
# ## Step 2 — Train the model
#
# We create a LinearRegressionGD instance, choosing:
#   - learning_rate = 0.1  (a reasonable starting point for scaled data)
#   - n_iterations  = 1000 (enough for the loss to plateau on this dataset)
#
# You'll experiment with these values in Step 4.

# %% Train
model = LinearRegressionGD(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

print(f"Final training loss (MSE): {model.loss_history[-1]:,.2f}")

# %% [markdown]
# ## Step 3 — Evaluate on the test set
#
# Two standard regression metrics:
#
#   RMSE (Root Mean Squared Error)
#     - Same units as the target (dollars here), so it's interpretable:
#       "on average our predictions are off by £X"
#     - Penalises large errors more than small ones.
#
#   R² (Coefficient of Determination)
#     - Ranges from 0 to 1 (can be negative for very bad models).
#     - 1.0 = perfect predictions, 0.0 = no better than predicting the mean.
#     - Rule of thumb: > 0.7 is considered decent for real-world tabular data.

# %% Predict and score
# evaluate() lives in src/evaluate.py — the same function train.py uses,
# so metrics are computed identically in the notebook and in production.
metrics = evaluate(model, X_test, y_test)
print_metrics(metrics)

rmse = metrics["rmse"]
r2   = metrics["r2"]

# %% [markdown]
# ## Step 4 — Plot the loss curve
#
# If gradient descent is working correctly the loss should fall steeply at first
# and then level off (converge). If it's zigzagging or going UP, the learning
# rate is too high.

# %% Loss curve
plt.figure(figsize=(8, 4))
plt.plot(model.loss_history)
plt.title("Training Loss over Iterations")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 5 — Tune hyperparameters
#
# We train the same model three times with different learning rates and overlay
# the loss curves so you can see the effect directly.
#
# Expected behaviour:
#   - 0.001  → converges slowly, loss still decreasing at iteration 1000
#   - 0.01   → converges at a moderate pace
#   - 0.1    → converges quickly (this is usually the sweet spot here)
#   - 1.0    → often explodes (loss goes to infinity) — try it and see!

# %% Compare learning rates
learning_rates = [0.001, 0.01, 0.1]

plt.figure(figsize=(10, 5))

for lr in learning_rates:
    m = LinearRegressionGD(learning_rate=lr, n_iterations=1000)
    m.fit(X_train, y_train)
    plt.plot(m.loss_history, label=f"lr={lr}")

plt.title("Loss Curves for Different Learning Rates")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 6 — Sanity check against sklearn
#
# sklearn's LinearRegression uses an exact algebraic solver (not gradient descent),
# so it finds the mathematically optimal weights in one shot.
#
# Our GD model should get very close to the same R² — if it doesn't, the model
# hasn't converged yet (try more iterations or a better learning rate).

# %% sklearn comparison
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train)
sklearn_pred = sklearn_model.predict(X_test)

sklearn_rmse = np.sqrt(mean_squared_error(y_test, sklearn_pred))
sklearn_r2   = r2_score(y_test, sklearn_pred)

print("── Our GD model ──────────────────")
print(f"  RMSE : ${rmse:,.2f}")
print(f"  R²   : {r2:.4f}")
print()
print("── sklearn LinearRegression ──────")
print(f"  RMSE : ${sklearn_rmse:,.2f}")
print(f"  R²   : {sklearn_r2:.4f}")
print()

# The gap between them tells you how close your gradient descent got to the
# true optimal solution. A gap < 0.01 R² is excellent.
gap = abs(r2 - sklearn_r2)
print(f"R² gap: {gap:.4f}  {'✓ converged well' if gap < 0.01 else '⚠ try more iterations or adjust lr'}")

# %% [markdown]
# ## Recap — what to try next
#
# - Change n_iterations to 500 vs 2000 and see when the curve flattens.
# - Try learning_rate=1.0 and observe the loss explosion.
# - The R² for this dataset typically lands around 0.75–0.78.
#   The ceiling is limited by the data, not the model — some of the variance
#   in insurance charges (e.g. individual health history) simply isn't in our features.
