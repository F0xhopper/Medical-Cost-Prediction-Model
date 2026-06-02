import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Run the model on test data and return RMSE and R².

    Keeping evaluation in its own module means train.py stays clean and
    you can call this from tests without importing the training loop.
    """
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {"rmse": rmse, "r2": r2}


def print_metrics(metrics: dict) -> None:
    print(f"  RMSE : ${metrics['rmse']:,.2f}")
    print(f"  R²   : {metrics['r2']:.4f}")
