"""
Entry point for training the insurance cost model.

Run from the project root:
    python train.py
"""

import joblib
from pathlib import Path

from src.preprocessing import load_and_preprocess
from src.model import LinearRegressionGD
from src.evaluate import evaluate, print_metrics

LEARNING_RATE = 0.1
N_ITERATIONS = 1000
DATA_PATH = "data/insurance.csv"
MODELS_DIR = Path("models")


def main():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess(DATA_PATH)
    print(f"  Train: {X_train.shape[0]} samples  |  Test: {X_test.shape[0]} samples\n")

    print("Training LinearRegressionGD...")
    model = LinearRegressionGD(learning_rate=LEARNING_RATE, n_iterations=N_ITERATIONS)
    model.fit(X_train, y_train)
    print(f"  Final training loss (MSE): {model.loss_history[-1]:,.2f}\n")

    print("Evaluating on test set...")
    metrics = evaluate(model, X_test, y_test)
    print_metrics(metrics)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "linear_regression.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print(f"\nSaved model and scaler to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
