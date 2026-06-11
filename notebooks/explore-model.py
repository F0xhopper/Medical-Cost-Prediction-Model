import joblib

model = joblib.load("models/linear_regression.pkl")

print("Learned weights:", model.weights)

print("Learned bias:", model.bias)

print("Final training loss (MSE):", model.loss_history[-1])
