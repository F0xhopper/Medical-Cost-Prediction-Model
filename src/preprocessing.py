import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(data_path: str = "data/insurance.csv"):
    df = pd.read_csv(data_path)

    df["sex"] = df["sex"].map({"female": 0, "male": 1})
    df["age_squared"] = df["age"] ** 2
    df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
    df["smoker_bmi"] = df["smoker"] * df["bmi"]

    df = pd.get_dummies(df, columns=["region"], drop_first=True)

    X = df.drop(columns=["charges"])
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
