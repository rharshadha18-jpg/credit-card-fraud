import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_split(path="data/creditcard.csv", test_size=0.2, random_state=42):
    df = pd.read_csv(path)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Split before scaling to prevent leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Fit scaler on training data only, then apply to both sets
    scaler = StandardScaler()
    X_train[["Amount", "Time"]] = scaler.fit_transform(X_train[["Amount", "Time"]])
    X_test[["Amount", "Time"]]  = scaler.transform(X_test[["Amount", "Time"]])

    return X_train, X_test, y_train, y_test
