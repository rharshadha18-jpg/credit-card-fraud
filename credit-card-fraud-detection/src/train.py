import joblib
import os
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.preprocess import load_and_split

def get_models():
    return {
        "lr_weighted": LogisticRegression(class_weight="balanced", max_iter=1000),
        "rf_weighted": RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42),
        "rf_smote":    RandomForestClassifier(n_estimators=100, random_state=42),
        "xgb":         XGBClassifier(scale_pos_weight=577, eval_metric="aucpr", random_state=42),
    }

def train_all(data_path="data/creditcard.csv"):
    X_train, X_test, y_train, y_test = load_and_split(data_path)

    os.makedirs("models", exist_ok=True)
    trained = {}

    for name, model in get_models().items():
        Xtr, ytr = X_train.copy(), y_train.copy()

        if name == "rf_smote":
            Xtr, ytr = SMOTE(random_state=42).fit_resample(Xtr, ytr)

        model.fit(Xtr, ytr)
        joblib.dump(model, f"models/{name}.pkl")
        trained[name] = model
        print(f"Trained and saved: {name}")

    joblib.dump((X_test, y_test), "models/test_data.pkl")
    joblib.dump((X_train, y_train), "models/train_data.pkl")
    return trained, X_test, y_test

if __name__ == "__main__":
    train_all()
