import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
SELECTED_FEATURES = ["koi_period", "koi_duration", "koi_prad", "koi_depth"]

def evaluate(model, X_test, y_test, label):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")
    print(f"\n {label}")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1-score: {f1:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))


def classify_user_input(model, feature_names):
    """Prompt for feature values, run the trained model, and print a verdict."""
    print("\nEnter feature values to classify a candidate:")
    values = []
    for feature in feature_names:
        while True:
            try:
                raw = input(f"{feature}: ").strip()
            except EOFError:
                print("\nNo manual input detected; skipping interactive prediction.")
                return
            if not raw:
                print("Please enter a numeric value.")
                continue
            try:
                values.append(float(raw))
                break
            except ValueError:
                print("Please enter a numeric value.")

    sample = pd.DataFrame([values], columns=feature_names, dtype=float)
    pred = model.predict(sample)[0]
    verdict = "Candidate exoplanet" if pred == 1 else "Likely false positive"
    print(f"\nPrediction: {verdict}")

df = pd.read_csv("KOI.csv", comment="#")

if "koi_pdisposition" in df.columns:
    df = df[df["koi_pdisposition"].isin(["CANDIDATE", "FALSE POSITIVE"])].copy()
    y = (df["koi_pdisposition"] == "CANDIDATE").astype(int)
elif "koi_disposition" in df.columns:
    df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
    y = (df["koi_disposition"] == "CONFIRMED").astype(int)
else:
    raise ValueError("Expected 'koi_pdisposition' or 'koi_disposition' in KOI.csv")

missing_features = [c for c in SELECTED_FEATURES if c not in df.columns]
if missing_features:
    raise ValueError(f"Dataset missing expected features: {missing_features}")

X = df[SELECTED_FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
rf.fit(X_train, y_train)
evaluate(rf, X_test, y_test, "Random Forest")
import joblib
joblib.dump(rf, "model.joblib")  # saves the trained RandomForest to a file
classify_user_input(rf, SELECTED_FEATURES)