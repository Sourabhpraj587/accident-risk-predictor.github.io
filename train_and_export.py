"""
Vehicle Accident Risk Prediction - training + website export
Trains a Random Forest, prints metrics, and exports the forest to model.json
so the website (index.html) can predict in the browser. GitHub Pages needs no server.

Run:   python train_and_export.py                       (synthetic demo data)
       python train_and_export.py --csv merged_road_safety.csv   (real Kaggle data)
"""
import argparse, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURES = ["driver_age", "weather", "speed_limit"]
TARGET = "high_risk"
WEATHER = {0: "Fine", 1: "Rain", 2: "Snow", 3: "Fog", 4: "High winds"}


def make_synthetic(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(17, 80, n)
    weather = rng.choice(list(WEATHER), n, p=[0.70, 0.15, 0.04, 0.04, 0.07])
    speed = rng.choice([20, 30, 40, 50, 60, 70], n, p=[0.05, 0.40, 0.12, 0.13, 0.20, 0.10])
    score = (speed - 30) / 20 + np.where(weather > 0, 0.6, 0) + np.where(weather >= 2, 0.5, 0)
    score += np.where(age < 25, 0.7, 0) + np.where(age > 70, 0.5, 0)
    prob = 1 / (1 + np.exp(-(score - 1.6 + rng.normal(0, 0.8, n))))
    return pd.DataFrame({"driver_age": age, "weather": weather, "speed_limit": speed,
                         TARGET: (rng.random(n) < prob).astype(int)})


def load_kaggle(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"Age_of_Driver": "driver_age", "Weather_Conditions": "weather",
                            "Speed_limit": "speed_limit"})
    df = df[(df["driver_age"] > 0) & (df["weather"] > 0)]
    # NOTE: real dataset uses codes 1-9 for weather; map them to 0-4 for the website
    # (1 Fine->0, 2 Rain->1, 3 Snow->2, 7 Fog->3, 4/5/6/8 -> 1/1/4...). Adjust to taste:
    wmap = {1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 4, 7: 3, 8: 0, 9: 0}
    df["weather"] = df["weather"].map(wmap).fillna(0).astype(int)
    df[TARGET] = (df["Accident_Severity"] <= 2).astype(int)
    return df[FEATURES + [TARGET]].dropna()


def export_forest(model, path="model.json"):
    trees = []
    for est in model.estimators_:
        t = est.tree_
        v = t.value[:, 0, :]
        p1 = (v[:, 1] / v.sum(axis=1)).round(4)       # P(high risk) at each node
        trees.append({"f": t.feature.tolist(),          # feature index (-2 = leaf)
                      "t": np.round(t.threshold, 3).tolist(),
                      "l": t.children_left.tolist(),
                      "r": t.children_right.tolist(),
                      "p": p1.tolist()})
    with open(path, "w") as f:
        json.dump({"features": FEATURES, "trees": trees}, f, separators=(",", ":"))


def main(csv=None):
    df = load_kaggle(csv) if csv else make_synthetic()
    print(f"Rows: {len(df)} | High-risk share: {df[TARGET].mean():.1%}")
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # smaller forest keeps model.json light for the web
    model = RandomForestClassifier(n_estimators=60, max_depth=8, min_samples_leaf=20,
                                   class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)
    print(f"Test accuracy: {acc:.3f}")
    print(confusion_matrix(y_te, pred))
    print(classification_report(y_te, pred, target_names=["Low risk", "High risk"]))

    export_forest(model)
    with open("metrics.json", "w") as f:
        json.dump({"accuracy": round(float(acc), 3), "rows": int(len(df))}, f)
    print("Saved model.json and metrics.json -> upload them with index.html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    main(ap.parse_args().csv)
