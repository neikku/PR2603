"""
Zaženi enkrat pred app.py:
    python train_model.py
Generira data/model2.pkl in data/model2_meta.pkl
"""
import json, re, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV

CURRENT_YEAR = 2026

def extract_num(v):
    if pd.isna(v): return np.nan
    m = re.search(r"\d+", str(v).replace(",","").replace("\xa0",""))
    return float(m.group()) if m else np.nan

def extract_year(v):
    if pd.isna(v): return np.nan
    m = re.search(r"(19|20)\d{2}", str(v))
    return float(m.group()) if m else np.nan

def extract_engine(v):
    if pd.isna(v): return np.nan
    m = re.search(r"[\d,]+", str(v).replace("\xa0",""))
    return float(m.group().replace(",","")) if m else np.nan

PRICE_MAP = {
    "Very good price": 0, "Good price": 1, "Fair price": 2,
    "Increased price": 3, "High price": 4
}

rows = []
for file in Path("data/top5").glob("*.json"):
    with open(file, encoding="utf-8") as f:
        cars = json.load(f)
    for car in cars:
        a = car.get("attributes") or {}
        d = car.get("dealerDetails") or {}
        feats = car.get("features") or []
        rows.append({
            "price":         car.get("price.total.amount"),
            "brand":         car.get("brand"),
            "model":         car.get("model"),
            "km":            a.get("Mileage"),
            "power":         a.get("Power"),
            "engine":        a.get("Cubic Capacity"),
            "year":          a.get("First Registration"),
            "owners":        a.get("Number of Vehicle Owners"),
            "fuel":          a.get("Fuel"),
            "gear":          a.get("Transmission"),
            "condition":     a.get("Vehicle condition"),
            "feature_count": len(feats),
            "rating":        PRICE_MAP.get((car.get("priceRating") or {}).get("rating")),
        })

df = pd.DataFrame(rows)
for c in ["price", "km", "power", "owners", "feature_count", "rating"]:
    df[c] = df[c].apply(extract_num)
df["engine"] = df["engine"].apply(extract_engine)
df["year"]   = df["year"].apply(extract_year)
df["age"]    = (CURRENT_YEAR - df["year"]).clip(lower=0)

df = df.dropna(subset=["price"])
low, high = df.price.quantile(.03), df.price.quantile(.97)
df = df[(df.price > low) & (df.price < high)]
print(f"Vrstic za trening: {len(df)}")

X = df.drop(columns=["price"])
y = np.log1p(df.price)

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

prep = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), cat_cols),
])

pipe = Pipeline([
    ("prep", prep),
    ("model", StackingRegressor(
        estimators=[
            ("xt", ExtraTreesRegressor(n_estimators=300, random_state=42)),
            ("gb", HistGradientBoostingRegressor(max_iter=300)),
        ],
        final_estimator=RidgeCV()
    ))
])

scores = cross_val_score(pipe, X, y, cv=KFold(5, shuffle=True, random_state=42),
                         scoring="neg_mean_absolute_error")
print(f"5-fold CV MAE (log): {-scores.mean():.4f}")

pipe.fit(X, y)
pred = np.expm1(pipe.predict(X))
mae = mean_absolute_error(np.expm1(y), pred)
print(f"Train MAE: {mae:.0f} EUR")

pickle.dump(pipe, open("data/model2.pkl", "wb"))
pickle.dump({"num_cols": num_cols, "cat_cols": cat_cols}, open("data/model2_meta.pkl", "wb"))
print("Shranjeno: data/model2.pkl, data/model2_meta.pkl")
