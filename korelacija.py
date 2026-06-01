import json
import re
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


# ==========================================================
# CONFIG
# ==========================================================

DATA_FOLDER = Path("data/top5")


# ==========================================================
# HELPERS
# ==========================================================

def extract_num(x):
    """Izvleče prvo število iz stringa (npr. '74 kW (101 hp)' -> 74)."""
    if pd.isna(x):
        return None
    m = re.search(r"\d+", str(x).replace(",", "").replace("\xa0", ""))
    return int(m.group()) if m else None


def extract_engine_size(x):
    """Izvleče prostornino motorja v ccm (npr. '1,499 ccm' -> 1499)."""
    if pd.isna(x):
        return None
    m = re.search(r"[\d,]+", str(x).replace("\xa0", ""))
    if m:
        return int(m.group().replace(",", ""))
    return None


def clean_dataframe(df):
    CURRENT_YEAR = 2026
    df = df.copy()

    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    for col in ["mileage", "power", "owners"]:
        df[col] = df[col].apply(extract_num)

    df["engine_size"] = df["engine_size"].apply(extract_engine_size)

    df["year"] = df["registration"].astype(str).str.extract(r"(\d{4})").astype(float)

    df["age"] = (CURRENT_YEAR - df["year"]).clip(lower=0)
    df["km_per_year"] = df["mileage"] / (df["age"] + 1)
    df["power_per_engine"] = df["power"] / (df["engine_size"] + 1)
    df["features_per_year"] = df["feature_count"] / (df["age"] + 1)
    df["owners_per_year"] = df["owners"] / (df["age"] + 1)
    df["power_age"] = df["power"] / (df["age"] + 1)

    return df


def create_feature_importance(df, model_name):
    model_df = df.copy()

    for col in model_df.select_dtypes(include="object"):
        model_df[col] = LabelEncoder().fit_transform(model_df[col].astype(str))

    model_df = model_df.drop(columns=["registration"], errors="ignore")
    model_df = model_df.dropna()

    if len(model_df) < 50:
        print(f"{model_name}: not enough rows")
        return

    X = model_df.drop(columns=["price"])
    y = model_df["price"]

    model = RandomForestRegressor(
        n_estimators=800, max_depth=18, min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    importance.plot(kind="barh", ax=ax)
    ax.set_title(f"{model_name} — Feature Importance")
    plt.tight_layout()
    plt.show()


def create_graphs(df, model_name):
    numeric = [
        "price", "mileage", "power", "engine_size", "year", "age",
        "km_per_year", "power_per_engine", "features_per_year",
        "owners_per_year", "power_age", "owners", "feature_count"
    ]

    # ── Korelacijska matrika ──────────────────────────────────────────────────
    corr = df[numeric].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                annot_kws={"size": 8}, ax=ax)
    ax.set_title(f"{model_name} — Korelacija")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # ── Glavni grafi ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    sns.scatterplot(data=df, x="mileage", y="price", ax=ax[0, 0])
    ax[0, 0].set_title("Prevoženi km vs Cena")
    ax[0, 0].set_xlabel("Km")
    ax[0, 0].set_ylabel("Cena (EUR)")

    sns.scatterplot(data=df, x="year", y="price", ax=ax[0, 1])
    ax[0, 1].set_title("Leto registracije vs Cena")
    ax[0, 1].set_xlabel("Leto")
    ax[0, 1].set_ylabel("Cena (EUR)")

    sns.regplot(data=df, x="power", y="price", ax=ax[1, 0])
    ax[1, 0].set_title("Moč motorja (kW) vs Cena")
    ax[1, 0].set_xlabel("Moč (kW)")
    ax[1, 0].set_ylabel("Cena (EUR)")

    sns.boxplot(data=df, x="transmission", y="price", ax=ax[1, 1])
    ax[1, 1].set_title("Menjalnik vs Cena")
    ax[1, 1].set_xlabel("Menjalnik")
    ax[1, 1].set_ylabel("Cena (EUR)")

    plt.suptitle(model_name, fontsize=14)
    plt.tight_layout()
    plt.show()

    # ── Prostornina motorja ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(data=df, x="engine_size", y="price", ax=ax)
    ax.set_title(f"{model_name} — Prostornina motorja vs Cena")
    ax.set_xlabel("Prostornina (ccm)")
    ax.set_ylabel("Cena (EUR)")
    plt.tight_layout()
    plt.show()


# ==========================================================
# LOAD EACH MODEL SEPARATELY
# ==========================================================

datasets = {}

for file in DATA_FOLDER.glob("*.json"):
    model_name = file.stem
    print("\nLoading:", model_name)

    with open(file, "r", encoding="utf-8") as f:
        cars = json.load(f)

    rows = []
    for car in cars:
        attrs = car.get("attributes", {})
        rows.append({
            "price":        car.get("price.total.amount"),
            "mileage":      attrs.get("Mileage"),
            "power":        attrs.get("Power"),
            "engine_size":  attrs.get("Cubic Capacity"),
            "fuel":         attrs.get("Fuel"),
            "transmission": attrs.get("Transmission"),
            "registration": attrs.get("First Registration"),
            "owners":       attrs.get("Number of Vehicle Owners"),
            "feature_count": len(car.get("features", [])),
        })

    datasets[model_name] = clean_dataframe(pd.DataFrame(rows))


# ── Zaženi za vsak model ──────────────────────────────────────────────────────
for model_name, df in datasets.items():
    print(f"\n======================\n{model_name.upper()}")
    print(f"Cars: {len(df)}")
    print(f"Avg price: {df['price'].mean():.2f}")
    create_graphs(df, model_name)
    create_feature_importance(df, model_name)

print("\nDone.")