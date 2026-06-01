import json
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OrdinalEncoder,
    RobustScaler
)

from sklearn.impute import (
    SimpleImputer
)

from sklearn.model_selection import (
    cross_val_predict,
    KFold
)

from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor
)

from sklearn.linear_model import RidgeCV


DATA = Path("data/top5")

CURRENT_YEAR = 2026

SUMMARY = []

MODELS = {}


# ====================================

def extract_num(v):
    """Izvleče prvo število (npr. '74 kW (101 hp)' -> 74.0)."""
    if pd.isna(v):
        return np.nan
    m = re.search(r"\d+", str(v).replace(",", "").replace("\xa0", ""))
    return float(m.group()) if m else np.nan


def extract_year(v):

    if pd.isna(v):
        return np.nan

    m = re.search(
        r"(19|20)\d{2}",
        str(v)
    )

    if m:
        return float(
            m.group()
        )

    return np.nan


PRICE_MAP = {

    "Very good price":0,
    "Good price":1,
    "Fair price":2,
    "Increased price":3,
    "High price":4

}


# ====================================

datasets = {}

for file in DATA.glob(
    "*.json"
):

    print(
        "\nLoading",
        file.stem
    )

    with open(
        file,
        encoding="utf-8"
    ) as f:

        cars = json.load(
            f
        )

    freq = {}

    for c in cars:

        for f in (

            c.get(
                "features"
            )

            or []

        ):

            freq[f] = (

                freq
                .get(
                    f,
                    0
                )

                + 1

            )

    keep = {

        k

        for k,v

        in freq.items()

        if v >= 25

    }

    rows = []

    for car in cars:

        a = (
            car.get(
                "attributes"
            )
            or {}
        )

        d = (
            car.get(
                "dealerDetails"
            )
            or {}
        )

        score = (
            d.get(
                "score"
            )
            or {}
        )

        feats = (
            car.get(
                "features"
            )
            or []
        )

        row = {

            "price":
            car.get(
                "price.total.amount"
            ),

            "km":
            a.get(
                "Mileage"
            ),

            "power":
            a.get(
                "Power"
            ),

            "engine":
            a.get(
                "Cubic Capacity"
            ),

            "year":
            a.get(
                "First Registration"
            ),

            "owners":
            a.get(
                "Number of Vehicle Owners"
            ),

            "fuel":
            a.get(
                "Fuel"
            ),

            "gear":
            a.get(
                "Transmission"
            ),

            "trim":
            a.get(
                "Trim line"
            ),

            "category":
            a.get(
                "Category"
            ),

            "condition":
            a.get(
                "Vehicle condition"
            ),

            "weight":
            a.get(
                "Weight"
            ),

            "co2":
            a.get(
                "CO₂ emissions (comb.)"
            ),

            "cons":
            a.get(
                "Energy consumption (comb.)"
            ),

            "dealer":
            score.get(
                "total"
            ),

            "reviews":
            d.get(
                "activeRatingCount"
            ),

            "rating":

            PRICE_MAP.get(

                (

                    car
                    .get(
                        "priceRating"
                    )

                    or {}

                )

                .get(
                    "rating"
                )

            ),

            "feature_count":
            len(
                feats
            )

        }

        for f in keep:

            row[f] = (

                1

                if f in feats

                else 0

            )

        rows.append(
            row
        )

    datasets[
        file.stem
    ] = pd.DataFrame(
        rows)


# ====================================

for name, df in datasets.items():

    print(
        "\nTRAIN",
        name
    )

    numeric = [

        c

        for c

        in df.columns

        if c not in [

            "fuel",
            "gear",
            "trim",
            "category",
            "condition"

        ]

    ]

    for c in numeric:

        if c == "year":

            df[c] = (
                df[c]
                .apply(
                    extract_year
                )
            )

        else:

            df[c] = (
                df[c]
                .apply(
                    extract_num
                )
            )

    df["age"] = (

        CURRENT_YEAR
        -
        df["year"]

    )

    df = df.dropna(
        subset=["price"]
    )

    low = df.price.quantile(.03)
    high = df.price.quantile(.97)

    df = df[
        (
            df.price > low
        )
        &
        (
            df.price < high
        )
    ]

    X = df.drop(
        columns=["price"]
    )

    y = np.log1p(
        df.price
    )

    cat = (

        X

        .select_dtypes(
            include="object"
        )

        .columns

    )

    num = [

        x

        for x

        in X.columns

        if x not in cat

    ]

    prep = (

        ColumnTransformer([

            (

                "num",

                Pipeline([

                    (
                        "imp",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),

                    (
                        "scale",
                        RobustScaler()
                    )

                ]),

                num

            ),

            (

                "cat",

                Pipeline([

                    (
                        "imp",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),

                    (
                        "enc",

                        OrdinalEncoder(

                            handle_unknown="use_encoded_value",

                            unknown_value=-1

                        )

                    )

                ]),

                cat

            )

        ])

    )

    model = StackingRegressor(

        estimators=[

            (

                "xt",

                ExtraTreesRegressor(

                    n_estimators=300,

                    random_state=42

                )

            ),

            (

                "gb",

                HistGradientBoostingRegressor(

                    max_iter=300

                )

            )

        ],

        final_estimator=RidgeCV()

    )

    pipe = Pipeline([

        (
            "prep",
            prep
        ),

        (
            "model",
            model
        )

    ])

    pred = (

        cross_val_predict(

            pipe,

            X,

            y,

            cv=KFold(
                5,
                shuffle=True,
                random_state=42
            )

        )

    )

    pipe.fit(
        X,
        y
    )

    pred = np.expm1(
        pred
    )

    actual = np.expm1(
        y
    )

    mae = mean_absolute_error(
        actual,
        pred
    )

    median = np.median(

        np.abs(
            pred
            -
            actual
        )

    )

    pct = (

        np.mean(

            np.abs(
                pred
                -
                actual
            )

            /

            actual

        )

        *

        100

    )

    mean_price = actual.mean()

    norm = (

        mae

        /

        mean_price

        *

        100

    )

    p90 = np.percentile(

        np.abs(
            pred
            -
            actual
        ),

        90

    )

    r2 = r2_score(
        actual,
        pred
    )

    SUMMARY.append({

        "model":name,

        "mean_price":
        round(mean_price),

        "avg_error":
        round(mae),

        "median_error":
        round(median),

        "p90_error":
        round(p90),

        "avg_pct":
        round(pct,2),

        "normalized_mae":
        round(norm,2),

        "r2":
        round(r2,3)

    })

    MODELS[
        name
    ] = pipe

    pickle.dump(

        pipe,

        open(
            f"{name}.pkl",

            "wb"

        )

    )


summary = pd.DataFrame(
    SUMMARY
)

print(
    summary
)

plt.figure(
    figsize=(14,6)
)

sns.barplot(
    data=summary,
    x="model",
    y="avg_pct"
)

plt.title(
    "Average Error %"
)

plt.show()


plt.figure(
    figsize=(14,6)
)

sns.barplot(
    data=summary,
    x="model",
    y="median_error"
)

plt.title(
    "Median Error €"
)

plt.show()

print("\nDONE")