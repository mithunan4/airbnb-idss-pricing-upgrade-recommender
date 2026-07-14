import pandas as pd
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/processed/clean_listings.csv")
MODEL_PATH = Path("models/price_model.joblib")
METRICS_PATH = Path("outputs/metrics/model_metrics.txt")


def main():
    df = pd.read_csv(DATA_PATH)

    target = "price"

    features = [
        "neighbourhood_cleansed",
        "room_type",
        "property_type",
        "latitude",
        "longitude",
        "accommodates",
        "bathrooms",
        "bedrooms",
        "beds",
        "minimum_nights",
        "maximum_nights",
        "availability_365",
        "number_of_reviews",
        "review_scores_rating",
        "host_is_superhost",
        "has_wifi",
        "has_parking",
        "has_air_conditioning",
        "has_kitchen",
        "has_washer",
        "has_dryer",
        "has_self_check_in",
        "has_pool",
        "has_hot_tub",
        "allows_pets",
    ]

    X = df[features]
    y = df[target]

    categorical_features = [
        "neighbourhood_cleansed",
        "room_type",
        "property_type",
    ]

    numeric_features = [col for col in features if col not in categorical_features]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_features),
            ("numeric", numeric_pipeline, numeric_features),
        ]
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            max_depth=15,
            min_samples_leaf=5,
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    results = []
    best_model = None
    best_mae = float("inf")

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        results.append(
            f"{model_name}\n"
            f"MAE: ${mae:.2f}\n"
            f"RMSE: ${rmse:.2f}\n"
            f"R2: {r2:.3f}\n"
        )

        if model_name == "Linear Regression":
        	best_model = pipeline

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        f.write("\n".join(results))
        f.write(f"\nProduction model saved to: {MODEL_PATH}\n")

    print("\n".join(results))
    print(f"Production model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
