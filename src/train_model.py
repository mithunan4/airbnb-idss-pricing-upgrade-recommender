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

PRICE_MODEL_PATH = Path("models/price_model.joblib")
OCCUPANCY_MODEL_PATH = Path("models/occupancy_model.joblib")
RANDOM_FOREST_BENCHMARK_PATH = Path("models/random_forest_price_benchmark.joblib")

METRICS_PATH = Path("outputs/metrics/model_metrics.txt")


PRICE_FEATURES = [
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


# availability_365 is excluded here to avoid leakage because occupancy_rate
# is created from calendar availability data.
OCCUPANCY_FEATURES = [
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


CATEGORICAL_FEATURES = [
    "neighbourhood_cleansed",
    "room_type",
    "property_type",
]


def build_preprocessor(features):
    numeric_features = [col for col in features if col not in CATEGORICAL_FEATURES]

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
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, numeric_features),
        ]
    )

    return preprocessor


def train_and_evaluate_model(df, target, features, model_name, model, model_path):
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    if target == "occupancy_rate":
        predictions = predictions.clip(0, 1)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    return {
        "model_name": model_name,
        "target": target,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "model_path": model_path,
    }


def main():
    df = pd.read_csv(DATA_PATH)

    results = []

    price_linear = train_and_evaluate_model(
        df=df,
        target="price",
        features=PRICE_FEATURES,
        model_name="Linear Regression Price Model",
        model=LinearRegression(),
        model_path=PRICE_MODEL_PATH,
    )
    results.append(price_linear)

    price_rf_benchmark = train_and_evaluate_model(
        df=df,
        target="price",
        features=PRICE_FEATURES,
        model_name="Random Forest Price Benchmark",
        model=RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            max_depth=15,
            min_samples_leaf=5,
        ),
        model_path=RANDOM_FOREST_BENCHMARK_PATH,
    )
    results.append(price_rf_benchmark)

    occupancy_linear = train_and_evaluate_model(
        df=df,
        target="occupancy_rate",
        features=OCCUPANCY_FEATURES,
        model_name="Linear Regression Occupancy Model",
        model=LinearRegression(),
        model_path=OCCUPANCY_MODEL_PATH,
    )
    results.append(occupancy_linear)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    output_lines = []

    for result in results:
        output_lines.append(result["model_name"])
        output_lines.append(f"Target: {result['target']}")

        if result["target"] == "price":
            output_lines.append(f"MAE: ${result['mae']:.2f}")
            output_lines.append(f"RMSE: ${result['rmse']:.2f}")
        else:
            output_lines.append(f"MAE: {result['mae']:.3f} occupancy rate")
            output_lines.append(f"RMSE: {result['rmse']:.3f} occupancy rate")
            output_lines.append(f"MAE in nights/month: {result['mae'] * 30:.2f}")

        output_lines.append(f"R2: {result['r2']:.3f}")
        output_lines.append(f"Saved to: {result['model_path']}")
        output_lines.append("")

    with open(METRICS_PATH, "w") as f:
        f.write("\n".join(output_lines))

    print("\n".join(output_lines))


if __name__ == "__main__":
    main()
