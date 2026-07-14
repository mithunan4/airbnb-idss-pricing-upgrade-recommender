import pandas as pd
import numpy as np
from pathlib import Path


RAW_PATH = Path("data/raw/listings.csv.gz")
PROCESSED_PATH = Path("data/processed/clean_listings.csv")


def clean_price(price):
    if pd.isna(price):
        return np.nan
    return float(str(price).replace("$", "").replace(",", ""))


def extract_bathrooms(bathroom_text):
    if pd.isna(bathroom_text):
        return np.nan
    text = str(bathroom_text).lower()
    if "half" in text:
        return 0.5
    number = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    return float(number) if number else np.nan


def has_amenity(amenities, keyword):
    if pd.isna(amenities):
        return 0
    return int(keyword.lower() in str(amenities).lower())


def main():
    df = pd.read_csv(RAW_PATH)

    columns = [
        "id",
        "name",
        "neighbourhood_cleansed",
        "latitude",
        "longitude",
        "room_type",
        "property_type",
        "accommodates",
        "bathrooms_text",
        "bedrooms",
        "beds",
        "price",
        "minimum_nights",
        "maximum_nights",
        "availability_365",
        "number_of_reviews",
        "review_scores_rating",
        "host_is_superhost",
        "host_response_rate",
        "amenities",
    ]

    df = df[columns].copy()

    df["price"] = df["price"].apply(clean_price)
    df["bathrooms"] = df["bathrooms_text"].apply(extract_bathrooms)

    df["host_is_superhost"] = df["host_is_superhost"].map({"t": 1, "f": 0}).fillna(0)

    df["host_response_rate"] = (
        df["host_response_rate"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .replace("nan", np.nan)
        .astype(float)
    )

    df["has_wifi"] = df["amenities"].apply(lambda x: has_amenity(x, "wifi"))
    df["has_parking"] = df["amenities"].apply(lambda x: has_amenity(x, "parking"))
    df["has_air_conditioning"] = df["amenities"].apply(lambda x: has_amenity(x, "air conditioning"))
    df["has_kitchen"] = df["amenities"].apply(lambda x: has_amenity(x, "kitchen"))
    df["has_washer"] = df["amenities"].apply(lambda x: has_amenity(x, "washer"))
    df["has_dryer"] = df["amenities"].apply(lambda x: has_amenity(x, "dryer"))
    df["has_self_check_in"] = df["amenities"].apply(lambda x: has_amenity(x, "self check-in"))
    df["has_pool"] = df["amenities"].apply(lambda x: has_amenity(x, "pool"))
    df["has_hot_tub"] = df["amenities"].apply(lambda x: has_amenity(x, "hot tub"))
    df["allows_pets"] = df["amenities"].apply(lambda x: has_amenity(x, "pets allowed"))

    df = df.drop(columns=["bathrooms_text", "amenities"])

    df = df.dropna(subset=["price", "neighbourhood_cleansed", "room_type", "property_type"])

    df = df[df["price"] > 0]
    df = df[df["price"] <= df["price"].quantile(0.99)]

    numeric_cols = [
        "accommodates",
        "bathrooms",
        "bedrooms",
        "beds",
        "minimum_nights",
        "maximum_nights",
        "availability_365",
        "number_of_reviews",
        "review_scores_rating",
        "host_response_rate",
    ]

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print(f"Cleaned data saved to {PROCESSED_PATH}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")


if __name__ == "__main__":
    main()
