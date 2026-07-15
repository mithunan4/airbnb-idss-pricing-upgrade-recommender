import pandas as pd
import numpy as np
from pathlib import Path


LISTINGS_RAW_PATH = Path("data/raw/listings.csv.gz")
CALENDAR_RAW_PATH = Path("data/raw/calendar.csv.gz")
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


def build_occupancy_labels():
    calendar = pd.read_csv(CALENDAR_RAW_PATH)

    # available = "f" means the night is unavailable.
    # We use unavailable nights as a proxy for occupied/booked nights.
    calendar["occupied_proxy"] = (calendar["available"] == "f").astype(int)

    occupancy = (
        calendar.groupby("listing_id")["occupied_proxy"]
        .mean()
        .reset_index()
        .rename(columns={"listing_id": "id", "occupied_proxy": "occupancy_rate"})
    )

    occupancy["expected_occupied_nights_30"] = occupancy["occupancy_rate"] * 30

    return occupancy


def main():
    listings = pd.read_csv(LISTINGS_RAW_PATH)
    occupancy = build_occupancy_labels()

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
        "amenities",
    ]

    df = listings[columns].copy()

    df["price"] = df["price"].apply(clean_price)
    df["bathrooms"] = df["bathrooms_text"].apply(extract_bathrooms)
    df["host_is_superhost"] = df["host_is_superhost"].map({"t": 1, "f": 0}).fillna(0)

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

    df = df.dropna(
        subset=[
            "price",
            "neighbourhood_cleansed",
            "room_type",
            "property_type",
        ]
    )

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
    ]

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    df = df.merge(occupancy, on="id", how="left")

    df["occupancy_rate"] = df["occupancy_rate"].fillna(df["occupancy_rate"].median())
    df["expected_occupied_nights_30"] = df["expected_occupied_nights_30"].fillna(
        df["expected_occupied_nights_30"].median()
    )

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print(f"Cleaned data saved to {PROCESSED_PATH}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Average occupancy rate: {df['occupancy_rate'].mean():.2%}")


if __name__ == "__main__":
    main()
