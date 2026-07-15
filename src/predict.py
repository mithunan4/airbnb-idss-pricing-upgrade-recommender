import pandas as pd
import joblib
from pathlib import Path
import numpy as np


PRICE_MODEL_PATH = Path("models/price_model.joblib")
OCCUPANCY_MODEL_PATH = Path("models/occupancy_model.joblib")


def load_models():
    """
    Loads both production models:
    1. Price model: predicts nightly Airbnb price
    2. Occupancy model: predicts occupancy rate
    """
    return {
        "price_model": joblib.load(PRICE_MODEL_PATH),
        "occupancy_model": joblib.load(OCCUPANCY_MODEL_PATH),
    }


def load_model():
    """
    Keeps old compatibility for files that only need the price model.
    """
    return joblib.load(PRICE_MODEL_PATH)


def predict_price(price_model, listing_inputs):
    """
    Predicts nightly price from listing features.
    The trained price model predicts log(price), so np.expm1 converts it back to dollars.
    """
    input_df = pd.DataFrame([listing_inputs])
    predicted_log_price = price_model.predict(input_df)[0]
    predicted_price = np.expm1(predicted_log_price)
    predicted_price = max(predicted_price, 0)

    return round(predicted_price, 2)


def predict_occupancy_rate(occupancy_model, listing_inputs):
    """
    Predicts occupancy rate between 0 and 1.
    """
    input_df = pd.DataFrame([listing_inputs])
    predicted_occupancy = occupancy_model.predict(input_df)[0]

    # Keep prediction within valid range
    predicted_occupancy = max(0, min(1, predicted_occupancy))

    return round(predicted_occupancy, 4)


def predict_occupied_nights(occupancy_model, listing_inputs, scenario_adjustment=0):
    """
    Converts predicted occupancy rate into expected occupied nights per month.

    scenario_adjustment allows the user to simulate better/worse demand.
    Example:
    +10 means occupancy is increased by 10%
    -10 means occupancy is decreased by 10%
    """
    occupancy_rate = predict_occupancy_rate(occupancy_model, listing_inputs)

    adjusted_occupancy_rate = occupancy_rate * (1 + scenario_adjustment / 100)
    adjusted_occupancy_rate = max(0, min(1, adjusted_occupancy_rate))

    occupied_nights = adjusted_occupancy_rate * 30

    return round(occupied_nights, 2), round(adjusted_occupancy_rate, 4)


if __name__ == "__main__":
    models = load_models()

    sample_listing = {
        "neighbourhood_cleansed": "Waterfront Communities-The Island",
        "room_type": "Entire home/apt",
        "property_type": "Entire rental unit",
        "latitude": 43.64,
        "longitude": -79.38,
        "accommodates": 2,
        "bathrooms": 1.0,
        "bedrooms": 1.0,
        "beds": 1.0,
        "minimum_nights": 2,
        "maximum_nights": 365,
        "availability_365": 180,
        "number_of_reviews": 20,
        "review_scores_rating": 4.8,
        "host_is_superhost": 0,
        "has_wifi": 1,
        "has_parking": 0,
        "has_air_conditioning": 1,
        "has_kitchen": 1,
        "has_washer": 1,
        "has_dryer": 1,
        "has_self_check_in": 0,
        "has_pool": 0,
        "has_hot_tub": 0,
        "allows_pets": 0,
    }

    price = predict_price(models["price_model"], sample_listing)

    occupied_nights, occupancy_rate = predict_occupied_nights(
        models["occupancy_model"],
        sample_listing,
    )

    monthly_revenue = price * occupied_nights

    print(f"Predicted nightly price: ${price}")
    print(f"Predicted occupancy rate: {occupancy_rate:.1%}")
    print(f"Predicted occupied nights/month: {occupied_nights}")
    print(f"Estimated monthly revenue: ${monthly_revenue:.2f}")
