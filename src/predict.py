import pandas as pd
import joblib
from pathlib import Path


MODEL_PATH = Path("models/price_model.joblib")


def load_model():
    return joblib.load(MODEL_PATH)


def predict_price(model, listing_inputs):
    input_df = pd.DataFrame([listing_inputs])
    predicted_price = model.predict(input_df)[0]
    return round(predicted_price, 2)


if __name__ == "__main__":
    model = load_model()

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

    price = predict_price(model, sample_listing)
    print(f"Predicted nightly price: ${price}")
