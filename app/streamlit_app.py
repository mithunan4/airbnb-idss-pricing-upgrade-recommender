import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from predict import load_model, predict_price
from recommender import recommend_upgrades


st.set_page_config(page_title="Airbnb Upgrade IDSS", layout="wide")

st.title("Airbnb Pricing & Upgrade Recommendation IDSS")

st.write(
    "This dashboard helps Airbnb hosts estimate nightly price and decide which listing upgrades "
    "are most worthwhile based on predicted revenue impact, ROI, payback period, and budget."
)

model = load_model()

st.sidebar.header("Listing Inputs")

neighbourhood = st.sidebar.selectbox(
    "Neighbourhood",
    [
        "Waterfront Communities-The Island",
        "Niagara",
        "Annex",
        "Kensington-Chinatown",
        "Church-Yonge Corridor",
        "Bay Street Corridor",
    ],
)

room_type = st.sidebar.selectbox(
    "Room Type",
    ["Entire home/apt", "Private room", "Shared room", "Hotel room"],
)

property_type = st.sidebar.selectbox(
    "Property Type",
    ["Entire rental unit", "Entire condo", "Private room in rental unit", "Entire home"],
)

accommodates = st.sidebar.slider("Accommodates", 1, 10, 2)
bedrooms = st.sidebar.slider("Bedrooms", 0, 5, 1)
bathrooms = st.sidebar.slider("Bathrooms", 0.5, 4.0, 1.0, step=0.5)
beds = st.sidebar.slider("Beds", 1, 6, 1)

review_score = st.sidebar.slider("Review Score", 0.0, 5.0, 4.8, step=0.1)
number_of_reviews = st.sidebar.slider("Number of Reviews", 0, 300, 20)
availability_365 = st.sidebar.slider("Availability Days Per Year", 0, 365, 180)

st.sidebar.header("Current Amenities")

has_wifi = st.sidebar.checkbox("WiFi", value=True)
has_parking = st.sidebar.checkbox("Parking", value=False)
has_air_conditioning = st.sidebar.checkbox("Air Conditioning", value=True)
has_kitchen = st.sidebar.checkbox("Kitchen", value=True)
has_washer = st.sidebar.checkbox("Washer", value=True)
has_dryer = st.sidebar.checkbox("Dryer", value=True)
has_self_check_in = st.sidebar.checkbox("Self Check-in", value=False)
has_pool = st.sidebar.checkbox("Pool", value=False)
has_hot_tub = st.sidebar.checkbox("Hot Tub", value=False)
allows_pets = st.sidebar.checkbox("Allows Pets", value=False)

st.sidebar.header("Decision Settings")

occupied_nights = st.sidebar.slider("Expected Occupied Nights Per Month", 1, 30, 20)
budget = st.sidebar.slider("Upgrade Budget ($)", 0, 10000, 3000, step=100)
min_roi = st.sidebar.slider("Minimum ROI (%)", 0, 100, 5)
max_payback = st.sidebar.slider("Maximum Payback Period (Months)", 1, 36, 12)

listing_inputs = {
    "neighbourhood_cleansed": neighbourhood,
    "room_type": room_type,
    "property_type": property_type,
    "latitude": 43.64,
    "longitude": -79.38,
    "accommodates": accommodates,
    "bathrooms": bathrooms,
    "bedrooms": bedrooms,
    "beds": beds,
    "minimum_nights": 2,
    "maximum_nights": 365,
    "availability_365": availability_365,
    "number_of_reviews": number_of_reviews,
    "review_scores_rating": review_score,
    "host_is_superhost": 0,
    "has_wifi": int(has_wifi),
    "has_parking": int(has_parking),
    "has_air_conditioning": int(has_air_conditioning),
    "has_kitchen": int(has_kitchen),
    "has_washer": int(has_washer),
    "has_dryer": int(has_dryer),
    "has_self_check_in": int(has_self_check_in),
    "has_pool": int(has_pool),
    "has_hot_tub": int(has_hot_tub),
    "allows_pets": int(allows_pets),
}

predicted_price = predict_price(model, listing_inputs)
monthly_revenue = predicted_price * occupied_nights

col1, col2, col3 = st.columns(3)

col1.metric("Predicted Nightly Price", f"${predicted_price:,.2f}")
col2.metric("Estimated Monthly Revenue", f"${monthly_revenue:,.2f}")
col3.metric("Occupied Nights Used", occupied_nights)

st.subheader("Upgrade Recommendations")

recommendations = recommend_upgrades(
    model=model,
    listing_inputs=listing_inputs,
    occupied_nights=occupied_nights,
    budget=budget,
    min_roi=min_roi,
    max_payback_months=max_payback,
)

if recommendations.empty:
    st.warning("No upgrade recommendations available.")
else:
    display_cols = [
        "upgrade",
        "current_price",
        "upgraded_price",
        "price_increase",
        "monthly_revenue_increase",
        "upgrade_cost",
        "roi_percent",
        "payback_months",
        "within_budget",
        "recommended",
    ]

    st.dataframe(recommendations[display_cols], use_container_width=True)

    recommended_options = recommendations[recommendations["recommended"] == True]

    if not recommended_options.empty:
        best = recommended_options.iloc[0]
        st.success(
            f"Recommended decision: {best['upgrade']} because it fits the budget, "
            f"meets the ROI threshold, and has an estimated payback period of "
            f"{best['payback_months']} months."
        )
    else:
        st.info(
            "No upgrades meet all decision thresholds. Try increasing the budget, "
            "lowering the ROI threshold, or extending the payback limit."
        )

st.subheader("Decision Impact")

st.write(
    "Changing the listing inputs or decision settings updates the predicted price, revenue estimate, "
    "and upgrade ranking. This helps the host compare upgrade scenarios before investing money."
)
