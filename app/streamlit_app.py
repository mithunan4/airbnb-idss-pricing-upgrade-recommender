import os
import sys
from pathlib import Path

import streamlit as st

# -----------------------------
# Project Paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

# Allow app file to import from src folder
sys.path.append(str(PROJECT_ROOT / "src"))

from predict import load_models, predict_price, predict_occupied_nights
from recommender import recommend_upgrades


# -----------------------------
# Page Setup
# -----------------------------

st.set_page_config(
    page_title="Airbnb Upgrade IDSS",
    layout="wide",
)

st.title("Airbnb Pricing & Upgrade Recommendation IDSS")

st.write(
    "This dashboard helps Airbnb hosts estimate nightly price, forecast occupancy, "
    "and decide which listing upgrades are most worthwhile based on predicted revenue impact, "
    "ROI, payback period, and budget."
)


# -----------------------------
# Neighbourhood Coordinates
# -----------------------------

NEIGHBOURHOOD_COORDS = {
    "Waterfront Communities-The Island": {"latitude": 43.64, "longitude": -79.38},
    "Niagara": {"latitude": 43.64, "longitude": -79.41},
    "Annex": {"latitude": 43.67, "longitude": -79.40},
    "Kensington-Chinatown": {"latitude": 43.65, "longitude": -79.40},
    "Church-Yonge Corridor": {"latitude": 43.66, "longitude": -79.38},
    "Bay Street Corridor": {"latitude": 43.66, "longitude": -79.39},
}


# -----------------------------
# Load Models
# -----------------------------

@st.cache_resource
def get_models():
    return load_models()


models = get_models()
price_model = models["price_model"]
occupancy_model = models["occupancy_model"]


# -----------------------------
# Sidebar Inputs
# -----------------------------

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

minimum_nights = st.sidebar.slider("Minimum Nights", 1, 30, 2)
maximum_nights = st.sidebar.slider("Maximum Nights", 30, 365, 365)

review_score = st.sidebar.slider("Review Score", 0.0, 5.0, 4.8, step=0.1)
number_of_reviews = st.sidebar.slider("Number of Reviews", 0, 300, 20)
availability_365 = st.sidebar.slider("Availability Days Per Year", 0, 365, 180)

host_is_superhost = st.sidebar.checkbox("Superhost", value=False)


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

budget = st.sidebar.slider(
    "Upgrade Budget ($)",
    0,
    10000,
    3000,
    step=100,
)

min_roi = st.sidebar.slider(
    "Minimum Monthly ROI (%)",
    0,
    100,
    5,
)

max_payback = st.sidebar.slider(
    "Maximum Payback Period (Months)",
    1,
    36,
    12,
)

scenario_adjustment = st.sidebar.slider(
    "Demand Scenario Adjustment (%)",
    -50,
    50,
    0,
    step=5,
)

st.sidebar.caption(
    "Demand scenario adjustment lets the host test optimistic or pessimistic occupancy conditions."
)


# -----------------------------
# Editable Upgrade Costs
# -----------------------------

st.sidebar.header("Upgrade Cost Assumptions")

upgrade_costs = {
    "has_parking": st.sidebar.number_input("Parking Cost ($)", 1, 20000, 800, step=100),
    "has_self_check_in": st.sidebar.number_input("Self Check-in Cost ($)", 1, 20000, 300, step=100),
    "allows_pets": st.sidebar.number_input("Pet-Friendly Setup Cost ($)", 1, 20000, 200, step=100),
    "has_hot_tub": st.sidebar.number_input("Hot Tub Cost ($)", 1, 20000, 3500, step=100),
    "has_pool": st.sidebar.number_input("Pool Cost ($)", 1, 50000, 8000, step=500),
    "has_air_conditioning": st.sidebar.number_input("Air Conditioning Cost ($)", 1, 20000, 1500, step=100),
    "has_washer": st.sidebar.number_input("Washer Cost ($)", 1, 20000, 1000, step=100),
    "has_dryer": st.sidebar.number_input("Dryer Cost ($)", 1, 20000, 1000, step=100),
}


# -----------------------------
# Model Inputs
# -----------------------------

coords = NEIGHBOURHOOD_COORDS[neighbourhood]

listing_inputs = {
    "neighbourhood_cleansed": neighbourhood,
    "room_type": room_type,
    "property_type": property_type,
    "latitude": coords["latitude"],
    "longitude": coords["longitude"],
    "accommodates": accommodates,
    "bathrooms": bathrooms,
    "bedrooms": bedrooms,
    "beds": beds,
    "minimum_nights": minimum_nights,
    "maximum_nights": maximum_nights,
    "availability_365": availability_365,
    "number_of_reviews": number_of_reviews,
    "review_scores_rating": review_score,
    "host_is_superhost": int(host_is_superhost),
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


# -----------------------------
# Prediction Summary
# -----------------------------

predicted_price = predict_price(price_model, listing_inputs)

predicted_occupied_nights, predicted_occupancy_rate = predict_occupied_nights(
    occupancy_model,
    listing_inputs,
    scenario_adjustment=scenario_adjustment,
)

monthly_revenue = predicted_price * predicted_occupied_nights

col1, col2, col3, col4 = st.columns(4)

col1.metric("Predicted Nightly Price", f"${predicted_price:,.2f}")
col2.metric("Predicted Occupancy Rate", f"{predicted_occupancy_rate:.1%}")
col3.metric("Predicted Occupied Nights", f"{predicted_occupied_nights:.2f}")
col4.metric("Estimated Monthly Revenue", f"${monthly_revenue:,.2f}")


# -----------------------------
# Upgrade Recommendations
# -----------------------------

st.subheader("Upgrade Recommendations")

recommendations = recommend_upgrades(
    models=models,
    listing_inputs=listing_inputs,
    budget=budget,
    min_roi=min_roi,
    max_payback_months=max_payback,
    scenario_adjustment=scenario_adjustment,
    upgrade_costs=upgrade_costs,
)

if recommendations.empty:
    st.warning(
        "All selected amenities are already included, so there are no remaining upgrades to evaluate."
    )
else:
    display_cols = [
        "upgrade",
        "price_increase",
        "occupied_nights_change",
        "monthly_revenue_increase",
        "upgrade_cost",
        "roi_percent",
        "payback_months",
        "recommended",
    ]

    display_table = recommendations[display_cols].rename(
        columns={
            "upgrade": "Upgrade",
            "price_increase": "Price Change ($/night)",
            "occupied_nights_change": "Occupied Nights Change",
            "monthly_revenue_increase": "Monthly Revenue Increase ($)",
            "upgrade_cost": "Upgrade Cost ($)",
            "roi_percent": "Monthly ROI (%)",
            "payback_months": "Payback Period (Months)",
            "recommended": "Recommended",
        }
    )

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
    )

    recommended_options = recommendations[recommendations["recommended"] == True]

    if not recommended_options.empty:
        best = recommended_options.iloc[0]

        st.success(
            f"Recommended upgrade: {best['upgrade']} because it fits the budget, "
            f"meets the ROI threshold, and has an estimated payback period of "
            f"{best['payback_months']} months."
        )
    else:
        st.info(
            "No upgrades meet all decision thresholds. Try increasing the budget, "
            "lowering the ROI threshold, or extending the payback limit."
        )


# -----------------------------
# Visualization
# -----------------------------

st.subheader("Monthly ROI by Upgrade")

st.write(
    "Higher ROI means the upgrade is expected to generate more monthly revenue "
    "relative to its estimated cost. Upgrades with negative ROI are not recommended."
)

if recommendations.empty:
    st.info("No ROI chart available because all upgrade options are already selected.")
else:
    chart_data = recommendations[["upgrade", "roi_percent"]].copy()
    chart_data = chart_data.rename(
        columns={
            "upgrade": "Upgrade",
            "roi_percent": "ROI (%)",
        }
    )

    chart_data = chart_data.set_index("Upgrade")
    st.bar_chart(chart_data)


st.subheader("Revenue Impact by Upgrade")

st.write(
    "This chart shows how much each upgrade is expected to change monthly revenue "
    "after combining predicted nightly price and predicted occupied nights."
)

if recommendations.empty:
    st.info("No revenue impact chart available because all upgrade options are already selected.")
else:
    revenue_chart = recommendations[["upgrade", "monthly_revenue_increase"]].copy()
    revenue_chart = revenue_chart.rename(
        columns={
            "upgrade": "Upgrade",
            "monthly_revenue_increase": "Monthly Revenue Increase ($)",
        }
    )

    revenue_chart = revenue_chart.set_index("Upgrade")
    st.bar_chart(revenue_chart)


# -----------------------------
# Model Performance
# -----------------------------

st.subheader("Model Performance")

metrics_path = Path("outputs/metrics/model_metrics.txt")

if metrics_path.exists():
    with open(metrics_path, "r") as f:
        st.text(f.read())
else:
    st.info("Model metrics are not available yet. Run src/train_model.py to generate them.")


# -----------------------------
# Decision Impact Explanation
# -----------------------------

st.subheader("Decision Impact")

st.write(
    "Changing the listing inputs, amenity assumptions, upgrade costs, or decision settings updates "
    "the predicted price, predicted occupancy, estimated revenue, ROI, payback period, and final "
    "upgrade recommendation. This helps the host compare upgrade scenarios before investing money."
)