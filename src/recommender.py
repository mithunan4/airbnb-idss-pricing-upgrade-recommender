import pandas as pd
from predict import load_models, predict_price, predict_occupied_nights


# Estimated upgrade costs in dollars
UPGRADE_COSTS = {
    "has_parking": 800,
    "has_self_check_in": 300,
    "allows_pets": 200,
    "has_hot_tub": 3500,
    "has_pool": 8000,
    "has_air_conditioning": 1500,
    "has_washer": 1000,
    "has_dryer": 1000,
}


UPGRADE_LABELS = {
    "has_parking": "Add parking",
    "has_self_check_in": "Add self check-in",
    "allows_pets": "Allow pets",
    "has_hot_tub": "Add hot tub",
    "has_pool": "Add pool",
    "has_air_conditioning": "Add air conditioning",
    "has_washer": "Add washer",
    "has_dryer": "Add dryer",
}


def estimate_monthly_revenue(nightly_price, occupied_nights):
    return nightly_price * occupied_nights


def calculate_roi(monthly_revenue_increase, upgrade_cost):
    if upgrade_cost <= 0:
        return 0
    return (monthly_revenue_increase / upgrade_cost) * 100


def calculate_payback_period(upgrade_cost, monthly_revenue_increase):
    if monthly_revenue_increase <= 0:
        return None
    return upgrade_cost / monthly_revenue_increase


def recommend_upgrades(
    models,
    listing_inputs,
    budget=3000,
    min_roi=5,
    max_payback_months=12,
    scenario_adjustment=0,
):
    price_model = models["price_model"]
    occupancy_model = models["occupancy_model"]

    current_price = predict_price(price_model, listing_inputs)

    current_occupied_nights, current_occupancy_rate = predict_occupied_nights(
        occupancy_model,
        listing_inputs,
        scenario_adjustment=scenario_adjustment,
    )

    current_monthly_revenue = estimate_monthly_revenue(
        current_price,
        current_occupied_nights,
    )

    recommendations = []

    for upgrade_feature, upgrade_cost in UPGRADE_COSTS.items():
        # Skip upgrades that the listing already has
        if listing_inputs.get(upgrade_feature, 0) == 1:
            continue

        upgraded_listing = listing_inputs.copy()
        upgraded_listing[upgrade_feature] = 1

        upgraded_price = predict_price(price_model, upgraded_listing)

        upgraded_occupied_nights, upgraded_occupancy_rate = predict_occupied_nights(
            occupancy_model,
            upgraded_listing,
            scenario_adjustment=scenario_adjustment,
        )

        upgraded_monthly_revenue = estimate_monthly_revenue(
            upgraded_price,
            upgraded_occupied_nights,
        )

        price_increase = upgraded_price - current_price
        occupied_nights_change = upgraded_occupied_nights - current_occupied_nights
        occupancy_rate_change = upgraded_occupancy_rate - current_occupancy_rate
        monthly_revenue_increase = upgraded_monthly_revenue - current_monthly_revenue

        roi = calculate_roi(monthly_revenue_increase, upgrade_cost)
        payback_period = calculate_payback_period(
            upgrade_cost,
            monthly_revenue_increase,
        )

        within_budget = upgrade_cost <= budget
        meets_roi = roi >= min_roi
        meets_payback = (
            payback_period is not None and payback_period <= max_payback_months
        )

        recommended = within_budget and meets_roi and meets_payback

        recommendations.append(
            {
                "upgrade": UPGRADE_LABELS[upgrade_feature],
                "feature": upgrade_feature,
                "current_price": round(current_price, 2),
                "upgraded_price": round(upgraded_price, 2),
                "price_increase": round(price_increase, 2),
                "current_occupancy_rate": round(current_occupancy_rate, 4),
                "upgraded_occupancy_rate": round(upgraded_occupancy_rate, 4),
                "occupancy_rate_change": round(occupancy_rate_change, 4),
                "current_occupied_nights": round(current_occupied_nights, 2),
                "upgraded_occupied_nights": round(upgraded_occupied_nights, 2),
                "occupied_nights_change": round(occupied_nights_change, 2),
                "current_monthly_revenue": round(current_monthly_revenue, 2),
                "upgraded_monthly_revenue": round(upgraded_monthly_revenue, 2),
                "monthly_revenue_increase": round(monthly_revenue_increase, 2),
                "upgrade_cost": upgrade_cost,
                "roi_percent": round(roi, 2),
                "payback_months": (
                    round(payback_period, 2) if payback_period is not None else None
                ),
                "within_budget": within_budget,
                "recommended": recommended,
            }
        )

    recommendations_df = pd.DataFrame(recommendations)

    if recommendations_df.empty:
        return recommendations_df

    recommendations_df = recommendations_df.sort_values(
        by=["recommended", "roi_percent", "monthly_revenue_increase"],
        ascending=[False, False, False],
    )

    return recommendations_df


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

    results = recommend_upgrades(
        models=models,
        listing_inputs=sample_listing,
        budget=3000,
        min_roi=5,
        max_payback_months=12,
        scenario_adjustment=0,
    )

    print(results)
