"""
Feature engineering notes.

Most feature engineering is currently handled inside data_cleaning.py.

Created features include:
- numeric bathroom count from bathrooms_text
- amenity indicator variables such as parking, pool, hot tub, pets, washer, dryer
- superhost binary indicator
- occupancy_rate from calendar availability data
- expected_occupied_nights_30 from occupancy_rate

This file is kept as a placeholder for future refactoring if the feature pipeline is split
out from the cleaning script.
"""
