# Airbnb Pricing & Upgrade Recommendation IDSS

This project is an interactive Intelligent Decision Support System (IDSS) that helps Airbnb hosts estimate nightly price and decide which listing upgrades are most worthwhile based on predicted revenue impact, ROI, payback period, and budget.

## Stakeholder

The stakeholder is an Airbnb host who needs to decide which listing upgrades or amenities to invest in.

## Decision Supported

The system supports decisions such as:

- Which upgrade should I invest in?
- How will an amenity change affect predicted nightly price?
- Which upgrades fit my budget?
- Which upgrades have the best ROI and payback period?

## Data Source

Data comes from Inside Airbnb for Toronto, Ontario, Canada.

Files used locally:

- `listings.csv.gz`
- `calendar.csv.gz`
- `neighbourhoods.csv`

Raw data is stored locally in `data/raw/` and is not committed to GitHub.

## Model

The production model is Linear Regression. It predicts nightly Airbnb price based on listing features such as location, property type, room type, bedrooms, bathrooms, review score, and amenities.

Random Forest was also tested as a benchmark model to compare predictive performance.

## IDSS Functionality

The dashboard allows the user to change listing inputs and decision settings, including:

- bedrooms
- bathrooms
- beds
- amenities
- expected occupied nights
- upgrade budget
- minimum ROI threshold
- maximum payback period

The system updates predicted nightly price, estimated monthly revenue, and upgrade recommendations in response to these changes.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
