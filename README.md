# Airbnb Pricing & Upgrade Recommendation IDSS

This project is an interactive Intelligent Decision Support System (IDSS) that helps Airbnb hosts estimate nightly price and evaluate which listing upgrades are most worthwhile based on predicted revenue impact, ROI, payback period, and budget.

## Stakeholder

The stakeholder is an independent Airbnb host who personally funds listing upgrades and needs to decide which amenity to add and what nightly price to charge.

## Decision Supported

The system supports decisions such as:

- Which upgrade should the host invest in?
- How would an amenity affect predicted nightly price and monthly revenue?
- Which upgrades fit within the host's budget?
- Which upgrades satisfy the host's minimum monthly ROI and maximum payback requirements?

## Data Source

Data comes from the **Inside Airbnb Toronto** dataset.

Raw files downloaded:

- `listings.csv.gz`
- `calendar.csv.gz`
- `neighbourhoods.csv`

The current pipeline uses:

- `listings.csv.gz` for listing characteristics, location, amenities, reviews, prices, and booking rules.
- `calendar.csv.gz` for availability information used to estimate occupancy.

Calendar nights marked unavailable are used as a proxy for occupied nights. Some unavailable dates may be manually blocked by hosts rather than booked, which can cause occupancy estimates to be overstated.

These two files are processed and merged into:

- `data/processed/clean_listings.csv`

The two production models and the Linear Regression price benchmark are trained using clean_listings.csv.

`neighbourhoods.csv` is downloaded and stored locally but is not currently used by the model pipeline.

Raw and processed data files are stored locally and are not committed to GitHub.

## Models

### Nightly Price Model

The production price model is a **Random Forest Regressor**.

It predicts nightly Airbnb price using listing features such as:

- neighbourhood
- latitude and longitude
- property type
- room type
- accommodates
- bedrooms
- bathrooms
- beds
- amenities
- review score
- Superhost status
- booking rules

Random Forest was selected because it outperformed the Linear Regression benchmark and captures non-linear relationships between listing characteristics.

### Occupancy Model

A separate Linear Regression model estimates expected occupancy rate using listing characteristics and amenities. The model is trained using an occupancy proxy derived from calendar availability data.

Projected monthly revenue is calculated as:

`Predicted nightly price × Predicted occupied nights`

### Benchmark Model

A **Linear Regression** price model is retained as a benchmark for comparison with the Random Forest production model.

## IDSS Functionality

The Streamlit dashboard allows users to modify listing characteristics and decision settings, including:

- neighbourhood
- property type
- room type
- accommodates
- bedrooms
- bathrooms
- beds
- minimum nights
- maximum nights
- review score
- number of reviews
- availability days per year
- Superhost status
- current amenities
- demand scenario
- upgrade budget
- upgrade cost assumptions
- minimum monthly ROI threshold
- maximum payback period

The dashboard dynamically updates:

- predicted nightly price
- estimated occupancy rate
- predicted occupied nights
- projected monthly revenue
- estimated revenue impact of each upgrade
- Monthly ROI
- payback period
- ranked upgrade recommendations

Because changing user inputs directly changes the model predictions and upgrade rankings, the system provides interactive decision support rather than a static report.

## Repository Structure

```text
app/
    streamlit_app.py

src/
    data_cleaning.py
    feature_engineering.py
    train_model.py
    predict.py
    recommender.py

models/
    trained model files (.joblib)

data/
    raw and processed local datasets

outputs/
    model evaluation metrics
```

## Data Setup

Download the Toronto `listings.csv.gz` and `calendar.csv.gz` files from Inside Airbnb and place them in:

```text
data/raw/listings.csv.gz
data/raw/calendar.csv.gz
```

The `neighbourhoods.csv` file may also be downloaded for reference, but it is not used by the current pipeline.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the processed dataset:

```bash
python src/data_cleaning.py
```

Train the models:

```bash
python src/train_model.py
```

Launch the Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Streamlit
- joblib