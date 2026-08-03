# Freight Rate Prediction Challenge

## Overview

This project was completed as part of the Machine Learning Engineer assessment.

The objective was to develop a machine learning model capable of predicting freight rates using historical shipment data and generate predictions for an unseen validation dataset.

## Dataset

The project uses three datasets:

* **train-test.csv** – historical labeled freight loads used for model development.
* **validation.csv** – unlabeled dataset requiring predictions.
* **december-chart-inputs.csv** – fixed December route used to generate the required prediction chart.

## Project Workflow

1. Data exploration
2. Data cleaning
3. Feature engineering
4. Model training
5. Validation
6. Final prediction generation
7. Assessment scoring

## Data Cleaning

The following preprocessing steps were applied:

* Missing values in **weight** were replaced using the median.
* Missing values in **market_index** were replaced using the median.
* Invalid negative shipment weights were treated as missing values.
* Date values were converted into datetime format.

## Feature Engineering

The following features were created:

* Route (Pickup → Delivery)
* Month
* Day
* Day of Week
* Cyclical Month Encoding
* Cyclical Day Encoding
* Latitude Difference
* Longitude Difference

## Model

The final model is a **CatBoost Regressor**.

CatBoost was selected because it:

* Handles categorical variables efficiently
* Performs well on mixed numerical and categorical data
* Requires minimal preprocessing
* Provides excellent performance on structured tabular datasets

## Validation Strategy

The development dataset was split into training and validation sets using an 80/20 split.

Model performance was evaluated using:

* RMSE
* MAE

Final validation results:

* RMSE: **526.47**
* MAE: **107.38**

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/train_model.py
```

Generate validation predictions:

```bash
python src/predict.py
```

Generate December predictions:

```bash
python src/predict_december.py
```

Run the scorer:

```bash
python score.py \
  --predictions outputs/validation_predictions.csv \
  --december-predictions data/december-chart-inputs.csv
```

## Output Files

The project generates:

* validation_predictions.csv
* candidate_december.png
* feature_importance.csv

## Author

Titus Ronoh
