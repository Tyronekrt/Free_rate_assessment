#!/bin/bash

python src/train.py
python src/predict_validation.py
python src/predict_december.py

python score.py \
    --predictions outputs/validation_predictions.csv \
    --december-predictions data/december-chart-inputs.csv