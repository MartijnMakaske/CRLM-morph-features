# Imports

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import xgboost as xgb
import torch
import os

device = "cpu"

# Load the training data
data = pd.read_csv('./clinical_model/training_data/training_data.csv')
labels = pd.read_csv("./clinical_model/training_data/training_targets_OS.csv")

labels = labels.squeeze()

X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=.2)

# Set up XGBoost classifier
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,  # Number of boosting rounds (trees)
    eval_metric=['mae', 'rmse'],
    random_state=42,
    device=device,
)


# Train the model
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# Save model
model.save_model("./clincial_model/XGBoost_models/best_model.json")