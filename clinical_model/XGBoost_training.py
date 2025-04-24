# Imports

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import xgboost as xgb


# Load the training data
data = pd.read_csv('training_data/training_data.csv')
labels = pd.read_csv("training_data/training_labels_OS.csv")

labels = labels.squeeze()

print(labels.shape)
print(data.shape)

X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=.2)

# Set up XGBoost classifier
model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=4,
    eval_metric='mlogloss',
    use_label_encoder=False,
    tree_method='gpu_hist',  # change to 'hist' if no GPU
    predictor='gpu_predictor',
    random_state=42
)

# Train the model
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='mlogloss',
    early_stopping_rounds=10,
    verbose=True
)

# Save model
model.save_model("XGBoost_models/best_model.json")