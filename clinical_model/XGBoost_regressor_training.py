# Imports

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import xgboost as xgb
import torch
import os

device = "cpu"

# Load the training data
data = pd.read_csv('./clinical_model/training_data/training_data.csv')
labels = pd.read_csv("./clinical_model/training_data/training_targets_OS.csv")
print(data.shape, labels.shape)

labels = labels.squeeze()

#X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=.2)
all_eval_results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
for train_index, test_index in kf.split(data):
    # Set up XGBoost classifier
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,  # Fewer trees
        max_depth=3,  # Shallower trees
        learning_rate=0.01,  # Slower learning rate
        subsample=0.8,  # Subsample data
        colsample_bytree=0.8,  # Subsample features
        reg_alpha=1,  # L1 regularization
        reg_lambda=10,  # L2 regularization
        random_state=42,
        early_stopping_rounds=20,
        eval_metric=['mae', 'rmse']
    )

    # Train the model
    model.fit(
        data.iloc[train_index],
        labels[train_index],
        eval_set=[(data.iloc[train_index], labels[train_index]), (data.iloc[test_index], labels[test_index])],
        #verbose=True
    )
        
    # Load evals result by calling the evals_result() function
    all_eval_results.append(model.evals_result())

    y_pred = model.predict(data.iloc[test_index])
    print(f"fold: {fold} -------------------------------------------")
    print("RMSE:", root_mean_squared_error(labels[test_index], y_pred))
    print("MAE:", mean_absolute_error(labels[test_index], y_pred))
    print("R²:", r2_score(labels[test_index], y_pred))

    fold += 1



# Initialize a figure for MAE and RMSE
plt.figure(figsize=(12, 6))

# Plot MAE for all folds
plt.subplot(1, 2, 1)
for fold_idx, evals_result in enumerate(all_eval_results):
    epochs = range(len(evals_result['validation_0']['mae']))
    plt.plot(epochs, evals_result['validation_0']['mae'], label=f'Fold {fold_idx + 1} Training MAE', color='blue')
    plt.plot(epochs, evals_result['validation_1']['mae'], label=f'Fold {fold_idx + 1} Validation MAE', linestyle='--', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('MAE over Epochs (All Folds)')
plt.legend()

# Plot RMSE for all folds
plt.subplot(1, 2, 2)
for fold_idx, evals_result in enumerate(all_eval_results):
    epochs = range(len(evals_result['validation_0']['rmse']))
    plt.plot(epochs, evals_result['validation_0']['rmse'], label=f'Fold {fold_idx + 1} Training RMSE', color='blue')
    plt.plot(epochs, evals_result['validation_1']['rmse'], label=f'Fold {fold_idx + 1} Validation RMSE', linestyle='--', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('RMSE over Epochs (All Folds)')
plt.legend()

plt.tight_layout()
plt.show()

xgb.plot_importance(model)
plt.show()  # Explicitly show the plot

# Save model
#model.save_model("./clincial_model/XGBoost_models/best_model.json")

