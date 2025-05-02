import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

import optuna

device = "cpu"

# Load the training data
data = pd.read_csv('./clinical_model/training_data/training_data.csv')              # Features (X)
labels = pd.read_csv("./clinical_model/training_data/training_targets_OS_log.csv")      # Survival Time (T)
occurence = pd.read_csv("./clinical_model/training_data/training_targets_OS_occurence.csv")  # Event indicator (1 = event, 0 = censored)

labels = labels.squeeze()
occurence = occurence.squeeze()


# Split data into training and validation sets
data_train, data_val, labels_train, labels_val, occurence_train, occurence_val = train_test_split(
    data, labels, occurence, test_size=0.2, random_state=42
)

# Set lower and upper bounds for training and validation sets
y_lower_train = labels_train.copy()
y_upper_train = np.where(occurence_train == 1, labels_train, np.inf)

y_lower_val = labels_val.copy()
y_upper_val = np.where(occurence_val == 1, labels_val, np.inf)

# Create DMatrix for training and validation
dtrain = xgb.DMatrix(data_train)
dtrain.set_float_info("label_lower_bound", y_lower_train)
dtrain.set_float_info("label_upper_bound", y_upper_train)

dval = xgb.DMatrix(data_val)
dval.set_float_info("label_lower_bound", y_lower_val)
dval.set_float_info("label_upper_bound", y_upper_val)

# Define hyperparameter search space
base_params = {'verbosity': 0,
              'objective': 'survival:aft',
              'eval_metric': 'aft-nloglik',
              'tree_method': 'hist',
              'seed': 42}  # Hyperparameters common to all trials

def objective(trial):
    params = {'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
              'aft_loss_distribution': trial.suggest_categorical('aft_loss_distribution',
                                                                  ['normal', 'logistic', 'extreme']),
              'aft_loss_distribution_scale': trial.suggest_loguniform('aft_loss_distribution_scale', 0.1, 10.0),
              'max_depth': trial.suggest_int('max_depth', 3, 8),
              'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
              'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0)}  # Search space
    params.update(base_params)
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'valid-aft-nloglik')
    bst = xgb.train(params, dtrain, num_boost_round=10000,
                    evals=[(dtrain, 'train'), (dval, 'valid')],
                    early_stopping_rounds=50, verbose_eval=False, callbacks=[pruning_callback]
                    )
    if bst.best_iteration >= 25:
        return bst.best_score
    else:
        return np.inf  # Reject models with < 25 trees

# Run hyperparameter search
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)
print('Completed hyperparameter tuning with best aft-nloglik = {}.'.format(study.best_trial.value))
params = {}
params.update(base_params)
params.update(study.best_trial.params)

# Re-run training with the best hyperparameter combination
print('Re-running the best trial... params = {}'.format(params))
bst = xgb.train(params, dtrain, num_boost_round=10000,
                evals=[(dtrain, 'train'), (dval, 'valid')],
                early_stopping_rounds=50
                )

preds = bst.predict(dval, output_margin= True)
print(preds.shape())

# Filter for samples where occurence = 1
actual_values = labels_val[occurence_val == 1]
predicted_values = preds[occurence_val == 1]


# Print predicted and actual values
print("Predicted vs Actual values for samples where occurence = 1:")
for actual, predicted in zip(actual_values, predicted_values):
    print(f"Actual: {actual}, Predicted: {predicted}")

mae = mean_absolute_error(actual_values, predicted_values)

from lifelines.utils import concordance_index
c_index = concordance_index(labels_val, -preds, occurence_val)  # negate if lower time = higher risk
print("C-index:", c_index)
print("mean error:", mae)


"""
# Train model
num_round = 500
evals = [(dtrain, "train"), (dval, "validation")]
bst = xgb.train(params, dtrain, num_boost_round=num_round, evals=evals)

# Make predictions 
#preds = bst.predict(dtrain)
"""
xgb.plot_importance(bst)
plt.show()  # Explicitly show the plot
