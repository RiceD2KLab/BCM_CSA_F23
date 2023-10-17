###########################################################
# This file was created during the initial experiment that was provided as a task from the sponsor.
# In this file we test several models with the features extracted by expert knowledge.
# The results are in the results fodler under initial_study folder.

# The following script is used to run the regression pipeline.
###########################################################
# Imports
import hydra
from omegaconf import DictConfig
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import os
from pathlib import Path

log = logging.getLogger(__name__)

root = os.getcwd()

def data_preprocessing(data, features, target):
    """Normalize the features, drop nan, categorize the target, and split on test/train.

    Args:
        df (pd.DataFrame): input dataset
        features (list): list of features
        target (str): target column name
    """
    # As a reference, the AHI is defined as the number of apneas and hypopneas per hour of sleep.
    # AHI >=5 (all sleep apnea)
    # AHI>=15 (moderate sleep apnea)
    # AHI >= 30 (severe sleep apnea)

    data[target] = pd.cut(data[target], bins=[0, 5, 15, 30, 1000], labels=[0, 1, 2, 3])
    
    # Drop nan on rows
    data = data.dropna()

    # Normalize the features, using sklearn StandardScaler
    scaler = StandardScaler()
    scaler.fit(data[features])
    data_scaled = scaler.transform(data[features])

    scaled_data = pd.DataFrame(data_scaled, columns=features)

    # Split on test/train 80/20
    # split startified on target
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, data[target], test_size=0.2, random_state=42, stratify=data[target])
    return X_train, X_test, y_train, y_test


@hydra.main(config_path="conf", config_name="config.yaml")
def regression(cfg: DictConfig) -> None:
    """Function that takes in hydra config and runs the regression pipeline.

    Args:
        cfg (DictConfig): pass the desired dataset, model, features, and target from config.yaml 
        Example: python my_app.py --multirun dataset=shhs1 model=linear_regression,lasso,ridge,knn,svr target=ahi_a0h3a, ahi_a0h4
    """
    dataset_path = os.path.join(root, cfg.dataset.path)
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name
    print(f"Data: {dataset_name}, Model: {model_name}")
    log.info(f"Data: {dataset_name}, Model: {model_name}, Features: {cfg.features.names}, Target: {cfg.target.name}")

    # Read data
    data = pd.read_csv(dataset_path)
    data = data[cfg.features.names + [cfg.target.name]]
    # print("Data: ", data.head(2))
    # Preprocess data
    X_train, X_test, y_train, y_test = data_preprocessing(data, cfg.features.names, cfg.target.name)    

    # Train model
    if model_name == "linear_regression":
        log.info("Training linear regression model")
        model = LinearRegression()
    elif model_name == 'lasso':
        log.info("Training LASSO model")
        model = LASSO()
    elif model_name == 'ridge':
        log.info("Training Ridge Regression model")
        model = Ridge()
    elif model_name == 'knn':
        log.info("Training KNN model")
        model = KNeighborsRegressor()
    elif model_name == 'svr':
        log.info('Training SVM model')
        model = SVR()

    model.fit(X_train, y_train)

    # Evaluate model with MSE and R2 score
    mse = mean_squared_error(y_test, model.predict(X_test), average='weighted')
    log.info("MSE: %s",  mse)
    r_squared = r2_score(y_test, model.predict(X_test), average='weighted')
    log.info("R Squared Score: %s",  r_squared)


if __name__ == "__main__":
    regression()

