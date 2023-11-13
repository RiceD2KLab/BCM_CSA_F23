# 4.1
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from xgboost import XGBRegressor
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor

def split_data(data, features, target):
    """Split on test/train/val.

    Args:
        df (pd.DataFrame): input dataset
        features (list): list of features
        target (str): target column name
    Returns:
        tuple: X_train, X_test, y_train, y_test, X_val, y_val
    """
    # Split on test/train 80/20
    # split startified on target
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=1)
    # make validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test, X_val, y_val


def train_model(model, X_train, y_train, X_test, y_test, X_val, y_val):
    """Train model and return predictions.

    Args:
        model (sklearn model): model to train
        X_train (pd.DataFrame): training data
        y_train (pd.DataFrame): training labels
        X_test (pd.DataFrame): test data
        y_test (pd.DataFrame): test labels
        X_val (pd.DataFrame): validation data
        y_val (pd.DataFrame): validation labels
    
    Returns:
        tuple: mae, model
    """
    from sklearn.metrics import mean_absolute_error
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    # Evaluaate
    mae = mean_absolute_error(y_val, y_pred)
    return mae, model


def find_best_data(folder_loc, target):
    """GIven datasets, return the best dataset and model for some target variable

    Args:
        folder_loc (string): location of datasets
        target (string): target variable
    """
    datasets = os.listdir(folder_loc)

    datasets = [dataset for dataset in datasets if target in dataset]


    best_model = None
    model_name = None
    best_mae = 100000
    best_dataset = None

    for dataset in tqdm(datasets):
    # Load data
        df = pd.read_csv(folder_loc + dataset)

        features = df.columns.tolist()
        features.remove(target)

        # Split data
        X_train, X_test, y_train, y_test, X_val, y_val = split_data(df, features, target)

        # Train xgboost

        mae_xgb, model_xgb = train_model(XGBRegressor(random_state=1), X_train, y_train, X_test, y_test, X_val, y_val)

        # Train random forest

        mae_rf, model_rf = train_model(RandomForestRegressor(random_state=1), X_train, y_train, X_test, y_test, X_val, y_val)

        # Train linear regression

        mae_lr, model_lr = train_model(LinearRegression(), X_train, y_train, X_test, y_test, X_val, y_val)

        # Train lasso

        mae_lasso, model_lasso = train_model(Lasso(random_state=1), X_train, y_train, X_test, y_test, X_val, y_val)

        # Train ridge

        mae_ridge, model_ridge = train_model(Ridge(random_state=1), X_train, y_train, X_test, y_test, X_val, y_val)

        # Train decision tree

        mae_dt, model_dt = train_model(DecisionTreeRegressor(random_state=1), X_train, y_train, X_test, y_test, X_val, y_val)

        # Save best model

        if mae_xgb < best_mae:
            best_mae = mae_xgb
            best_model = model_xgb
            model_name = 'xgb'
            best_dataset = dataset
        if mae_rf < best_mae:
            best_mae = mae_rf
            best_model = model_rf
            model_name = 'rf'
            best_dataset = dataset
        if mae_lr < best_mae:
            best_mae = mae_lr
            best_model = model_lr
            model_name = 'lr'
            best_dataset = dataset
        if mae_lasso < best_mae:
            best_mae = mae_lasso
            best_model = model_lasso
            model_name = 'lasso'
            best_dataset = dataset
        if mae_ridge < best_mae:
            best_mae = mae_ridge
            best_model = model_ridge
            model_name = 'ridge'
            best_dataset = dataset
        if mae_dt < best_mae:
            best_mae = mae_dt
            best_model = model_dt
            model_name = 'dt'
            best_dataset = dataset
    return best_mae, best_model, model_name, best_dataset
