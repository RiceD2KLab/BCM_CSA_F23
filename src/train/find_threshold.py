###########################################################
# This file consist of experiments for finiding a new threshold value for 
# diagonissi of CSA. The script is ran throug a command line interface.
# The results can be found in the results folder under the subfolder
# threshold. 
###########################################################
# Imports
import hydra
import pickle
from omegaconf import DictConfig
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import numpy as np
import os
from pathlib import Path
import random
import numpy as np

random.seed(0)
np.random.seed(0)

log = logging.getLogger(__name__)

root = os.getcwd()

def normalize_data(data, features):
    # Normalize the features, using sklearn StandardScaler
    scaler = StandardScaler(random_state=1)
    scaler.fit(data[features])
    data_scaled = scaler.transform(data[features])

    scaled_data = pd.DataFrame(data_scaled, columns=features)

    return scaled_data

def oversample_data(data, features, target):
    oversample = SMOTE(random_state=1)
    X, y = oversample.fit_resample(data[features], data[target])
    return X, y
    

def split_data(data, features, target):
    """Split on test/train/val.

    Args:
        df (pd.DataFrame): input dataset
        features (list): list of features
        target (str): target column name
    """
    # Split on test/train 80/20
    # split startified on target
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=1, stratify=data[target])
    # make validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1, stratify=y_train)
    return X_train, X_test, y_train, y_test, X_val, y_val


def hyperparameter_tuning(model_name, X_train, y_train, X_val, y_val, feature_slection, n_iter=50):
    # for each model find the best parameters and return the model
    # hyperparameter tuning is done with RandomizedSearchCV

    os.makedirs(os.path.join(root, './models/threshold'), exist_ok=True)
    model_path = os.path.join(root, './models/threshold', model_name + '_' + feature_slection + '.pkl')

    # check if model is already trained by checking if the model file startswith model name
    if os.path.isfile(model_path):
        log.info("Model already trained")
        model = pickle.load(open(model_path, 'rb'))
    
    else:

        if model_name == "logistic_regression":
            log.info("Training logistic regression model")
            model = LogisticRegression(class_weight='balanced', random_state=1)
            # Create regularization penalty space
            penalty = ['l1', 'l2']
            # Create regularization hyperparameter distribution using uniform distribution
            C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            # Create hyperparameter options
            hyperparameters = dict(C=C, penalty=penalty)
            # Create randomized search 5-fold cross validation and 100 iterations
            clf = RandomizedSearchCV(model, hyperparameters, random_state=1, n_iter=n_iter, cv=5, verbose=0, n_jobs=-1)
            # Fit randomized search
            best_model = clf.fit(X_train, y_train)
            # View best hyperparameters
            log.info('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
            log.info('Best C:', best_model.best_estimator_.get_params()['C'])
            log.info('Best Score:', best_model.best_score_)
            model = best_model.best_estimator_

        elif model_name == 'random_forest':
            log.info("Training random forest model")
            model = RandomForestClassifier(class_weight='balanced', random_state=1)
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 200)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 100, num = 20)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]
            # Create the random grid
            random_grid = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}
            # Random search of parameters, using 3 fold cross validation,
            # search across 100 different combinations, and use all available cores
            clf = RandomizedSearchCV(model, random_grid, random_state=1, n_iter=n_iter, cv=3, verbose=0, n_jobs=-1)
            # Fit randomized search
            best_model = clf.fit(X_train, y_train)
            # View best hyperparameters
            log.info('Best n_estimators:', best_model.best_estimator_.get_params()['n_estimators'])
            log.info('Best max_features:', best_model.best_estimator_.get_params()['max_features'])
            log.info('Best max_depth:', best_model.best_estimator_.get_params()['max_depth'])
            log.info('Best min_samples_split:', best_model.best_estimator_.get_params()['min_samples_split'])
            log.info('Best min_samples_leaf:', best_model.best_estimator_.get_params()['min_samples_leaf'])
            log.info('Best bootstrap:', best_model.best_estimator_.get_params()['bootstrap'])
            log.info('Best Score:', best_model.best_score_)
            model = best_model.best_estimator_

        elif model_name == 'svc':
            log.info("Training svc model")
            model = SVC(class_weight='balanced', random_state=1)
            # Create the random grid
            random_grid = {'C': [0.1, 1, 10, 100, 1000],
                        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                        'kernel': ['rbf', 'poly', 'sigmoid']}
            # Random search of parameters, using 3 fold cross validation,
            # search across 100 different combinations, and use all available cores
            clf = RandomizedSearchCV(model, random_grid, random_state=1, n_iter=n_iter, cv=3, verbose=0, n_jobs=-1)
            # Fit randomized search
            best_model = clf.fit(X_train, y_train)
            # View best hyperparameters
            log.info('Best C:', best_model.best_estimator_.get_params()['C'])
            log.info('Best gamma:', best_model.best_estimator_.get_params()['gamma'])
            log.info('Best kernel:', best_model.best_estimator_.get_params()['kernel'])
            log.info('Best Score:', best_model.best_score_)
            model = best_model.best_estimator_

        elif model_name == 'decision_tree':
            log.info("Training Decision Tree model")
            model = DecisionTreeClassifier(class_weight='balanced', random_state=1)
            # Create the random grid
            random_grid = {'criterion': ['gini', 'entropy'],
                        'splitter': ['best', 'random'],
                        'max_depth': [int(x) for x in np.linspace(10, 100, num = 10)],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]}
            # Random search of parameters, using 3 fold cross validation,
            # search across 100 different combinations, and use all available cores
            clf = RandomizedSearchCV(model, random_grid, random_state=1, n_iter=n_iter, cv=3, verbose=0, n_jobs=-1)
            # Fit randomized search
            best_model = clf.fit(X_train, y_train)
            # View best hyperparameters
            log.info('Best criterion:', best_model.best_estimator_.get_params()['criterion'])
            log.info('Best splitter:', best_model.best_estimator_.get_params()['splitter'])
            log.info('Best max_depth:', best_model.best_estimator_.get_params()['max_depth'])
            log.info('Best min_samples_split:', best_model.best_estimator_.get_params()['min_samples_split'])
            log.info('Best min_samples_leaf:', best_model.best_estimator_.get_params()['min_samples_leaf'])
            log.info('Best Score:', best_model.best_score_)
            model = best_model.best_estimator_
    
        elif model_name == 'xgboost':
            log.info("Training XGBoost model")
            # scale_pos_weight = total_negative_examples / total_positive_examples
            scale_pos = np.sum(y_train == 0) / np.sum(y_train == 1)
            model = XGBClassifier(scale_pos_weight=scale_pos, random_state=1)
            # Create the random grid
            random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 1000, num = 200)],
                        'max_depth': [int(x) for x in np.linspace(10, 100, num = 10)],
                        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
                        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1]}
            # Random search of parameters, using 3 fold cross validation,
            # search across 100 different combinations, and use all available cores
            clf = RandomizedSearchCV(model, random_grid, random_state=1, n_iter=n_iter, cv=3, verbose=0, n_jobs=-1)
            # Fit randomized search
            best_model = clf.fit(X_train, y_train)
            # View best hyperparameters
            log.info('Best n_estimators:', best_model.best_estimator_.get_params()['n_estimators'])
            log.info('Best max_depth:', best_model.best_estimator_.get_params()['max_depth'])
            log.info('Best learning_rate:', best_model.best_estimator_.get_params()['learning_rate'])
            log.info('Best subsample:', best_model.best_estimator_.get_params()['subsample'])
            log.info('Best colsample_bytree:', best_model.best_estimator_.get_params()['colsample_bytree'])
            log.info('Best Score:', best_model.best_score_)
            model = best_model.best_estimator_


    # save model in ../models folder

    pickle.dump(model, open(model_path, 'wb'))
    
    # test model on validation set
    y_pred = model.predict(X_val)
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    log.info("F1 Validation score weighted: %s",  f1_weighted)

    return model


@hydra.main(config_path="../conf", config_name="config_threshold.yaml")
def classify(cfg: DictConfig) -> None:
    """Function that takes in hydra config and runs the classification pipeline.

    Args:
        cfg (DictConfig): pass the desired dataset, model, features, and target from config.yaml 
        Example: python my_app.py --multirun dataset=shhs1 model=logistic_regression,svc target=ahi_a0h3a,ahi_a0h4
    """
    target_variable = cfg.target.name
    threshold_1 = cfg.threshold_cahi.value
    threshold_2 = cfg.threshold_c_o.value
    feature_selection_method = cfg.dataset.path.split("/")[-1]
    dataset_path = cfg.dataset.path + "_" + target_variable + "_threshold_" + str(threshold_1) + "_" + str(threshold_2) + '.csv'
    dataset_path = os.path.join(root, dataset_path)
    dataset_name = cfg.dataset.name

    model_name = cfg.model.name
    print(f"Data: {dataset_name}, Model: {model_name} Data path: {dataset_path}")
    log.info(f"Data: {dataset_name}, Model: {model_name}, Target: {cfg.target.name}")

    # Read data
    data = pd.read_csv(dataset_path)
    features = data.columns.tolist() 
    features.remove(target_variable)

    # Oversample the dataset
    X, y = oversample_data(data, features, target_variable)

    data = pd.concat([X, y], axis=1)

    # Preprocess data
    X_train, X_test, y_train, y_test, X_val, y_val = split_data(data, features, target_variable)    

    model = hyperparameter_tuning(model_name, X_train, y_train, X_val, y_val, feature_selection_method, n_iter=10)

    # Evaluate model, F1 score
    f1_weighted = f1_score(y_test, model.predict(X_test), average='weighted')
    log.info("F1 Test score weighted: %s",  f1_weighted)

    # log the confusion matrix

    cm = confusion_matrix(y_test, model.predict(X_test))
    log.info("Confusion matrix: %s",  cm)
    print(f"### {model_name}  {threshold_1}  {threshold_2}  {dataset_name} ###")
    print("### DONE ###")


if __name__ == "__main__":
    classify()