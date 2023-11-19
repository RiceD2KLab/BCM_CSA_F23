# Configuration File Documentation

This document provides descriptions of the configuration files used in this project. These files store various settings and parameters used in data processing, model training, and threshold definitions.

## Project Configuration

### `config.yaml`

This configuration file contains project-level settings, including dataset paths, model names, and global project parameters.

### `config_threshold.yaml`

This configuration file is used for specifying threshold-related settings, such as values for classification thresholds.

## Dataset Configuration

### `dataset/shhs1.yaml`

This configuration file defines settings related to the 'shhs1' dataset, including data preprocessing and feature extraction options.

### `dataset/shhs2.yaml`

This configuration file defines settings for the 'shhs2' dataset, including data preprocessing and feature extraction options.

### `dataset/feature_selection_dt.yaml`

This configuration file contains options for feature selection using a decision tree-based method.

### `dataset/feature_selection_mi.yaml`

This configuration file contains options for feature selection using mutual information.

### `dataset/feature_selection_rf.yaml`

This configuration file contains options for feature selection using a random forest-based method.

### `dataset/feature_selection_fs.yaml`

This configuration file contains options for feature selection using a random forward selection method.

### `dataset/feature_selection_bs.yaml`

This configuration file contains options for feature selection using a random backward selection method.

### `features/bmi.yaml`

This configuration file defines settings for BMI (Body Mass Index) feature extraction.

## Model Configuration

### `model/logistic_regression.yaml`

This configuration file defines settings for logistic regression model training.

### `model/random_forest.yaml`

This configuration file contains settings for random forest model training.

### `model/svc.yaml`

This configuration file defines settings for Support Vector Classifier (SVC) model training.

### `model/linear_regression.yaml`

This configuration file contains settings for linear regression model training.

### `model/decision_tree.yaml`

This configuration file defines settings for decision tree model training.

### `model/xgboost.yaml`

This configuration file contains settings for XGBoost model training.

### `model/knn.yaml`

This configuration file defines settings for k-nearest neighbors (KNN) model training.

### `model/lasso.yaml`

This configuration file contains settings for Lasso regression model training.

### `model/ridge.yaml`

This configuration file contains settings for Ridge regression model training.

### `model/svr.yaml`

This configuration file defines settings for Support Vector Regressor (SVR) model training.

## Target Configuration

### `target/ahi_a0h3a.yaml`

This configuration file defines parameters related to the target 'ahi_a0h3a.'

### `target/ahi_a0h4.yaml`

This configuration file defines parameters related to the target 'ahi_a0h4.'

### `target/hf15.yaml`

This configuration file defines parameters related to the target 'hf15.'

## Threshold Configuration

### `threshold_cahi/`

These files contain configurations related to different threshold levels for the Central Apnea Index (CAI).

### `threshold_c_o/`

These files contain configurations related to different threshold levels for the Obstructive Apnea Index (OAI).