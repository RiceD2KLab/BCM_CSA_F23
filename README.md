# BCM_CSA_F23

## Description
This is a repository for the BCM CSA F23 class.

## Installation
`conda create -n csa python=3.9`

`pip install -r requirements.txt`

### Generating results for Initial Experiment
`python classify.py --multirun dataset=shhs1 model=logistic_regression,svc,random_forest target=ahi_a0h3a,ahi_a0h4`

`python regression.py --multirun dataset=shhs1 model=linear_regression,ridge,lasso,knn,svr target=ahi_a0h3a,ahi_a0h4`

### Generating results for the Threshold experiments

`python3 ./src/models/find_threshold.py --multirun dataset=feature_selection_dt,feature_selection_mi,feature_selection_rf model=logistic_regression,svc,random_forest,decision_tree,xgboost target=hf15 threshold_cahi=threshold_1,threshold_2,threshold_3,threshold_4,threshold_5,threshold_6,threshold_7,threshold_8,threshold_9 threshold_c_o=threshold_1,threshold_2,threshold_3,threshold_4,threshold_5`

## Directory structure

```nohighlight
├── LICENSE
├── README.md               <- The top-level README for developers using this project.
├── data
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
│
├── docs                    <- A default Sphinx project; see sphinx-doc.org for details
│
├── models                  <- Trained and serialized models, model predictions, or model summaries
│   ├── threshold           <- Trained models during finding the Threshold for Central Sleep Apnea diagnosis experiments.
│   └── cheap_features      <- Trained models during finding cheap features to predict Central Sleep Apnea Index.
|
├── notebooks               <- Jupyter notebooks. 
│
├── references              <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated graphics and figures to be used in reporting
│
├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
│                               generated with `pip freeze > requirements.txt`
│
├── src                     <- Source code for use in this project.
│   ├── __init__.py         <- Makes src a Python module
│   │
|   ├── conf                <- Config folder for storing Hydra configurations.
|   │   ├── dataset         <- Intermediate data that has been transformed.
|   │   ├── model           <- The original, immutable data dump.
|   │   ├── features        <- The final, canonical data sets for modeling.
|   │   ├── target          <- Intermediate data that has been transformed.
|   │   ├── threshold_cahi  <- Intermediate data that has been transformed.
|   │   ├── config_threshold.yaml        <- Intermediate data that has been transformed.
|   │   └── config.yaml     <- Intermediate data that has been transformed.
|   |
│   ├── models              <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │   
│   ├── classify.py         <- Script to run the classification experiments.
│   ├── find_threshold.py   <- Script to run the threshold experiments.
|   ├── regression.py       <- Script to run the regression experiments. 

```

## Configuration
The configuration file is `config.yaml`. It contains the following parameters:
- `dataset`: The dataset to use. Possible values are `shhs1` and `shhs2`.
- `model`: The model to use. Possible values are `logistic_regression`, `svc`, `random_forest`, 'linear_regression', 'ridge', 'lasso', 'knn', and 'svr'.
- `target`: The target to use. Possible values are `ahi_a0h3a` and `ahi_a0h4`.
- `features`: The features of the model. ['bmi_s1', 'waist', 'neck20']

This project uses Hydra for managing configurations. Hydra is a framework that simplifies the development of complex applications by enabling their configuration to be dynamically composed and overridden.

Here is a brief explanation of the configuration files used in this project:

config.yaml: This is the main configuration file. It contains the default settings for the application.

__init__.py: This file is used to initialize Python packages. It can be left empty but is often used to perform setup needed for the package.

config_threshold.yaml: This file contains the threshold settings for the application.

.DS_Store: This is a file that's automatically created by Mac OS X for its own use. It's not related to the application's configuration.

explore.ipynb: This is a Jupyter notebook used for exploratory data analysis.

shhs1.yaml, shhs2.yaml, feature_selection_dt.yaml, feature_selection_mi.yaml, feature_selection_rf.yaml: These files contain various configurations related to feature selection and the SHHS dataset.

bmi.yaml: This file contains configurations related to BMI (Body Mass Index) features.

logistic_regression.yaml, random_forest.yaml, svc.yaml, linear_regression.yaml, decision_tree.yaml, xgboost.yaml, knn.yaml, lasso.yaml, ridge.yaml, svr.yaml: These files contain configurations for different machine learning models used in the application.

ahi_a0h3a.yaml, ahi_a0h4.yaml, hf15.yaml: These files contain configurations related to different target variables.

threshold_9.yaml, threshold_8.yaml, threshold_7.yaml, threshold_6.yaml, threshold_5.yaml, threshold_4.yaml, threshold_3.yaml, threshold_2.yaml, threshold_1.yaml: These files contain configurations related to different threshold levels for the Central Apnea Index (CAI).

threshold_c_o\threshold_1.yaml, threshold_c_o\threshold_2.yaml, threshold_c_o\threshold_3.yaml, threshold_c_o\threshold_4.yaml, threshold_c_o\threshold_5.yaml: These files contain configurations related to different threshold levels for the Obstructive Apnea Index (OAI).

## Notebooks

This folder contains Jupyter notebooks used for data exploration, preprocessing, feature selection, and result analysis. The following is a list of the files in this folder and their descriptions:

### `0.1-replicate_study.ipynb`: 

This notebook covers the replication of Table 1 and Table 3 of the following study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4909617/#:~:text=The%20prevalence%20of%20CSA%20(defined,those%20aged%2065%20and%20older.&text=In%20a%20later%20cohort%20of,was%20appreciably%20higher%20(7.5%25). This was our first data task assigned by our sponsor to help us understand the goal of the project as well as important features to take note of. 

Table 1  describes demographics and sleep characteristics of the sleep study participants, classified into no sleep apnea, CSA, and OSA. You can find the data in [Table 1](../study/table1.csv "Table 1").

Table 3 describes co-existing diseases and prescription drug use of sleep study participants. You can find the data in [Table 3](../study/table3.csv "Table 3").

### `1.0-data-exploration.ipynb`: This notebook contains code for exploring the dataset and visualizing the data.

### `2.0-feature_selection.ipynb`: This notebook contains code for selecting the best features for the model.

### `3.0-data-preprocessing.ipynb`: 

This notebook contains code for preprocessing the data, including handling missing values and encoding categorical variables.

In this notebook we preprocess the datasets that will be used for training the models for finding a new threshold for CSA diagnosis. We will use 50 different threholds indicator variables allong with the feature selection variables. We will try to predict if a patient had hearth failure or not. The dataset the will yield the best results will be our new proposed threshold

### `4.1-cheap-features-modes.ipynb`: This notebook contains code for selecting cheap features and analyzing the results.

### `4.2-threshold-result-analysis.ipynb`: This notebook contains code for analyzing the results using different threshold values.

### `4.3 Comparing Cheap Features and all features.ipynb`: This notebook contains code for comparing the results of using cheap features versus using all features.

### Utils

This folder contains utility functions used in the notebooks.