# BCM_CSA_F23

## Description
This is a repository for the BCM CSA F23 class.

## Installation
`conda create -n csa python=3.9`

`pip install -r requirements.txt`

### Generating results for Initial Experiment
`python clasify.py --multirun dataset=shhs1 model=logistic_regression,svc,random_forest target=ahi_a0h3a,ahi_a0h4`

`python regression.py --multirun dataset=shhs1 model=linear_regression,ridge,lasso,knn,svr target=ahi_a0h3a,ahi_a0h4`

### Generating results for the Threshold experiments

`python ./src/models/find_threshold.py --multirun dataset=feature_selection_dt,feature_selection_mi,feature_selection_rf model=logistic_regression,svc,random_forest,decision_tree,xgboost target=hf15 threshold_cahi=threshold_1,threshold_2,threshold_3,threshold_4,threshold_5,threshold_6,threshold_7,threshold_8,threshold_9 threshold_c_o=threshold_1,threshold_2,threshold_3,threshold_4,threshold_5`

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

## Notebooks
0.1 and 0.2 are the experiments for replication of the Central Sleep Apnea paper. 0.1 focuses on the demographic data, and 0.2 focuses on drug use data. 

1.0 is the Exploratory Data Analysis (EDA)

2.0 is the feature selection

3.0 is the preprocessing for cheap feature analysis and the threshold analysis

4.1, 4.2, and 4.3 are for results and analysis. 4.1 is for cheap feature analysi, 4.2 is for threshold analysis, and 4.3 is to compare the model performance between cheap features and all features from feature selection. 
