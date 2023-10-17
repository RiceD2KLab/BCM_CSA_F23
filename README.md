# BCM_CSA_F23

## Description
This is a repository for the BCM CSA F23 class.

## Installation
`pip install -r requirements.txt`

## Usage
`python clasify.py --multirun dataset=shhs1 model=logistic_regression,svc,random_forest target=ahi_a0h3a,ahi_a0h4`

## Directory structure

```nohighlight
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- Make this project pip installable with `pip install -e`
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py

```

## Configuration
The configuration file is `config.yaml`. It contains the following parameters:
- `dataset`: The dataset to use. Possible values are `shhs1` and `shhs2`.
- `model`: The model to use. Possible values are `logistic_regression`, `svc`, and `random_forest`.
- `target`: The target to use. Possible values are `ahi_a0h3a` and `ahi_a0h4`.
- `features`: The features of the model. ['bmi_s1', 'waist', 'neck20']