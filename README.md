# BCM_CSA_F23

## Description
This is a repository for the BCM CSA F23 class.

## Installation
`conda create -n csa -python=3.9`
`conda activate csa`
`pip install -r requirements.txt`

## Usage
`python clasify.py --multirun dataset=shhs1 model=logistic_regression,svc,random_forest target=ahi_a0h3a,ahi_a0h4`

## Configuration
The configuration file is `config.yaml`. It contains the following parameters:
- `dataset`: The dataset to use. Possible values are `shhs1` and `shhs2`.
- `model`: The model to use. Possible values are `logistic_regression`, `svc`, and `random_forest`.
- `target`: The target to use. Possible values are `ahi_a0h3a` and `ahi_a0h4`.
- `features`: The features of the model. ['bmi_s1', 'waist', 'neck20']