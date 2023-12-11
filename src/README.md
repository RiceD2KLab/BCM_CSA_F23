# Methods

In this folder is the core of our analysis here we explore the two main objectives of our study: (1) Finding cost-effective features for predicting CSA and (2) Finding a new threshold for diagnosing CSA.

## Generating results for Initial Experiment
`python classify.py --multirun dataset=shhs1 model=logistic_regression,svc,random_forest target=ahi_a0h3a,ahi_a0h4`

`python regression.py --multirun dataset=shhs1 model=linear_regression,ridge,lasso,knn,svr target=ahi_a0h3a,ahi_a0h4`

## Generating results for the Threshold experiments

`python3 ./src/models/find_threshold.py --multirun dataset=feature_selection_dt,feature_selection_mi,feature_selection_rf model=logistic_regression,svc,random_forest,decision_tree,xgboost target=hf15 threshold_cahi=threshold_1,threshold_2,threshold_3,threshold_4,threshold_5,threshold_6,threshold_7,threshold_8,threshold_9 threshold_c_o=threshold_1,threshold_2,threshold_3,threshold_4,threshold_5`

## Results



## Utils

This folder contains utility functions used in the notebooks.
