# Notebooks Folder

This folder contains Jupyter notebooks used for data exploration, preprocessing, feature selection, and result analysis. The following is a list of the files in this folder and their descriptions:

## `0.1-replicate_study.ipynb`: 

This notebook covers the replication of Table 1 and Table 3 of the following study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4909617/#:~:text=The%20prevalence%20of%20CSA%20(defined,those%20aged%2065%20and%20older.&text=In%20a%20later%20cohort%20of,was%20appreciably%20higher%20(7.5%25). This was our first data task assigned by our sponsor to help us understand the goal of the project as well as important features to take note of. 

Table 1  describes demographics and sleep characteristics of the sleep study participants, classified into no sleep apnea, CSA, and OSA. You can find the data in [Table 1](../study/table1.csv "Table 1").

Table 3 describes co-existing diseases and prescription drug use of sleep study participants. You can find the data in [Table 3](../study/table3.csv "Table 3").

## `1.0-data-exploration.ipynb`: This notebook contains code for exploring the dataset and visualizing the data.

## `2.0-feature_selection.ipynb`: This notebook contains code for selecting the best features for the model.

## `3.0-data-preprocessing.ipynb`: 

This notebook contains code for preprocessing the data, including handling missing values and encoding categorical variables.

In this notebook we preprocess the datasets that will be used for training the models for finding a new threshold for CSA diagnosis. We will use 50 different threholds indicator variables allong with the feature selection variables. We will try to predict if a patient had hearth failure or not. The dataset the will yield the best results will be our new proposed threshold

## `4.1-cheap-features-modes.ipynb`: This notebook contains code for selecting cheap features and analyzing the results.

## `4.2-threshold-result-analysis.ipynb`: This notebook contains code for analyzing the results using different threshold values.

## `4.3 Comparing Cheap Features and all features.ipynb`: This notebook contains code for comparing the results of using cheap features versus using all features.

## Utils

This folder contains utility functions used in the notebooks.
