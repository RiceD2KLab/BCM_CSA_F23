# Notebooks Folder

This folder contains Jupyter notebooks used for data exploration, preprocessing, feature selection, and result analysis. The following is a list of the files in this folder and their descriptions:

## `00_replicate_study.ipynb`: 

This notebook covers the replication of Table 1 and Table 3 of the following study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4909617/#:~:text=The%20prevalence%20of%20CSA%20(defined,those%20aged%2065%20and%20older.&text=In%20a%20later%20cohort%20of,was%20appreciably%20higher%20(7.5%25). This was our first data task assigned by our sponsor to help us understand the goal of the project as well as important features to take note of. 

Table 1  describes demographics and sleep characteristics of the sleep study participants, classified into no sleep apnea, CSA, and OSA. You can find the data in [Table 1](../study/table1.csv "Table 1").

Table 3 describes co-existing diseases and prescription drug use of sleep study participants. You can find the data in [Table 3](../study/table3.csv "Table 3").

## `01_exploratory-data-analysis.ipynb`: 

This notebook contains code for exploring the dataset and visualizing the data. We looked at the BMI, Waist Circumference, Neck Circumference of the patients in the study as well as the severity of CSA within the patients.

## `02_feature-selection.ipynb`: 

This notebook contains code for selecting the best features for the model. 

For the target variable of ahi_c0h4a (in which we want to find out cheaper ways to diagnose CSA), we utilized decision tree, random forest, mutual information, forward selection AIC, forward selection BIC, backward selection AIC

For the target variable of hf15 (in which we want to question the current CSA threshold), we utilized MRMR 10, MRMR 20, random forest, decision tree, mutual information, forward selection AIC, forward selection BIC, backward selection AIC

## `03_data-preprocessing.ipynb`: 

This notebook contains code for preprocessing the data, including handling missing values and encoding categorical variables.

In this notebook we preprocess the datasets that will be used for training the models for finding a new threshold for CSA diagnosis. We will use 50 different threholds indicator variables allong with the feature selection variables. We will try to predict if a patient had hearth failure or not. The dataset the will yield the best results will be our new proposed threshold

## `04a_cheap-features-models.ipynb`: 

This notebook contains code for selecting the best features and model for predicting our first goal (finding cheap features to diagnose CSA).

We also tuned the hyperparameters after finding our best model.



## `04b_threshold-result-analysis.ipynb`: 

This notebook contains code for analyzing the results using different threshold values for our second goal (questioning the current threshold for diagnosing CSA).

## `05a_explainability.ipynb`: 

In this notebook, we explore variables that contribute the most to our models using SHAP in an attempt to improve our explainability. 

## '05b_error-analysis.ipynb'

In this notebook we will isolate and analyze factors that give the most error and plot some visualizations examining them.

## '05c_interpretation.ipynb'

In this notebook we will interpret important features using different methods and visualize the most important, using techniques such as Random Forest and SHAP values. 

We also interpret the results from our threshold analysis.



