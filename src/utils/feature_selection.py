import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_classification
from mrmr import mrmr_classif

def write_to_csv(dataset, filepath, name):
  """Writes a dataset to a given file at a given file path
  
  Args:
    dataset (pd.DataFrame): input dataset
    filepath (string): prefix file path (ends with /)
    name (string): csv file name
  
  Returns:
    None
  """
  csv_name = filepath + name + ".csv"
  dataset.to_csv(csv_name, index=False)

def write_features_to_csv(df, shhs1, model_name, target_vars, filepath, model_importances=None, model_features=None):
  """Takes important features from given model and writes data to a new csv

  Args:
    df (pd.DataFrame): the imputed input dataset 
    shhs1 (pd.DataFrame): Sleep Hearth Health Study dataset
    model (sklearn model): model to find important features from
    model_name (string): model name
    target_vars ([string]): target variable(s)
  
  Returns:
    length of the important features
  """
  # populate important features CSV from SHHS data
  if model_features is None: 
    model_features = np.array(df.columns)[model_importances > 0]
  model_dataset = df[model_features]
  for var in target_vars:
    model_dataset.loc[:, var] = shhs1[var]

  # write important features to CSV file
  write_to_csv(model_dataset, filepath, model_name.replace(" ", "_") + "_" + target_vars[0])

  return len(model_features)

def get_backward_selection_features(x, y, IC_metric, threshold=1.8, debug=False):
  """
  """
  backward_selection_features = list(x.columns)

  # Start with an arbitrarily large initial AIC/BIC
  previous_IC = float('inf')

  #run backward selection
  while len(backward_selection_features) > 0:
    removed_feature_IC = float('inf')
    feature_to_remove = None

    for feature in backward_selection_features:
      candidate_features = backward_selection_features.copy()
      candidate_features.remove(feature)
      
      x_backward = x[candidate_features]
      y_backward = y.copy()
      
      x_backward = sm.add_constant(x_backward)
      model = sm.OLS(y_backward, x_backward).fit()
      if IC_metric == "AIC":
        IC = model.aic
      elif IC_metric == "BIC":
        IC = model.bic
      
      # Update the best feature to remove if the current one is better (lower AIC/BIC)
      if IC < removed_feature_IC:
          removed_feature_IC = IC
          feature_to_remove = feature

    # Break the loop if change in AIC/BIC is smaller than the threshold or if removing a feature increases the AIC/BIC
    if abs(removed_feature_IC - previous_IC) < threshold or removed_feature_IC > previous_IC:
      if debug:
        print(f'No significant improvement in {IC_metric} or removing any more features deteriorates the model. Stopping backward selection.')
      break

    backward_selection_features.remove(feature_to_remove)
    if debug:
      print(f"Removed feature: {feature_to_remove}, {IC_metric} without feature: {removed_feature_IC}")

    # Update the previous AIC/BIC for the next iteration
    previous_IC = removed_feature_IC
  
  return backward_selection_features

def get_forward_selection_features(x, y, IC_metric, threshold, debug=False):
  """
  """
  forward_selection_features = []

  # Start with an arbitrarily large initial IC value
  previous_IC = float('inf')

  #run forward selection
  while len(forward_selection_features) < len(x.columns):
    remaining_features = list(set(x.columns) - set(forward_selection_features))
    best_feature_IC = float('inf')
    best_feature = None

    for feature in remaining_features:
        candidate_features = forward_selection_features.copy()
        candidate_features.append(feature)

        x_forward = x[candidate_features]
        y_forward = y.copy()

        x_forward = sm.add_constant(x_forward)
        model = sm.OLS(y_forward, x_forward).fit()
        if IC_metric == "AIC":
          IC = model.aic
        elif IC_metric == "BIC":
          IC = model.bic

        # Update the best feature if the current one is better (lower AIC/BIC)
        if IC < best_feature_IC:
            best_feature_IC = IC
            best_feature = feature

    # Break the loop if change in AIC/BIC is smaller than the threshold or if no significant decrease in AIC/BIC
    if abs(best_feature_IC - previous_IC) < threshold or best_feature is None:
        if debug:
            print(f'No significant improvement in {IC_metric} or no further significant feature found. Stopping forward selection.')
        break

    forward_selection_features.append(best_feature)
    if debug:
        print(f"Added feature: {best_feature}, {IC_metric} with features: {best_feature_IC}")

    # Update the previous AIC/BIC for the next iteration
    previous_IC = best_feature_IC
  
  return forward_selection_features

def get_features(method_name, x, y):
    """Creates and fits a feature selection model given the name, x, and y data.
    
    Available method names:
        1. decision tree
        2. random forest
        3. mutual information
        4. forward selection
        5. backward selection
        6. MRMR

    Args:
        method_name (str): feature selection name
        x (pd.DataFrame): the imputed input dataset (exludes target variable column)
        y (pd.DataFrame): the data column of the target variable
        
    Returns:
      (model_importances, model_features),
    """
    if method_name == "decision tree":
      fs = DecisionTreeRegressor(random_state = 131)
      fs.fit(x, y)
      return fs.feature_importances_, None

    elif method_name == "random forest":
      fs = RandomForestRegressor(n_estimators=100, random_state = 52)  # You can change the number of trees if needed
      fs.fit(x, y)
      return fs.feature_importances_, None

    elif method_name == "mutual information":
      mi = mutual_info_regression(x, y, random_state = 568)
      return mi, None

    elif method_name == "forward selection AIC":
      forward_selection_features_aic = get_forward_selection_features(x, y, "AIC", 3)
      return None, forward_selection_features_aic

    elif method_name == "forward selection BIC":
      forward_selection_features_bic = get_forward_selection_features(x, y, "BIC", 2)
      return None, forward_selection_features_bic

    elif method_name == "MRMR 10":
      selected_features_10 = mrmr_classif(X=x, y=y, K=10)
      return None, selected_features_10

    elif method_name == "MRMR 20":
      selected_features_20 = mrmr_classif(X=x, y=y, K=20)
      return None, selected_features_20
    
    elif method_name == "backward selection AIC":
      backward_selection_features = get_backward_selection_features(x, y, "AIC", 1.8)
      return None, backward_selection_features

    else:
      print(f'method name {method_name} did not match any existing methods.')
      