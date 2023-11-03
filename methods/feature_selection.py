import numpy as np

def find_important_features(x_imputed, shhs1, model, model_name, target_var):
  model_importances = model.feature_importances_
  model_features = np.array(x_imputed.columns)[model_importances > 0]
  model_dataset = x_imputed[model_features]
  model_dataset[target_var] = shhs1[target_var]
  print(len(model_features))
  csv_name = model_name + "_" + target_var + ".csv"
  model_dataset.to_csv(csv_name, index=False)