import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to extract information from the log file
def extract_log(log_file):
    """Extracts the best configuration that has the lowest false negative rate from a log file.

    Sample log folder
    [2023-10-19 09:50:30,844][__main__][INFO] - Data: Decision Tree Feature Selection Dataset, Model: random_forest, Target: hf15
    [2023-10-19 09:50:30,958][__main__][INFO] - Model already trained
    [2023-10-19 09:50:31,098][__main__][INFO] - F1 Validation score weighted: 0.9960578118524658
    [2023-10-19 09:50:31,200][__main__][INFO] - F1 Test score weighted: 0.9966257589337173
    [2023-10-19 09:50:31,301][__main__][INFO] - Confusion matrix: [[1627    3]
    [   8 1622]]

    Args:
        log_file (string): file name.

    Returns:
        Metrics from the log file.
    """
    f1_test = None  # Initialize with a default value
    f1_val = None   # Initialize with a default value
    false_negative_rate = None  # Initialize with a default value
    cm = None  # Initialize with a default value

    with open(log_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if '[INFO]' in line:
                line = ' '.join(line.split('[INFO]')[1:])
                if 'F1 Test score weighted' in line:
                    f1_test = float(re.findall(r'\d+\.\d+', line)[0])
                if 'F1 Validation score weighted' in line:
                    f1_val = float(re.findall(r'\d+\.\d+', line)[0])
                if 'Confusion matrix' in line:
                    cm = re.findall(r'\d+', line + lines[i+1])
                    cm = np.array(cm).reshape(2, 2).astype(int)
                    false_negative_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1])

    return f1_test, f1_val, false_negative_rate, cm

# Function to extract information from the folder name
def extract_info_folder(folder):
    """Extract configuration information from the folder name.

    Args:
        folder (str): Folder name containing the configuration information.

    Returns:
        tuple: Contains dataset name, model name, target metric, and threshold values i and j.
    """
    settings = folder.split(',')
    dataset = settings[0].split('=')[1].strip()
    model = settings[1].split('=')[1].strip()
    target = settings[2].split('=')[1].strip()
    threshold_1 = settings[3].split('=')[1].strip().split('_')[1]
    threshold_2 = settings[4].split('=')[1].strip().split('_')[1]

    # Extract i and j values
    j_value = int(threshold_1)
    i_value = int(threshold_2)

    return dataset, model, target, j_value, i_value

# Function to create and save a heatmap as an image
def create_heatmap(data, title, filename):
    """Create and save a heatmap from given data.

    Args:
        data (DataFrame): Data to be plotted in the heatmap.
        title (str): Title of the heatmap.
        filename (str): File path where the heatmap will be saved as an image.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(data, annot=True, fmt=".5f", cmap="YlGnBu")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# With this function, we extract all the variables containing the cheapest features and ahi_c0h4 along with the target variable.
# Then we impute the data
# The function will return a dataframe with the imputed data
def read_cheap_features(path1, target):
    shhs1 = pd.read_csv('shhs1-dataset-0.20.0 (1).csv', encoding='cp1252', engine='python')
    var_dict = pd.read_csv('shhs-data-dictionary-0.20.0-variables.csv', encoding='cp1252', engine='python')
    sleep_monitoring_col = var_dict[var_dict['folder'].str.contains(r'sleep monitoring', case=False, na=False)]['id']
    x = shhs1.drop(columns=['ahi_c0h4a', 'pptidr'])
    for col in sleep_monitoring_col:
        if col in x.columns:
            x = x.drop(columns=col)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    imputer = KNNImputer(n_neighbors=5)  # You can change the number of neighbors if needed
    x_imputed_scaled = imputer.fit_transform(x_scaled)
    x_imputed = scaler.inverse_transform(x_imputed_scaled)
    x_imputed = pd.DataFrame(x_imputed, columns=x.columns)

    x_imputed['ahi_c0h4'] = shhs1['ahi_c0h4']
    df_without_target = pd.read_csv(path1)
    cheap_features = df_without_target.columns.tolist()
    cheap_features.append(target)

    df = x_imputed[cheap_features]

    return df

# In this function, we selected the rows in the dataframe.
# We first set the threshold 1-9 for the ahi_c0h4
# Under each threshold, the function will return a csv file containing the rows where ahi_c0h4 equals 1
def set_threshold(df):
    for threshold in range(1,10):
        temp = df.copy()
        temp['ahi_c0h4'] = temp['ahi_c0h4'].apply(lambda x: 0 if x < threshold else 1)
        temp['hf15'] = temp['hf15'].apply(lambda x: 1 if x == 1 else 0)
        selected_data = temp[(temp['ahi_c0h4'] == 1)]
        #output_path = '../../data/processed/threshold2/'
        output_path = 'ProcessedData4/'
        output_name = 'ahic0h4'+'_threshold_' + str(threshold) + '.csv'
        selected_data.to_csv(output_path + output_name, index=False)
    return 'completed'

def split_data(data, features, target):
    """Split on test/train/val.

    Args:
        df (pd.DataFrame): input dataset
        features (list): list of features
        target (str): target column name
    """
    # Split on test/train 80/20
    # split startified on target
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=1)
    # make validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test, X_val, y_val

def tunelr(X_train, y_train, X_val, y_val, n_iter=15):
        model = LogisticRegression(class_weight='balanced', random_state=1)
        penalty = ['l2']
        C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        hyperparameters = dict(C=C, penalty=penalty)
        clf = RandomizedSearchCV(model, hyperparameters, random_state=1, n_iter=n_iter, cv=5, verbose=0, n_jobs=-1)
        best_model = clf.fit(X_train, y_train)
        best_panel = best_model.best_estimator_.get_params()['penalty']
        best_C = best_model.best_estimator_.get_params()['C']
        best_score = best_model.best_score_
        model = best_model.best_estimator_
        y_pred = model.predict(X_val)
        f1_weighted = f1_score(y_val, y_pred, average='weighted')
        return best_score, model, f1_weighted

def tunerf(X_train, y_train, X_val, y_val, n_iter=50):
    model = RandomForestClassifier(class_weight='balanced', random_state=1)
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 200)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 100, num = 20)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

    clf = RandomizedSearchCV(model, random_grid, random_state=1, n_iter=n_iter, cv=3, verbose=0, n_jobs=-1)
            # Fit randomized search
    best_model = clf.fit(X_train, y_train)
    best_score = best_model.best_score_
    model = best_model.best_estimator_
    y_pred = model.predict(X_val)
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    return best_score, model, f1_weighted

def tunesvc(X_train, y_train, X_val, y_val, n_iter=15):    
    model = SVC(class_weight='balanced', random_state=1)
            # Create the random grid
    random_grid = {'C': [0.1, 1, 10, 100, 1000],
                        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                        'kernel': ['rbf', 'poly', 'sigmoid']}
            # Random search of parameters, using 3 fold cross validation,
            # search across 100 different combinations, and use all available cores
    clf = RandomizedSearchCV(model, random_grid, random_state=1, n_iter=n_iter, cv=3, verbose=0, n_jobs=-1)
            # Fit randomized search
    best_model = clf.fit(X_train, y_train)
            # View best hyperparameters
    best_score = best_model.best_score_
    model = best_model.best_estimator_
    y_pred = model.predict(X_val)
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    return best_score, model, f1_weighted

def tunedt(X_train, y_train, X_val, y_val, n_iter=15):
    model = DecisionTreeClassifier(class_weight='balanced', random_state=1)
            # Create the random grid
    random_grid = {'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_depth': [int(x) for x in np.linspace(10, 100, num = 10)],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]}
            # Random search of parameters, using 3 fold cross validation,
            # search across 100 different combinations, and use all available cores
    clf = RandomizedSearchCV(model, random_grid, random_state=1, n_iter=n_iter, cv=3, verbose=0, n_jobs=-1)
            # Fit randomized search
    best_model = clf.fit(X_train, y_train)
    best_score = best_model.best_score_
    model = best_model.best_estimator_
    y_pred = model.predict(X_val)
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    return best_score, model, f1_weighted

def tunexgb(X_train, y_train, X_val, y_val, n_iter=15):
    scale_pos = np.sum(y_train == 0) / np.sum(y_train == 1)
    model = XGBClassifier(scale_pos_weight=scale_pos, random_state=1)
            # Create the random grid
    random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 1000, num = 200)],
                    'max_depth': [int(x) for x in np.linspace(10, 100, num = 10)],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1]}
            # Random search of parameters, using 3 fold cross validation,
            # search across 100 different combinations, and use all available cores
    clf = RandomizedSearchCV(model, random_grid, random_state=1, n_iter=n_iter, cv=3, verbose=0, n_jobs=-1)
            # Fit randomized search
    best_model = clf.fit(X_train, y_train)
            # View best hyperparameters
    best_score = best_model.best_score_
    model = best_model.best_estimator_
    y_pred = model.predict(X_val)
    f1_weighted = f1_score(y_val, y_pred, average='weighted')
    return best_score, model, f1_weighted    

def oversample_data(data, features, target):
    oversample = SMOTE(random_state=1)
    X, y = oversample.fit_resample(data[features], data[target])
    return X, y

# path: the path of the folder containing the selected data under different threshold
def find_threshold(path):
    results = []
    best_false_negative = 100000
    min_score = 0
    min_f1 = 0
    model_lst = ['lr', 'rf', 'dt', 'xgb']
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path)
        df['hf15'] = df['hf15'].astype(int)
        features = df.columns.tolist()
        features.remove('hf15')
        target = 'hf15'
        X, y = oversample_data(df, features, 'hf15')
        data = pd.concat([X, y], axis=1)
        X_train, X_test, y_train, y_test, X_val, y_val = split_data(data, features, target)
        for i in model_lst:
            if i=='lr':
                res = tunelr(X_train, y_train, X_val, y_val, n_iter=15)
            elif i=='rf':
                res = tunerf(X_train, y_train, X_val, y_val, n_iter=15)
            elif i == 'dt':
                res = tunedt(X_train, y_train, X_val, y_val, n_iter=15)
            else:
                res = tunexgb(X_train, y_train, X_val, y_val, n_iter=15)
                
            model = res[1]
            f1_val = res[2]
                
            f1_test = f1_score(y_test, model.predict(X_test), average='weighted')
            cm = confusion_matrix(y_test, model.predict(X_test))
            false_negative = cm[0][1]
            if false_negative < best_false_negative:
                best_false_negative = false_negative
                best_dataset = file
                best_cm = cm
                best_f1_test = f1_test
                best_f1_val = f1_val
                best_model = model
            print(file, model, f1_test, f1_val, cm, false_negative)
            result = {
            'File': file,
            'Model': str(model),
            'F1 Test': f1_test,
            'F1 Validation': f1_val,
            'Confusion Matrix': str(cm),
            'False Negative': false_negative
        }
            results.append(result)

    return best_dataset, best_model, results
            
def feature_importance(dataset, model_res):
    path = 'ProcessedData4/' + dataset
    df = pd.read_csv(path)
    df['hf15'] = df['hf15'].astype(int)
    features = df.columns.tolist()
    features.remove('hf15')
    X, y = oversample_data(df, features, 'hf15')
    data = pd.concat([X, y], axis=1)
    
    X_train, X_test, y_train, y_test, X_val, y_val = split_data(data, features, 'hf15')
    model = model_res
    model.fit(X_train, y_train)
    importances = model.feature_importances_
    feature_importances = pd.DataFrame({'feature':features, 'importance': importances})
    feature_importances = feature_importances.sort_values('importance', ascending=False)
    return feature_importances

def feature_importance_SHAP(dataset, model_res):
    path = 'ProcessedData4/' + dataset
    df = pd.read_csv(path)
    df['hf15'] = df['hf15'].astype(int)
    features = df.columns.tolist()
    features.remove('hf15')
    X, y = oversample_data(df, features, 'hf15')
    data = pd.concat([X, y], axis=1)
    
    X_train, X_test, y_train, y_test, X_val, y_val = split_data(data, features, 'hf15')
    model = model_res
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model, X_train)
    shap_values = explainer(X_test, check_additivity=False)

    #Plot the summary plot
    shap.summary_plot(shap_values, X_test)