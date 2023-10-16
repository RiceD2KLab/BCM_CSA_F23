import hydra
from omegaconf import DictConfig
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import os

log = logging.getLogger(__name__)

root = os.getcwd()

class_mapping = {
    0: 'No Sleep Apnea',
    1: 'Mild Sleep Apnea',
    2: 'Moderate Sleep Apnea',
    3: 'Severe Sleep Apnea'
}

def data_preprocessing(data, features, target):
    """Normalize the features, drop nan, categorize the target, and split on test/train.

    Args:
        df (pd.DataFrame): input dataset
        features (list): list of features
        target (str): target column name
    """
    # As a reference, the AHI is defined as the number of apneas and hypopneas per hour of sleep.
    # AHI >=5 (all sleep apnea)
    # AHI>=15 (moderate sleep apnea)
    # AHI >= 30 (severe sleep apnea)

    data[target] = pd.cut(data[target], bins=[0, 5, 15, 30, 1000], labels=[0, 1, 2, 3])
    
    # Drop nan on rows
    data = data.dropna()

    # Normalize the features, using sklearn StandardScaler
    scaler = StandardScaler()
    scaler.fit(data[features])
    data_scaled = scaler.transform(data[features])

    scaled_data = pd.DataFrame(data_scaled, columns=features)

    # Split on test/train 80/20
    # split startified on target
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, data[target], test_size=0.2, random_state=42, stratify=data[target])
    return X_train, X_test, y_train, y_test


@hydra.main(config_path="conf", config_name="config.yaml")
def classify(cfg: DictConfig) -> None:
    """Function that takes in hydra config and runs the classification pipeline.

    Args:
        cfg (DictConfig): pass the desired dataset, model, features, and target from config.yaml 
        Example: python my_app.py --multirun dataset=shhs1 model=logistic_regression,svc target=ahi_a0h3a,ahi_a0h4
    """
    seed = 42
    dataset_path = os.path.join(root, cfg.dataset.path)
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name
    print(f"Data: {dataset_name}, Model: {model_name}")
    log.info(f"Data: {dataset_name}, Model: {model_name}, Features: {cfg.features.names}, Target: {cfg.target.name}")

    # Read data
    data = pd.read_csv(dataset_path)
    data = data[cfg.features.names + [cfg.target.name]]
    # print("Data: ", data.head(2))
    # Preprocess data
    X_train, X_test, y_train, y_test = data_preprocessing(data, cfg.features.names, cfg.target.name)    

    # Train model
    if model_name == "logistic_regression":
        log.info("Training logistic regression model")
        model = LogisticRegression(random_state=seed, **cfg.model.params)
    elif model_name == 'random_forest':
        log.info("Training random forest model")
        model = RandomForestClassifier(random_state=seed, **cfg.model.params)
    elif model_name == 'svc':
        log.info("Training svc model")
        model = SVC(random_state=seed, **cfg.model.params)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # log the distribution of the predictions vs the true labels
    # use the class mapping as a reference
    predictions_labels = pd.Series(predictions).map(class_mapping)
    y_test_labels = y_test.map(class_mapping)
    log.info("Predictions distribution: %s",  pd.Series(predictions_labels).value_counts())
    log.info("True labels distribution: %s",  y_test_labels.value_counts())

    # Evaluate model, F1 score
    f1_weighted = f1_score(y_test, predictions, average='weighted')
    log.info("F1 score weighted: %s",  f1_weighted)


if __name__ == "__main__":
    classify()