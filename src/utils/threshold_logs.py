import re
import pandas as pd
import numpy as np

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
    f1_test = None
    f1_val = None
    cm = None
    with open(log_file, 'r') as f:
        lines = f.readlines()
        # parses line by line in the file and assigns each line to their corresponding metrics
        for i, line in enumerate(lines):
            line = ' '.join(line.split('[INFO]')[1:])
            if 'F1 Test score weighted' in line:
                f1_test = float(re.findall(r'\d+\.\d+', line)[0])
            if 'F1 Validation score weighted' in line:
                f1_val = float(re.findall(r'\d+\.\d+', line)[0])
            if 'Confusion matrix' in line:
                cm = re.findall(r'\d+', line + lines[i+1])
                # print(cm)
                cm = np.array(cm).reshape(2,2).astype(int)
    return f1_test, f1_val, cm

def extract_info_folder(folder):
    """Takes in a folder of logs and extracts the threshold calculated by the model results.

    Args:
        folder: a directory containing output logs.

    Returns:
        None.
    """
    # string parsing the folder
    settings = folder.split(',')
    dataset = settings[0].split('=')[1].strip()
    model = settings[1].split('=')[1].strip()
    target = settings[2].split('=')[1].strip()
    threshold_1 = settings[3].split('=')[1].strip().split('_')[1]
    threshold_2 = settings[4].split('=')[1].strip().split('_')[1]

    # print everything in new line and left align
    print(f'Dataset: {dataset}')
    print(f'Model: {model}')
    print(f'Target: {target}')
    print(f'Threshold CAHI > {threshold_1}')
    print(f'Threshold CAHI > OAHI * 1/{threshold_2}')
