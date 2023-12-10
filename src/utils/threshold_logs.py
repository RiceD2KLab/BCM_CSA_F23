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