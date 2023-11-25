# Rice D2K COMP 449/DSCI 435/DSCI 535 Capstone Spring 2023, Team BCM Central Sleep Apnea

# TODO: UPDATE /src/notebooks/README.md when code is done

## Team Members
Students: Minyu Chen, Jingwen Hu, Liuxiao Kang, Zheran Li, Huailin Tang, Risto Trajanov, Benjamin Zhao

Faculty Mentor: Dr. Arko Barman

PHD Mentor: Kai Malcolm

Sponsor: Dr. Ritwick Agrawal

## Contents
1. [Repository Structure](#repository-structure)
2. [Installation](#installation)
3. [Running the software](#running-the-software)
4. [Description](#description)
5. [Goals](#goals)
6. [Data](#data)
7. [Data Science Pipeline](#data-science-pipeline)

## Repository structure

```nohighlight
BCM_CSA_F23/
├── data
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
│
├── models                  
│   ├── threshold           <- Trained models during finding the Threshold for Central Sleep Apnea diagnosis experiments.
│   └── cheap_features      <- Trained models during finding cheap features to predict Central Sleep Apnea Index.
│
├── references              <- Data dictionaries, manuals, and all other explanatory materials.
│
├── results                 <- Generated logs from the threshold fine tuning experiments using Hydra.
│   ├── intial_study
│   └── threshold  
│              
├── src                     
│   ├── conf                <- Config folder for storing Hydra configurations.
│   │   ├── dataset         
│   │   ├── model      
│   │   ├── features      
│   │   ├── target        
│   │   ├── threshold_cahi  <- Configuration folder stating the threshold of CSA (CSA>i).
│   │   └── threshold_c_o   <- Configuration folder stating the threshold of the fraction of OSA (CSA>1/i*OSA).
│   │
│   ├── models          
│   ├── notebooks             
│   └── utils
│
└── visualizations
    └── replication_study   <- Contains table data to the initial study replication experiment done 
```
- `data/` directory contains raw data from the Sleep Heart Health Study and our interim and processed datasets manipulating the raw dataset.
- `models/` contains python models generated for our objectives: finding cheap features and finding a new threshold for diagnosis.
- `results/` directory contains generated logs from the threshold fine tuning experiments using Hydra. They are sorted into `results/initial_study,` representing our findings from our initial feature selection, and `results/threshold,` representing our final results.
- `src/` directory contains all our source code.
    - `conf/` contains dynamic configuration files to run [Hydra](https://hydra.cc/docs/intro/) models that compares all our built models for predicting the new diagnosis threshold. The config.yaml sets the default application parameters, while other files like config_threshold.yaml provide specific settings for threshold experiments. For more read this [README.md](./src/conf/README.md)
    - `models/` contains python modules for custom model implementations.
    - `notebooks/` contain jupyter notebooks used for data exploration, preprocessing, feature selection, and result analysis.
    - `utils/` contains python modules for various routines such as preprocessing, feature selection, and metrics.
- `visualizations/` contains saved data visualizations generated from the experiments.

## Installation

The following steps explain how to set up a Python environment using Linux/Windows/MacOS + conda.

Step 1: [Install conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/download.html)

Conda is an environment manager handled within a terminal (in MacOS), or AnacondaPrompt/PowerShell (in Windows).

Step 2: Clone the repository
`$ git clone https://github.com/RiceD2KLab/BCM_CSA_F23.git`

Step 3: Create a conda Python 3.9 environment
1. `$ conda create -n csa python=3.9`
2. `$ cd BCM_CSA_F23` switch to the project repository directory
3. activate the conda environment: `$ conda activate csa`
4. update pip: `$ pip install update pip`

Step 4: Install required packages
`pip install -r requirements.txt`

Deactivate to base when done! `conda deactivate`

## Running the software

The dataset has been provided in this repository for anyone that would like replicate our results (in `/data/raw`). If you do not have stand alone hardware with a minimum of 26 GB system RAM and a dedicated GPU with a minimum of 16 GB of vRAM, we recommend using Google Colab to upload and run the jupyter notebooks.

# TODO: write more detailed instructions & update /src/README.md too.

# TODO: make a demo notebook, write instructions for Google Colab

### Initial Experiment

As of August 11, 2023, our software APIs usage and data science pipeline is documented in the Jupyter notebook demo_notebook.ipynb. This notebook is intended to demonstrate how to use our software for end-to-end training and evaulation. Note that we have placed a small subsample of data under data/demo_data for use with this notebook. The results shown in demo_notebook.ipynb are not representative of our final products and are meant for demonstration purposes, only. Within src/ we have a directory notebooks which contains individual notebooks that run our entire pipeline on the entire data set and capture our current results. For specific details about these notebooks, please see the source README.

### Generating results for Initial Experiment

`python classify.py --multirun dataset=shhs1 model=logistic_regression,svc,random_forest target=ahi_a0h3a,ahi_a0h4`

`python regression.py --multirun dataset=shhs1 model=linear_regression,ridge,lasso,knn,svr target=ahi_a0h3a,ahi_a0h4`

### Generating results for the Threshold experiments

`python3 ./src/models/find_threshold.py --multirun dataset=feature_selection_dt,feature_selection_mi,feature_selection_rf,feature_selection_bs,feature_selection_mrmr10,feature_selection_mrmr20,feature_selection_fs_AIC,feature_selection_fs_BIC model=logistic_regression,svc,random_forest,decision_tree,xgboost target=hf15 threshold_cahi=threshold_1,threshold_2,threshold_3,threshold_4,threshold_5,threshold_6,threshold_7,threshold_8,threshold_9 threshold_c_o=threshold_1,threshold_2,threshold_3,threshold_4,threshold_5`

## Description
As of March 2023, over 18 million people in the United States have been officially diagnosed
with sleep apnea, or about one in 15 people. However, it is estimated that many more
people have sleep apnea and are unaware of it [1]. 

The project outlines a comprehensive study on Central Sleep Apnea (CSA), aiming to identify cost-effective predictive features and establish a new diagnostic threshold that could potentially save lives and healthcare costs. The project emphasizes the potential of data science in healthcare, advocating for more accessible sleep testing methods, thus reducing the financial burden on patients and healthcare facilities. The success of this study could influence other healthcare domains, leveraging data science to challenge and improve existing diagnostic standards.

[1] A. M. Barbara Phillips, David Gozal, “The statistics of sleep apnea,” American Journal of Respiratory and Critical Care Medicine, vol. 192, 2015.

### Useful terms

- *Central Apnea Hypopnea Index (CAHI)*: 
Unit: Events Per Hour
Meaning: Number of times the patient stopped breathing for 10 seconds and the main cause was no signal for breathing from the brain.

- *Obstructive Apnea Hypopnea Index (OAHI)*: 
Unit: Events Per Hour
Meaning: Number of times the patient stopped breathing for 10 seconds and the main cause was that something was obstructing the breathing pathways.

- *Diagnosis Threshold*:
The number of events per night needed for a patient to be classified as diagnosed with Central Apnea or Obstructive Apnea. For example CAHI > 5

- *Cost Effective Features*:
The cost of a study to measure the events per hour is too expensive ($2K to $10k) so we are looking for medical measurements that can accurately predict the outcomes of the study (events per hour) for a given patent.

## Goals
1. Predict central sleep apnea using a less invasive and less expensive procedure
2.  Explore which of the variables contribute most to the mortality
3.  Determine if a small central apnea-hypopnea index leads to increased mortality risk, and if so, lower the threshold of diagnosis of CSA to more appropriately capture this risk.

## Data

For this project we use data from [The Sleep Heart Health Study (SHHS)](https://sleepdata.org/datasets/shhs). The SHHS
dataset is a comprehensive multi-center cohort study, initiated and overseen by the National
Heart, Lung, and Blood Institute. The primary objective of the SHHS study is to explore the
cardiovascular and broader health implications of sleep-disordered breathing. Specifically,
it seeks to establish whether there exists a link between sleep-related breathing issues and
an elevated risk of coronary heart disease, stroke, overall mortality, and hypertension.

The main datasets being used in this project as follows:

- shhs1 (Visit 1): This dataset captures data for initial clinic visit and polysomnogram
conducted between the years 1995 and 1998.
- shhs2 (Visit 2): This dataset captures data from the follow-up clinic visit and
polysomnogram which took place from 2001 to 2003.

Data citation:
Zhang GQ, Cui L, Mueller R, Tao S, Kim M, Rueschman M, Mariani S, Mobley D, Redline S. The National Sleep Research Resource: towards a sleep data commons. J Am Med Inform Assoc. 2018 Oct 1;25(10):1351-1358. doi: 10.1093/jamia/ocy064. PMID: 29860441; PMCID: PMC6188513.

Quan SF, Howard BV, Iber C, Kiley JP, Nieto FJ, O'Connor GT, Rapoport DM, Redline S, Robbins J, Samet JM, Wahl PW. The Sleep Heart Health Study: design, rationale, and methods. Sleep. 1997 Dec;20(12):1077-85. PMID: 9493915.

The Sleep Heart Health Study (SHHS) was supported by National Heart, Lung, and Blood Institute cooperative agreements U01HL53916 (University of California, Davis), U01HL53931 (New York University), U01HL53934 (University of Minnesota), U01HL53937 and U01HL64360 (Johns Hopkins University), U01HL53938 (University of Arizona), U01HL53940 (University of Washington), U01HL53941 (Boston University), and U01HL63463 (Case Western Reserve University). The National Sleep Research Resource was supported by the National Heart, Lung, and Blood Institute (R24 HL114473, 75N92019R002).

## Data Science Pipeline

![Data Science Pipeline](https://github.com/RiceD2KLab/BCM_CSA_F23/assets/22122979/5f67c314-9f80-44ae-a03f-de9b805a1a07)

1. We begin with raw data, where the initial step involves visualizing the unprocessed information to gain an initial understanding of the dataset.
2. Subsequently, a series of essential data preprocessing steps are conducted, including data
normalization, handling of missing values, and feature selection.
3. After these preparatory stages, the dataset is once again visualized, enabling a comparison of the data’s evolution and providing insights into how these preprocessing steps have influenced the data’s distribution and structure.
4. The pipeline then delves into more specific analyses, focusing on Apnea-Hypopnea Index (AHI), Cost-Effective Feature Analysis, and Diagnosis Threshold Analysis, potentially involving advanced statistical and machine learning techniques to extract meaningful information and critical insights from the data.

This comprehensive approach helps streamline the data science process, making it more effective and efficient in uncovering valuable insights and patterns within the dataset.

