from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import pandas as pd


def subsets(nums):
    """Generete powerset from a input set

    Args:
        nums (list): set of numbers

    Returns:
        list: The power set of the input set.
    """

    rez = set()

    def rec(sub):
        if not sub:
            return
        rez.add(tuple(sub))

        for i in sub:
            new_sub = sub.copy()
            new_sub.remove(i)
            rec(new_sub)

    rec(nums)
    rez = list(rez)
    return rez


def write_ahi_to_csv(new_df, threshold, threshold_2, filepath, dataset):
    """Calculates AHI given two thresholds and writes it to a CSV file

    Args:
        new_df (pd.DataFrame): input dataset
        threshold (int): first varying threshold
        threshold_2 (int): second varying threshold
        filepath (string): prefix file path
        dataset (string): dataset file name

    Returns:
        None
    """
    new_df["CSA"] = 0
    # calculate AHI with our definition of diagnosis
    new_df["CSA"][
        (new_df["ahi_c0h4"] >= threshold)
        & (new_df["ahi_c0h4"] > new_df["ahi_o0h4"] / threshold_2)
    ] = 1
    new_df.drop(["ahi_c0h4", "ahi_o0h4", "ahi_a0h4"], axis=1, inplace=True)
    # drop na
    new_df.dropna(inplace=True)
    # save the new dataframe
    dataset_name = dataset.split(".")[0]
    new_df.to_csv(
        filepath
        + dataset_name
        + "_threshold_"
        + str(threshold)
        + "_"
        + str(threshold_2)
        + ".csv",
        index=False,
    )


def generate_subset_dataset(features_df, subset, shhs1, cahi, abbreviations, filepath):
    """Given a subset category, find the subset data and write it to a file.

    Args:
        features_df (pd.DataFrame): the input dataset
        subset (array): the input generated subset
        shhs1 (pd.Dataframe): the data
        cahi (int): cahi to predict with
        abbreviations (array): array of names of the cheap features
        filepath (str): filepath to write csv to

    Returns:
        None
    """
    cheap_features_labels = features_df[
        features_df["folder"].str.startswith(tuple(subset))
    ]["id"].values
    match_columns = shhs1.columns.intersection(cheap_features_labels)
    features = match_columns.copy()
    match_columns = match_columns.tolist()
    match_columns.append("nsrrid")
    dataset = shhs1[match_columns].copy()
    dataset = pd.merge(dataset, cahi, on="nsrrid", how="inner")

    # if a column is missing more than 50% of the values, drop it
    dataset.dropna(thresh=dataset.shape[0] * 0.5, axis=1, inplace=True)

    features = dataset.columns.tolist()

    imputer = KNNImputer(
        n_neighbors=5
    )  # You can change the number of neighbors if needed
    imputer_cat = KNNImputer(n_neighbors=1)

    # extract categorical variables those are columns with two unique values
    categorical = [col for col in features if dataset[col].nunique() == 2]
    numerical = [col for col in features if col not in categorical]

    # scale numerical data
    scaler = StandardScaler()
    dataset[numerical] = scaler.fit_transform(dataset[numerical])
    dataset[numerical] = imputer.fit_transform(dataset[numerical])
    if categorical:
        dataset = imputer_cat.fit_transform(dataset)

    # rescale the numerical data

    dataset = pd.DataFrame(dataset, columns=features)
    dataset[numerical] = scaler.inverse_transform(dataset[numerical])

    dataset_name = [abbreviations[feature] for feature in subset]
    dataset_name = "_".join(dataset_name)

    dataset.to_csv(f"{filepath}{dataset_name}.csv", index=False)