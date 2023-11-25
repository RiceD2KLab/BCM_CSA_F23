import numpy as np
import pandas as pd


# Common function to plot histogram and median
def plot_histogram(axis, data_s1, title, xlabel, data_s2=None):
    """Plot a histogram with median lines for one or two datasets.

    Args:
        axis (matplotlib.axes.Axes): The axis object of the matplotlib plot.
        data_s1 (pd.Series): Data for the first dataset.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        data_s2 (pd.Series, optional): Data for the second dataset.

    Returns:
        None
    """
    # Plotting for the first dataset (shhs1)
    median_s1 = data_s1.median()
    right_amt = 1.1
    axis.hist(
        data_s1, bins=25, alpha=1, edgecolor="black", color="steelblue", label="shhs1"
    )
    axis.axvline(median_s1, color="red", linestyle="-", label="Median (shhs1)")
    axis.text(
        median_s1 * right_amt,
        axis.get_ylim()[1] * 0.95,
        f"Median (shhs1): {median_s1:.2f}",
        color="red",
        ha="left",
        fontsize=10,
    )

    # Plotting for the second dataset (shhs2) if provided
    if data_s2 is not None:
        median_s2 = data_s2.median()
        axis.hist(
            data_s2,
            bins=25,
            alpha=0.7,
            edgecolor="black",
            color="lightsteelblue",
            label="shhs2",
        )
        axis.axvline(median_s2, color="red", linestyle=":", label="Median (shhs2)")
        axis.text(
            median_s2 * right_amt,
            axis.get_ylim()[1] * 0.91,
            f"Median (shhs1): {median_s2:.2f}",
            color="red",
            ha="left",
            fontsize=10,
        )

    axis.set_title(title, fontsize=20)
    axis.set_xlabel(xlabel, fontsize=18)
    axis.set_ylabel("Number of Participants", fontsize=18)  # Set Y-axis label
    axis.tick_params(axis="both", labelsize=14)  # Set tick label font size


def plot_bar(ax, data_col, title, bins, bin_labels, colors):
    """Plots a sub-bar graph given a dataframe column and an aesthetics configuration

    Args:
        ax: subplot to draw on
        data_col (list): column of data that contains frequency data
        title (string)
        bins ([int])
        bin_labels ([string])
        colors ([string])

    Returns:
        None
    """
    counts, _ = np.histogram(data_col, bins=bins)
    bars = ax.bar(bin_labels, counts, color=colors)

    ax.set_xlabel("Events per Hour", fontsize=18)
    ax.set_ylabel("Number of Participants", fontsize=18)
    ax.set_title(title, fontsize=20)

    # Adjust tick font sizes
    ax.tick_params(axis="both", labelsize=14)

    # Annotate each bar with its count
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height - 10,
            str(int(height)),
            ha="center",
            va="bottom",
            fontsize=14,
        )


def model_mae_dataset_table(dict, table_type):
    # cheap feature name mapping
    cheap_feature = {
        "Ant": "Anthropometry",
        "Cli": "Clinical Data",
        "Dem": "Demographics",
        "Gen": "General Health",
        "Lif": "Lifestyle and Behavioral Health",
        "Med": "Medical History",
        "Tre": "Sleep Treatment",
    }

    # feature selection name mapping
    feature_selection = {
        "mutual_information_ahi_c0h4a.csv": "Mutual Information",
        "backward_selection_AIC_ahi_c0h4a.csv": "Backward Selection",
        "forward_selection_AIC_ahi_c0h4a.csv": "Forward Selection",
        "forward_selection_BIC_ahi_c0h4a.csv": "Forward Selection",
        "random_forest_ahi_c0h4a.csv": "Random Forest",
        "decision_tree_ahi_c0h4a.csv": "Decision Tree",
    }

    # Convert dictionaries to dataframes
    df = pd.DataFrame(dict).T.reset_index()[["index", "mae", "dataset"]]

    # Rename columns for clarity
    df.columns = ["Model", "MAE", "Dataset"]

    # Replace model abbreviations
    df["Model"] = df["Model"].replace(
        {
            "xgb": "XGBoost",
            "rf": "Random Forest",
            "lr": "Linear Regression",
            "lasso": "Lasso Regression",
            "ridge": "Ridge Regression",
            "dt": "Decision Tree",
        }
    )

    # Replace dataset names based on the table type
    if table_type == "cheap_feature":
        df["Dataset"] = df["Dataset"].apply(
            lambda x: ", ".join(
                [
                    cheap_feature.get(item, item)
                    for item in x.replace(".csv", "").split("_")
                ]
            )
        )
    elif table_type == "feature_selection":
        df["Dataset"] = df["Dataset"].replace(feature_selection)

    # Find the row with the minimum MAE
    min_mae_row = df["MAE"].idxmin()

    # Function to highlight the row with the minimum MAE
    def highlight_min_row(row):
        if row.name == min_mae_row:
            return ["background-color: green"] * len(row)
        return [""] * len(row)

    # Use Pandas styling to create a table visualization and highlight the row with the lowest MAE
    styled_table = df.style.apply(highlight_min_row, axis=1).set_table_styles(
        [
            {
                "selector": "th",
                "props": [("font-size", "12pt"), ("text-align", "center")],
            },
            {"selector": "td", "props": [("text-align", "center")]},
        ]
    )

    return styled_table


def model_mae_dataset_table_all(model_maes, table_type):
    # Mapping for cheap feature names
    cheap_feature = {
        "Ant": "Anthropometry",
        "Cli": "Clinical Data",
        "Dem": "Demographics",
        "Gen": "General Health",
        "Lif": "Lifestyle and Behavioral Health",
        "Med": "Medical History",
        "Tre": "Sleep Treatment",
    }

    # Mapping for feature selection names
    feature_selection = {
        "mutual_information_ahi_c0h4a.csv": "Mutual Information",
        "backward_selection_AIC_ahi_c0h4a.csv": "Backward Selection",
        "forward_selection_AIC_ahi_c0h4a.csv": "Forward Selection",
        "forward_selection_BIC_ahi_c0h4a.csv": "Forward Selection",
        "random_forest_ahi_c0h4a.csv": "Random Forest",
        "decision_tree_ahi_c0h4a.csv": "Decision Tree",
    }

    # Preparing a list to store all rows of the table
    rows = []

    # Iterating over each model and its MAEs
    for model, datasets in model_maes.items():
        for dataset, mae in datasets.items():
            # Determine dataset name based on table_type
            if table_type == "cheap_feature":
                dataset_name = ", ".join(
                    [
                        cheap_feature.get(item, item)
                        for item in dataset.replace(".csv", "").split("_")
                    ]
                )
            elif table_type == "feature_selection":
                dataset_name = feature_selection.get(dataset, dataset)

            # Add a row for each model-dataset-MAE combination
            rows.append([model, mae, dataset_name])

    # Creating DataFrame from the rows
    df = pd.DataFrame(rows, columns=["Model", "MAE", "Dataset"])

    # Replace model abbreviations
    df["Model"] = df["Model"].replace(
        {
            "xgb": "XGBoost",
            "rf": "Random Forest",
            "lr": "Linear Regression",
            "lasso": "Lasso Regression",
            "ridge": "Ridge Regression",
            "dt": "Decision Tree",
        }
    )

    # Find the row with the minimum MAE
    min_mae_row = df["MAE"].idxmin()

    # Function to highlight the row with the minimum MAE
    def highlight_min_row(row):
        if row.name == min_mae_row:
            return ["background-color: green"] * len(row)
        return [""] * len(row)

    # Use Pandas styling to create a table visualization and highlight the row with the lowest MAE
    styled_table = df.style.apply(highlight_min_row, axis=1).set_table_styles(
        [
            {
                "selector": "th",
                "props": [("font-size", "12pt"), ("text-align", "center")],
            },
            {"selector": "td", "props": [("text-align", "center")]},
        ]
    )

    return styled_table
