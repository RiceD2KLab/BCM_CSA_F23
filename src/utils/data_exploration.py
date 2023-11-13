import numpy as np

def calc_confidence_interval(df, col_name):
    """Calculates the confidence interval of a data column.

    Args:
        df (pd.DataFrame): input dataset
        col_name (str): column name in df
    Returns:
        [percentage, lower_bound, upper_bound]: a list of floats representing the CI.
    """
    # Step 1: Define the specific value we want to calculate the percentage for
    specific_value = 1

    # Step 2: Calculate the percentage of the specific value in the column
    percentage = (df[col_name] == specific_value).mean() * 100

    # Step 3: Calculate the standard error and margin of error for the percentage
    n = len(df)
    standard_error = np.sqrt((percentage / 100 * (1 - percentage / 100)) / n)
    margin_of_error = 1.96 * standard_error  # 1.96 corresponds to a 95% confidence interval

    # Step 4: Calculate the confidence interval
    lower_bound = percentage - margin_of_error
    upper_bound = percentage + margin_of_error
    return [percentage, lower_bound, upper_bound]