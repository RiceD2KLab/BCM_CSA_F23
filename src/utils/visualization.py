import numpy as np

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
    axis.hist(data_s1, bins=25, alpha=1, edgecolor='black', color='steelblue', label='shhs1')
    axis.axvline(median_s1, color='red', linestyle='-', label='Median (shhs1)')
    axis.text(median_s1 * right_amt, axis.get_ylim()[1] * 0.95, f'Median (shhs1): {median_s1:.2f}', color='red', ha='left', fontsize=10)
    
    # Plotting for the second dataset (shhs2) if provided
    if data_s2 is not None:
        median_s2 = data_s2.median()
        axis.hist(data_s2, bins=25, alpha=0.7, edgecolor='black', color='lightsteelblue', label='shhs2')
        axis.axvline(median_s2, color='red', linestyle=':', label='Median (shhs2)')
        axis.text(median_s2 * right_amt, axis.get_ylim()[1] * 0.91, f'Median (shhs1): {median_s2:.2f}', color='red', ha='left', fontsize=10)

    axis.set_title(title, fontsize=20)
    axis.set_xlabel(xlabel, fontsize=18)
    axis.set_ylabel('Number of Participants', fontsize=18)  # Set Y-axis label
    axis.tick_params(axis='both', labelsize=14)  # Set tick label font size

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

    ax.set_xlabel('Events per Hour', fontsize=18)
    ax.set_ylabel('Number of Participants', fontsize=18)
    ax.set_title(title, fontsize=20)

    # Adjust tick font sizes
    ax.tick_params(axis='both', labelsize=14)

    # Annotate each bar with its count
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height - 10, str(int(height)),
                ha='center', va='bottom', fontsize=14)