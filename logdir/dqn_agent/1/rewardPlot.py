import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

def plot_data_from_jsonl(file_path, rbf_func='multiquadric', smooth=5000):
    # Load data from jsonl file
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    # Determine which keys to plot
    keys_to_plot = []
    for key in data[0].keys():
        if 'achievement' in key or key == 'reward':  # Check only for 'achievement' and 'reward'
            if any(obj[key] != 0 for obj in data):   # Check if not all values are zero
                keys_to_plot.append(key)

    # Plot each key that passed the check
    num_plots = len(keys_to_plot)
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))
    if num_plots == 1:
        axs = [axs]  # Make sure axs is iterable

    for ax, key in zip(axs, keys_to_plot):
        # Extract the series to be plotted
        y_values = np.array([obj[key] for obj in data])
        x_values = np.arange(len(y_values))

        # Plot raw data
        ax.plot(x_values, y_values, label=f'Raw {key}', alpha=0.6)

        # Interpolate data to smooth it
        rbf = Rbf(x_values, y_values, function=rbf_func, smooth=smooth)
        yi_smooth = rbf(x_values)

        # Plot smoothed data
        ax.plot(x_values, yi_smooth, label=f'Smoothed {key}', linewidth=2)

        # Set labels and title
        ax.set_title(f'Plot for {key}')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_data_from_jsonl("logdir/dqn_agent/1/stats.jsonl")
