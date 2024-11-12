import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import os
from numpy.polynomial.polynomial import Polynomial

# Function to fit polynomial regression for each run
def fit_polynomial(data, degree=3):
    polynomials = []
    for run_name, group in data.groupby('run_name'):
        x = group['steps']
        y = group['avg_rewards']
        p = Polynomial.fit(x, y, degree)
        y_fit = p(x)  # Predict values
        polynomials.append(pd.DataFrame({'steps': x, 'avg_rewards_poly': y_fit, 'run_name': run_name}))
    return pd.concat(polynomials)



# Load all training metrics CSV files from each run's directory
dataframes = []
# for file in glob.glob("logdir/dqn_agent/*/training_metrics.csv"): #All files
for file in glob.glob("logdir/dqn_agent/Epsilon Tests/*/training_metrics.csv"): #All files
    df = pd.read_csv(file)
    df["run_name"] = os.path.basename(os.path.dirname(file))  # Identify each run by its directory name
    dataframes.append(df)

# Combine all data for comparison
combined_data = pd.concat(dataframes, ignore_index=True)
# print("Combined data sample:\n", combined_data.head())  # Verify data structure

# Fit polynomial regression to avg_rewards
combined_data_poly = fit_polynomial(combined_data, degree=2)

# Set color palette for consistent coloring between avg_rewards and epsilon
palette = sns.color_palette("husl", n_colors=combined_data['run_name'].nunique())
run_names = combined_data['run_name'].unique()
palette_dict = dict(zip(run_names, palette))

# Plot average rewards and epsilon for all runs on the same plot with dual y-axes
fig, ax1 = plt.subplots(figsize=(12, 8))

# Manually store handles and labels for a combined legend
handles = []
labels = []

# Plot average rewards (with transparency) and polynomial trend on primary y-axis
for run_name, color in palette_dict.items():
    # Plot average rewards with transparency
    avg_reward_plot = sns.lineplot(
        data=combined_data[combined_data['run_name'] == run_name],
        x="steps",
        y="avg_rewards",
        ax=ax1,
        color=color,
        alpha=0.3,
        label='',
    )
    # handles.append(avg_reward_plot.lines[-1])
    # labels.append(f"{run_name} - Avg Reward")

    # Plot polynomial trend line
    poly_plot = sns.lineplot(
        data=combined_data_poly[combined_data_poly['run_name'] == run_name],
        x="steps",
        y="avg_rewards_poly",
        ax=ax1,
        color=color,
        label='',
    )
    handles.append(poly_plot.lines[-1])
    labels.append(f"{run_name} Avg Reward")

# Add a solid flat line at 1.35 labeled "Random"
ax1.axhline(y=1.35, color="gray", linestyle="-", label="Random")  # Solid line

# Position the label on the left side
ax1.text(combined_data['steps'].min()+20000, 1.4, "Random", color="gray", 
         va="center", ha="right", fontsize=10)

ax1.set_xlabel("Training Steps")
ax1.set_ylabel("Average Reward")
ax1.tick_params(axis='y')

# Create secondary y-axis for epsilon
ax2 = ax1.twinx()
for run_name, color in palette_dict.items():
    epsilon_plot = sns.lineplot(
        data=combined_data[combined_data['run_name'] == run_name],
        x="steps",
        y="epsilon",
        ax=ax2,
        color=color,
        linestyle="--",
        label='',
    )
    handles.append(epsilon_plot.lines[-1])
    labels.append(f"{run_name} Epsilon")

ax2.set_ylabel("Epsilon")
ax2.tick_params(axis='y', labelcolor="green")

# Add a single combined legend to the plot
fig.legend(handles=handles, labels=labels, title="Legend", loc="center left", bbox_to_anchor=(0.7, 0.25))


# Title
plt.title("Average Rewards with High and Low Epsilon Decay")

plt.show()

# Separate plot for loss
fig, ax = plt.subplots(figsize=(12, 8))

# Plot loss for each run
for run_name, color in palette_dict.items():
    sns.lineplot(
        data=combined_data[combined_data['run_name'] == run_name],
        x="steps",
        y="loss",  # Assuming 'loss' column is present in your data
        ax=ax,
        color=color,
        label=f"{run_name} Loss"
    )

# Customize plot
ax.set_xlabel("Training Steps")
ax.set_ylabel("Loss")
plt.title("Loss Across Different Runs")
plt.legend(title="Run Name")
plt.show()