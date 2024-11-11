import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load all training metrics CSV files from each run's directory
dataframes = []
for file in glob.glob("logdir/dqn_agent/*/training_metrics.csv"):
    df = pd.read_csv(file)
    df["run_name"] = os.path.basename(os.path.dirname(file))  # Identify each run by its directory name
    dataframes.append(df)

# Combine all data for comparison
combined_data = pd.concat(dataframes, ignore_index=True)
print("Combined data sample:\n", combined_data.head())  # Verify data structure

# Plot average rewards and epsilon for all runs on the same plot with dual y-axes
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot average rewards on primary y-axis
sns.lineplot(data=combined_data, x="steps", y="avg_rewards", hue="run_name", ax=ax1)
ax1.set_xlabel("Training Steps")
ax1.set_ylabel("Average Reward")
ax1.tick_params(axis='y')

# Create secondary y-axis for epsilon
ax2 = ax1.twinx()
sns.lineplot(data=combined_data, x="steps", y="epsilon", hue="run_name", ax=ax2, palette="husl", linestyle="--")
ax2.set_ylabel("Epsilon")
ax2.tick_params(axis='y', labelcolor="green")

# Add legends and title
ax1.legend(loc="upper left", title="Avg Reward by Run")
ax2.legend(loc="upper right", title="Epsilon by Run")
plt.title("Average Rewards and Epsilon Decay Across Different Runs")

plt.show()
