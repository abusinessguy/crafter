import argparse
import pickle
import random
from collections import deque
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from src.crafter_wrapper import Env  # Import the custom environment wrapper


# Define the Q-network for estimating Q-values
class DQNetwork(nn.Module):
    def __init__(self, action_space):
        super(DQNetwork, self).__init__()
        # CNN layers for processing 84x84 pixel inputs with history length (stacked frames)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Fully connected layer to process CNN output before splitting
        self.fc1 = nn.Linear(64 * 7 * 7, 512)

        # Separate streams for value and advantage
        self.value_fc = nn.Linear(512, 1)  # Outputs a single scalar for V(s)
        self.advantage_fc = nn.Linear(512, action_space)  # Outputs a value for each action A(s, a)

    def forward(self, x):
        # Pass through convolutional layers with ReLU activation
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Flatten the output before feeding it into the fully connected layer
        x = x.view(x.size(0), -1)  # Shape should be (batch_size, 3136)
        x = torch.relu(self.fc1(x))

        # Compute value and advantage streams
        value = self.value_fc(x)  # Output shape: (batch_size, 1)
        advantage = self.advantage_fc(x)  # Output shape: (batch_size, action_space)

        # Combine value and advantage to get the Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values



# Replay buffer to store past experiences for training stability
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# DQN agent that interacts with the environment and learns from experiences
class DQNAgent:
    # def __init__(self, action_space, device, buffer_size=10000, batch_size=64, gamma=0.99, lr=1e-4, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    def __init__(self, action_space, device, buffer_size, batch_size, gamma, lr, epsilon_start, epsilon_end, epsilon_decay):
        self.action_space = action_space
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-network and target network
        self.q_network = DQNetwork(action_space).to(device)
        self.target_network = DQNetwork(action_space).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.update_target_network()  # Sync target network with q_network

    # Update the target network to follow the current Q-network's weights
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    # Select an action using epsilon-greedy exploration
    def select_action(self, state):
        # Ensure `state` has the correct shape by adding a batch dimension if needed
        if state.dim() == 3:  # If state is (channels, height, width)
            state = state.unsqueeze(0)  # Add batch dimension -> (1, channels, height, width)
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)  # Random action
        else:
            with torch.no_grad():
                return self.q_network(state).argmax().item()  # Best action

    # Other methods like `store_experience` and `train` would follow here
    # Store experience in the replay buffer
    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add((state, action, reward, next_state, done))

    # Sample a batch from the buffer and update the Q-network
    def train(self):
        if self.replay_buffer.size() < self.batch_size:
            return 0
        
        # Sample a batch of experiences
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Current Q-values for the selected actions
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Q-values for the next state from the target network
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute the loss and backpropagate
        loss = nn.functional.mse_loss(current_q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # # Decrease epsilon to reduce randomness over time
        # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()  # Return loss for tracking


# Evaluation function to assess agent performance
def eval(agent, env, crt_step, opt):
    episodic_returns = []
    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        episodic_returns.append(0)
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            episodic_returns[-1] += reward

    avg_return = torch.tensor(episodic_returns).mean().item()
    print(f"[{crt_step:06d}] eval results: R/ep={avg_return:.2f}, std={torch.tensor(episodic_returns).std().item():.2f}.")
    return avg_return  # Return average reward for logging


# Save evaluation statistics to file
def _save_stats(episodic_returns, crt_step, path):
    episodic_returns = torch.tensor(episodic_returns)
    avg_return = episodic_returns.mean().item()
    print(f"[{crt_step:06d}] eval results: R/ep={avg_return:.2f}, std={episodic_returns.std().item():.2f}.")
    with open(path + "/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return}, f)


# Main training loop
def main(opt):
     # Set random seeds
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)

    # Define unique log directory based on seed
    logdir = f"{opt.logdir}/gamma_{opt.gamma}_lr_{opt.lr}_epsilon_{opt.epsilon_decay}"
    Path(logdir).mkdir(parents=True, exist_ok=True)

    # Initialize environment, agent, and parameters
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {opt.device}")
    env = Env("train", opt)
    eval_env = Env("eval", opt)
    agent = DQNAgent(
        action_space=env.action_space.n,
        device=opt.device,
        buffer_size=opt.buffer_size,
        batch_size=opt.batch_size,
        gamma=opt.gamma,
        lr=opt.lr,
        epsilon_start=opt.epsilon_start,
        epsilon_end=opt.epsilon_end,
        epsilon_decay=opt.epsilon_decay
    )

    # Store hyperparameters as text for plotting
    hyperparams_text = (
        f"Batch Size: {opt.batch_size}, Buffer Size: {opt.buffer_size}, "
        f"Learning Rate: {opt.lr}, Gamma: {opt.gamma}\n"
        f"Epsilon Start: {opt.epsilon_start}, Epsilon End: {opt.epsilon_end}, "
        f"Epsilon Decay: {opt.epsilon_decay}"
    )

    # Initialize lists to store steps and rewards for plotting
    steps = []
    avg_rewards = []
    epsilon_values = []  # New list to store epsilon values
    loss_values = []
    loss = 0

    # Training loop
    step_cnt, done = 0, True
    obs = env.reset()  # Initialize observation

    while step_cnt < opt.steps:
        if done:  # Reset environment at the start of each episode
            obs = env.reset()

        # Select and execute action
        action = agent.select_action(obs)
        next_obs, reward, done, _ = env.step(action)

        # Store experience and train the agent
        agent.store_experience(obs, action, reward, next_obs, done)
        loss += agent.train()

        # Decay epsilon explicitly here, ensuring it happens once per step
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        obs = next_obs
        step_cnt += 1
        # print(f"Step {step_cnt}: Epsilon = {agent.epsilon}")

        # Periodic evaluation and target network update
        if step_cnt % opt.eval_interval == 0:
            avg_reward = eval(agent, eval_env, step_cnt, opt)
            agent.update_target_network()  # Sync target network

            # Log steps and avg_reward for plotting
            steps.append(step_cnt)
            avg_rewards.append(avg_reward)
            epsilon_values.append(agent.epsilon)
            loss_values.append(loss/opt.eval_interval)

            loss = 0

    # Save metrics to CSV in the log directory for the current run
    metrics_df = pd.DataFrame({
        "steps": steps,
        "avg_rewards": avg_rewards,
        "epsilon": epsilon_values,
        "loss": loss_values,
    })
    metrics_df.to_csv(f"{logdir}/training_metrics.csv", index=False)



# Command-line arguments
def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logdir/dqn_agent", help="Logging directory for the model")
    parser.add_argument("--steps", type=int, default=500_000, help="Total training steps")
    parser.add_argument("--history-length", type=int, default=4, help="Frames to stack")
    parser.add_argument("--eval-interval", type=int, default=10_000, help="Steps between evaluations")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Episodes per evaluation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")

    # DQN-specific hyperparameters
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--buffer-size", type=int, default=10000, help="Replay buffer size")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor for Q-learning")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Starting value of epsilon for exploration")
    parser.add_argument("--epsilon-end", type=float, default=0.01, help="Minimum value of epsilon for exploration")
    parser.add_argument("--epsilon-decay", type=float, default=0.999989, help="Decay rate of epsilon per step")

    return parser.parse_args()

if __name__ == "__main__":
    main(get_options())
