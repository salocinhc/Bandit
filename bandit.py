import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data
data = pd.read_csv('metrics.csv', delimiter=';')

# Normalize each metric (min-max scaling example)
normalized_data = (data - data.min()) / (data.max() - data.min())

# Initialize parameters
n_arms = 3
n_metrics_per_arm = 3
n_rounds = len(data)  # Number of rows in the dataset
alpha = 0.1  # Learning rate for Q-value update
learning_rate_weights = 0.05  # Learning rate for weight optimization
epsilon = 0.1  # Exploration rate for epsilon-greedy
c = 2  # Exploration constant for UCB
selection_method = "epsilon_greedy"  # Choose either "epsilon_greedy" or "UCB"
base_output_folder = 'output_results/'  # Base folder for all outputs

# Function to ensure the output folder exists
def ensure_output_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to save results to CSV files
def save_results(output_folder, reward_history, Q_values_history, count_matrix, Q_values, avg_reward_history, weights_matrix):
    ensure_output_directory(output_folder)
    
    # Save reward history
    reward_df = pd.DataFrame(reward_history, columns=['Reward'])
    reward_df.to_csv(output_folder + 'reward_history.csv', index=False)

    # Save Q-values history
    q_values_df = pd.DataFrame(Q_values_history, columns=[f'Q_Arm_{i+1}' for i in range(n_arms)])
    q_values_df.to_csv(output_folder + 'Q_values_history.csv', index=False)

    # Save arm selection counts
    counts_df = pd.DataFrame(count_matrix, columns=['Selection Count'], index=[f'Arm_{i+1}' for i in range(n_arms)])
    counts_df.to_csv(output_folder + 'arm_selection_counts.csv')

    # Save final Q-values
    final_q_values_df = pd.DataFrame(Q_values, columns=['Final Q-value'], index=[f'Arm_{i+1}' for i in range(n_arms)])
    final_q_values_df.to_csv(output_folder + 'final_Q_values.csv')

    # Save average reward over time
    avg_reward_df = pd.DataFrame(avg_reward_history, columns=['Average Reward'])
    avg_reward_df.to_csv(output_folder + 'average_reward_history.csv', index=False)

    # Save weights matrix
    weights_df = pd.DataFrame(weights_matrix, columns=[f'Metric_{i+1}' for i in range(n_metrics_per_arm)], index=[f'Arm_{i+1}' for i in range(n_arms)])
    weights_df.to_csv(output_folder + 'final_weights_matrix.csv')

# Function to plot the results and save plots to the output folder
def plot_results(reward_history, avg_reward_history, count_matrix, Q_values_history, output_folder):
    ensure_output_directory(output_folder)
    
    plt.figure(figsize=(15, 5))

    # Plot rewards over time
    plt.subplot(1, 3, 1)
    plt.plot(reward_history, label='Reward')
    plt.plot(avg_reward_history, label='Average Reward', linestyle='--')
    plt.xlabel('Round')
    plt.ylabel('Reward')
    plt.title('Reward Over Time')
    plt.legend()
    plt.savefig(output_folder + 'reward_plot.png')

    # Plot selection counts for each arm
    plt.subplot(1, 3, 2)
    plt.bar(range(n_arms), count_matrix, color=['blue', 'green', 'red'])
    plt.xticks(range(n_arms), [f'Arm {i+1}' for i in range(n_arms)])
    plt.ylabel('Number of Selections')
    plt.title('Arm Selection Counts')
    plt.savefig(output_folder + 'selection_counts_plot.png')

    # Plot Q-values for each arm over time
    plt.subplot(1, 3, 3)
    for arm in range(n_arms):
        plt.plot(Q_values_history[:, arm], label=f'Arm {arm+1}')
    plt.xlabel('Round')
    plt.ylabel('Q-value')
    plt.title('Q-values Over Time')
    plt.legend()
    plt.savefig(output_folder + 'Q_values_plot.png')

    # Show the plots
    plt.tight_layout()
    plt.show()

# Function to run the bandit algorithm for a single seed
def run_bandit_experiment(seed, output_folder, epsilon):
    np.random.seed(seed)
    
    # Initialize Q-values, rewards, and count matrix
    Q_values = np.zeros(n_arms)
    count_matrix = np.zeros(n_arms)  # Track the number of times each arm is selected

    # Initialize a weight matrix (one set of weights for each arm)
    weights_matrix = np.ones((n_arms, n_metrics_per_arm)) / n_metrics_per_arm

    # To track rewards over time
    reward_history = []
    avg_reward_history = []
    Q_values_history = np.zeros((n_rounds, n_arms))  # To store Q-values over time

    # Initialize epsilon for epsilon-greedy strategy
    epsilon_value = epsilon  # Use a local epsilon value that can be decayed

    # Loop over each round
    for i in range(n_rounds):
        metrics_row = normalized_data.iloc[i]
        
        # Reshape the row into 3 arms and 3 metrics
        metrics_arms = metrics_row.values.reshape(n_arms, n_metrics_per_arm)
        
        # Choose an arm based on the selected method
        if selection_method == "epsilon_greedy":
            chosen_arm = choose_arm_epsilon_greedy(epsilon_value, Q_values)
            epsilon = max(0.01, epsilon_value * 0.99)  # Optional: decay epsilon over time
        elif selection_method == "UCB":
            chosen_arm = choose_arm_ucb(Q_values, count_matrix, i, c)
        
        # Get the metrics for the chosen arm
        chosen_arm_metrics = metrics_arms[chosen_arm]
        
        # Compute the reward for the chosen arm based on its current weights
        reward = compute_reward(chosen_arm_metrics, weights_matrix[chosen_arm])
        reward_history.append(reward)
        
        # Update the Q-value for the chosen arm
        Q_values[chosen_arm] = Q_values[chosen_arm] + alpha * (reward - Q_values[chosen_arm])
        
        # Track how many times each arm has been selected
        count_matrix[chosen_arm] += 1
        
        # Store Q-values over time
        Q_values_history[i] = Q_values.copy()
        
        # Target reward can be the current best Q-value
        target_reward = np.max(Q_values)
        
        # Optimize the weights for the chosen arm based on the observed reward
        weights_matrix[chosen_arm] = update_weights(weights_matrix[chosen_arm], 
                                                    chosen_arm_metrics, 
                                                    reward, 
                                                    target_reward, 
                                                    learning_rate_weights)
        
        # Update average reward
        avg_reward = np.mean(reward_history)
        avg_reward_history.append(avg_reward)

    # After running the algorithm, save the results and plot
    save_results(output_folder, reward_history, Q_values_history, count_matrix, Q_values, avg_reward_history, weights_matrix)
    plot_results(reward_history, avg_reward_history, count_matrix, Q_values_history, output_folder)

    # Print the final results
    print(f"Seed {seed}: Final Q-values:", Q_values)
    print(f"Seed {seed}: Count matrix (times each arm was chosen):", count_matrix)
    print(f"Seed {seed}: Weights matrix (optimized weights for each arm):", weights_matrix)
    print(f"Seed {seed}: Average reward:", np.mean(reward_history))

# Function to automate multiple seed runs
def run_multiple_seeds(seeds, base_output_folder, epsilon):
    for seed in seeds:
        seed_output_folder = base_output_folder + f'seed_{seed}/'
        print(f"Running experiment for seed {seed}...")
        run_bandit_experiment(seed, seed_output_folder, epsilon)

# Function to choose an arm using epsilon-greedy
def choose_arm_epsilon_greedy(epsilon_value, Q_values):
    if np.random.rand() < epsilon:
        return np.random.randint(n_arms)  # Explore
    else:
        return np.argmax(Q_values)  # Exploit

# Function to choose an arm using UCB
def choose_arm_ucb(Q_values, count_matrix, current_round, c):
    ucb_values = np.zeros(n_arms)
    for arm in range(n_arms):
        if count_matrix[arm] > 0:
            ucb_values[arm] = Q_values[arm] + c * np.sqrt(np.log(current_round + 1) / count_matrix[arm])
        else:
            ucb_values[arm] = float('inf')  # Select arms that have not been tried yet
    return np.argmax(ucb_values)

# Function to compute reward (custom for each metric)
def compute_reward(metrics_row, weights):
    # For metric 1 and 2: lower is better (1 - value)
    # For metric 3: higher is better (value)
    custom_metrics = np.array([1 - metrics_row[0], 1 - metrics_row[1], metrics_row[2]])
    
    # Compute the weighted reward based on the custom metric adjustments
    reward = np.dot(custom_metrics, weights)
    return reward

# Function to update weights based on observed reward (gradient ascent)
def update_weights(weights, metrics, reward, target_reward, lr):
    gradient = (target_reward - reward) * metrics
    new_weights = weights + lr * gradient
    new_weights = np.clip(new_weights, 0, None)  # Keep weights non-negative
    new_weights /= np.sum(new_weights)  # Normalize to sum to 1
    return new_weights

# Main execution
seeds = [42, 123, 999]  # Example seed list
run_multiple_seeds(seeds, base_output_folder, epsilon)
