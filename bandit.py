import numpy as np
import pandas as pd

# Load the CSV input with a header
def load_metrics_from_csv(file_name='metrics.csv'):
    """Loads the metrics from the CSV file with a header. Each row is structured as:
    l1; td1; rs1; l2; td2; rs2; l3; td3; rs3
    """
    # Read CSV file (semicolon-separated values) and skip the header row
    data = pd.read_csv(file_name, delimiter=';', header=0)

    # Convert all values to float (handles string to float conversion and possible spaces)
    data = data.applymap(lambda x: float(x.strip()) if isinstance(x, str) else float(x))

    # Reshape each row from a flat list of 9 metrics into a 3x3 matrix (3 arms, 3 metrics each)
    metrics = data.values.reshape(-1, 3, 3)  # Reshape into (num_samples, 3 arms, 3 metrics)
    #print(metrics)

    return metrics[0]  # Return the first row for single execution

# Normalize the metrics using Min-Max Normalization
def normalize_metrics(metrics):
    """Normalize metrics for each arm using min-max normalization."""
    normalized_metrics = np.zeros_like(metrics)

    # Calculate the minimum and maximum for each column across all rows
    min_values = metrics.min(axis=0)  # Minimum of each column
    max_values = metrics.max(axis=0)  # Maximum of each column

    for i in range(metrics.shape[0]):  # For each row (or arm)
        # Invert and normalize metric 1 (column 0)
        normalized_metrics[i, 0] = (max_values[0] - metrics[i, 0]) / (max_values[0] - min_values[0] + 1e-9)  # Inverted normalization

        # Invert and normalize metric 2 (column 1)
        normalized_metrics[i, 1] = (max_values[1] - metrics[i, 1]) / (max_values[1] - min_values[1] + 1e-9)  # Inverted normalization

        # Normalize metric 3 (column 2) - higher is better
        normalized_metrics[i, 2] = (metrics[i, 2] - min_values[2]) / (max_values[2] - min_values[2] + 1e-9)


    # Invert metrics 1 and 2, normalize
    #for i in range(metrics.shape[0]):  # For each arm
        # Invert and normalize metric 1 and metric 2
        #print("RANGE:" , metrics.shape[0])
        #min_l = metrics[i, 0].min()
        #max_l = metrics[i, 0].max()
     #   normalized_metrics[i, 0] = (max_l - metrics[i, 0]) / (max_l - min_l + 1e-9)  # Inverted normalization
        #print("MIN_0: " , min_l)
        #print("MAX_0: ", max_l)

        #min_td = metrics[i, 1].min()
        #max_td = metrics[i, 1].max()
      #  normalized_metrics[i, 1] = (max_td - metrics[i, 1]) / (max_td - min_td + 1e-9)  # Inverted normalization
        #print("MIN_1: ", min_td)
        #print("MAX_1: " , max_td)

       
        # Normalize metric 3 (higher is better)
        #min_rs = metrics[i, 2].min()
        #max_rs = metrics[i, 2].max()
       # normalized_metrics[i, 2] = (metrics[i, 2] - min_rs) / (max_rs - min_rs + 1e-9)
        #print("MIN_2: ", min_rs)
        #print("MAX_2: " , max_rs)


    return normalized_metrics

# Normalize rewards only when necessary
def normalize_rewards(rewards):
    """Normalize rewards using min-max normalization."""
    min_reward = rewards.min()
    max_reward = rewards.max()

    print(f"Normalizing rewards: min={min_reward}, max={max_reward}, rewards={rewards}")  # Debugging line

    # If all rewards are the same, return them without normalization
    if max_reward == min_reward:
        print("All rewards are the same. Using raw rewards for update.")  # Debugging line
        return rewards  # Return the original rewards without normalization
    else:
        return (rewards - min_reward) / (max_reward - min_reward)

# Reward Function
def compute_reward(weights, metrics):
    """Compute the weighted sum of normalized metrics."""
    return np.dot(metrics, weights)

# Epsilon-Greedy for Arm Selection
def epsilon_greedy(rewards, epsilon=0.1):
    """Epsilon-greedy selection of arm."""
    if np.random.rand() < epsilon:
        # Explore: Select a random arm
        return np.random.randint(0, len(rewards))
    else:
        # Exploit: Select arm with highest reward
        return np.argmax(rewards)

# Simulate feedback to update weights
def get_reward_feedback(selected_arm, true_rewards):
    """Simulate the actual reward received from the environment."""
    return true_rewards[selected_arm]

# Update weights using Stochastic Gradient Descent (SGD)
def update_weights(weights, learning_rate, feedback_reward, predicted_reward, metrics):
    """Update the weights based on the feedback reward and predicted reward."""
    gradient = (feedback_reward - predicted_reward) * metrics  # Gradient of reward function
    weights += learning_rate * gradient  # Update weights
    return weights

# Training Loop
def train_k_armed_bandit(file_name='metrics.csv', true_rewards=None, epsilon=0.1, learning_rate=0.01, epochs=100):
    # Step 1: Load and normalize metrics
    metrics = load_metrics_from_csv(file_name)
    #normalized_metrics = normalize_metrics(metrics)
    normalized_metrics=metrics

    # Step 2: Initialize weights (e.g., start with equal weights for each metric)
    weights = np.array([0.33, 0.33, 0.33])  # Initial weights for l, td, rs

    # Step 3: Training loop
    for epoch in range(epochs):
        # Compute predicted rewards based on current weights
        predicted_rewards = compute_reward(weights, normalized_metrics)

        # Select an arm using epsilon-greedy strategy
        selected_arm = epsilon_greedy(predicted_rewards, epsilon)

        # Get actual reward feedback for the selected arm (simulated)
        feedback_reward = get_reward_feedback(selected_arm, true_rewards)

        # Normalize feedback reward only if necessary
        normalized_feedback_reward = normalize_rewards(np.array([feedback_reward]))  # Wrap in array for normalization

        # Print debug information about feedback reward
        print(f"Epoch {epoch + 1}/{epochs}: Selected Arm {selected_arm}, Feedback Reward: {feedback_reward}, Normalized Feedback Reward: {normalized_feedback_reward}")

        # Update weights based on feedback using gradient descent
        weights = update_weights(weights, learning_rate, normalized_feedback_reward[0], predicted_rewards[selected_arm], normalized_metrics[selected_arm])

        # Output progress
        print(f"Weights: {weights}")

    return weights

# Main function
def k_armed_bandit_with_training(file_name='metrics.csv', epsilon=0.1, learning_rate=0.01, epochs=100):
    # Assume true_rewards are the actual rewards we are trying to maximize (this could come from real feedback or simulation)
    true_rewards = np.array([5, 10, 15])  # Example true rewards for each arm (these should be varied)

    # Train the K-Armed Bandit and get optimized weights
    optimized_weights = train_k_armed_bandit(file_name, true_rewards, epsilon, learning_rate, epochs)

    # Use optimized weights for exploitation after training
    print("\nTraining complete. Optimized weights:", optimized_weights)

# Run the K-Armed Bandit algorithm with training
k_armed_bandit_with_training('metrics.csv', epsilon=0.1, learning_rate=0.01, epochs=100)
