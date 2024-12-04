import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributions as dist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_reward(sequence_embedding, selected_song_features):
    structure_weight = 0.3 # tempo, key, loudness, acousticness, instrumentalness, speechiness
    emotion_weight = 0.45 # danceability, valence, energy, liveliness
    total_similarity_weight = 0.25 # To account for not specified: duration, popularity

    structure_dim = 6   
    emotion_dim = 4    
    total_similarity_dim = 2 

    # Ensure that the embeddings are 2D
    if sequence_embedding.ndimension() == 1:
        sequence_embedding = sequence_embedding.unsqueeze(0) 
    if selected_song_features.ndimension() == 1:
        selected_song_features = selected_song_features.unsqueeze(0)

    # Extract feature groups from the embeddings based on their positions
    # Structure features: tempo (11), key (4), loudness (5), speechiness (6), acousticness (7), instrumentalness (8)
    sequence_structure = sequence_embedding[:, [4, 5, 6, 7, 8, 11]]
    song_structure = selected_song_features[:, [4, 5, 6, 7, 8, 11]]
    
    # Emotion features: danceability (2), energy (3), valence (10), liveness (9)
    sequence_emotion = sequence_embedding[:, [2, 3, 9, 10]]  # indices [2, 3, 9, 10]
    song_emotion = selected_song_features[:, [2, 3, 9, 10]]
    
    # Total similarity features
    sequence_total_similarity = sequence_embedding[:, :]
    song_total_similarity = selected_song_features[:, :]
    
    # Normalize
    sequence_structure = F.normalize(sequence_structure, p=2, dim=-1)
    sequence_emotion = F.normalize(sequence_emotion, p=2, dim=-1)
    sequence_total_similarity = F.normalize(sequence_total_similarity, p=2, dim=-1)
    song_structure = F.normalize(song_structure, p=2, dim=-1)
    song_emotion = F.normalize(song_emotion, p=2, dim=-1)
    song_total_similarity = F.normalize(song_total_similarity, p=2, dim=-1)
    
    # Compute cosine similarity
    structure_similarity = torch.matmul(sequence_structure, song_structure.T)  # Shape (batch_size, batch_size)
    emotion_similarity = torch.matmul(sequence_emotion, song_emotion.T)
    total_similarity = torch.matmul(sequence_total_similarity, song_total_similarity.T)
    
    # Normalize
    structure_similarity = (structure_similarity + 1) / 2
    emotion_similarity = (emotion_similarity + 1) / 2
    total_similarity = (total_similarity + 1) / 2

    structure_similarity = torch.sigmoid(structure_similarity)
    emotion_similarity = torch.sigmoid(emotion_similarity)
    total_similarity = torch.sigmoid(total_similarity)

    decay_factor = 0.9 ** (len(sequence_embedding) - 4)  # Assume `4` is the index of the current song in the sequence.
    reward = (structure_weight * structure_similarity + emotion_weight * emotion_similarity + total_similarity_weight * total_similarity) * decay_factor
        
    reward = reward.mean()

    diversity_penalty = 0.2 * (1 - structure_similarity - emotion_similarity)  # Simple penalty based on similarity
    reward = reward - diversity_penalty

    # print(reward.item())
    
    return reward.item()  # Return as scalar value


def compute_gae(rewards, values, next_values, dones, gamma=0.99, tau=0.95):
    deltas = rewards + gamma * next_values * (1 - dones) - values
    advantages = []
    gae = 0
    for delta in reversed(deltas.tolist()):
        gae = delta + gamma * tau * gae
        advantages.insert(0, gae)
    return torch.tensor(advantages)


class PPO(nn.Module):
    def __init__(self, embedding_size, num_songs, hidden_size=128):
        super(PPO, self).__init__()
        self.embedding_size = embedding_size
        self.num_songs = num_songs

        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_songs)  # Outputs probabilities over all songs
        )

        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single value estimate of the sequence state
        )

    def forward(self, state):
        action_probs = F.softmax(self.actor(state), dim=-1)  # Probabilities over all songs
        state_value = self.critic(state)  # Value estimate for the state
        return action_probs, state_value

    def get_action(self, state):
        # Sample an action (song) based on the action probabilities
        action_probs, _ = self(state)
        dist_ = dist.Categorical(action_probs)
        action = dist_.sample()  # Sample action from the distribution
        return action.item()

    def get_action_prob(self, state):
        # Return the action probabilities (for advantage calculation)
        action_probs, _ = self(state)
        return action_probs


class Env:
    def __init__(self, sequence_embeddings, song_features):
        self.sequence_embeddings = sequence_embeddings
        self.song_features = song_features
        self.current_sequence_idx = 0  # Start with the first sequence

    def reset(self):
        self.current_sequence_idx = np.random.randint(0, len(self.sequence_embeddings))
        return self.sequence_embeddings[self.current_sequence_idx]

    def step(self, action):

        selected_song_features = self.song_features[action]
        sequence_embedding = self.sequence_embeddings[self.current_sequence_idx]
        
        # Calculate reward
        reward = calculate_reward(sequence_embedding, selected_song_features)
        
        # End after 1 step
        done = True
        
        # Next state is None because only 1 step
        next_state = None
        
        return next_state, reward, done, {}

    def get_action_space(self):
        return self.song_features.shape[0]

    def get_state_space(self):
        return self.sequence_embeddings.shape[1]



def ppo_loss(action_probs, state_value, advantage, action, old_log_prob=None, epsilon=0.2):
    # Convert advantage to tensor (detach from the computation graph)
    advantage = advantage.detach()

    # Compute log probability for the chosen actions
    dist = torch.distributions.Categorical(action_probs)
    log_prob = dist.log_prob(action)

    # If old_log_prob is available, use it for calculating the ratio
    if old_log_prob is not None:
        ratio = torch.exp(log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        surrogate_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
    else:
        # If this is the first step, we use the log_prob and advantage directly
        surrogate_loss = -(log_prob * advantage).mean()

    # Value loss (mean squared error between predicted value and advantage)
    value_loss = (state_value - advantage).pow(2).mean()

    # Entropy loss (encourages exploration)
    entropy_loss = -dist.entropy().mean()

    # Total loss
    total_loss = surrogate_loss + 0.5 * value_loss - 0.01 * entropy_loss
    return total_loss


def predict_song(model, state, use_deterministic=False):
    """
    Predict the next song based on the current state (sequence embedding).
    
    :param model: The trained PPO model.
    :param state: The current sequence embedding (input to the model).
    :param use_deterministic: Whether to use a deterministic policy (choose the song with highest probability) or stochastic (sample from the distribution).
    :return: The predicted song index.
    """
    # Set the model to evaluation mode
    model.eval()

    # Ensure the state is a tensor
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension if needed

    # Get the action probabilities and state value from the model
    action_probs, _ = model(state_tensor)

    # If using deterministic action selection (choose the song with the highest probability)
    if use_deterministic:
        action = torch.argmax(action_probs, dim=-1)  # Select song with highest probability
    else:
        # Otherwise, sample an action from the probability distribution
        dist_ = torch.distributions.Categorical(action_probs)
        action = dist_.sample()  # Sample action (song)

    return action.item()  # Return the index of the predicted song


# Simulating 1000 sequence embeddings, each of size 12
# num_sequences = 1000
# embedding_size = 12
# sequence_embeddings = torch.randn(num_sequences, embedding_size)  # (1000, 12)

# Simulating a song bank with N songs, each of size 12
# num_songs = 500
# song_bank = torch.randn(num_songs, embedding_size)  # (500, 12)

# Retreive Embedding Data
df = pd.read_csv('./data/sequence_embeddings.csv')
df = df.drop(columns=['track_names'])
sequence_embeddings = df.values  # shape: [num_samples, embedding_size]
nan_mask = np.isnan(sequence_embeddings).any(axis=1)
sequence_embeddings = sequence_embeddings[~nan_mask]
sequence_embeddings = torch.tensor(sequence_embeddings, dtype=torch.float32)
embedding_size = 12
num_sequences = sequence_embeddings.shape[0]

# Retrieve Song Bank 
file_paths = [
    './data/tracks_dataset1.csv', 
    './data/tracks_dataset2.csv', 
    './data/tracks_dataset3.csv', 
    './data/tracks_dataset4.csv', 
    './tracks_dataset5.csv'
    ]
dfs = [pd.read_csv(file) for file in file_paths]
song_features_df_og = pd.concat(dfs, axis=0, ignore_index=True)
song_features_df = song_features_df_og.drop(columns=['track_name', 'track_id'], errors='ignore')
song_features_df = song_features_df.dropna()
song_features = torch.tensor(song_features_df.values, dtype=torch.float32)
num_songs = song_features.shape[0]

# # Initialize PPO, optimizer, and other components
# ppo_model = PPO(embedding_size=embedding_size, num_songs=num_songs)  # Example: 500 songs in the bank
# optimizer = optim.Adam(ppo_model.parameters(), lr=1e-5)

# Create the environment
env = Env(sequence_embeddings, song_features)

# Initialize PPO model and optimizer
ppo_model = PPO(embedding_size=embedding_size, num_songs=num_songs, hidden_size=128)
optimizer = optim.Adam(ppo_model.parameters(), lr=1e-5)

# Training loop PPO
losses = []
num_episodes = 3000
for episode in range(num_episodes):
    state = env.reset()  # Get the current sequence embedding
    
    # Initialize variables for PPO training
    old_log_prob = None
    total_loss = 0
    
    # PPO loop (1 step per episode in this simple environment)
    action_probs, state_value = ppo_model(torch.tensor(state, dtype=torch.float32))  # Forward pass
    dist = torch.distributions.Categorical(action_probs)
    
    action = dist.sample()  # Sample action (song)
    log_prob = dist.log_prob(action)  # Log probability of the action
    
    # Take the selected action and get the reward
    next_state, reward, done, _ = env.step(action.item())  # Perform the action
    
    # Compute advantage (in this simple case, just reward)
    # advantage = torch.tensor([reward], dtype=torch.float32)

    # GAE for the collected episode
    reward = torch.tensor(reward, dtype=torch.float32)
    next_value = torch.cat([state_value[1:], torch.tensor([0.0])])  # Next state value is 0 for the terminal state
    dones = torch.tensor([done], dtype=torch.float32)  # Done flag
    advantage = compute_gae(reward, state_value, next_value, dones, 0.99, 0.95)
    
    # Compute PPO loss and update model
    loss = ppo_loss(action_probs, state_value, advantage, action, old_log_prob=old_log_prob)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ppo_model.parameters(), max_norm=0.5)
    optimizer.step()
    
    total_loss += loss.item()
 
    # Update old_log_prob for next iteration
    old_log_prob = log_prob

    losses.append(total_loss)
    
    # Print training stats
    if episode % 100 == 0:
        print(f'Episode {episode}/{num_episodes} - Loss: {total_loss/100}')
        total_loss = 0


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


plt.plot(moving_average(losses, 50))
plt.title("Training Loss Over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.show()

predicted_song_idx = predict_song(ppo_model, sequence_embeddings[0], use_deterministic=True)
print(f"Predicted song index: {predicted_song_idx}")
print(f"Predicted Song: \n {song_features_df_og.iloc[predicted_song_idx]}")
