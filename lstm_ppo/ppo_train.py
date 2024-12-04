import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributions as dist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from ppo_model import *
from ppo_env import *


def predict_best_song_in_cluster(env, model, sequence_embedding):
    """
    Predict best song given sequence embedding from the predicted cluster

    Args:
        env (Env): environment 
        model (PPO): trained PPO model
        sequence_embedding (torch.Tensor): sequence embedding of previous songs

    Returns:
        tuple: best_song_idx (int), best_song_name (str), cluster (int)
    """
    model.eval()

    # predict cluster index & get songs in cluster
    cluster = model.get_action(sequence_embedding)
    cluster_songs = torch.where(env.song_clusters == cluster)[0]

    # similarity between sequence embedding & all songs in cluster
    best_similarity = -float('inf')
    best_song_idx = None
    for song_idx in cluster_songs:
        song_embedding = env.song_features[song_idx]
        similarity = F.cosine_similarity(sequence_embedding.unsqueeze(0), song_embedding.unsqueeze(0))
        # similarity = weighted_similarity(sequence_embedding.unsqueeze(0), song_embedding.unsqueeze(0))

        if similarity > best_similarity:
            best_similarity = similarity
            best_song_idx = song_idx

    best_song_idx = best_song_idx.item()  # int
    # best_song_idx = int(best_song_idx) # int

    # get best song 
    best_song_features = env.song_features[best_song_idx]
    best_song_name = song_features_df_og.iloc[best_song_idx]['track_name']
    print(f"Predicted cluster: {cluster}, Best Song: {best_song_name}, Similarity: {best_similarity.item()}")
    # print(f"Predicted cluster: {cluster}, Best Song: {best_song_name}, Similarity: {best_similarity}")
    return best_song_idx, best_song_name, cluster


# embedding data of different sequence sizes
dfs = [
    pd.read_csv('./data/sequence_embeddings_len3.csv'),
    pd.read_csv('./data/sequence_embeddings_len4.csv'),
    pd.read_csv('./data/sequence_embeddings_len5.csv'),
    pd.read_csv('./data/sequence_embeddings_len8.csv')
    ]

scaler = MinMaxScaler()

# get song bank from data 
file_paths = ['./data/candidates_dataset_all_genres.csv']
songs_dfs = [pd.read_csv(file) for file in file_paths]
random.shuffle(songs_dfs) # in case in any sorted order
song_features_df_og = pd.concat(songs_dfs, axis=0, ignore_index=True)

# drop non-numerical features
song_features_df = song_features_df_og.drop(columns=['track_name', 'track_id', 'artist'], errors='ignore')
song_features_df = song_features_df.dropna()
song_features_np = song_features_df.to_numpy()  # numpy array
song_features_scaled = scaler.fit_transform(song_features_np)
song_features = torch.tensor(song_features_scaled, dtype=torch.float32)
num_songs = song_features.shape[0]

# isolate instrumentalness & speechiness & create clusters based on those features
is_features_df = song_features_df.drop(columns=['popularity', 'danceability', 'energy', 'key', 'loudness', 'acousticness', 'tempo', 'liveness', 'valence', 'duration_s'], errors='ignore')
is_features_np = is_features_df.to_numpy()  # numpy array
is_features_scaled = scaler.fit_transform(is_features_np)
is_features = torch.tensor(is_features_scaled, dtype=torch.float32)
num_clusters = 20
kmeans = KMeans(n_clusters=num_clusters, random_state=6)
song_clusters = kmeans.fit_predict(is_features)
song_clusters = torch.tensor(song_clusters, dtype=torch.long)

# Training loop PPO for all sequence sizes
losses_dict = {}
df_idx = 0
for df_og in dfs:

    df_shuffled = df_og.sample(frac=1, random_state=66).reset_index(drop=True)
    train_df_og, test_df_og = train_test_split(df_shuffled, test_size=0.1, random_state=66)

    # train Set
    train_df = train_df_og.drop(columns=['track_names'])
    sequence_embeddings = train_df.values
    nan_mask = np.isnan(sequence_embeddings).any(axis=1)
    sequence_embeddings = sequence_embeddings[~nan_mask]
    sequence_embeddings_scaled = scaler.fit_transform(sequence_embeddings)
    sequence_embeddings = torch.tensor(sequence_embeddings_scaled, dtype=torch.float32)
    embedding_size = 12
    num_sequences = sequence_embeddings.shape[0]

    # test Set
    test_df = test_df_og.drop(columns=['track_names'])
    test_sequence_embeddings = test_df.values
    nan_mask = np.isnan(test_sequence_embeddings).any(axis=1)
    test_sequence_embeddings = test_sequence_embeddings[~nan_mask]
    test_sequence_embeddings_scaled = scaler.fit_transform(test_sequence_embeddings)
    test_sequence_embeddings = torch.tensor(test_sequence_embeddings_scaled, dtype=torch.float32)

    # initialize environment, PPO model & optimizer
    env = Env(sequence_embeddings, song_features, song_clusters)
    ppo_model = PPO(embedding_size=embedding_size, num_songs=num_songs, num_clusters=num_clusters, hidden_size=2048)
    optimizer = optim.Adam(ppo_model.parameters(), lr=1e-5, weight_decay=0)

    losses = []
    recent_actions = []
    num_episodes = 4500
    for episode in range(num_episodes):
        # PPO loop : 1 step (for every episode)

        # start with random sequence embedding
        state = env.reset()

        old_log_prob = None
        total_loss = 0
        
        action_probs, state_value = ppo_model(torch.tensor(state, dtype=torch.float32))  # forward pass
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()  # sample cluster
        log_prob = dist.log_prob(action)  # log prob of action
        
        # reward (don't need next_state or done)
        next_state, reward, done, _ = env.step(action.item())
        reward = torch.tensor(reward, dtype=torch.float32)
        next_value = torch.cat([state_value[1:], torch.tensor([0.0])])  # next_value is 0 for last state
        dones = torch.tensor([done], dtype=torch.float32)

        # force the PPO to learn all clusters and not overfit to a couple
        reward, recent_actions = diversify_reward(action, reward, recent_actions)
        advantage = reward - state_value

        # ppo_loss
        # convert advantage to tensor (detach from the computation graph)
        advantage = advantage.detach()

        # log probability for actions chosen
        dist_ = torch.distributions.Categorical(action_probs)
        log_prob = dist_.log_prob(action)

        # old_log_prob is available -> use to calculate ratio
        if old_log_prob is not None:
            ratio = torch.exp(log_prob - old_log_prob)
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            surrogate_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
        else:
            # else -> log_prob & advantage directly
            surrogate_loss = -(log_prob * advantage).mean()

        # value loss
        gamma = 0.9
        td_error = reward + gamma * next_value * (1 - done) - state_value
        value_loss = td_error.pow(2).mean()

        # entropy loss
        entropy_loss = -dist_.entropy().mean()

        # loss
        loss = (surrogate_loss + 0.5 * value_loss - 0.5 * entropy_loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ppo_model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        old_log_prob = log_prob
        losses.append(total_loss)
        
        # print
        if episode % 100 == 0:
            print(f'Episode {episode}/{num_episodes} - Loss: {total_loss}')
            total_loss = 0
    
    losses_dict[df_idx] = losses

    # keep track of the songs & clusters chosen
    clusters_chosen = []
    songs_chosen = []
    print(song_clusters)
    for i in range(test_sequence_embeddings.shape[0]):
        idx, name, cluster = predict_best_song_in_cluster(env, ppo_model, test_sequence_embeddings[i])
        
        if name not in songs_chosen:
            songs_chosen.append(name)

        if cluster not in clusters_chosen:
            clusters_chosen.append(cluster)

        playlist = test_df_og['track_names'].iloc[i].split("|")
        print("Full Playlist:")
        for playlist_idx, song in enumerate(playlist):
            print(f"{playlist_idx}. {song.strip()}")
        print(f"Next: {name}")

    print(f"Total Different Clusters: {len(clusters_chosen)}")
    print(f"Total Different Songs: {len(songs_chosen)}")
    
    df_idx += 1


# plot loss over episodes
ids_to_size = ["3 Sequences", "4 Sequences", "5 Sequences", "8 Sequences"]
plt.figure(figsize=(10, 6))
for dataset_id, losses in losses_dict.items():
    plt.plot(np.convolve(losses, np.ones(100)/100, mode='valid'), label=ids_to_size[dataset_id])  # plot each dataset's loss curve
plt.title("Training Loss Over 4500 Episodes")
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.ylim(bottom=0)
plt.legend()
plt.show()
