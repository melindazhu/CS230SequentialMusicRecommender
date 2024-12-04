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
from PPO_Kmeans_basic_PPO import *


class Env:
    def __init__(self, sequence_embeddings, song_features, song_clusters):
        """
        Custom playlist environment for PPO
        
        Args:
            sequence_embeddings (torch.Tensor): embeddings of the sequences of songs played before selecting the next song
            song_features (torch.Tensor): features of all the songs in the song bank
            song_clusters (torch.Tensor): ids that represent which cluster a song belongs to
        """
        self.sequence_embeddings = sequence_embeddings
        self.song_features = song_features
        self.song_clusters = song_clusters # cluster id
        self.current_sequence_idx = 0

    def reset(self):
        """
        Reset the environment
        
        Returns:
            torch.Tensor: random sequence embedding
        """
        self.current_sequence_idx = np.random.randint(0, len(self.sequence_embeddings))
        return self.sequence_embeddings[self.current_sequence_idx]

    def step(self, action):
        """
        Step by selecting a cluster and receiving a reward
        
        Args:
            action (int): index of the selected cluster
        
        Returns:
            tuple: next_state, reward, done, info
        """
        cluster = action
        
        # get the songs in the chosen cluster
        cluster_songs = torch.where(self.song_clusters == cluster)[0]

        current_sequence_embedding = self.sequence_embeddings[self.current_sequence_idx]
        
        reward = calculate_reward(current_sequence_embedding, cluster_songs, self.song_features, self.song_clusters)
        
        done = True  # done after 1 step
        next_state = None  # no next state
        return next_state, reward, done, {}


def diversify_reward(action, reward, recent_actions, diversity_penalty=0.5, N=20):
    """
    Punish the reward if there are repeated clusters chosen in order to increase diversity

    Args:
        action (int): current action (cluster index)
        reward (float): initial reward calculated prior
        recent_actions (list): most recent actions
        diversity_penalty (float): penalty applied when action has been chosen recently
        N (int): max number of recent actions to consider

    Returns:
        tuple: (reward, recent_actions) - modified reward & list of recent actions
    """
    # penalty if the song has been chosen recently
    if action in recent_actions:
        reward -= diversity_penalty
    else:
        recent_actions.append(action)
        # Keep the size of the recent_actions list to N
        if len(recent_actions) > N:
            recent_actions.pop(0)

    return reward, recent_actions



def weighted_similarity(sequence_embedding, song_embedding):
    """Calculate the weighted similarity between the song and sequence embeddings based on feature importance.
    
    Args:
        sequence_embedding (torch.Tensor): The embedding of the sequence of previously played songs.
        song_embedding (torch.Tensor): The embedding of the current song.

    Returns:
        float: custom weighted similarity between the embeddings
    """
    # weights
    structure_weight_important = 0.85  # instrumentalness, speechiness
    structure_weight_other = 0.10  # tempo, key, loudness, acousticness
    emotion_weight = 0.05  # danceability, valence, energy, liveliness

    # structure features: instrumentalness (7), speechiness (8)
    sequence_structure_important = sequence_embedding[:, [7, 8]]
    song_structure_important = song_embedding[:, [7, 8]]
    
    # other structure features: tempo (11), key (4), loudness (5), acousticness (6)
    sequence_structure_other = sequence_embedding[:, [4, 5, 6, 11]]
    song_structure_other = song_embedding[:, [4, 5, 6, 11]]
    
    # emotion features: danceability (2), energy (3), valence (10), liveliness (9)
    sequence_emotion = sequence_embedding[:, [2, 3, 9, 10]]
    song_emotion = song_embedding[:, [2, 3, 9, 10]]
    
    # normalize
    sequence_structure_important = F.normalize(sequence_structure_important, p=2, dim=-1)
    sequence_structure_other = F.normalize(sequence_structure_other, p=2, dim=-1)
    sequence_emotion = F.normalize(sequence_emotion, p=2, dim=-1)
    
    song_structure_important = F.normalize(song_structure_important, p=2, dim=-1)
    song_structure_other = F.normalize(song_structure_other, p=2, dim=-1)
    song_emotion = F.normalize(song_emotion, p=2, dim=-1)
    
    # cosine similarities
    structure_similarity_important = torch.matmul(sequence_structure_important, song_structure_important.T)
    structure_similarity_other = torch.matmul(sequence_structure_other, song_structure_other.T)
    emotion_similarity = torch.matmul(sequence_emotion, song_emotion.T)
    
    # normalize
    structure_similarity_important = (structure_similarity_important + 1) / 2
    structure_similarity_other = (structure_similarity_other + 1) / 2
    emotion_similarity = (emotion_similarity + 1) / 2
    
    # sigmoid for smooth scaling
    structure_similarity_important = torch.sigmoid(structure_similarity_important)
    structure_similarity_other = torch.sigmoid(structure_similarity_other)
    emotion_similarity = torch.sigmoid(emotion_similarity)

    weighted_similarity = (structure_weight_important * structure_similarity_important +
                           structure_weight_other * structure_similarity_other +
                           emotion_weight * emotion_similarity)

    return weighted_similarity.item()


def calculate_reward(sequence_embedding, cluster_songs, song_features, all_clusters):
    """
    Calculate the reward for selecting a particular cluster based on the custom similarity
    between the sequence embedding and the selected cluster's average embedding

    Args:
        sequence_embedding (torch.Tensor): sequence embedding of previous songs
        cluster_songs (torch.Tensor): song indices in cluster
        song_features (torch.Tensor): features of songs
        all_clusters (torch.Tensor): cluster ids for all songs in bank

    Returns:
        float: reward for the selected action
    """

    # similarity between the sequence and the average of the current cluster
    cluster_avg_embedding = song_features[cluster_songs].mean(dim=0, keepdim=True)  # shape (1, embedding_dim)
    cluster_similarity = weighted_similarity(sequence_embedding.unsqueeze(0), cluster_avg_embedding)
    
    # average similarity for all other clusters
    best_similarity_other_clusters = -float('inf')
    other_cluster_similarities = []
    for cluster in all_clusters.unique():
        if cluster == cluster_songs[0]:  # Skip current
            continue
        
        # average embedding of all songs in cluster & get similarity with sequence embedding's average
        cluster_songs_idx_in_other_cluster = torch.where(all_clusters == cluster)[0]
        cluster_songs_in_other_cluster = song_features[cluster_songs_idx_in_other_cluster]
        other_cluster_avg_embedding = cluster_songs_in_other_cluster.mean(dim=0, keepdim=True)  # shape (1, embedding_dim)
        other_cluster_similarity = weighted_similarity(sequence_embedding.unsqueeze(0), other_cluster_avg_embedding)
        other_cluster_similarities.append(other_cluster_similarity)
        
    # average
    similarity_other_clusters = np.mean(other_cluster_similarities)

    reward = 0
    if cluster_similarity > similarity_other_clusters:
        reward = 1 + cluster_similarity
    else:
        reward = -1 + cluster_similarity

    return reward
