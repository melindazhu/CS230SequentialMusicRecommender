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
from ppo_env import *


class PPO(nn.Module):
    def __init__(self, embedding_size, num_songs, num_clusters, hidden_size=128):
        """
        Proximal Policy Optimization (PPO) model

        Args:
            embedding_size (int): sequence embedding size
            num_songs (int): number of songs in the song bank
            num_clusters (int): number of clusters created from the song bank
            hidden_size (int, optional): hidden layer size for both networks
        """
        super(PPO, self).__init__()
        self.embedding_size = embedding_size
        self.num_songs = num_songs
        self.num_clusters = num_clusters

        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_clusters)  # Outputs probabilities over all songs
        )

        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, 1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1)  # Single value estimate of the sequence state
        )

        self.apply(self.init_weights)

    def init_weights(self, module):
        """
        Custom weight initialization using He initialization
        
        Args:
            module (nn.Module): module being initialized
        """
        if isinstance(module, nn.Linear):
            # He Initialization for ReLU activations
            init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                init.zeros_(module.bias)

    def forward(self, state):
        """
        Forward pass for both networks
        
        Args:
            state (torch.Tensor): current state (sequence embedding)
        
        Returns:
            tuple: action (song) probabilities, state value
        """
        logits = self.actor(state)  # Raw output

        # Apply temperature scaling
        # 2.2 is good
        temperature = 2.21
        scaled_logits = logits / temperature
        action_probs = F.softmax(scaled_logits, dim=-1)  # Probabilities over all clusters
        state_value = self.critic(state)
        return action_probs, state_value

    def get_action(self, state):
        """
        Select a song using the actor network
        
        Args:
            state (torch.Tensor): current state (sequence embedding)
        
        Returns:
            int: index of the selected song in the song bank dataset
        """
        # Sample an action (song) based on the action probabilities
        action_probs, _ = self(state)
        dist_ = torch.distributions.Categorical(action_probs)
        cluster = dist_.sample()  # Sample action from the distribution
        return cluster.item()

    def get_action_prob(self, state):
        """
        Get action probabilities for the current state
        
        Args:
            state (torch.Tensor): current state (sequence embedding)
        
        Returns:
            torch.Tensor: action probabilities over all possible actions (songs)
        """
        action_probs, _ = self(state)
        return action_probs
      
