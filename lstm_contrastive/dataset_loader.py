import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class SongSequenceDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file).values
        self.sequence_length = 8
        self.feature_dim = 12
        self.sequences = self._create_sequences()
        print(f"Training data dimension: {self.sequences.shape}")

    def _create_sequences(self):
        # Organizes data into sequences of 3 songs, each with 12 features
        num_samples = len(self.data) // self.sequence_length
        return self.data[:num_samples * self.sequence_length].reshape(-1, self.sequence_length, self.feature_dim)

    def _create_positive_sequence(self, anchor_sequence):
        positive_sequence = anchor_sequence.copy()
        idx = np.random.randint(0, self.sequence_length - 1)
        positive_sequence[idx], positive_sequence[idx + 1] = positive_sequence[idx + 1], positive_sequence[idx]
        return positive_sequence

    def _create_negative_sequence(self):
        neg_idx = np.random.randint(0, len(self.sequences))
        return self.sequences[neg_idx]

    def __getitem__(self, idx):
        anchor_sequence = self.sequences[idx]
        positive_sequence = self._create_positive_sequence(anchor_sequence)
        negative_sequence = self._create_negative_sequence()

        anchor = torch.tensor(anchor_sequence, dtype=torch.float32)
        positive = torch.tensor(positive_sequence, dtype=torch.float32)
        negative = torch.tensor(negative_sequence, dtype=torch.float32)

        return anchor, positive, negative

    def __len__(self):
        return len(self.sequences)
