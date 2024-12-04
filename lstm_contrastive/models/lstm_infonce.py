import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(SequenceEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)  # simple linear layer to get embeddings

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        # project hidden state to embedding
        embedding = self.fc(hn[-1])  # Get the last layer's hidden state
        return F.normalize(embedding, p=2, dim=1, eps=1e-6)  # L2 normalization - good for cosine sim.
