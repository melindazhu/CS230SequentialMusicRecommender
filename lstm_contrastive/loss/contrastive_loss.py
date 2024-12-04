import torch
import torch.nn.functional as F

def contrastive_loss(anchor, positive, negatives, temperature=0.07):
    """
    anchor: Anchor sequence embeddings of shape (batch_size, embedding_dim).
    positive: Positive sequence embeddings of shape (batch_size, embedding_dim).
    negatives: Negative sequence embeddings of shape (batch_size, embedding_dim).
    temperature: Temperature scaling factor for similarity scores.
    """
    pos_sim = torch.sum(anchor * positive, dim=1) / temperature
    neg_sim = torch.mm(anchor, negatives.t()) / temperature
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)
