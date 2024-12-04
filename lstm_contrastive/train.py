"""
Train an embeddings model. Designed for HPO.
To fill in: `csv_file` to change the length of sequence used.
"""
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.lstm_infonce import SequenceEncoder
from loss.contrastive_loss import contrastive_loss
from CS230SequentialMusicRecommender.lstm_contrastive.dataset_loader import SongSequenceDataset

# HPs
input_dim = 12 # features per song
hidden_dim = 50
embedding_dim = 12 # output predicted "features"
batch_size = 32
num_epochs = 70
temperatures = [0.001]
csv_file = 'data/training_data_seqs_no_names_len8.csv'

# Initialize dataset and data loader
dataset = SongSequenceDataset(csv_file)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer
model = SequenceEncoder(input_dim=input_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_histories = {}  # for plotting losses for different HPs

# Training loop
for temp in temperatures:
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        valid_batches = 0
        for anchor_batch, positive_batch, negative_batch in dataloader:
            # Get embeddings for anchor, positive, and negative sequences
            anchor_embedding = model(anchor_batch)
            positive_embedding = model(positive_batch)
            negative_embedding = model(negative_batch)
            
            # Compute contrastive loss
            loss = contrastive_loss(anchor_embedding, positive_embedding, negative_embedding, temp)

            if torch.isnan(loss):
                print("NaN detected in loss. Skipping this batch.")
                continue
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1

        print(f"Epoch {epoch} had {valid_batches} batches.")
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")
        
        # plot avg losses for the (anchor, positive, negative) embedding training
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
        else:
            avg_loss = float('nan')
        loss_history.append(avg_loss)
    loss_histories[temp] = loss_history

torch.save(model.state_dict(), "sequence_encoder_len8.pth")
print("Training complete and model saved.")

plt.figure(figsize=(10, 6))
for temp, losses in loss_histories.items():
    epochs = range(1, len(losses) + 1)
    plt.plot(epochs, losses, linestyle='-', label=f"temp={temp}")

# Filter out NaN values for plotting
plt.title("Noise-Contrastive Estimation (InfoNCE) Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig("infoNCE_loss.png", dpi=300)
plt.show()
