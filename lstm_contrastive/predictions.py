"""
[MILESTONE/BASELINE ONLY] - do not use anymore
"""
import numpy as np
import pandas as pd
import torch
import pandas as pd
import matplotlib.pyplot as plt
from models.lstm_infonce import SequenceEncoder  # Replace with your actual model import
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def concatenate_files():
    csv_files = [
        "data/raw_csvs/tracks_dataset1.csv",
        "data/raw_csvs/tracks_dataset2.csv",
        "data/raw_csvs/tracks_dataset3.csv",
        "data/raw_csvs/tracks_dataset4.csv",
        "data/raw_csvs/tracks_dataset5.csv",
    ]

    dataframes = [pd.read_csv(file) for file in csv_files]
    full_dataset = pd.concat(dataframes, ignore_index=True)

    # ensure only the 12 numeric features are retained
    numeric_columns = ['popularity', 'danceability', 'energy', 'key', 'loudness',
                    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_s']
    full_dataset = full_dataset[numeric_columns]
    full_dataset.to_csv("data/raw_csvs/concatenated_with_categoricals.csv", index=False)

    print("Concatenated dataset shape:", full_dataset.shape)

model_path = "sequence_encoder.pth"
input_dim = 12
hidden_dim = 50

model = SequenceEncoder(input_dim=input_dim, hidden_dim=hidden_dim, embedding_dim=12)
model.load_state_dict(torch.load(model_path))
model.eval()

csv_file = "data/raw_csvs/concatenated.csv"
data = pd.read_csv(csv_file).values
data = data[:2613]

if np.isnan(data).any():
    nan_mask = np.isnan(data)
    col_means = np.nanmean(data, axis=0)
    data[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

# Ensure the data is reshaped into sequences of 3 songs
sequence_length = 3
feature_dim = 12

if data.shape[0] % sequence_length != 0:
    raise ValueError("The number of rows in the dataset is not a multiple of the sequence length.")

sequences = data.reshape(-1, sequence_length, feature_dim)

top_1_scores = []
top_3_scores = []

for sequence_idx, selected_sequence in enumerate(sequences):
    sequence_tensor = torch.tensor(selected_sequence, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        embedding = model(sequence_tensor).squeeze(0).numpy()

    # compute cosine similarity
    similarities = cosine_similarity(embedding.reshape(1, -1), normalized_data).flatten()

    # find top 3 most similar songs
    top_k = 3
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    top_k_scores = similarities[top_k_indices]

    top_1_scores.append(top_k_scores[0])
    top_3_scores.append(top_k_scores[2])

random_indices = np.random.choice(len(top_1_scores), size=350, replace=False)
top_1_scores = [top_1_scores[i] for i in random_indices]
top_3_scores = [top_3_scores[i] for i in random_indices]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(range(len(top_1_scores)), top_1_scores, color='blue', alpha=0.7, s=20)
plt.title("Top-1 Cosine Similarity Scores for Predicted 4th Song")
plt.xlabel("Sequence Index")
plt.ylabel("Cosine Similarity")
plt.ylim(0.2, 1.0)

plt.subplot(1, 2, 2)
plt.scatter(range(len(top_3_scores)), top_3_scores, color='green', alpha=0.7, s=20)
plt.title("Top-3 Cosine Similarity Scores for Predicted 4th Song")
plt.xlabel("Sequence Index")
plt.ylabel("Cosine Similarity")
plt.ylim(0.2, 1.0)

plt.tight_layout()
plt.savefig("heatmap.png", dpi=300)
plt.show()
