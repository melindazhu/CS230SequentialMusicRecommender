"""
Make "predictions" i.e. generate embeddings based on the 
saved .pth file. To edit: `model_path`, `input_csv`, `output_csv`.
"""
import pandas as pd
import torch
from models.lstm_infonce import SequenceEncoder  # Import your saved model class

model_path = "sequence_encoder_len8.pth"
input_csv = "data/training_data_seqs_len8.csv"
output_csv = "sequence_embeddings_len8.csv"
input_dim = 12  # fatures per song
hidden_dim = 50
embedding_dim = 12  # output embedding size
sequence_length = 8  # number of songs in each sequence

model = SequenceEncoder(input_dim=input_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
model.load_state_dict(torch.load(model_path))
model.eval()

df = pd.read_csv(input_csv)

track_names = df["track_names"].values
features = df.drop(columns=["track_names"]).values

# reshape features into (num_sequences, sequence_length, input_dim)
num_features = features.shape[1] // sequence_length
features = features.reshape(-1, sequence_length, num_features)

embeddings = []
with torch.no_grad(): 
    for sequence in features:
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        embedding = model(sequence_tensor)
        embeddings.append(embedding.squeeze(0).numpy())


embedding_columns = [f'embedding_dim_{i+1}' for i in range(embedding_dim)]
embedding_df = pd.DataFrame(embeddings, columns=embedding_columns)
output_df = pd.concat([pd.DataFrame({"track_names": track_names}), embedding_df], axis=1)

output_df.to_csv(output_csv, index=False)
print(f"embeddings saved successfully")
