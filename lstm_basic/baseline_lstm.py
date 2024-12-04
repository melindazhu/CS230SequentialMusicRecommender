import pandas as pd
import numpy as np
import tensorflow as tf
import spotipy
import data_preparation
import matplotlib.pyplot as plt
from spotipy.oauth2 import SpotifyOAuth
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(df):
    # Drop identification columns
    data = df.drop(columns=['track_id', 'track_name'])
    data.dropna(inplace=True)

    # Normalize the data
    scaler = MinMaxScaler()
    # scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Prepare sequences of length 3
    sequence_length = 3
    X = []
    y = []

    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])  # Use a window of X previous songs
        y.append(scaled_data[i + sequence_length])    # Predict the next song's features

    X = np.array(X)
    y = np.array(y) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

    return X_train, X_test, y_train, y_test, scaler

# Based Off MSE Loss https://keras.io/api/losses/
def custom_similarity_loss(y, y_hat, last_loss_weight, overall_loss_weight):
    
    last_y = y[-1]  # Last timestep in the sequence
    last_loss = tf.reduce_mean(tf.square(last_y - y_hat), axis=-1)

    # overall_similarity = tf.reduce_sum(y * y_hat, axis=-1) # Possible Cosine Similarity for Overall 
    overall_loss = tf.reduce_mean(tf.square(y - y_hat), axis=-1)


    # overall_loss = 1 - overall_similarity
    total_loss = last_loss_weight * last_loss + overall_loss_weight * overall_loss

    return total_loss


# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
def build_basic_model(input_shape):
    model = Sequential()

    # keras.io/api/optimizers/learning_rate_schedules/cosine_decay/
    warm_up_learning_rate = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.1, decay_steps=100, alpha=0.01)
    optimizer = tf.keras.optimizers.Adam(learning_rate=warm_up_learning_rate)

    # Basic LSTM Model with 4 LSTM Layers & 4 Dropout Layers
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.5))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(input_shape[1], activation='linear'))

    model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: custom_similarity_loss(y_true, y_pred, 0.5, 0.5), metrics=[metrics.MeanSquaredError(), metrics.CosineSimilarity()])

    return model


def train_model(X_train, y_train, X_test, y_test):
    model = build_basic_model(X_train.shape[1:])

    # Train model with 5000 epochs 
    # history = model.fit(X_train, y_train, epochs=5000, batch_size=5, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)
    history = model.fit(X_train, y_train, epochs=100, batch_size=12, validation_data=(X_test, y_test), verbose=1)
    print(model)

    return model, history


def predict_next_song(model, song_features, scaler):
    # Make sure shapes are correct 
    song_features = np.expand_dims(np.array(song_features), axis=0)

    # Predict
    predicted_features = model.predict(song_features)
    print(predicted_features.shape)

    # Inverse the scaling to get back to original feature values
    predicted_features = scaler.inverse_transform(predicted_features)

    return predicted_features


# Access Spotify to retreive a song with the closest parameters
def retrieve_song(predicted_features):
    predicted_song = None
    return predicted_song


def plot_results(history):
    plt.plot(history.history['cosine_similarity'], label='Training Cosine Similarity')
    plt.plot(history.history['val_cosine_similarity'], label='Validation Cosine Similarity')
    plt.xlabel('Epochs')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.show()


def main():
    # set up client
    # sp_client = redacted

    model_input = None
    try:
        recently_played = sp_client.current_user_recently_played(limit=15)
        model_input = data_preparation(sp_client, recently_played)
    except Exception as e:
        print("Error:", e)
    print(model_input)

    # Assuming df is your DataFrame containing track metadata
    df = pd.DataFrame()
    for i in range(1,5):
        df = df.append(pd.read_csv(f'/Users/rachael/Documents/School/Masters/Fall 2024/230/data/tracks_dataset{i}.csv'))

    # Preprocess Data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Build & Train
    model, history = train_model(X_train, y_train, X_test, y_test)
    plot_results(history)

    # Predict Song Features
    song_features = X_test[-1]
    # predicted_features = predict_next_song(model, song_features, scaler)
    predicted_features = predict_next_song(model, model_input, scaler)

    print("Predicted Features for Next Song:", predicted_features)


if __name__ == '__main__':
    main()
