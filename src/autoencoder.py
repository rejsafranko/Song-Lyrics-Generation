import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, RepeatVector
import pandas as pd
import numpy as np


def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    # Additional cleaning steps as needed
    return text

def generate_text(seed_text, model, max_sequence_length, tokenizer, num_words_to_generate):
    generated_text = seed_text
    for _ in range(num_words_to_generate):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length, padding='post')
        predicted = model.predict(token_list, verbose=0)
        predicted_word = tokenizer.index_word[np.argmax(predicted)]
        generated_text += " " + predicted_word
    return generated_text


# Define the autoencoder model
def create_autoencoder_model(max_sequence_length, total_words):
    # Encoder
    inputs = Input(shape=(max_sequence_length,))
    encoded = Embedding(total_words, 128, input_length=max_sequence_length)(inputs)
    encoded = LSTM(128)(encoded)

    # Decoder
    decoded = RepeatVector(max_sequence_length)(encoded)
    decoded = LSTM(128, return_sequences=True)(decoded)
    decoded = Dense(total_words, activation='softmax')(decoded)

    # Create autoencoder model
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return autoencoder


def main():

    # Load the CSV file into a DataFrame
    data_path = "/Users/Lenovo/Desktop/NEUMRE/projekt/Song-Lyrics-Generation/data/processed/dataset.csv"
    data = pd.read_csv(data_path)
    data = data.sample(n=1000, random_state=42)

    corpus = data["Lyric"].astype(str)
    # Tokenize the lyrics
    tokenizer = Tokenizer()
    # Access text data from a specific column (assuming the column name is 'lyrics')
    tokenizer.fit_on_texts(corpus.str.lower())
    #corpus = clean_text(lyrics)
    total_words = len(tokenizer.word_index) + 1

    # Convert text to sequences
    input_sequences = tokenizer.texts_to_sequences(corpus)
    max_sequence_length = max(len(seq) for seq in input_sequences)
    padded_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')

    # # Save tokenizer
    # with open('models/autoencoder/' + 'tokenizer.pickle', 'wb') as handle:
    #     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    X_train, X_val = train_test_split(padded_sequences, test_size=0.1, random_state=42)

    # Create the autoencoder model
    autoencoder_model = create_autoencoder_model(max_sequence_length, total_words)

    # Train the autoencoder model
    autoencoder_model.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=10, batch_size=32)

    # Generate text
    seed_text = "blue ocean"
    generated_sequence = generate_text(seed_text, autoencoder_model, max_sequence_length, tokenizer, num_words_to_generate=50)
    print(generated_sequence)

    # Select a sequence from the dataset for testing
    test_sequence = X_train[0]  # Replace with the desired index from your dataset

    # Reconstruct the sequence
    reconstructed_sequence = autoencoder_model.predict(np.array([test_sequence]))[0]

    # Decode the reconstructed sequence back to text using the tokenizer
    decoded_sequence = ' '.join([tokenizer.index_word.get(idx, '') for idx in reconstructed_sequence])
    print("Original Sequence:", ' '.join([tokenizer.index_word.get(idx, '') for idx in test_sequence]))
    print("Reconstructed Sequence:", decoded_sequence)


if __name__ == "__main__":
    main()