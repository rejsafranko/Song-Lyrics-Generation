import os
import numpy as np
import pandas as pd
import wandb
import pickle
from tqdm import tqdm
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dropout, LSTM, Dense, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential

from dotenv import load_dotenv
load_dotenv()
wandb_key = os.getenv("WANDB_API_KEY")

def generate_sequences(tokenized_sentences):
    for i in tqdm(tokenized_sentences, desc="Generating sequences"):
        for t in range(1, len(i)):
            n_gram_sequence = i[: t + 1]
            yield n_gram_sequence

def preprocess_dataset(dataset_path):
    # Load processed dataset.
    df = pd.read_csv(dataset_path)

    # Tokenization.
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df["lyrics"].astype(str).str.lower())

    total_words = len(tokenizer.word_index) + 1
    tokenized_sentences = tokenizer.text_to_sequences(df["lyrics"].astype(str))

    tokenizer_save_path = os.path.join("models/lstm/", 'tokenizer.pkl')
    with open(tokenizer_save_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    # Slash sequences into n gram sequence.
    sequence_generator = generate_sequences(tokenized_sentences)

    # Find the maximum sequence length.
    max_sequence_len = max(len(seq) for seq in tqdm(sequence_generator, desc="Calculating max sequence length"))

    # Create a new generator for sequences.
    sequence_generator = generate_sequences(tokenized_sentences)

    # Pad sequences in smaller batches to save memory.
    batch_size = 1000
    padded_sequences = []

    for batch in tqdm(iter(lambda: list(sequence_generator)[:batch_size], []), desc="Padding sequences"):
        padded_sequences.extend(keras.preprocessing.sequence.pad_sequences(batch, maxlen=max_sequence_len, padding="pre"))

    input_sequences = np.array(padded_sequences)

    # Create predictors and labels.
    X, labels = input_sequences[:, :-1], input_sequences[:, -1]
    y = keras.utils.to_categorical(labels, num_classes=total_words)

    dataset = dict()
    dataset["features"] = X
    dataset["labels"] = y

    return dataset, total_words, max_sequence_len


def build_model(total_words, max_sequence_len):
    model = Sequential()
    model.add(Embedding(total_words, 40, input_length=max_sequence_len - 1))
    model.add(Bidirectional(LSTM(250)))
    model.add(Dropout(0.1))
    model.add(Dense(total_words, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def main():
    dataset, total_words, max_sequence_len = preprocess_dataset(
        "data/processed/dataset.csv"
    )

    with open("data/processed/lstm_parameters.txt", "w") as file:
        file.write(f"Total Words: {total_words}\n")
        file.write(f"Max Sequence Length: {max_sequence_len}")
        file.close()

    model = build_model(total_words, max_sequence_len)

    # EarlyStopping Callback.
    earlystop = EarlyStopping(
        monitor="loss", min_delta=0, patience=3, verbose=0, mode="auto"
    )

    # Set up wandb monitoring for the training process.
    wandb.login(key=wandb_key)
    run = wandb.init(project="Training LSTM for Lyrics Generation", job_type="training", anonymous="allow")
    wandb_callback = wandb.keras.WandbCallback()

    model.fit(
        dataset["features"],
        dataset["labels"],
        epochs=10,
        verbose=1,
        callbacks=[earlystop, wandb_callback],
    )

    wandb.finish()

    model.save("models/lstm/lstm_lyrics_generator.h5")


if __name__ == "__main__":
    main()
