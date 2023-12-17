import numpy as np
import pandas as pd
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dropout, LSTM, Dense, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential


def prepare_dataset(dataset_path):
    # Load preprocessed dataset for LSTM.
    df = pd.read_csv(dataset_path)

    # Tokenization.
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df["lyrics"].astype(str).str.lower())

    total_words = len(tokenizer.word_index) + 1
    tokenized_sentences = tokenizer.text_to_sequences(df["lyrics"].astype(str))

    # Slash sequences into n gram sequence.
    input_sequences = list()
    for i in tokenized_sentences:
        for t in range(1, len(i)):
            n_gram_sequence = i[: t + 1]
            input_sequences.append(n_gram_sequence)

    # Pre-padding.
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(
        pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
    )

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
    dataset, total_words, max_sequence_len = prepare_dataset(
        "../data/processed/train_lstm.csv"
    )
    model = build_model(total_words, max_sequence_len)

    # EarlyStopping Callback.
    earlystop = EarlyStopping(
        monitor="loss", min_delta=0, patience=3, verbose=0, mode="auto"
    )

    model.fit(
        dataset["features"],
        dataset["labels"],
        epochs=10,
        verbose=1,
        callbacks=[earlystop],
    )

    model.save("../models/lstm_lyrics_generator.h5")


if __name__ == "__main__":
    main()
