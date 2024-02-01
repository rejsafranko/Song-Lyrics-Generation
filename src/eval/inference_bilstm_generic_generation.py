import numpy as np
import tensorflow as tf
import pickle
from keras.preprocessing.sequence import pad_sequences
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed_text", type=str)
    parser.add_argument("--next_words", type=int)
    return parser.parse_args()


def read_parameters(file_path: str):
    """Utility function which reads the total_words and max_seq_len parameters of the trained model."""
    with open(file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        key, value = line.split(":")
        value = value.strip()
        if key == "Total Words":
            total_words = int(value)
        elif key == "Max Sequence Length":
            max_sequence_length = int(value)

    return total_words, max_sequence_length


def complete_the_song(seed_text, next_words, tokenizer, model, max_sequence_len):
    """Utility function for generating a select number of words from the initial seed text."""
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len - 1, padding="pre"
        )

        predictions = model.predict(token_list, verbose=0)
        predicted_class_index = np.argmax(predictions)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_class_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


def main(args):
    with open("models/lstm/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    total_words, max_sequence_len = read_parameters(
        "data/processed/lstm_parameters.txt"
    )
    model = tf.keras.models.load_model("models/lstm/lstm_lyrics_generator.h5")
    output = complete_the_song(
        seed_text=args.seed_text,
        next_words=args.next_words,
        tokenizer=tokenizer,
        model=model,
        max_sequence_len=max_sequence_len,
    )
    print(output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
