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


def generate_lyric_vae_with_seed(
    vae, tokenizer, seed_text, maxlen=200, n_words=50, temperature=1.0
):
    """
    Generate lyrics using a Variational Autoencoder (VAE) with a seed text.

    Parameters:
    - vae: The trained VAE model.
    - tokenizer: Keras Tokenizer used for text sequences.
    - seed_text: The seed text for generating the lyrics.
    - maxlen: Maximum sequence length.
    - n_words: Number of words to generate.
    - temperature: Parameter controlling the randomness of the generation.

    Returns:
    - generated_lyric: The generated lyrics.
    """

    # Tokenize the seed text
    seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
    seed_padded = pad_sequences(
        [seed_sequence], maxlen=maxlen, padding="pre", truncating="pre"
    )

    # Generate new lyrics word by word
    generated_lyric = seed_text
    for _ in range(n_words):
        # Predict the next word based on the seed text
        predicted = vae.predict(seed_padded, verbose=0)[0]

        # Apply temperature to the predicted logits
        predicted = np.log(predicted) / temperature
        exp_preds = np.exp(predicted)
        predicted_probs = exp_preds / np.sum(exp_preds)

        # Sample from the predicted distribution
        next_index = np.argmax(np.random.multinomial(1, predicted_probs, 1)) + 1

        # Convert the index back to text
        decoded_lyric_text = tokenizer.sequences_to_texts([[next_index]])[0]

        # Update the seed text for the next iteration
        seed_text += " " + decoded_lyric_text
        seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
        seed_padded = pad_sequences(
            [seed_sequence], maxlen=maxlen, padding="pre", truncating="pre"
        )

        # Update the generated lyric
        generated_lyric += " " + decoded_lyric_text

    return generated_lyric


def main(args):
    with open("models/vae/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    model = tf.keras.models.load_model("models/vae/variational_autoencoder.h5")

    output = generate_lyric_vae_with_seed(
        model, tokenizer, args.seed_text, args.next_words, temperature=0.2
    )
    print(output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
