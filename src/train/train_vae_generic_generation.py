import os
import pandas as pd
import wandb
import pickle
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from dotenv import load_dotenv

load_dotenv()
wandb_key = os.getenv("WANDB_API_KEY")


def preprocess_data(dataset_path: str):
    """Utility function to read the CSV, tokenize, padd and scale the input data a return a dataset dictionary."""
    df = pd.read_csv(dataset_path)

    # Extract lyrics from the dataset.
    lyrics = df["Lyric"].tolist()

    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df["Lyric"].astype(str).str.lower())
    total_words = len(tokenizer.word_index) + 1
    tokenized_sentences = tokenizer.texts_to_sequences(df["Lyric"].astype(str))

    tokenizer_save_path = os.path.join("models/vae/", "tokenizer.pkl")
    with open(tokenizer_save_path, "wb") as f:
        pickle.dump(tokenizer, f)

    maxlen = 200  # Set your desired maximum sequence length.

    # Pad or truncate sequences.
    padded_sequences = pad_sequences(tokenized_sentences, maxlen=maxlen, padding="pre")

    # Split the data into training and testing sets.
    X_train, X_val = train_test_split(padded_sequences, test_size=0.2, random_state=42)

    # Standardize the input data.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    dataset = dict()
    dataset["train"] = X_train
    dataset["valid"] = X_val

    return dataset, maxlen, tokenizer


class CustomVariationalLayer(Layer):
    def vae_loss(self, x, x_decoded_mean, z_mean, z_log_var, maxlen):
        xent_loss = maxlen * binary_crossentropy(x, x_decoded_mean)
        kl_loss = -0.5 * K.sum(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1
        )
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        z_mean = inputs[2]
        z_log_var = inputs[3]
        loss = self.vae_loss(x, x_decoded_mean, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        return x


def build_vae_model(maxlen, tokenizer):
    """Utility function to build and configure the variational autoencoder model."""

    def sampling(args):
        """Utility function which applies the reparameterization trick."""
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim), mean=0.0, stddev=epsilon_std)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    # Define VAE parameters.
    latent_dim = 32
    intermediate_dim = 256
    epsilon_std = 1.0
    embedding_dim = 100
    num_latent_vars = 3

    # Encoder.
    inputs = Input(shape=(maxlen,))
    embedding_layer = Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=embedding_dim,
        input_length=maxlen,
    )(inputs)
    flatten_layer = Flatten()(embedding_layer)
    h = Dense(intermediate_dim, activation="relu")(flatten_layer)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    # Sample z using reparameterization trick.
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decoder
    decoder_h = Dense(intermediate_dim, activation="relu")
    decoder_mean = Dense(maxlen, activation="sigmoid")

    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # Instantiate VAE model.
    y = CustomVariationalLayer()([inputs, x_decoded_mean, z_mean, z_log_var, maxlen])
    model = Model(inputs, y)
    model.compile(optimizer=Adam(learning_rate=0.00001), loss=None)

    return model


def main():
    dataset, maxlen, tokenizer = preprocess_data("data/processed/dataset.csv")
    model = build_vae_model(maxlen, tokenizer)

    wandb.login(key=wandb_key)
    run = wandb.init(
        project="Training variational autoencoder for Lyrics Generation",
        job_type="training",
        anonymous="allow",
    )

    # Callbacks.
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    tensorboard = TensorBoard(
        log_dir="./logs", histogram_freq=0, write_graph=True, write_images=False
    )
    wandb_callback = wandb.keras.WandbCallback()

    model.fit(
        dataset["train"],
        epochs=10,
        batch_size=512,
        validation_data=(dataset["valid"], None),
        callbacks=[early_stopping, tensorboard, wandb_callback],
    )

    wandb.finish()
    model.save("models/vae/variational_autoencoder.h5")


if __name__ == "__main__":
    main()
