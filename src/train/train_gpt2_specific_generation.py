import os
import pandas as pd
import torch
import wandb
import re
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
wandb_key = os.getenv("WANDB_API_KEY")


class SongLyrics(Dataset):
    """The dataset class for loading the CSV data, encoding it and storing it into an array."""

    def __init__(self, df, truncate=False, gpt2_type="gpt2", max_length=1024):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.lyrics = []

        for row in df:
            self.lyrics.append(
                torch.tensor(
                    self.tokenizer.encode(f"<|{df}|>{row[:max_length]}<|endoftext|>")
                )
            )
        if truncate:
            self.lyrics = self.lyrics[:20000]
        self.lyrics_count = len(self.lyrics)

    def __len__(self):
        return self.lyrics_count

    def __getitem__(self, item):
        return self.lyrics[item]


def preprocess_lyrics(lyrics):
    """Utility function for applying lyric transformations."""
    # Removing bracketed text.
    pattern = r"\[.*?\]"
    lyrics = re.sub(pattern, "", lyrics)

    # Removing newline symbols.
    lyrics = re.sub("\n", "", lyrics)

    # Removing specific parenthesized text.
    pattern = r"\((chorus|CHORUS|verse|VERSE|intro|INTRO)(.*?)\)"
    lyrics = re.sub(pattern, "", lyrics)

    def special_symbols_resolver(s):
        """Utility function to resolve special symbols."""
        replacements = {
            "à": "a",
            "á": "a",
            "â": "a",
            "ã": "a",
            "ä": "a",
            "ç": "c",
            "ö": "o",
            "ú": "u",
            "ü": "u",
            "œ": "oe",
            "Â": "A",
            "‰": "",
            "™": "",
            "´": "",
            "·": "",
            "¦": "",
            "": "",
            "": "",
            "˜": "",
            "“": "",
            "†": "",
            "…": "",
            "′": "",
            "″": "",
            "�": "",
            "í": "i",
            "é": "e",
            "ï": "i",
            "ó": "o",
            ";": ",",
            "‘": "'",
            "’": "'",
            ":": ",",
            "е": "e",
        }
        for symbol, replacement in replacements.items():
            s = s.replace(symbol, replacement)
        return s

    # Apply the special symbols resolver.
    lyrics = special_symbols_resolver(lyrics)

    # Additional cleaning.
    replace_with_space = ["\u2005", "\u200b", "\u205f", "\xa0", "-"]
    remove_list = [
        "\)",
        "\(",
        "–",
        '"',
        "”",
        '"',
        "\[.*\]",
        ".*\|.*",
        "—",
        "(VERSE)",
        "(CHORUS ONE)",
    ]

    for string in remove_list:
        lyrics = re.sub(string, "", lyrics)
    for string in replace_with_space:
        lyrics = re.sub(string, " ", lyrics)

    return lyrics


def preprocess_dataset(dataset_path):
    """Utility function for loading the data, preprocessing the data and creating a SongLyrics dataset object."""
    df = pd.read_csv(dataset_path)
    df = df[(df["Artist"] == "alicia-keys")]
    df = df[
        df["Lyric"].apply(lambda x: len(x.split(" ")) < 450)
    ]  # Drop the songs with lyrics too long.
    df["Lyric"] = df["Lyric"].apply(preprocess_lyrics)
    # Test set.
    test_set_size = int(len(df) * 0.05)
    test_set = df.sample(n=test_set_size)
    df = df.loc[~df.index.isin(test_set.index)]
    test_set = test_set.reset_index()
    df = df.reset_index()
    # Keep last 30 words in a new column, then remove them from original column.
    test_set["TrueFinalLyric"] = test_set["Lyric"].str.split().str[-30:].apply(" ".join)
    test_set["Lyric"] = test_set["Lyric"].str.split().str[:-30].apply(" ".join)

    dataset = SongLyrics(df["Lyric"], truncate=True, gpt2_type="gpt2")

    return dataset


def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    """Utility function for handling batch size accumulation."""
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None


def train(
    dataset,
    model,
    batch_size=16,
    epochs=15,
    lr=2e-5,
    warmup_steps=200,
    output_dir="models/gpt2",
    output_prefix="wreckgar",
    save_model_on_epoch=False,
):
    wandb.login(key=wandb_key)
    run = wandb.init(
        project="Training GPT2 for Lyrics Generation",
        job_type="training",
        anonymous="allow",
    )

    device = torch.device("cuda")
    model = model.cuda()
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):
        print(f"Training epoch {epoch}")

        epoch_loss = 0  # Reset loss for each epoch

        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            epoch_loss += loss.item()  # Accumulate the loss

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None

        # Log the average loss per epoch
        wandb.log({"epoch": epoch, "loss": epoch_loss / len(train_dataloader)})

        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )

        print(f"Epoch {epoch} loss: {epoch_loss / len(train_dataloader)}")

    wandb.finish()

    return model


def main():
    dataset = preprocess_dataset("data/processed/dataset.csv")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model: GPT2LMHeadModel = train(dataset, model)
    model.save_pretrained("models/gpt2")


if __name__ == "__main__":
    main()
