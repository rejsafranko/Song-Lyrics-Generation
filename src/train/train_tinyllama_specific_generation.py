import os
import torch
import re
import pandas as pd
import re
import wandb
from peft import get_peft_model, PeftModel, LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from datasets import Dataset
from argparse import ArgumentParser
from dotenv import load_dotenv

load_dotenv()
wandb_key = os.getenv("WANDB_API_KEY")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--artist_name")
    return parser.parse_args()


def concatenate_lyrics(df, artist_name):
    """Utility function which concatenates an artist's lyrics."""
    # Filter the DataFrame for the given artist.
    artist_df = df[df["Artist"] == artist_name]

    # Concatenate all lyrics into a single string.
    all_lyrics = "\n".join(artist_df["Lyric"])

    return all_lyrics


def special_symbols_resolver(s):
    """Utility function which removes special symbols or, if possible,
    standardizes them to common used symbol."""
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


def preprocess_dataset(dataset_path, artist_name):
    """Utility function which loads the CSV, applies text data transformations and prepares the data split."""
    df = pd.read_csv(dataset_path)
    lyrics = concatenate_lyrics(df, artist_name)

    pattern = r"\[.*?\]"
    lyrics = re.sub(pattern, "", lyrics)

    pattern = r"\((chorus|CHORUS|verse|VERSE|intro|INTRO)(.*?)\)"
    lyrics = re.sub(pattern, "", lyrics)

    lyrics = special_symbols_resolver(lyrics)

    replace_with_space = ["\u2005", "\u200b", "\u205f", "\xa0", "-"]
    replace_letters = {}
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

    cleaned_lyrics = lyrics

    for string in remove_list:
        cleaned_lyrics = re.sub(string, "", cleaned_lyrics)
    for string in replace_with_space:
        cleaned_lyrics = re.sub(string, " ", cleaned_lyrics)

    split_point = int(len(cleaned_lyrics) * 0.95)
    train_data = cleaned_lyrics[:split_point]
    test_data = cleaned_lyrics[split_point:]
    train_data_seg = []
    for i in range(0, len(train_data), 500):
        text = train_data[i : min(i + 500, len(train_data))]
        train_data_seg.append(text)
    train_data_seg = Dataset.from_dict({"text": train_data_seg})

    dataset = dict()
    dataset["train"] = train_data_seg
    dataset["test"] = test_data

    return dataset


def main(args):
    # Prepare the dataset.
    dataset = preprocess_dataset("data/processed/dataset.csv", args.artist_name)

    # Load the model and set up the configuration.
    model_name = "PY007/TinyLlama-1.1B-step-50K-105b"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    lora_alpha = 32
    lora_dropout = 0.05
    lora_rank = 32

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_rank,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, peft_config)

    wandb.login(key=wandb_key)
    run = wandb.init(
        project="Fine-tuning TinyLLama for artist lyric generation",
        job_type="training",
        anonymous="allow",
    )

    training_arguments = TrainingArguments(
        output_dir="models/tinyllama",
        per_device_train_batch_size=3,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        save_steps="steps",
        logging_steps=10,
        learning_rate=2e-3,
        max_grad_norm=0.3,
        max_steps=200,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        max_seq_length=500,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
    )
    peft_model.config.use_cache = False

    trainer.train()
    wandb.finish()

    # Merge the low-rank adapter with the base model and save it.
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, return_dict=True, torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base_model, "models/tinyllama")
    model = model.merge_and_unload()
    model.save_pretrained("models/tinyllama" + "/merged")


if __name__ == "__main__":
    args = parse_args()
    main(args)
