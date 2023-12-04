import pandas as pd
import os
import re
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--interim_path")
    return parser.parse_args()


def format_lyric(lyric):
    # Add leading and trailing whitespace based on newline position.
    lyric = lyric.strip("\n")
    # Replace multiple \n with single \n.
    lyric = re.sub(r"(\n{2,})", "\n", lyric)
    # Wrap the single \n with a single leading and trailing whitespace.
    if "\n" in lyric:
        lyric = lyric.replace("\n", " \n ")
    # Replace \t with single whitespace.
    if "\t" in lyric:
        lyric = re.sub(r"(\t{1,})", " ", lyric)

    # Replace multiple whitespaces with a single whitespace.
    lyric = re.sub(" +", " ", lyric)

    return lyric


def split_lyrics(row, chunk_size=250):
    lyrics = row["Lyric"].split(" \n ")
    lines = [line for line in lyrics]
    for i in range(len(lines)):
        if i != len(lines):
            lines[i] = lines[i] + " \n"
    new_rows = []
    current_row = ""
    for line in lines:
        words = line.split(" ")
        for word in words:
            word_lenght = 0
            if word[-2:] != "\n":
                word_lenght = len(word)
            else:
                word_lenght = len(word) - 2
            if len(current_row.split(" ")) + word_lenght < chunk_size:
                current_row += word + " "
            else:
                new_rows.append({"Artist": row["Artist"], "Lyric": current_row[:-1]})
                current_row = word + " "
    if len(current_row) > 0:
        new_rows.append({"Artist": row["Artist"], "Lyric": current_row})

    return new_rows


def process_dataset(df):
    df["Lyric"] = df["Lyric"].apply(format_lyric)
    df["number_of_words"] = df["Lyric"].apply(lambda x: len(str(x).split()))
    # Delete useless row with no lyrics.
    df = df[df["number_of_words"] > 1]
    # Balance the number of words.
    df_balancing = df[df["number_of_words"] > 3422]
    df_balancing.reset_index(drop=True, inplace=True)
    new_rows = []
    for idx, row in df_balancing.iterrows():
        new_rows.extend(split_lyrics(row))
    new_rows = pd.DataFrame(new_rows, columns=["Artist", "Lyric"])
    # Concatenate the datasets.
    df_part_one = df[df["number_of_words"] <= 3422]
    df_part_two = new_rows

    df_part_one.drop(["number_of_words"], axis=1)
    df = pd.concat([df_part_one, df_part_two])
    df = df.sample(frac=1)
    df.reset_index(drop=True, inplace=True)
    return df


def main(args):
    df = pd.read_csv(args.interim_path)
    save_path = os.path.join("data/processed", "dataset.csv")
    processed_df = process_dataset(df)
    processed_df.to_csv(save_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
