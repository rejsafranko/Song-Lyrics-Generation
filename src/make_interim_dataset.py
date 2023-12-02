import pandas as pd
import os
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--csv_file_path")
    parser.add_argument("--txt_dir_path")
    return parser.parse_args()


def handle_csv_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df = df.drop(["SName", "SLink", "language"], axis=1)
    df = df.dropna()
    df["ALink"] = df["ALink"].apply(lambda x: x.replace("/", ""))
    return df


def handle_txt_data(txt_dir_path):
    file_data = []
    for file_name in os.listdir(txt_dir_path):
        file_path = os.path.join(txt_dir_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                file_content = file.read()
                file_data.append({"ALink": file_name[:-4], "Lyric": file_content})
    df = pd.DataFrame(file_data)
    return df


def main(args):
    df_from_csv = handle_csv_data(args.csv_file_path)
    df_from_txt = handle_txt_data(args.txt_dir_path)
    merged_df = pd.concat([df_from_csv, df_from_txt])
    merged_df = merged_df.reset_index(drop=True)
    merged_df = merged_df.rename(columns={"ALink": "Artist"})
    csv_save_path = os.path.join("data/interim", "merged_data.csv")
    merged_df.to_csv(csv_save_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
