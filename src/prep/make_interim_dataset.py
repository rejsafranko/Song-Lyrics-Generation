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
    df = df[df["language"] == "en"]
    df = df.drop(["SName", "SLink", "language"], axis=1)
    df = df.dropna()
    df["ALink"] = df["ALink"].apply(lambda x: x.replace("/", ""))
    df.reset_index(drop=True, inplace=True)
    return df


def handle_txt_data(txt_dir_path):
    def transform_artist_name(x):
        x = x.replace("_", "-")
        x = x.lower()
        return x

    def merge_values(df):
        # Identify values where one is contained in the other.
        to_merge = {}
        for idx, row in df.iterrows():
            for other_idx, other_row in df.iterrows():
                if idx != other_idx and row["ALink"] in other_row["ALink"]:
                    if other_idx not in to_merge:
                        to_merge[other_idx] = []
                    to_merge[other_idx].append(idx)

        # Merge values.
        for other_idx, merge_indices in to_merge.items():
            merged_values = "\n".join(df.at[idx, "Lyric"] for idx in merge_indices)
            df.at[other_idx, "Lyric"] += "\n" + merged_values

        # Drop the rows with duplicates and reset the index.
        df.drop(
            index=[idx for indices in to_merge.values() for idx in indices],
            inplace=True,
        )
        df.reset_index(drop=True, inplace=True)

        return df

    file_data = []
    for file_name in os.listdir(txt_dir_path):
        file_path = os.path.join(txt_dir_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                file_content = file.read()
                if file_name[:-4] != ".git":
                    file_data.append({"ALink": file_name[:-4], "Lyric": file_content})

    df = pd.DataFrame(file_data)
    df["ALink"] = df["ALink"].apply(transform_artist_name)
    df = df.groupby("ALink")["Lyric"].agg(lambda x: "\n".join(x)).reset_index()
    df = merge_values(df)

    return df


def main(args):
    df_from_csv = handle_csv_data(args.csv_file_path)
    save_path = os.path.join("data/interim", "from_csv.csv")
    df_from_csv.to_csv(save_path)

    df_from_txt = handle_txt_data(args.txt_dir_path)
    save_path = os.path.join("data/interim", "from_txt.csv")
    df_from_txt.to_csv(save_path)

    merged_df = pd.concat([df_from_csv, df_from_txt])
    merged_df = merged_df.reset_index(drop=True)
    merged_df = merged_df.rename(columns={"ALink": "Artist"})
    save_path = os.path.join("data/interim", "merged_data.csv")
    merged_df.to_csv(save_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
