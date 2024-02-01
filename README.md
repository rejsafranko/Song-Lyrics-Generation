# Song-Lyrics-Generation
Create dataset:

```python src/prep/make_interim_dataset.py -csv_file_path data/raw/csv/lyrics-data.csv -txt_dir_path data/raw/txt```

```python src/prep/make_processed_dataset.py --interim_path data/interim/merged_data.csv```


Run training and inference scripts:

```python src/train/train_[model]_[task].py -csv_file_path data/raw/csv/lyrics-data.csv -txt_dir_path data/raw/txt```

```python src/train/inference_[model]_[task].py --seed_text [text] <--next_words [number of words to generate]>```
