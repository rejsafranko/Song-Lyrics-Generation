# Song Lyric Generation with Deep Models

Welcome to the Song Lyric Generation project! This repository contains the code and resources for a research project focused on generating song lyrics using deep learning models. Our goal is to explore different models and techniques to generate lyrics that can mimic specific singers' styles as well as produce generic lyrics.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Models Used](#models-used)
- [Data Preprocessing](#data-preprocessing)
- [Evaluation](#evaluation)
- [How to Use](#how-to-use)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This research project aims to create song lyric generators using various deep learning models. We implemented four generators in total:
- **Generic Lyric Generators**:
  - BiLSTM
  - Variational Autoencoder (VAE)
- **Specific Singer Style Generators**:
  - GPT-2
  - TinyLlama

The models were trained on different datasets and evaluated through human assessments to ensure the generated lyrics are meaningful and stylistically accurate.

## Project Structure






Create dataset:

```python src/prep/make_interim_dataset.py -csv_file_path data/raw/csv/lyrics-data.csv -txt_dir_path data/raw/txt```

```python src/prep/make_processed_dataset.py --interim_path data/interim/merged_data.csv```


Run training and inference scripts:

```python src/train/train_[model]_[task].py -csv_file_path data/raw/csv/lyrics-data.csv -txt_dir_path data/raw/txt```

```python src/train/inference_[model]_[task].py --seed_text [text] <--next_words [number of words to generate]>```
