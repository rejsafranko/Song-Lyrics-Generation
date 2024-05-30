# Song Lyric Generation with Deep Models

Welcome to the Song Lyric Generation project! This repository contains the code and resources for a research project focused on generating song lyrics using deep learning models. Our goal is to explore different models and techniques to generate lyrics that can mimic specific singers' styles as well as produce generic lyrics.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Models Used](#models-used)
- [Data Preprocessing](#data-preprocessing)
- [Evaluation](#evaluation)
- [How to Use](#how-to-use)
- [License](#license)

## Introduction

This research project aims to create song lyric generators using various deep learning models. We implemented four generators in total. Two for generic lyric generation and two for specific artist lyric generation.
The models were trained on a song lyric datset created from 2 different data sources and evaluated through human assessments to ensure the generated lyrics are meaningful and stylistically accurate.

## Project Structure
```
.
├── data
│ ├── raw # Raw dataset files
|   ├── csv # Raw csv song file
|   └── txt # Raw artist lyrics as txt files
│ ├── interim # Interim dataset files after initial preprocessing
│ └── processed # Final processed dataset files ready for model training
├── models # Saved model checkpoints
├── src
│ ├── prep # Scripts for data preprocessing
│ ├── train # Scripts for training the models
│ └── eval # Scripts for evaluating the models
├── docs
│ └── project documentation.pdf # Research report in Croatian
├── README.md # Project README file
└── requirements.txt # Python dependencies
```

## Models Used

### Generic Lyric Generation
1. **BiLSTM**: A Bidirectional Long Short-Term Memory model used for sequential data processing.
2. **Variational Autoencoder (VAE)**: A generative model that learns to represent data in a latent space.

### Specific Singer Style Generation
1. **GPT-2**: A powerful transformer-based model pre-trained on a large corpus of text, fine-tuned to mimic a specific singer's style.
2. **TinyLlama**: A smaller, efficient model similar to GPT-2, optimized for stylistic lyric generation.

## Data Preprocessing

Data preprocessing is crucial for training effective models. The preprocessing pipeline involves:
1. **Making an Interim Dataset**: Initial cleaning and formatting of raw data.
2. **Making a Processed Dataset**: Further processing to prepare data for model training, such as tokenization and sequence generation.

Scripts for data preprocessing are located in the `src/prep` directory.

## Evaluation

The models were evaluated using human evaluations. We focused on the meaningfulness and stylistic accuracy of the generated lyrics. The evaluation criteria included:
- Coherence and fluency of the lyrics
- Stylistic resemblance to the target singer (for specific singer models)
- Overall creativity and originality

In our report we also provide GPT-4 evaluation to compare how a SOTA large language model evaluates a task only humans can effectively evaluate.

## How to Use

### Prerequisites

Ensure you have Python installed. Install the required dependencies using:

```bash
pip install -r requirements.txt


Create dataset:

```python src/prep/make_interim_dataset.py -csv_file_path data/raw/csv/lyrics-data.csv -txt_dir_path data/raw/txt```

```python src/prep/make_processed_dataset.py --interim_path data/interim/merged_data.csv```


Run training and inference scripts:

```python src/train/train_[model]_[task].py -csv_file_path data/raw/csv/lyrics-data.csv -txt_dir_path data/raw/txt```

```python src/train/inference_[model]_[task].py --seed_text [text] <--next_words [number of words to generate]>```
