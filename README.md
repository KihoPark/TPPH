# Transformer Point Processes for High-Dimensional Events (TPPH)

In our paper, we propose a transformer-based point process model for high-dimensional event history data.
This repository provides the code for our model.

## Data
The datasets (Amazon, Earthquake, Stack Overflow, Taobao, and Taxi) refined by [EasyTPP](https://arxiv.org/abs/2307.08097) can be downloaded from this [link](https://drive.google.com/drive/folders/1f8k82-NL6KFKuNMsUwozmbzDSFycYvz7).

The LastFM dataset can be downloaded from this [GitHub repository](https://github.com/shchur/ifl-tpp).

The Netflix dataset is not publicly available.

## Requirements
You need to install the Python packages `torch` and `pickle` to run the code. Additionally, using a GPU is recommended for efficient model training.

## Training the TPPH Model on Datasets
To train and evaluate the TPPH model on the Amazon dataset, run:

    python src/run/train.py --data_name amazon --data_dir <directory_of_data> --log_dir <directory_of_logs>