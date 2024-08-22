## Introduction

The project aims to investigate how sequence context impacts SNP genotype calling errors in WGS data. We use autoencoder (AE) models to detect anomalies in genotype call data. By reconstructing input sequences and analyzing the reconstruction errors, we identify potential errors influenced by sequence context.

For more detailed information, please refer to the [original research article (under review)](#) or [preprint](https://www.biorxiv.org/content/10.1101/2024.03.23.586433v1).

# Exploring the Impact of Sequence Context on Errors in SNP Genotype Calling with Whole Genome Sequencing Data Using an AI-based Autoencoder Approach

This repository contains the implementation of a research project exploring the impact of sequence context on SNP genotype calling errors using Whole Genome Sequencing (WGS) data. The project leverages an AI-based autoencoder approach to detect anomalies in genotype calling errors.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Prepare the Input Data](#1-prepare-the-input-data)
  - [2. Train the Models](#2-train-the-models)
  - [3. Anomaly Detection and Labeling](#3-anomaly-detection-and-labeling)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The project aims to investigate how sequence context impacts SNP genotype calling errors in WGS data. We use autoencoder (AE) models to detect anomalies in genotype call data. By reconstructing input sequences and analyzing the reconstruction errors, we identify potential errors influenced by sequence context. 

Three different AE models, designated as S, M, and L, are trained on different scales of data to capture varying levels of sequence context effects. The project also includes an anomaly detection algorithm to assign final labels to the detected reconstruction errors.

## Features

- AI-based Autoencoder models (S, M, and L) for detecting genotype calling errors.
- Anomaly detection algorithm to classify reconstruction errors.
- Integration with Neptune.ai for monitoring model training (optional).

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/kotlarzkrzysztof/AE-AnomalySNP.git
    cd AE-AnomalySNP
    ```

2. Install the required Python packages:
    ```bash
    pip install numpy pandas tensorflow scikit-learn matplotlib seaborn neptune
    ```
## Usage

### 1. Prepare the Input Data

To prepare the input data, follow the approach suggested in the paper, which includes performing FAMD (Factorial Analysis for Mixed Data) analysis using a suitable R package. The prepared data will be used for training the autoencoder models.

- Install the required R packages:
    ```r
    install.packages("FactoMineR")
    ```

- Use the following R script to perform FAMD analysis and prepare the data. Make sure You are using the suitable number of PCs `ncp` and optionally add the label variable `sup.var`:
    ```r
    library(FactoMineR)

    # Load your data
    data <- read.csv("your_data.csv")

    # Perform FAMD
    result <- FAMD(data, ncp = 28, graph = FALSE)

    # Save the processed data
    write.csv(result$ind$coord, "famd_data.csv")
    ```

### 2. Train the Models

There are three AE models of diffrent complexity: S, M, and L. Each model has the same code structure but and requires updating the paths to Your training and test dataset. 

- To train a model, select model with the complexity of Your choice `scripts/ae_models/AE_model_{S,M,L}.py` and modify the paths in the script to point to the appropriate dataset. For example:

    ```python
    data_train = pd.read_csv('sample_train_data.csv')
    
    data_test = pd.read_csv('sample_test_data.csv')

    # In result_test function:

    data_test_org = pd.read_csv('data/sample_test_data.csv')
    ```
- Set the number of analysed PCs from the FAMD step. **Note**: by default every second PCs is analised:

    ```python
    N_PCAs = 23
    ```

- Optionally, if you wish to monitor the training using Neptune.ai, include your Neptune token:
    ```python
    import neptune

    run = neptune.init_run(
        project="your_project_name",
        api_token="your_neptune_api_token",
    )
    ```

    or by using the global variables:
    ```python
    os.environ["NEPTUNE_API_TOKEN"] = "your_neptune_api_token"
    os.environ["NEPTUNE_PROJECT"] = "your_project_name"
    ```

### 3. Anomaly Detection and Labeling

Once the models are trained, use the provided anomaly detection script to assign labels to the reconstruction errors. 

In this project, two models—**Isolation Forest** and **Support Vector Machine (SVM)**—are used for anomaly detection. Both models can be fine-tuned using grid search to optimize their parameters.

- Two scripts are aviable IsolationForest: `scripts/anomaly_detection/GRID_ISO_label_script.py` and SVM: `scripts/anomaly_detection/GRID_SVM_label_script.py`:

- **Isolation Forest:** A model specifically designed for anomaly detection in high-dimensional datasets. Grid parameter include `n_estimators`.
  
- **SVM (Support Vector Machine):** Used here for one-class classification in anomaly detection, with grid parameters such as `kernel`, and `gamma` that can be fine-tuned.

Adjust the grid search parameters in the `iso_params` or `svm_params` variables to suit your dataset and objectives.

## Contributing

Contributions are welcome! Please feel free to open an issue to discuss potential changes or improvements with the author of the article.

**Note:** To obtain the original dataset used in this research, please contact the corresponding author of the article.
