# Regression on the tabular data. General Machine Learning

Welcome to the Regression on the tabular data. General Machine Learning! This module contains tools and scripts for performing exploratory data analysis (EDA) on a given dataset, training an regression model, and making predictions on test data using the trained model.

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training](#model-training)
- [Making Predictions](#making-predictions)
- [Prediction Results](#prediction-results)

## Project Overview

This project aims to provide a simple and streamlined approach to perform data analysis, build a predictive model, and make predictions on new data using the trained model. The module is designed to be user-friendly and requires minimal setup to get started.

## Getting Started

### Prerequisites

Before you begin, make sure you have the following installed on your system:

- Python (>=3.10)
- Jupyter Notebook
- pip (Python package manager)

### Installation

1. Clone current repository from GitHub:

   ```
   git clone {path to the remote repository}
   ```

2. Navigate to the project directory:

   ```
   cd second_task
   ```

3. Install the required Python packages by running:

   ```
   pip install -r requirements.txt
   ```

## Exploratory Data Analysis

To perform exploratory data analysis on the provided dataset (train.csv), follow these steps:

1. Open Jupyter Notebook:

   ```
   jupyter notebook
   ```

2. In the Jupyter Notebook interface, navigate to the project directory and open the file "Exploratory_Data_Analysis.ipynb."

3. Follow the instructions and code examples provided in the notebook to explore the dataset and gain insights into its characteristics.

## Model Training

The Data Science Module includes a Python script (train.py) that enables you to train an XGBoost Regressor model using the prepared training data.

To train the model, follow these steps:

1. Open a terminal or command prompt.

2. Navigate to the project directory.

3. The training script (train.py) provides several command-line arguments to offer flexibility and customization during model training. These arguments allow you to control various aspects of the training process. Here are the available arguments and their purposes:

    - `-h`, `--help`: Display the help message and exit. It provides information about the available arguments and their usage.

    - `--train-path TRAIN_PATH`, `--train TRAIN_PATH`: *(Required)* This argument specifies the path to the input training data file (e.g., train.csv). The script will use this file to train the XGBoost Regressor model.

    - `--delimiter DELIMITER`: Specify the data delimiter used in the training data file. If not provided, the script will assume the delimiter to be the default value (e.g., comma for CSV files).

    - `--objective OBJECTIVE`: Use this argument to specify the learning task and the corresponding objective function for the XGBoost Regressor (e.g., regression, classification).

    - `--normalization`: When included, this argument instructs the script to perform data normalization on the dataset before training the model. Normalization can help improve convergence and performance in certain cases.

    - `--weights-name WEIGHTS_NAME`, `--name WEIGHTS_NAME`: Specify the name for the saved XGBoost model weights or checkpoint. The trained model will be saved with this name for later use or evaluation.

    - `--test-split TEST_SPLIT`, `--split TEST_SPLIT`: Specify the test data split ratio. It determines the portion of the training data that will be used for model evaluation during training. The remaining data will be used for actual model training.

    - `--seed SEED`: Set the random seed for reproducibility. Providing a fixed seed ensures consistent results across multiple runs.

    - `--metric METRIC`: Specify the evaluation metric used to assess the model's performance during training. This metric is used to monitor the model's progress and make decisions, such as early stopping based on the metric's behavior.

    These arguments provide fine-grained control over the training process and enable you to experiment with different hyperparameters to optimize the XGBoost Regressor model's performance on your specific dataset.

    You can use the training script with various combinations of these arguments to adapt the model to your specific requirements. For detailed information about each argument and its possible values, consult the script's documentation or use the `--help` argument when executing the script:

    ```
    python train.py --help
    ```

4. Run the training script:

   ```
   python train.py
   ```

5. The script will process the training data, train the XGBoost Regressor model, and save the trained model to disk.

## Making Predictions

Once the model is trained, you can use it to make predictions on new test data. The module includes a Python script (predict.py) for this purpose.

To make predictions, follow these steps:

1. Place the test data in a CSV file with the same format as the training data.

2. Open a terminal or command prompt.

3. Navigate to the project directory.

4. The inference script (predict.py) provides several command-line arguments to allow flexible usage of the script for making predictions using the trained XGBoost Regressor model. These arguments provide control over various aspects of the inference process. Here are the available arguments and their purposes:

    - `-h`, `--help`: Display the help message and exit. It provides information about the available arguments and their usage.

    - `--normalization`: When included, this argument instructs the script to perform data normalization on the dataset before making predictions. This normalization should be consistent with the one used during training.

    - `--weights-name WEIGHTS_NAME`, `--name WEIGHTS_NAME`: *(Required)* Specify the name of the saved XGBoost model weights or checkpoint. The script will load the trained model with this name to make predictions on new data.

    - `--input INPUT`, `-i INPUT`: *(Required)* This argument specifies the path to the input data file on which you want to make predictions. The script will use this data to make predictions using the trained XGBoost Regressor model.

    - `--delimiter DELIMITER`: Specify the data delimiter used in the input data file. If not provided, the script will assume the delimiter to be the default value (e.g., comma for CSV files).

    - `--output OUTPUT`, `-o OUTPUT`: *(Required)* Use this argument to define the path where the prediction results will be saved. The script will create a file at this location and write the predicted values into it.

    These arguments allow you to easily customize the inference process and adapt it to different data formats and use cases. By providing the appropriate values for these arguments, you can use the script to make predictions on new data using the trained XGBoost Regressor model.

    For detailed information about each argument and its possible values, you can refer to the script's documentation or use the `--help` argument when executing the script:

    ```
    python predict.py --help
    ```

5. Run the prediction script:

   ```
   python predict.py --input test.csv --output results.csv
   ```

   Replace "test.csv" with the filename of your test data, and "results.csv" with the desired filename for the prediction results.

## Prediction Results

After running the prediction script, you will find the prediction results in the specified output file (results.csv). This file will contain the predicted target values for the test data based on the trained XGBoost Regressor model.
