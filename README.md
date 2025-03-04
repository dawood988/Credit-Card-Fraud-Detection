# Credit Card Fraud Detection

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Workflow](#project-workflow)
   - [Data Preprocessing](#1-data-preprocessing)
   - [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
   - [Model Building](#3-model-building)
   - [Model Evaluation](#4-model-evaluation)
4. [Installation & Dependencies](#installation--dependencies)
5. [Running the Project](#running-the-project)
6. [Results & Visualizations](#results--visualizations)
7. [Conclusion](#conclusion)
8. [Future Enhancements](#future-enhancements)
9. [Author](#author)
10. [License](#license)

## Overview

Credit card fraud detection is a crucial application of machine learning aimed at identifying fraudulent transactions from legitimate ones. This project uses multiple machine learning models, including Logistic Regression, Random Forest, AdaBoost, CatBoost, XGBoost, and LightGBM, to classify fraudulent transactions.

## Dataset

The dataset used contains anonymized credit card transaction data. Key features include:

- `Time`: Transaction timestamp.
- `Amount`: Transaction amount.
- `V1 - V28`: Principal Component Analysis (PCA) transformed features.
- `Class`: Target variable (0 = Non-fraudulent, 1 = Fraudulent).

## Project Workflow

### 1. Data Preprocessing

- Load dataset using Pandas.
- Detect encoding and read the CSV file.
- Handle missing values and duplicate entries.
- Address class imbalance using undersampling.

### 2. Exploratory Data Analysis (EDA)

- Display summary statistics.
- Analyze fraud and non-fraud transaction distributions.
- Plot correlation matrix heatmap.
- Compare transaction amounts in fraud vs. non-fraud cases.

### 3. Model Building

- Split data into training and testing sets.
- Implement the following models:
  - Logistic Regression
  - Random Forest
  - AdaBoost
  - CatBoost
  - XGBoost
  - LightGBM
- Perform hyperparameter tuning using GridSearchCV.

### 4. Model Evaluation

- Evaluate models using:
  - Accuracy
  - ROC-AUC score
  - Classification report (Precision, Recall, F1-score)

## Installation & Dependencies

### Prerequisites

Ensure you have the following libraries installed:

```sh
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost
```

## Running the Project

1. Clone this repository:
   ```sh
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```sh
   cd credit-card-fraud-detection
   ```
3. Run the script:
   ```sh
   python fraud_detection.py
   ```

## Results & Visualizations

- **Class Distribution Plot**: Showcases the imbalance in the dataset.
- **Correlation Heatmap**: Highlights important relationships between features.
- **Model Performance Metrics**: Displays accuracy, ROC-AUC, and classification report for each model.

## Conclusion

This project demonstrates the effectiveness of various machine learning models in detecting fraudulent transactions. The results highlight the importance of handling imbalanced datasets and using ensemble models for improved performance.

## Future Enhancements

- Implement **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.
- Use **Deep Learning (LSTMs or Autoencoders)** for anomaly detection.
- Deploy the model as an **API using Flask or FastAPI**.

## Author

- **Dawood M D**

## License

This project is open-source and available under the [MIT License](LICENSE).

