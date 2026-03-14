# Credit Card Fraud Detection ML System

This project builds a machine learning system to detect fraudulent credit card transactions.

## Dataset

Credit Card Fraud Detection dataset.

- 284,807 transactions
- 492 fraud cases
- Highly imbalanced dataset

## Technologies

- Python
- pandas
- scikit-learn
- FastAPI

## Project Structure

data/ – dataset  
notebooks/ – exploratory data analysis  
models/ – trained model  
app/ – FastAPI application  

## Model

Random Forest classifier used to detect fraudulent transactions.

Evaluation metrics:

- Precision
- Recall
- F1-score
- ROC AUC

## API

FastAPI endpoint:

POST /predict

Input: transaction feature list

Output:


Model Performance
- Fraud Precision: 0.94
- Fraud Recall: 0.82
- Fraud F1-score: 0.87
- ROC AUC: 0.96


