# Credit Card Fraud Detection ML System

This project builds a machine learning system to detect fraudulent credit card transactions.

The system trains a fraud detection model using a real-world dataset and exposes the model through a FastAPI API endpoint for predictions.

---

# Project Overview

Credit card fraud is a major financial problem. Machine learning can help detect suspicious transactions automatically.

This project demonstrates a full **end-to-end machine learning workflow**:

1. Data loading
2. Exploratory Data Analysis (EDA)
3. Model training
4. Model evaluation
5. Model saving
6. API deployment

---

# Dataset

Credit Card Fraud Detection Dataset

Dataset characteristics:

* Total transactions: **284,807**
* Fraud transactions: **492**
* Features: **V1 – V28, Time, Amount**
* Target variable: **Class**

Class labels:

* **0 → Normal transaction**
* **1 → Fraud transaction**

The dataset is **highly imbalanced**, which makes fraud detection a challenging machine learning problem.

---

# Technologies Used

Python
pandas
NumPy
scikit-learn
Matplotlib
Seaborn
FastAPI
Uvicorn
Joblib

---

# Project Structure

```
credit-card-fraud-detection-ml-khatantamir

app/
    main.py                # FastAPI application

data/
    raw/                   # dataset placeholder

models/
    fraud_model.pkl        # trained ML model

notebooks/
    eda.ipynb              # exploratory data analysis

src/
    train.py               # model training script

requirements.txt           # project dependencies
README.md                  # project documentation
```

---

# Exploratory Data Analysis

EDA was performed in a Jupyter Notebook.

Analysis included:

* dataset shape and structure
* fraud vs normal distribution
* class imbalance visualization
* transaction feature exploration
* correlation analysis

Key observation:

Fraud cases represent only **~0.17% of the dataset**, creating a severe class imbalance.

---

# Machine Learning Model

Model used:

**Random Forest Classifier**

Training pipeline:

1. Load dataset
2. Separate features and target
3. Train/Test split (80/20)
4. Train Random Forest model
5. Generate predictions
6. Evaluate model performance
7. Save trained model using Joblib

Training script:

```
src/train.py
```

---

# Model Performance

Fraud detection results:

| Metric    | Score |
| --------- | ----- |
| Precision | 0.94  |
| Recall    | 0.82  |
| F1 Score  | 0.87  |
| ROC AUC   | 0.96  |

These results show strong performance in detecting fraudulent transactions.

---

# Confusion Matrix

The confusion matrix shows how many fraud and normal transactions were correctly classified.

Key insight:

The model successfully detects most fraudulent transactions while maintaining very low false positives.

---

# API (FastAPI)

The trained model is deployed using **FastAPI**.

API endpoint:

```
POST /predict
```

The API receives transaction feature values and returns whether the transaction is fraud or normal.

Example response:

```
{
  "prediction": "Fraud"
}
```

---

# Run the Project Locally

Install dependencies:

```
pip install -r requirements.txt
```

Start the API server:

```
uvicorn app.main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000/docs
```

FastAPI automatically generates interactive API documentation.

---

# Example Prediction Request

```
POST /predict
```

Example request body:

```
[
0.0,
-1.359807,
-0.072781,
2.536346,
1.378155,
-0.338321,
0.462388,
0.239599,
0.098698,
0.363787,
0.090794,
-0.551600,
-0.617801,
-0.991390,
-0.311169,
1.468177,
-0.470401,
0.207971,
0.025791,
0.403993,
0.251412,
-0.018307,
0.277838,
-0.110474,
0.066928,
0.128539,
-0.189115,
0.133558,
149.62
]
```

Example response:

```
{
  "prediction": "Normal"
}
```

---

# Key Skills Demonstrated

Machine Learning Model Development
Exploratory Data Analysis
Handling Imbalanced Datasets
Model Evaluation (Precision, Recall, F1, ROC AUC)
Model Serialization with Joblib
Building REST APIs using FastAPI
End-to-End Machine Learning Project Structure

---

# Author

Khatantamir

Data Science / Machine Learning Portfolio Project
