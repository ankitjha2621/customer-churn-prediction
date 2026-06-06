# Customer Churn Prediction Dashboard

## Overview

Customer Churn Prediction Dashboard is a machine learning-powered application that predicts whether a customer is likely to leave a telecom service.

The project uses historical customer data and a Logistic Regression model to identify churn risk and help businesses improve customer retention.

---

## Features

- Customer churn prediction
- Interactive Streamlit dashboard
- Risk score visualization
- Business insights and recommendations
- Modern SaaS-style UI
- Real-time prediction

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Plotly
- Joblib

---

## Machine Learning Pipeline

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Label Encoding
5. Train-Test Split
6. Logistic Regression Training
7. Model Evaluation
8. Deployment using Streamlit

---

## Model Performance

Accuracy: 82.26%

---

## Project Structure

customer-churn-prediction/
├── app/
├── data/
├── models/
├── notebooks/
├── src/
├── train.py
├── requirements.txt
└── README.md

---
## System Architecture

```text
┌──────────────────────┐
│ Telecom Customer Data│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Data Preprocessing   │
│ - Cleaning           │
│ - Encoding           │
│ - Missing Values     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Logistic Regression  │
│ Model Training       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Saved Model (.pkl)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Streamlit Dashboard  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Churn Prediction     │
│ Risk Analysis        │
│ Recommendations      │
└──────────────────────┘
```
## Application Screenshots

### Dashboard
![Dashboard](<Screenshot 2026-06-06 103140.png>)

### Analytics Dashboard

![Analytics](<Screenshot 2026-06-06 103231.png>)
## Future Improvements

- Random Forest
- XGBoost
- Real Telecom Data Integration
- Cloud Deployment
- Customer Segmentation