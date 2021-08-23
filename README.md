# Predict Telco Churn

This project follows the CRISP-DM process.

## Data Source

https://www.kaggle.com/blastchar/telco-customer-churn

## Business Understanding

**Goal**:  
Predict which customer will churn the telco-company.  
Select which customer groups (which properties) have the highest chance to leave.

**Motivation**:  
If the company knows which customer will leave, it can react and adjust to hold the customer.

**Requirements**:  
There should be at least 3 properties which indicate that a customer will quit.

**Example Anwers**:  
"This Customer will churn"  
"The following properties are non-optimal: short term contracts, senior customer, only telephone service"

## Results from data understanding

## Structure

- [Data understanding and preparation notebook](notebooks/1-data_understanding_prerparation.ipynb)
- [Predictions with ML algorithms](notebooks/2a-prediction_ml_algos.ipynb)
- [Predictions with neural networks](notebooks/2b-prediction_nn.ipynb)
