# FUTURE_DS_03
# Loan Approval Prediction Model
This repository contains the implementation of a Loan Approval Prediction Model using machine learning techniques to predict whether a loan application will be approved based on various factors such as applicant income, loan amount, credit history, and more. The goal is to develop a model that can predict the loan approval status (Loan_Status) as either 1 (approved) or 0 (not approved).

# Project Overview
The dataset contains several features, including:

- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area
- Gender, Marital Status, Education, Self-Employment Status
- Loan Status (target variable)
  
The model predicts the Loan Status (Loan_Status) based on these features using machine learning algorithms.

# Technologies Used
- Python: For data processing and building the machine learning model.
  # Libraries:
- pandas for data manipulation and handling missing values.
- sklearn for machine learning models, preprocessing, and evaluation.
- matplotlib and seaborn for data visualisation.
- numpy for numerical computations.
  # Data Preprocessing
The data preprocessing steps include:

# 1. Handling Missing Values
- Numerical columns are filled with the median value.
- Categorical columns are filled with the mode (most frequent value).
# 2. Label Encoding
- Categorical features like Gender, Married, Education, Self_Employed, and Property_Area are encoded into numeric values using LabelEncoder.
- The target variable Loan_Status is encoded into 1 (approved) and 0 (not approved).
# 3. Feature Scaling
- Numerical features are scaled using StandardScaler to ensure the data is standardized and suitable for machine learning models.
# Training the Model
We use machine learning techniques such as Logistic Regression, Random Forest, or Gradient Boosting to train the model.
# Model Evaluation
After training the model, we evaluate its performance using metrics such as:
- Accuracy: The percentage of correct predictions.
- Precision, Recall, F1-Score: Metrics that help to understand the model’s performance with respect to both positive and negative classes.
- Confusion Matrix: Helps in understanding false positives, false negatives, true positives, and true negatives.
  # Conclusion
This model successfully predicts whether a loan application will be approved based on various factors. It uses machine learning techniques like Logistic Regression for classification. You can experiment with other algorithms such as Random Forest or Gradient Boosting for potentially better performance.
