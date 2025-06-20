
# Credit Card Fraud Detection using Machine Learning

This project demonstrates a practical approach to detecting fraudulent credit card transactions using machine learning.

## Objective

Build a robust machine learning pipeline that can accurately classify credit card transactions as fraudulent or legitimate, with special attention to handling class imbalance and optimizing model performance.

## Dataset

The dataset is sourced from [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

- Contains transactions made by European cardholders in September 2013.
- Features are numerical due to PCA transformation, except for `Time` and `Amount`.
- Highly imbalanced: only ~0.17% of transactions are fraudulent.

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- SMOTE (from imbalanced-learn)
- Matplotlib, Seaborn
- Jupyter Notebook

## Project Workflow

1. **Data Preprocessing**
   - Clean and scale data using `StandardScaler`.
   - Handle class imbalance using **SMOTE**.

2. **Exploratory Data Analysis (EDA)**
   - Visualize class distribution.
   - Generate summary statistics and insights from feature distributions.

3. **Model Development**
   - Train/test split after SMOTE.
   - Model training using **XGBoost** (also compatible with Logistic Regression and Random Forest).
   - Performance evaluated using precision, recall, F1-score, ROC-AUC, PR Curve and KS Statistic.

4. **Evaluation**
   - Detailed classification report.
   - PR Curve visualization to assess model performance.
   - KS Statistic used to determine model's ability to distinguish between positive and negative classes

## Results

- Successfully addressed class imbalance with SMOTE.
- Achieved high recall and precision scores suitable for fraud detection scenarios.
- PR curve used to assess performance of the positive (fraudulent) class.
