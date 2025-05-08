#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 16:42:41 2025

@author: macbook
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  

# Load embedding dataset (Load the embedding you want to get results for that)
data = pd.read_csv('/Users/macbook/Documents/PhD_Documents/embedding_methods/total_data/Brent/embedding_Fasttext_mean_Brent.csv')

# Load sentiment dataset
#data = pd.read_csv('/Users/macbook/Documents/PhD_Documents/sentiment_methods/total_data/Brent/sentiment_Brent.csv')
data = data[data['Date'] < '2024-01-01']
data = data.drop(['Date', 'Unnamed: 0', 'RV_neg', 'RV_pos', 'RQ', 'BVP'], axis=1)

# Create target variable 'Direction'
data['Change'] = data['RV'].diff()
data['Direction'] = data['Change'].apply(lambda x: 1 if x > 0 else 0)  # Convert to numeric
data.drop(columns=['Change'], inplace=True)
data = data.iloc[1:]

# Create lagged features
def create_lagged_features(data, target_column, num_lags):
    lagged_data = pd.DataFrame(index=data.index)
    exclude_columns = ['Direction', 'RV', 'RV_daily', 'RV_weekly', 'RV_monthly']
    for column in data.columns:
        if column not in exclude_columns:
            for lag in range(1, num_lags + 1):
                lagged_data[f'{column}_lag{lag}'] = data[column].shift(lag)

   # lagged_data[['RV_daily', 'RV_weekly', 'RV_monthly']] = data[['RV_daily', 'RV_weekly', 'RV_monthly']]
    lagged_data[target_column] = data[target_column]
    return lagged_data.dropna()

# Run for HAR method
#data['RV_daily'] = data['RV'].shift(1)
#data['RV_weekly'] = data['RV'].shift(5)
#data['RV_monthly'] = data['RV'].shift(22)
#data = data[['RV_daily', 'RV_weekly', 'RV_monthly', 'Direction']]

# Run for different sentiment methods
#data = data[['finbert', 'Direction']]
#data = create_lagged_features(data, 'Direction', 10)

# Run for embedding methods
data = create_lagged_features(data, 'Direction', 10)

# Separate features and target
X = data.drop(columns=['Direction'])
y = data['Direction']

# Initialize scaler and models
scaler = StandardScaler()
models = {
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(probability=True),  # SVC should have probability=True for soft voting
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naïve Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression()
}

# Create a VotingClassifier ensemble
voting_clf = VotingClassifier(estimators=[  ('knn', models['KNN']),
                                         # ('svc', models['SVC']),
                                          #('rf', models['Random Forest']),
                                          #('dt', models['Decision Tree']),
                                          ('nb', models['Naïve Bayes']),
                                          ('lr', models['Logistic Regression'])],
                              voting='soft')  # Use 'hard' for hard voting or 'soft' for soft voting

# Rolling window parameters
train_window = 2486
test_window = 1

# Initialize an empty list to store all model results
all_model_results = []

# Train the ensemble model with a rolling window approach
print(f"\nTraining Ensemble Model with Rolling Window...\n")
all_y_true, all_y_pred = [], []
scaler_fitted = False  # Flag to fit scaler only once

for i in tqdm(range(train_window, len(X)), desc="Training Ensemble", unit="window"):
    X_train, y_train = X.iloc[i - train_window:i], y.iloc[i - train_window:i]
    X_test, y_test = X.iloc[i:i + test_window], y.iloc[i:i + test_window]
    
    # Fit scaler only once
    if not scaler_fitted:
        scaler.fit(X_train)
        scaler_fitted = True
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit the VotingClassifier ensemble model
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

# Save the results of the ensemble model
ensemble_results_df = pd.DataFrame({
    'Model': ['Ensemble'] * len(all_y_true),
    'Actual': all_y_true,
    'Predicted': all_y_pred
})

# Compute performance metrics for the ensemble model
accuracy = accuracy_score(all_y_true, all_y_pred)
precision = precision_score(all_y_true, all_y_pred, average="weighted")
recall = recall_score(all_y_true, all_y_pred, average="weighted")
f1 = f1_score(all_y_true, all_y_pred, average="weighted")
cm = confusion_matrix(all_y_true, all_y_pred)

# Store ensemble results
ensemble_model_results = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "confusion_matrix": cm
}

# Concatenate all model results (including the ensemble model results)
all_model_results.append(ensemble_results_df)
final_results_df = pd.concat(all_model_results, ignore_index=True)

# Save the final concatenated results to a CSV file
final_results_df.to_csv('/Users/macbook/Documents/PhD_Documents/Italy_conference_paper/Prediction/option2/all_models_ensemble_fasttext.csv', index=False)

# Display the ensemble model performance summary
print("\nEnsemble Model Performance:")
print(f"  - Accuracy:  {ensemble_model_results['accuracy']:.4f}")
print(f"  - Precision: {ensemble_model_results['precision']:.4f}")
print(f"  - Recall:    {ensemble_model_results['recall']:.4f}")
print(f"  - F1-score:  {ensemble_model_results['f1_score']:.4f}")
print("-" * 40)

# Plot confusion matrix for the ensemble model
plt.figure(figsize=(5, 4))
sns.heatmap(ensemble_model_results['confusion_matrix'], annot=True, fmt="d", cmap="Blues", xticklabels=["DOWN", "UP"], yticklabels=["DOWN", "UP"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Ensemble Model")
plt.show()
