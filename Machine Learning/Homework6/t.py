import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

#%%
# Load the dataset
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/Smarket.csv'
data = pd.read_csv(url)

#%%
# Check if the dataset is balanced or imbalanced
direction_counts = data['Direction'].value_counts()
print("Class distribution:\n", direction_counts)

# Plot the imbalanced dataset
plt.figure(figsize=(8, 5))
sns.barplot(x=direction_counts.index, y=direction_counts.values)
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Direction')
plt.ylabel('Count')
plt.show()

# If the dataset is imbalanced, apply SMOTE
X = data[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
y = data['Direction'].apply(lambda x: 1 if x == 'Up' else 0)  # Convert to binary labels

smote = SMOTE(random_state=5805)
X_res, y_res = smote.fit_resample(X, y)

# Plot the balanced dataset
plt.figure(figsize=(8, 5))
sns.barplot(x=['Down', 'Up'], y=np.bincount(y_res))
plt.title('Class Distribution After SMOTE')
plt.xlabel('Direction')
plt.ylabel('Count')
plt.show()

# Split the dataset into train-test 80-20
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, shuffle=True, random_state=5805)

# Standardize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Display the first 5 rows of the train and test set
print("First 5 rows of X_train:\n", X_train[:5])
print("First 5 rows of X_test:\n", X_test[:5])

# Train logistic regression model
log_reg = LogisticRegression(random_state=5805)
log_reg.fit(X_train, y_train)

# Get predictions
y_train_pred = log_reg.predict(X_train)
y_test_pred = log_reg.predict(X_test)

# Print the best score for train and test
train_score = log_reg.score(X_train, y_train)
test_score = log_reg.score(X_test, y_test)
print(f"Training Accuracy: {train_score:.4f}")
print(f"Testing Accuracy: {test_score:.4f}")

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot the ROC curve
y_test_prob = log_reg.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Display accuracy, recall, precision, and f1 score
accuracy = accuracy_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")


#%%

# Check if the dataset is balanced or imbalanced
direction_counts = data['Direction'].value_counts()
print("Class distribution:\n", direction_counts)

# Plot the imbalanced dataset
plt.figure(figsize=(8, 5))
sns.barplot(x=direction_counts.index, y=direction_counts.values)
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Direction')
plt.ylabel('Count')
plt.show()

# If the dataset is imbalanced, apply SMOTE
X = data[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
y = data['Direction'].apply(lambda x: 1 if x == 'Up' else 0)  # Convert to binary labels

smote = SMOTE(random_state=5805)
X_res, y_res = smote.fit_resample(X, y)

# Plot the balanced dataset
plt.figure(figsize=(8, 5))
sns.barplot(x=['Down', 'Up'], y=np.bincount(y_res))
plt.title('Class Distribution After SMOTE')
plt.xlabel('Direction')
plt.ylabel('Count')
plt.show()

# Split the dataset into train-test 80-20
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, shuffle=True, random_state=5805)

# Standardize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Display the first 5 rows of the train and test set
print("First 5 rows of X_train:\n", X_train[:5])
print("First 5 rows of X_test:\n", X_test[:5])

# Train logistic regression model
log_reg = LogisticRegression(random_state=5805)
log_reg.fit(X_train, y_train)

# Get predictions
y_train_pred = log_reg.predict(X_train)
y_test_pred = log_reg.predict(X_test)

# Print the best score for train and test
train_score = log_reg.score(X_train, y_train)
test_score = log_reg.score(X_test, y_test)
print(f"Training Accuracy: {train_score:.4f}")
print(f"Testing Accuracy: {test_score:.4f}")

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot the ROC curve
y_test_prob = log_reg.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Display accuracy, recall, precision, and f1 score
accuracy = accuracy_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")

# Perform grid search with cross-validation to find the best parameters
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'C': np.logspace(-3, 1, 10),
    'l1_ratio': np.linspace(0, 1, 30)
}

grid_search = GridSearchCV(LogisticRegression(solver='saga', random_state=5805), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Display the result of the grid search with the best parameters
print("Best parameters found by grid search:")
print(grid_search.best_params_)
print(f"Best cross-validated score: {grid_search.best_score_:.4f}")

#%%
# Fit the train dataset to a Logistic Regression model using the fine-tuned parameters
best_log_reg = grid_search.best_estimator_
best_log_reg.fit(X_train, y_train)

# Get predictions using the fine-tuned model
y_train_pred_best = best_log_reg.predict(X_train)
y_test_pred_best = best_log_reg.predict(X_test)

# Print the best score for train and test using the fine-tuned model
train_score_best = best_log_reg.score(X_train, y_train)
test_score_best = best_log_reg.score(X_test, y_test)
print(f"Training Accuracy (fine-tuned): {train_score_best:.4f}")
print(f"Testing Accuracy (fine-tuned): {test_score_best:.4f}")

# Plot the confusion matrix for the fine-tuned model
conf_matrix_best = confusion_matrix(y_test, y_test_pred_best)
sns.heatmap(conf_matrix_best, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Fine-Tuned)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot the ROC curve for the fine-tuned model
y_test_prob_best = best_log_reg.predict_proba(X_test)[:, 1]
fpr_best, tpr_best, _ = roc_curve(y_test, y_test_prob_best)
roc_auc_best = auc(fpr_best, tpr_best)

plt.figure(figsize=(8, 5))
plt.plot(fpr_best, tpr_best, color='green', lw=2, label=f'ROC curve (AUC = {roc_auc_best:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (Fine-Tuned)')
plt.legend(loc='lower right')
plt.show()

# Display accuracy, recall, precision, and f1 score for the fine-tuned model
accuracy_best = accuracy_score(y_test, y_test_pred_best)
recall_best = recall_score(y_test, y_test_pred_best)
precision_best = precision_score(y_test, y_test_pred_best)
f1_best = f1_score(y_test, y_test_pred_best)

print(f"Accuracy (fine-tuned): {accuracy_best:.4f}")
print(f"Recall (fine-tuned): {recall_best:.4f}")
print(f"Precision (fine-tuned): {precision_best:.4f}")
print(f"F1 Score (fine-tuned): {f1_best:.4f}")

# Compare performance
print("\nComparison of performance before and after grid search:")
print(f"Accuracy - Before: {accuracy:.4f}, After: {accuracy_best:.4f}")
print(f"Recall - Before: {recall:.4f}, After: {recall_best:.4f}")
print(f"Precision - Before: {precision:.4f}, After: {precision_best:.4f}")
print(f"F1 Score - Before: {f1:.4f}, After: {f1_best:.4f}")
