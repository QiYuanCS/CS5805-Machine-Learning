import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk import ConfusionMatrix
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc,roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import seaborn as sns

#%%
print('2')
z = np.linspace(-10, 10, 1000)
sigma_z = 1 / (1 + np.exp(-z))

loss_y1 = -np.log(sigma_z)
loss_y0 = -np.log(1 - sigma_z)

plt.figure(figsize=(10, 6))
plt.plot(sigma_z, loss_y1, label='J(w) if y = 1', linewidth=3, color='blue')
plt.plot(sigma_z, loss_y0, label='J(w) if y = 0', linewidth=3, linestyle='--', color='red')
plt.xlabel(r'$\sigma(z)$', fontsize=14)
plt.ylabel('J(w)', fontsize=14)
plt.title('Log-Loss function', fontsize=16)
plt.legend(fontsize=12, loc='upper right')
plt.grid()
plt.show()

#%%
print('3')
X, y = make_classification(n_samples=1000, n_features=2, n_clusters_per_class=2, n_informative=2,
                           n_repeated=0, n_redundant=0, random_state=5805)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)

model = LogisticRegression()
model.fit(X_train, y_train)

y_prediction = model.predict(X_test)
cm = confusion_matrix(y_test, y_prediction)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.show()

#%%
y_prediction_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prediction_proba)
auc = roc_auc_score(y_test, y_prediction_proba)

plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="upper right")
plt.show()

#%%
accuracy = accuracy_score(y_test, y_prediction)
recall = recall_score(y_test, y_prediction)
precision = precision_score(y_test, y_prediction)
print(f"Accuracy: {accuracy:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")


#%%
print('4')
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/Smarket.csv'
data = pd.read_csv(url)

#%%
direction_counts = data['Direction'].value_counts()
plt.figure(figsize=(8, 5))
plt.bar(direction_counts.index, direction_counts.values, color=['skyblue', 'orange'])
plt.title('Imbalanced Dataset')
plt.xlabel('Direction')
plt.ylabel('Count')
plt.show()

#%%
X = data[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
y = data['Direction'].apply(lambda x: 1 if x == 'Up' else 0)

smote = SMOTE(random_state=5805)
X_balanced, y_balanced = smote.fit_resample(X, y)
y_balanced_counts = pd.Series(y_balanced).value_counts()
plt.figure(figsize=(8, 5))
plt.bar(y_balanced_counts.index, y_balanced_counts.values, color=['skyblue', 'orange'])
plt.title('Balanced Dataset using SMOTE')
plt.xlabel('Direction')
plt.ylabel('Count')
plt.show()

#%%

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=5805, shuffle=True)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("First 5 rows of the training set:")
print(X_train[:5])
print("\nFirst 5 rows of the testing set:")
print(X_test[:5])


#%%
logreg = LogisticRegression(random_state=5805)
logreg.fit(X_train_scaled, y_train)

train_score = logreg.score(X_train_scaled, y_train)
test_score = logreg.score(X_test_scaled, y_test)
print(f"Training Score: {train_score:.3f}")
print(f"Testing Score: {test_score:.3f}")

#%%
y_pred = logreg.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(conf_matrix).plot()
plt.title("Confusion Matrix")
plt.show()

#%%
y_pred_prob = logreg.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

#%%
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Precision: {precision:.3f}")
print(f"F1 Score: {f1:.3f}")


#%%
param_grid = [
    {
        'penalty': ['l1', 'l2'],
        'C': np.logspace(-3, 1, 10),
        'solver': ['saga']
    },
    {
        'penalty': ['elasticnet'],
        'C': np.logspace(-3, 1, 10),
        'solver': ['saga'],
        'l1_ratio': np.linspace(0, 1, 30)
    }
]

grid_search = GridSearchCV(
    LogisticRegression(random_state=5805, max_iter=10000),
    param_grid,
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

print("Best parameters from GridSearchCV:")
print(grid_search.best_params_)
print(f"Best cross-validated score: {grid_search.best_score_:.4f}")



#%%
best_params = grid_search.best_params_
grid_best = LogisticRegression(
    **best_params,
    random_state=5805,
    max_iter=10000
)
grid_best.fit(X_train_scaled, y_train)
y_pred_best = grid_best.predict(X_test_scaled)

conf_matrix_best = confusion_matrix(y_test, y_pred_best)
ConfusionMatrixDisplay(conf_matrix_best).plot()
plt.title("Confusion Matrix")
plt.show()


#%%
y_test_prob = grid_best.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='red',
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic After Grid Search')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

                                                                        #%%
accuracy_best = accuracy_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)
print(f"Accuracy : {accuracy_best:.3f}")
print(f"Recall: {recall_best:.3f}")
print(f"Precision : {precision_best:.3f}")
print(f"F1 Score : {f1_best:.3f}")