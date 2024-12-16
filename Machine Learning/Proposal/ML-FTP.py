#Phrase-I
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import warnings
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')
df = pd.read_csv('car_prices.csv')

df.drop('vin', axis=1, inplace=True)
df.drop('state', axis=1, inplace=True)

df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows before removal: {duplicates}")

df.drop_duplicates(inplace=True)

duplicates_after = df.duplicated().sum()
print(f"Number of duplicate rows after removal: {duplicates_after}")

df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')
df.dropna(subset=['saledate'], inplace=True)

df['sale_year'] = df['saledate'].apply(lambda x: int(x.strftime('%Y')))
df['sale_month'] = df['saledate'].apply(lambda x: int(x.strftime('%m')))
df['Vehicle_Age'] = abs(df['year'] - (df['sale_year'] + df['sale_month'] / 12))

print(df.head())


df = df.drop(['saledate', 'sale_year', 'sale_month', 'year'], axis=1)

label_encoder = LabelEncoder()
categorical_cols = ['make', 'model', 'trim', 'body', 'color', 'interior', 'seller', 'transmission']
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

print("\nData after Label Encoding:")
print(df.head())

numerical_cols = ['condition', 'odometer', 'Vehicle_Age','mmr']
lof = LocalOutlierFactor(n_neighbors=20)
outliers = lof.fit_predict(df[numerical_cols])
df = df[outliers != -1]
print(f"The number of data points remaining after the removal of outliers in the dataset: {df.shape[0]}")

print(df.head())

pd.set_option('display.max_columns', None)
X = df.drop('sellingprice', axis=1)
y = df['sellingprice']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)


print("\nData after Standardization:")
print(X_scaled_df.head())

pca = PCA()
pca.fit(X_scaled_df)

print("\nExplained variance ratio of each principal component:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"Principal Component {i+1}: {ratio:.6f}")

plt.figure(figsize=(12, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.tight_layout()
plt.show()

loadings = pca.components_.T
print("\nPCA Loadings (first few components):")
for i, feature in enumerate(X.columns):
    print(f"{feature}: PC1 Loading={loadings[i, 0]:.6f}, PC2 Loading={loadings[i, 1]:.6f}")

U, s, Vt = np.linalg.svd(X_scaled_df, full_matrices=False)
condition_number = s[0] / s[-1]
print(f"\nCondition Number: {condition_number:.6f}")

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled_df.values, i) for i in range(X_scaled_df.shape[1])]
print("\nVariance Inflation Factor (VIF):")
print(vif_data)


X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=5805)

cov_matrix = np.cov(X_train.T)
cov_df = pd.DataFrame(cov_matrix, index=X_train.columns, columns=X_train.columns)

plt.figure(figsize=(12, 10))
sns.heatmap(cov_df, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Covariance Matrix Heatmap (Training Set)')
plt.show()

corr_matrix = X_train.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix Heatmap (Training Set)')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(y_train, bins=50, kde=True)
plt.title('Distribution of sellingprice (Training Set)')
plt.xlabel('sellingprice')
plt.ylabel('Frequency')
plt.show()

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

plt.figure(figsize=(8, 6))
sns.histplot(y_train_log, bins=50, kde=True)
plt.title('Distribution of Log-Transformed sellingprice')
plt.xlabel('Log(sellingprice)')
plt.ylabel('Frequency')
plt.show()

print(df.head())
numerical_feature = ['make','body','condition','model','seller','odometer','color','interior','transmission','mmr','Vehicle_Age']
X = df[numerical_feature]
y = df['sellingprice']
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

scaler = MinMaxScaler().set_output(transform="pandas")
X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X), columns = X.columns)

print(X_scaled.head())

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y, test_size = 0.3,
    random_state = 10
)

#%%Phrase-II
# Apply Linear Regression (reg) code to the following code

# T-test and Linear Regression Model Fitting
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Fit the sklearn Linear Regression model
reg = LinearRegression().fit(X_train, y_train)

print("\nInitial Model Summary:")
print(f"R-squared: {reg.score(X_train, y_train)}")

# Predict on the test set
y_pred = reg.predict(X_test)

# MSE evaluation
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE) on Test Set: {mse}")

# F-test analysis (using statsmodels)
X_train_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_const).fit()
print("\nF-test Analysis:")
print(f"F-statistic: {model.fvalue}")
print(f"F-pvalue: {model.f_pvalue}")

initial_results = {
    'R-squared': reg.score(X_train, y_train),
    'Adjusted R-squared': 1 - (1 - reg.score(X_train, y_train)) * (len(y_train) - 1) / (len(y_train) - X_train.shape[1] - 1),
    'AIC': model.aic,
    'BIC': model.bic,
    'MSE': mse
}

# Stepwise Regression (Forward Selection)
def forward_selection(data, target, significance_level=0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while (len(initial_features) > 0):
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features, dtype=float)
        for new_column in remaining_features:
            model_fs = sm.OLS(target, sm.add_constant(data[best_features + [new_column]])).fit()
            new_pval[new_column] = model_fs.pvalues[new_column]
        min_p_value = new_pval.min()
        if min_p_value < significance_level:
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features

best_features = forward_selection(X_train, y_train)
print("\nSelected features after Stepwise Regression:")
print(best_features)

X_train_fs = X_train[best_features]
reg_fs = LinearRegression().fit(X_train_fs, y_train)

print("\nFinal Model Summary after Stepwise Regression:")
print(f"R-squared: {reg_fs.score(X_train_fs, y_train)}")

X_test_fs = X_test[best_features]
y_pred_fs = reg_fs.predict(X_test_fs)
mse_fs = mean_squared_error(y_test, y_pred_fs)
print(f"\nMean Squared Error (MSE) on Test Set after Stepwise Regression: {mse_fs}")

# F-test analysis for final model
X_train_fs_const = sm.add_constant(X_train_fs)
model_fs = sm.OLS(y_train, X_train_fs_const).fit()
print("\nFinal Model F-test Analysis:")
print(f"F-statistic: {model_fs.fvalue}")
print(f"F-pvalue: {model_fs.f_pvalue}")

final_results = {
    'R-squared': reg_fs.score(X_train_fs, y_train),
    'Adjusted R-squared': 1 - (1 - reg_fs.score(X_train_fs, y_train)) * (len(y_train) - 1) / (len(y_train) - X_train_fs.shape[1] - 1),
    'AIC': model_fs.aic,
    'BIC': model_fs.bic,
    'MSE': mse_fs
}

results_df = pd.DataFrame([initial_results, final_results], index=['Initial Model', 'Final Model'])
print("\nComparison of Model Metrics:")
print(results_df)

# Calculate confidence intervals for coefficients using statsmodels
conf_int = model_fs.conf_int()
print("\nConfidence Intervals of the Coefficients:")
print(conf_int)

# T-test analysis for final model
print("\nT-test Analysis for Final Model Coefficients:")
print(model_fs.summary())

# Combine training and testing sets for visualization
total_actual = np.concatenate([y_train, y_test])
total_predicted = np.concatenate([np.full(len(y_train), np.nan), y_pred_fs])

plt.figure(figsize=(12, 6))
plt.plot(range(len(total_actual)), total_actual, linestyle='dotted', label='Actual Selling Price', color='blue')
plt.plot(range(len(total_predicted)), total_predicted, linestyle='--', label='Predicted Selling Price', color='red')
plt.axvline(x=len(y_train), color='green', linestyle='-', label='Train/Test Split')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Selling Price')
plt.title('Actual and Predicted Selling Price')
plt.show()

#%% Phrase-III
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
                             roc_curve, auc, classification_report, ConfusionMatrixDisplay)

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import LocalOutlierFactor
import json

pd.set_option('display.max_columns', None)


df = pd.read_csv('car_prices.csv')
df.drop('vin', axis=1, inplace=True)
df.drop('state', axis=1, inplace=True)

df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows before removal: {duplicates}")
df.drop_duplicates(inplace=True)
duplicates_after = df.duplicated().sum()
print(f"Number of duplicate rows after removal: {duplicates_after}")

df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')
df.dropna(subset=['saledate'], inplace=True)

df['sale_year'] = df['saledate'].apply(lambda x: int(x.strftime('%Y')))
df['sale_month'] = df['saledate'].apply(lambda x: int(x.strftime('%m')))

df['Vehicle_Age'] = abs(df['year'] - (df['sale_year'] + df['sale_month'] / 12))


df = df.drop(['saledate', 'sale_year', 'sale_month', 'year', 'seller', 'color', 'interior'], axis=1)

label_cols = ['model', 'trim', 'make', 'body']
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

scaler_cols = ['model', 'trim', 'make', 'body', 'odometer', 'mmr','sellingprice']
scaler = StandardScaler()
for col in scaler_cols:
    df[[col]] = scaler.fit_transform(df[[col]])

numerical_cols = ['condition', 'model', 'mmr', 'trim', 'make', 'body', 'odometer','sellingprice']

onehot_cols = ['transmission']
df = pd.get_dummies(df, columns=onehot_cols)



X = df.drop(['transmission_automatic', 'transmission_manual'], axis=1)
y = df['transmission_automatic']
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)

print(df.head())
print(skf)

from imblearn.over_sampling import RandomOverSampler
from collections import Counter

print("Original class distribution:", Counter(y))
ros = RandomOverSampler(random_state=5805)
X_res, y_res = ros.fit_resample(X, y)
print("Resampled class distribution:", Counter(y_res))
X, y = X_res, y_res
results = {}

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV,cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
#%%Decision Tree
# Initialize Decision Tree Classifier
# Proper train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5805, stratify=y, shuffle=True
)

dt = DecisionTreeClassifier(random_state=5805)

param_grid_dt = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'max_features': [None, 'sqrt'],
    'ccp_alpha': [0.0, 0.1]
}

grid_search_dt = GridSearchCV(
    estimator=dt,
    param_grid=param_grid_dt,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1
)

grid_search_dt.fit(X_train, y_train)

best_dt = grid_search_dt.best_estimator_
print("Best parameters for Decision Tree:")
print(grid_search_dt.best_params_)

# Evaluate on test set
y_pred_dt = best_dt.predict(X_test)

# Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
disp_dt.plot()
plt.title('Confusion Matrix - Decision Tree')
plt.show()

# Classification Report
report_dt = classification_report(y_test, y_pred_dt)
print("Classification Report for Decision Tree:\n", report_dt)

# Compute Specificity
tn, fp, fn, tp = cm_dt.ravel()
specificity_dt = tn / (tn + fp)
print(f"Specificity: {specificity_dt:.2f}")

# ROC and AUC
y_proba_dt = best_dt.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_proba_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

# Plot ROC Curve
plt.figure()
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend()
plt.show()

# Store results
results['Decision Tree'] = {
    'Accuracy': accuracy_score(y_test, y_pred_dt),
    'Precision': precision_score(y_test, y_pred_dt),
    'Recall': recall_score(y_test, y_pred_dt),
    'Specificity': specificity_dt,
    'F1 Score': f1_score(y_test, y_pred_dt),
    'AUC': roc_auc_dt
}


#%%
#Hyperparameter Tuning with Grid Search
# Initialize Logistic Regression
lr = LogisticRegression(random_state=5805, max_iter=1000)

# Define hyperparameter grid
param_grid_lr = {
    'penalty': ['l2', 'none'],
    'C': [0.1, 1.0, 10.0],
    'solver': ['lbfgs', 'saga']
}
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5805, stratify=y, shuffle=True
)
# Grid Search with Stratified K-Fold
grid_search_lr = GridSearchCV(estimator=lr, param_grid=param_grid_lr, cv=skf, scoring='accuracy', n_jobs=-1)
#grid_search_lr.fit(X, y)
grid_search_lr.fit(X_train, y_train)
# Best estimator
best_lr = grid_search_lr.best_estimator_
print("Best parameters for Logistic Regression:")
print(grid_search_lr.best_params_)
#Evaluation Metrics
# Predict using the best estimator
y_pred_lr = best_lr.predict(X_test)

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr)
disp_lr.plot()
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# Classification Report
report_lr = classification_report(y_test, y_pred_lr)
print("Classification Report for Logistic Regression:\n", report_lr)

# Compute Specificity
tn, fp, fn, tp = cm_lr.ravel()
specificity_lr = tn / (tn + fp)
print(f"Specificity: {specificity_lr:.2f}")

# ROC and AUC
y_proba_lr = best_lr.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Plot ROC Curve
plt.figure()
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.show()

# Store results
results['Logistic Regression'] = {
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'Recall': recall_score(y_test, y_pred_lr),
    'Specificity': specificity_lr,
    'F1 Score': f1_score(y_test, y_pred_lr),
    'AUC': roc_auc_lr
}

#%%
#%%K-Nearest Neighbors (KNN), using the elbow method to find the optimal number of neighbors.
#Finding Optimal K Using Elbow Method
from sklearn.model_selection import StratifiedKFold, GridSearchCV,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
error_rate = []
k_range = range(1, 21, 2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5805, stratify=y, shuffle=True
)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=skf, scoring='accuracy')
    error_rate.append(1 - scores.mean())

# Plotting the elbow graph
plt.figure()
plt.plot(k_range, error_rate, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Error Rate')
plt.xticks(k_range)
plt.show()

# Optimal K is where error rate is minimized
optimal_k = error_rate.index(min(error_rate)) + 1
print(f"Optimal number of neighbors (K): {optimal_k}")
#Hyperparameter Tuning with Grid Search
# Initialize KNN with optimal K
knn = KNeighborsClassifier(n_neighbors=optimal_k)

# Define hyperparameter grid
param_grid_knn = {
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Grid Search with Stratified K-Fold
grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=skf, scoring='accuracy', n_jobs=-1)
grid_search_knn.fit(X_train, y_train)

# Best estimator
best_knn = grid_search_knn.best_estimator_
print("Best parameters for KNN:")
print(grid_search_knn.best_params_)

#Evaluation Metrics
# Predict using the best estimator
y_pred_knn = best_knn.predict(X_test)

# Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot()
plt.title('Confusion Matrix - KNN')
plt.show()

# Classification Report
report_knn = classification_report(y_test, y_pred_knn)
print("Classification Report for KNN:\n", report_knn)

# Compute Specificity
tn, fp, fn, tp = cm_knn.ravel()
specificity_knn = tn / (tn + fp)
print(f"Specificity: {specificity_knn:.2f}")

# ROC and AUC
y_proba_knn = best_knn.predict_proba(X_test)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_proba_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# Plot ROC Curve
plt.figure()
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_knn:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - KNN')
plt.legend()
plt.show()

# Store results
results['KNN'] = {
    'Accuracy': accuracy_score(y_test, y_pred_knn),
    'Precision': precision_score(y_test, y_pred_knn),
    'Recall': recall_score(y_test, y_pred_knn),
    'Specificity': specificity_knn,
    'F1 Score': f1_score(y_test, y_pred_knn),
    'AUC': roc_auc_knn
}


#%%
import json
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
import numpy as np


kernels = ['linear']

# Define hyperparameters for each kernel
hyperparameters = {
    'linear': {'C': 1.0},
    'poly': {'C': 1.0, 'degree': 2, 'gamma': 'scale'},  # Reduced degree
    'rbf': {'C': 1.0, 'gamma': 'scale'}
}

# Split data once


for kernel in kernels:
    print(f"\nSVM with {kernel} kernel")

    # Get hyperparameters
    params = hyperparameters[kernel]

    # Initialize SVM model
    if kernel == 'linear':
        svm = LinearSVC(
            C=params['C'],
            dual=False,
            random_state=5805,
            max_iter=10000
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=5805, stratify=y, shuffle=True
    )
    # Train the model

    svm.fit(X_train, y_train)

    # Predict
    y_pred_svm = svm.predict(X_test)

    print("Used hyperparameters:", params)

    # Confusion Matrix
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
    disp_svm.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - SVM ({kernel} kernel)')
    plt.show()

    # Classification Report
    report_svm = classification_report(y_test, y_pred_svm)
    print(f"Classification Report for SVM ({kernel} kernel):\n", report_svm)

    # Specificity
    if cm_svm.shape == (2, 2):
        tn, fp, fn, tp = cm_svm.ravel()
        specificity_svm = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"Specificity: {specificity_svm:.2f}")
    else:
        print("Specificity is not defined for multi-class classification.")
        specificity_svm = np.nan

    # ROC and AUC (binary classification)
    if len(np.unique(y_test)) == 2:
        if kernel == 'linear':
            # For LinearSVC, use decision_function
            y_scores = svm.decision_function(X_test)
        else:
            y_scores = svm.decision_function(X_test)

        fpr_svm, tpr_svm, _ = roc_curve(y_test, y_scores)
        roc_auc_svm = auc(fpr_svm, tpr_svm)

        # Plot ROC Curve
        plt.figure()
        plt.plot(fpr_svm, tpr_svm, label=f'SVM ({kernel}) (AUC = {roc_auc_svm:.2f})')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - SVM ({kernel} kernel)')
        plt.legend(loc='lower right')
        plt.show()
    else:
        print("ROC AUC is not defined for multi-class classification.")
        roc_auc_svm = np.nan

    # Store results
    results[f'SVM ({kernel})'] = {
        'Accuracy': accuracy_score(y_test, y_pred_svm),
        'Precision': precision_score(y_test, y_pred_svm, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred_svm, average='weighted', zero_division=0),
        'Specificity': specificity_svm,
        'F1 Score': f1_score(y_test, y_pred_svm, average='weighted', zero_division=0),
        'AUC': roc_auc_svm
    }
    print(results[f'SVM ({kernel})'])


import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=5805)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5805, stratify=y, shuffle=True
)

nb = GaussianNB()

y_pred_nb = cross_val_predict(nb, X_test, y_test, cv=skf, method='predict')

cm_nb = confusion_matrix(y_test, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
disp_nb.plot()
plt.title('Confusion Matrix - Naïve Bayes')
plt.show()

report_nb = classification_report(y_test, y_pred_nb)
print("Classification Report for Naïve Bayes:\n", report_nb)

tn, fp, fn, tp = cm_nb.ravel()
specificity_nb = tn / (tn + fp)
print(f"Specificity: {specificity_nb:.2f}")

y_proba_nb = cross_val_predict(nb, X_test, y_test, cv=skf, method='predict_proba')[:, 1]
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_proba_nb)
roc_auc_nb = auc(fpr_nb, tpr_nb)

plt.figure()
plt.plot(fpr_nb, tpr_nb, label=f'Naïve Bayes (AUC = {roc_auc_nb:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Naïve Bayes')
plt.legend()
plt.show()


results['Naïve Bayes'] = {
    'Accuracy': accuracy_score(y_test, y_pred_nb),
    'Precision': precision_score(y_test, y_pred_nb),
    'Recall': recall_score(y_test, y_pred_nb),
    'Specificity': specificity_nb,
    'F1 Score': f1_score(y_test, y_pred_nb),
    'AUC': roc_auc_nb
}


#%%

#%%Bagging Classifier
#Bagging uses bootstrap aggregation to improve stability and accuracy.

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV,cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict


import matplotlib.pyplot as plt
import seaborn as sns
# Initialize base estimator

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5805, stratify=y, shuffle=True
)

base_estimator = DecisionTreeClassifier(random_state=5805)

# Initialize Bagging Classifier
bagging = BaggingClassifier(estimator=base_estimator,
                            n_estimators=50,
                            random_state=5805)

bagging.fit(X_train, y_train)
y_pred_bagging = bagging.predict(X_test)
# Cross-validated predictions


# Confusion Matrix
cm_bagging = confusion_matrix(y_test, y_pred_bagging)
disp_bagging = ConfusionMatrixDisplay(confusion_matrix=cm_bagging)
disp_bagging.plot()
plt.title('Confusion Matrix - Bagging')
plt.show()

# Classification Report
report_bagging = classification_report(y_test, y_pred_bagging)
print("Classification Report for Bagging:\n", report_bagging)

# Compute Specificity
tn, fp, fn, tp = cm_bagging.ravel()
specificity_bagging = tn / (tn + fp)
print(f"Specificity: {specificity_bagging:.2f}")

# ROC and AUC
y_proba_bagging = bagging.predict_proba(X_test)[:, 1]
fpr_bagging, tpr_bagging, _ = roc_curve(y_test, y_proba_bagging)
roc_auc_bagging = auc(fpr_bagging, tpr_bagging)

# Plot ROC Curve
plt.figure()
plt.plot(fpr_bagging, tpr_bagging, label=f'Bagging (AUC = {roc_auc_bagging:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Bagging')
plt.legend()
plt.show()

# Store results
results['Bagging'] = {
    'Accuracy': accuracy_score(y_test, y_pred_bagging),
    'Precision': precision_score(y_test, y_pred_bagging),
    'Recall': recall_score(y_test, y_pred_bagging),
    'Specificity': specificity_bagging,
    'F1 Score': f1_score(y_test, y_pred_bagging),
    'AUC': roc_auc_bagging
}

#%%
#%%AdaBoost (Boosting)
# AdaBoost is a boosting technique that adjusts weights iteratively to minimize errors.

# Initialize base estimator for AdaBoost
base_estimator_ada = DecisionTreeClassifier(max_depth=1, random_state=5805)

# Initialize AdaBoost Classifier
ada = AdaBoostClassifier(
    estimator=base_estimator_ada,
    n_estimators=100,
    random_state=5805
)

# Split data into training and testing sets (consistent with Bagging split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5805, stratify=y, shuffle=True
)

# Fit AdaBoost model
ada.fit(X_train, y_train)

# Predictions for test set
y_pred_ada = ada.predict(X_test)

# Confusion Matrix
cm_ada = confusion_matrix(y_test, y_pred_ada)
disp_ada = ConfusionMatrixDisplay(confusion_matrix=cm_ada)
disp_ada.plot()
plt.title('Confusion Matrix - AdaBoost')
plt.show()

# Classification Report
report_ada = classification_report(y_test, y_pred_ada)
print("Classification Report for AdaBoost:\n", report_ada)

# Compute Specificity
tn, fp, fn, tp = cm_ada.ravel()
specificity_ada = tn / (tn + fp)
print(f"Specificity: {specificity_ada:.2f}")

# ROC and AUC
y_proba_ada = ada.predict_proba(X_test)[:, 1]
fpr_ada, tpr_ada, _ = roc_curve(y_test, y_proba_ada)
roc_auc_ada = auc(fpr_ada, tpr_ada)

# Plot ROC Curve
plt.figure()
plt.plot(fpr_ada, tpr_ada, label=f'AdaBoost (AUC = {roc_auc_ada:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - AdaBoost')
plt.legend()
plt.show()

# Store results
results['AdaBoost'] = {
    'Accuracy': accuracy_score(y_test, y_pred_ada),
    'Precision': precision_score(y_test, y_pred_ada),
    'Recall': recall_score(y_test, y_pred_ada),
    'Specificity': specificity_ada,
    'F1 Score': f1_score(y_test, y_pred_ada),
    'AUC': roc_auc_ada
}


#%%
#%%Neural Network (Multi-layer Perceptron)
#Hyperparameter Tuning with Grid Search
# Initialize MLP Classifier
mlp = MLPClassifier(random_state=5805, max_iter=1000)

# Define hyperparameter grid
param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.001],
    'learning_rate': ['adaptive']
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5805, stratify=y, shuffle=True
)
# Grid Search with Stratified K-Fold
grid_search_mlp = GridSearchCV(estimator=mlp, param_grid=param_grid_mlp, cv=skf, scoring='accuracy', n_jobs=-1)
grid_search_mlp.fit(X_train, y_train)

# Best estimator
best_mlp = grid_search_mlp.best_estimator_
print("Best parameters for MLP Classifier:")
print(grid_search_mlp.best_params_)

#Evaluation Metrics
# Predict using the best estimator
y_pred_mlp = best_mlp.predict(X_test)

# Confusion Matrix
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
disp_mlp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp)
disp_mlp.plot()
plt.title('Confusion Matrix - Neural Network')
plt.show()

# Classification Report
report_mlp = classification_report(y_test, y_pred_mlp)
print("Classification Report for Neural Network:\n", report_mlp)

# Compute Specificity
tn, fp, fn, tp = cm_mlp.ravel()
specificity_mlp = tn / (tn + fp)
print(f"Specificity: {specificity_mlp:.2f}")

# ROC and AUC
y_proba_mlp = best_mlp.predict_proba(X_test)[:, 1]
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_proba_mlp)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

# Plot ROC Curve
plt.figure()
plt.plot(fpr_mlp, tpr_mlp, label=f'Neural Network (AUC = {roc_auc_mlp:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Neural Network')
plt.legend()
plt.show()

# Store results
results['Neural Network'] = {
    'Accuracy': accuracy_score(y_test, y_pred_mlp),
    'Precision': precision_score(y_test, y_pred_mlp),
    'Recall': recall_score(y_test, y_pred_mlp),
    'Specificity': specificity_mlp,
    'F1 Score': f1_score(y_test, y_pred_mlp),
    'AUC': roc_auc_mlp
}


#%%Create a Summary Table
# Create DataFrame from results
results_df = pd.DataFrame(results).T
print("Performance Metrics for Each Classifier:")
print(results_df)
#%%
#Visualize the Comparison
# Plotting metrics for comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'AUC']

results_df[metrics].plot(kind='bar', figsize=(12, 8))
plt.title('Classifier Performance Comparison')
plt.ylabel('Scores')
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


#%%

#%%Recommendation
#Based on the performance metrics, we can identify the classifier with the highest scores.
# Find the best classifier based on Accuracy
best_classifier = results_df['Accuracy'].idxmax()
print(f"The recommended classifier is: {best_classifier}")


#%%Graphical Representation of Classification Results
#Confusion Matrix Heatmap for the Best Classifier
# Retrieve the confusion matrix of the best classifier
if best_classifier == 'Decision Tree':
    cm_best = cm_dt
elif best_classifier == 'Logistic Regression':
    cm_best = cm_lr
elif best_classifier == 'KNN':
    cm_best = cm_knn
elif best_classifier == 'SVM':
    cm_best = cm_svm
elif best_classifier == 'Naïve Bayes':
    cm_best = cm_nb
elif best_classifier == 'Bagging':
    cm_best = cm_bagging
elif best_classifier == 'AdaBoost':
    cm_best = cm_ada
elif best_classifier == 'Neural Network':
    cm_best = cm_mlp


# Plot heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_classifier}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#%Plotting ROC Curves Together
# Plot ROC Curves of all classifiers
plt.figure(figsize=(10, 8))

# Decision Tree
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.2f})')

# Logistic Regression
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')

# KNN
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_knn:.2f})')

# SVM Kernels
for kernel in kernels:
    if kernel == 'linear':
        plt.plot(fpr_svm, tpr_svm, label=f'SVM (linear) (AUC = {roc_auc_svm:.2f})')

# Naïve Bayes
plt.plot(fpr_nb, tpr_nb, label=f'Naïve Bayes (AUC = {roc_auc_nb:.2f})')

# Bagging
plt.plot(fpr_bagging, tpr_bagging, label=f'Bagging (AUC = {roc_auc_bagging:.2f})')

# AdaBoost
plt.plot(fpr_ada, tpr_ada, label=f'AdaBoost (AUC = {roc_auc_ada:.2f})')


# Neural Network
plt.plot(fpr_mlp, tpr_mlp, label=f'Neural Network (AUC = {roc_auc_mlp:.2f})')

plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves of All Classifiers')
plt.legend()
plt.show()


#%% Phrase-IV
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn for clustering and preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
# mlxtend for Apriori algorithm
from mlxtend.frequent_patterns import apriori, association_rules

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

#%%
df = pd.read_csv('car_prices.csv')
df.drop('vin', axis=1, inplace=True)
df.drop('state', axis=1, inplace=True)

df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows before removal: {duplicates}")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Verify removal
duplicates_after = df.duplicated().sum()
print(f"Number of duplicate rows after removal: {duplicates_after}")

df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')
df.dropna(subset=['saledate'], inplace=True)

# Extract year using lambda function
df['sale_year'] = df['saledate'].apply(lambda x: int(x.strftime('%Y')))

# Extract month using lambda function
df['sale_month'] = df['saledate'].apply(lambda x: int(x.strftime('%m')))

df['Vehicle_Age'] = abs(df['year'] - (df['sale_year'] + df['sale_month'] / 12))

df = df.drop(['saledate', 'sale_year','sale_month','year'], axis=1)


print(df.head())
numerical_features = ['Vehicle_Age', 'condition', 'odometer', 'mmr', 'sellingprice']
df_num = df[numerical_features]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num)



print(X_scaled)

#%%
print(f"Shape of X_scaled: {X_scaled.shape}")
#%%
#K-Means Clustering with Silhouette Analysis and Elbow Method
#Determine the Optimal Number of Clusters (k):
# Verify the shape of the data


# Adjust K based on the number of samples
n_samples = X_scaled.shape[0]
max_k = min(10, n_samples - 1)  # Max clusters can't exceed n_samples - 1
K = range(2, max_k + 1)

wcss = []  # Within-cluster sum of squares
silhouette_scores = []

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=5805)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    labels = kmeans.labels_
    # Check if the number of unique labels is greater than 1
    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(X_scaled, labels)
        silhouette_scores.append(silhouette_avg)
    else:
        # Silhouette score cannot be computed if there is only one cluster
        silhouette_scores.append(np.nan)


# Plot the Elbow Method
plt.figure(figsize=(10, 5))
plt.plot(K, wcss, 'bx-')
plt.xlabel('Number of clusters k')
plt.ylabel('Within-cluster sum of squares (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.show()

# Plot the Silhouette Scores
plt.figure(figsize=(8, 4))
plt.plot(K, silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Average Silhouette Score')
plt.title('Silhouette Analysis for Optimal k')
plt.show()

#%%
# Apply K-Means with Optimal k
optimal_k = K[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters based on silhouette score: {optimal_k}")

# Apply K-Means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the dataset
df['Cluster_KMeans'] = clusters

# Display cluster counts
print("Cluster counts:\n", df['Cluster_KMeans'].value_counts())

# Visualize Clusters using PCA
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

# Create a DataFrame for visualization
vis_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
vis_df['Cluster_KMeans'] = clusters

# Plot the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=vis_df, x='PC1', y='PC2', hue='Cluster_KMeans', palette='viridis')
plt.title('K-Means Clusters Visualization')
plt.show()

# DBSCAN Clustering
# Find the optimal epsilon value
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

# Sort distances and plot k-distance graph
distances = np.sort(distances[:, 4], axis=0)
plt.figure(figsize=(8, 4))
plt.plot(distances)
plt.xlabel('Data Points sorted by distance')
plt.ylabel('Epsilon')
plt.title('K-Distance Graph for Epsilon Determination')
plt.show()

#%%
# Based on the graph, choose an appropriate eps value
# For example, eps = 0.5

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
db_clusters = dbscan.fit_predict(X_scaled)

# Add cluster labels to the dataset
df['Cluster_DBSCAN'] = db_clusters

# Display cluster counts
print("Cluster counts:\n", df['Cluster_DBSCAN'].value_counts())

# Visualize Clusters
# Add DBSCAN cluster labels to vis_df
vis_df['Cluster_DBSCAN'] = db_clusters

# Plot the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=vis_df, x='PC1', y='PC2', hue='Cluster_DBSCAN', palette='viridis')
plt.title('DBSCAN Clusters Visualization')
plt.show()

# Apriori Algorithm for Association Rule Mining
# Define categorical columns
categorical_cols = [col for col in df.columns if col not in numerical_features + ['Cluster_KMeans', 'Cluster_DBSCAN']]

# Select categorical features for association analysis
assoc_data = df[categorical_cols].copy()

# Convert boolean columns to strings (if any)
for col in assoc_data.select_dtypes(include='bool').columns:
    assoc_data[col] = assoc_data[col].astype(str)

# Convert data to one-hot encoded format
onehot_data = pd.get_dummies(assoc_data)

# Display the prepared data
print(onehot_data.head())

# Apply the Apriori algorithm
frequent_itemsets = apriori(onehot_data, min_support=0.05, use_colnames=True)

# Display frequent itemsets
print("Frequent Itemsets:\n", frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)

# Sort rules by lift
rules = rules.sort_values(by='lift', ascending=False)

# Display the rules
print("Association Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Interpretation of Rules
# Function to convert frozensets to strings for better readability
def frozenset_to_str(fset):
    return ', '.join(list(fset))

# Apply the function to antecedents and consequents
rules['antecedents'] = rules['antecedents'].apply(frozenset_to_str)
rules['consequents'] = rules['consequents'].apply(frozenset_to_str)

# Display the top 10 rules
print("Top 10 Association Rules:\n", rules.head(10))