import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score, roc_curve, auc

pd.set_option('display.precision', 2)

np.random.seed(5805)

#%%
print('1')
def entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def gini(p):
    return 2 * p * (1 - p)

def plot_entropy_gini():
    p = np.linspace(0.01, 0.99, 200)
    entropy_vals = entropy(p)
    gini_vals = gini(p)

    plt.figure(figsize=(10, 6))
    plt.plot(p, entropy_vals, label="Entropy", linewidth=3, color='blue')
    plt.plot(p, gini_vals, label="Gini", linewidth=3, color='orange')
    plt.xlabel("p", fontsize=14)
    plt.ylabel("Magnitude", fontsize=14)
    plt.title("Entropy versus Gini Index", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.show()

plot_entropy_gini()

#%%Phase II
print('4')

# Load the dataset
df = sns.load_dataset('titanic')
df.dropna(how='any', inplace=True)

numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
X = df[numerical_features].drop(columns=['survived'])
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805, stratify=y)

decision_tree = DecisionTreeClassifier(random_state=5805)
decision_tree.fit(X_train, y_train)

train_accuracy = accuracy_score(y_train, decision_tree.predict(X_train))
test_accuracy = accuracy_score(y_test, decision_tree.predict(X_test))

print(f'Train Accuracy: {train_accuracy:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')

print('Decision Tree Parameters:')
print(decision_tree.get_params())

plt.figure(figsize=(20,10))
plot_tree(decision_tree, filled=True, feature_names=X.columns, class_names=['Not Survived', 'Survived'])
plt.show()

#%%

#%%
print('5')
tuned_parameters = [{'max_depth': [1, 2, 3, 4, 5],
                     'min_samples_split': [20, 30, 40],
                     'min_samples_leaf': [10, 20, 30],
                     'criterion': ['gini', 'entropy', 'log_loss'],
                     'splitter': ['best', 'random'],
                     'max_features': ['sqrt', 'log2']}]

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=5805), tuned_parameters, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_decision_tree = grid_search.best_estimator_

print('Best Parameters:')
print(best_params)

train_accuracy_pruned = accuracy_score(y_train, best_decision_tree.predict(X_train))
test_accuracy_pruned = accuracy_score(y_test, best_decision_tree.predict(X_test))

print(f'Train Accuracy (Pruned): {train_accuracy_pruned:.2f}')
print(f'Test Accuracy (Pruned): {test_accuracy_pruned:.2f}')

plt.figure(figsize=(20,10))
plot_tree(best_decision_tree, filled=True, feature_names=X.columns, class_names=['Not Survived', 'Survived'])
plt.show()

#%%

#%%Post-Pruning
print('6')
path = decision_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=5805, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

train_scores = [accuracy_score(y_train, clf.predict(X_train)) for clf in clfs]
test_scores = [accuracy_score(y_test, clf.predict(X_test)) for clf in clfs]

plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, marker='o', label='Train', drawstyle="steps-post")
plt.plot(ccp_alphas, test_scores, marker='o', label='Test', drawstyle="steps-post")
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Alpha for Training and Testing sets')
plt.legend()
plt.show()

optimal_alpha_index = test_scores.index(max(test_scores))
optimal_alpha = ccp_alphas[optimal_alpha_index]
print(f'Optimal alpha: {optimal_alpha:.2f}')

post_pruned_tree = DecisionTreeClassifier(random_state=5805, ccp_alpha=optimal_alpha)
post_pruned_tree.fit(X_train, y_train)
train_accuracy_post_pruned = accuracy_score(y_train, post_pruned_tree.predict(X_train))
print(f'Training accuracy of the post-pruned tree: {train_accuracy_post_pruned:.2f}')
test_accuracy_post_pruned = accuracy_score(y_test, post_pruned_tree.predict(X_test))
print(f'Test accuracy of the post-pruned tree: {test_accuracy_post_pruned:.2f}')
plt.figure(figsize=(20,10))
plot_tree(post_pruned_tree, filled=True, feature_names=X.columns, class_names=['Not Survived', 'Survived'])
plt.show()

#%%
print('7')
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression(random_state=5805)
logistic_regression.fit(X_train, y_train)

train_accuracy_lr = accuracy_score(y_train, logistic_regression.predict(X_train))
test_accuracy_lr = accuracy_score(y_test, logistic_regression.predict(X_test))

print(f'Logistic Regression Train Accuracy: {train_accuracy_lr:.2f}')
print(f'Logistic Regression Test Accuracy: {test_accuracy_lr:.2f}')

#%%
print('8')
train_accuracy_pre_pruned = train_accuracy_pruned
test_accuracy_pre_pruned = test_accuracy_pruned
y_pred_train_pre_pruned = clf.predict(X_train)
y_pred_test_pre_pruned = clf.predict(X_test)
conf_matrix_pre_pruned = confusion_matrix(y_test, y_pred_test_pre_pruned)
recall_pre_pruned = recall_score(y_test, y_pred_test_pre_pruned)
fpr_pre_pruned, tpr_pre_pruned, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
roc_auc_pre_pruned = auc(fpr_pre_pruned, tpr_pre_pruned)

y_pred_train_post_pruned = post_pruned_tree.predict(X_train)
y_pred_test_post_pruned = post_pruned_tree.predict(X_test)
conf_matrix_post_pruned = confusion_matrix(y_test, y_pred_test_post_pruned)
recall_post_pruned = recall_score(y_test, y_pred_test_post_pruned)
fpr_post_pruned, tpr_post_pruned, _ = roc_curve(y_test, post_pruned_tree.predict_proba(X_test)[:, 1])
roc_auc_post_pruned = auc(fpr_post_pruned, tpr_post_pruned)

y_pred_train_lr = logistic_regression.predict(X_train)
y_pred_test_lr = logistic_regression.predict(X_test)
conf_matrix_lr = confusion_matrix(y_test, y_pred_test_lr)
recall_lr = recall_score(y_test, y_pred_test_lr)
fpr_lr, tpr_lr, _ = roc_curve(y_test, logistic_regression.predict_proba(X_test)[:, 1])
roc_auc_lr = auc(fpr_lr, tpr_lr)

metrics_data = {
    'Classifier': ['DT Pre-Pruned', 'DT Post-Pruned', 'Logistic Regression'],
    'Train Accuracy': [train_accuracy_pre_pruned, train_accuracy_post_pruned, train_accuracy_lr],
    'Test Accuracy': [test_accuracy_pre_pruned, test_accuracy_post_pruned, test_accuracy_lr],
    'Recall': [recall_pre_pruned, recall_post_pruned, recall_lr],
    'AUC': [roc_auc_pre_pruned, roc_auc_post_pruned, roc_auc_lr]
}

metrics_df = pd.DataFrame(metrics_data)
print(metrics_df)

plt.figure(figsize=(10, 6))
plt.plot(fpr_pre_pruned, tpr_pre_pruned, label=f'DT Pre-Pruned (AUC = {roc_auc_pre_pruned:.2f})')
plt.plot(fpr_post_pruned, tpr_post_pruned, label=f'DT Post-Pruned (AUC = {roc_auc_post_pruned:.2f})')
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

best_classifier = metrics_df.loc[metrics_df['Test Accuracy'].idxmax()]['Classifier']
print(f'The best classifier for this dataset is: {best_classifier}')
