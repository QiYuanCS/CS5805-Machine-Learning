import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.decomposition import PCA
#from statsmodels.tools.eval_measures import aic, bic

pd.set_option('display.precision', 3)
pd.set_option('display.max_columns', None)
np.random.seed(5805)

url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/Carseats.csv'
data = pd.read_csv(url)

#%%
print('#1-a')
sales_df = data.groupby(['ShelveLoc', 'US'])['Sales'].sum().unstack()

sales_df.plot(kind='barh', stacked=True)
plt.xlabel('Sales')
plt.ylabel('ShelveLoc')
plt.title('The ‘ShelveLoc’ versus Sales with respect to the ‘US’')
plt.grid()
plt.legend(title='US', loc='best')
plt.tight_layout()
plt.show()

#%%
print('#1-b')
df_encoded = pd.get_dummies(data, columns= ['ShelveLoc', 'Urban', 'US'], drop_first=True)
print(df_encoded.head())

#%%
print('#1-c')
df_encoded = pd.get_dummies(data, columns= ['ShelveLoc', 'US'], drop_first=True)

X = df_encoded.drop(columns=['Sales'])
y = df_encoded['Sales']
y = pd.to_numeric(y, errors='coerce')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=5805)
feature_scaled = ['CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'Age', 'Education']

scaler = StandardScaler()
X_train[feature_scaled] = scaler.fit_transform(X_train[feature_scaled])
X_test[feature_scaled] = scaler.transform(X_test[feature_scaled])

print("First 5 rows of X_train: ")
print(X_train.head())
print("\nFirst 5 rows of X_test: ")
print(X_test.head())

#%%
print('2-a')
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)

X_train_const = sm.add_constant(X_train)
X_train_const = X_train_const.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

model = sm.OLS(y_train, X_train_const).fit()
elimination_process = []
features = X_train_const.columns.tolist()

while len(features) > 1:
    pvalues = model.pvalues
    max_pvalue = pvalues.max()
    if max_pvalue < 0.01:
        break
    feature_to_remove = pvalues.idxmax()
    features.remove(feature_to_remove)
    X_train_const = X_train_const[features]
    model = sm.OLS(y_train, X_train_const).fit()
    elimination_process.append({
        'Eliminated Feature': feature_to_remove,
        'p-value': round(max_pvalue, 3),
        'AIC': round(model.aic, 3),
        'BIC': round(model.bic, 3),
        'Adjusted R-squared': round(model.rsquared_adj, 3)
    })
elimination_df = pd.DataFrame(elimination_process)
print("\nFeature Elimination Process:")
print(elimination_df)
print("\nFinal Selected Features:")
print(features)

#%%
print('2-b')
print(X_train.head())
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)
X_train_const = sm.add_constant(X_train)
X_train_const = X_train_const.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
model = sm.OLS(y_train, X_train_const).fit()
elimination_process = []
features = X_train_const.columns.tolist()

while len(features) > 1:
    pvalues = model.pvalues
    max_pvalue = pvalues.max()
    if max_pvalue < 0.01:
        break
    feature_to_remove = pvalues.idxmax()
    features.remove(feature_to_remove)
    X_train_const = X_train_const[features]
    model = sm.OLS(y_train, X_train_const).fit()
    elimination_process.append({
        'Eliminated Feature': feature_to_remove,
        'p-value': round(max_pvalue, 3),
        'AIC': round(model.aic, 3),
        'BIC': round(model.bic, 3),
        'Adjusted R-squared': round(model.rsquared_adj, 3)
    })
    print(f"\nOLS Regression Summary after eliminating '{feature_to_remove}':\n")
    print(model.summary())

elimination_df = pd.DataFrame(elimination_process)
print("\nFeature Elimination Process:")
print(elimination_df)
print("\nFinal Selected Features:")
print(features)

#%%
print('2-c')
final_features = ['CompPrice', 'Income', 'Advertising', 'Price', 'Age', 'ShelveLoc_Good', 'ShelveLoc_Medium']

X_train_final = sm.add_constant(X_train_const[final_features])
final_model = sm.OLS(y_train, X_train_final).fit()
print("Final OLS Regression Summary:\n")
print(final_model.summary())

X_test_final = sm.add_constant(X_test[final_features])
y_predict = final_model.predict(X_test_final)

y_test = y_test.reset_index(drop=True)
y_predict = y_predict.reset_index(drop=True)
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Sales', marker='o', linestyle='-')
plt.plot(y_predict, label='Predicted Sales', marker='x', linestyle='--')
plt.title('Actual Sales vs Predicted Sales')
plt.xlabel('Sample Index')
plt.ylim(-2.5, 16)
plt.ylabel('Sales')
plt.legend()
plt.show()

#%%
print('2-d')
mse = mean_squared_error(y_test, y_predict)
print(f"Test Set Mean Squared Errof: {mse:.3f}")

#%%
print('3-a')
df_encoded = pd.get_dummies(data, columns=['ShelveLoc', 'US', 'Urban'], drop_first=True)

X = df_encoded.drop(columns=['Sales'])
y = df_encoded['Sales']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
pca.fit(X_scaled)
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(explained_variance_ratio >= 0.95) + 1
print(f'Number of components needed to explain more than 95% of the variance: {n_components_95}')

#%%
print('3-b')
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='-')
plt.title('The cumulative explained variance versus the number of features')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative explained variance')
plt.grid()
plt.show()

#%%
print('3-c')
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
plt.axhline(y = 0.95, color='r', linestyle='--', label='95% Explained Variance')
n_components_95 = np.argmax(explained_variance_ratio >= 0.95) + 1
plt.axvline(x=n_components_95, color='g', linestyle='--', label=f'{n_components_95} components')
plt.title('The cumulative explained variance versus the number of features')
plt.xlabel('Number of Features')
plt.ylabel('Cumulative explained variance')
plt.legend()
plt.grid()
plt.show()

#%%
print('4-a')
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
feature_importances = rf.feature_importances_
features = X.columns
sorted_idx = np.argsort(feature_importances)[::-1]
sorted_features = features[sorted_idx]
sorted_importances = feature_importances[sorted_idx]

plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Features Importance from Random Forest')
plt.gca().invert_yaxis()
plt.show()

#%%
print('4-b')
threshold = 0.05
selected_features_rf = sorted_features[sorted_importances > threshold]

eliminated_features_rf = sorted_features[sorted_importances <= threshold]

print("Selected Features (Random Forest):", selected_features_rf)
print("Eliminated Features (Random Forest):", eliminated_features_rf)

selected_features_stepwise = ['CompPrice', 'Price', 'ShelveLoc_Good']
print("Are the selected features identical between Random Forest and Stepwise Regression?")

if set(selected_features_rf) == set(selected_features_stepwise):
    print("Yes, the selected features are identical.")
else:
    print("No, the selected features are different.")

#%%
print('4-c')
import statsmodels.api as sm
X_train_rf_selected = X_train[selected_features_rf]
X_test_rf_selected = X_test[selected_features_rf]

X_train_rf_const = sm.add_constant(X_train_rf_selected)
X_train_rf_const = X_train_rf_const.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

ols_model_rf = sm.OLS(y_train, X_train_rf_const).fit()
print(ols_model_rf.summary())

#%%
print('4-d')
X_test_rf_selected = X_test[selected_features_rf]
X_test_rf_const = sm.add_constant(X_test_rf_selected)
X_test_rf_const = X_test_rf_const.reset_index(drop=True)
y_predict_rf = ols_model_rf.predict(X_test_rf_const)
y_test = y_test.reset_index(drop=True)

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Sales', marker='o', linestyle='-', color='b')
plt.plot(y_predict_rf, label='Predicted Sales', marker='x', linestyle='--', color='r')
plt.title('Actual Sales vs Predicted Sales')
plt.xlabel('Sample Index')
plt.ylabel('Sales')
plt.legend()
plt.show()

#%%
print('4-e')
from sklearn.metrics import mean_squared_error
mse_rf = mean_squared_error(y_test, y_predict_rf)
print(f"Test Set Mean Squared Error (Random Forest Selected Features): {mse_rf:.3f}")


#%%
print('5')
r_squared_step2 = final_model.rsquared
adj_r_squared_step2 = final_model.rsquared_adj
aic_step2 = final_model.aic
bic_step2 = final_model.bic
mse_step2 = mse

r_squared_step4 = ols_model_rf.rsquared
adj_r_squared_step4 = ols_model_rf.rsquared_adj
aic_step4 = ols_model_rf.aic
bic_step4 = ols_model_rf.bic
mse_step4 = mse_rf

comparison_df = pd.DataFrame({
    'Metric': ['R-squared', 'Adjusted R-squared', 'AIC', 'BIC', 'MSE'],
    'Step 2 Stepwise Regression': [r_squared_step2, adj_r_squared_step2, aic_step2, bic_step2, mse_step2],
    'Step 4 Random Forest Feature Selection': [r_squared_step4, adj_r_squared_step4,aic_step4, bic_step4, mse_step4]
})

comparison_df.set_index('Metric', inplace = True)
comparison_df = comparison_df.round(3)

print("\n Comparison of Models from step2 and step4:")
print(comparison_df.to_string())

#%%
print('6')
X_test_selected = X_test[final_features]
X_test_selected_const = sm.add_constant(X_test_selected)
X_test_final = X_test_selected_const.reset_index(drop=True)

y_test = y_test.reset_index(drop=True)
predictions = final_model.get_prediction(X_test_final)
prediction_summary = predictions.summary_frame(alpha = 0.05)

plt.figure(figsize=(12, 6))
plt.plot(prediction_summary.index, prediction_summary['mean'], label='Predicted Sales', color='blue', linestyle='-')
plt.fill_between(prediction_summary.index, prediction_summary['obs_ci_lower'], prediction_summary['obs_ci_upper'], color='blue', alpha=0.2, label='CI')
plt.title('Sales prediction with Confidence Interval')
plt.xlabel('# of Samples')
plt.ylabel('Sales USD($)')
plt.ylim(-2.5, 16)
plt.legend()
plt.grid()
plt.show()

#%%
print('4-b')
model = SelectFromModel(rf, threshold=0.05, prefit=True)
X_train_scaled_rf_selected = model.transform(X_train_scaled)
selected_features_rf = X_train_scaled.columns[model.get_support()]

selected_features_rf = selected_features_rf.tolist()
X_train_scaled_rf_selected_const = sm.add_constant(X_train_scaled_rf_selected)
ols_model = sm.OLS(y_train, X_train_scaled_rf_selected_const).fit()
stepwise_selected_features = list(ols_model.pvalues[ols_model.pvalues < 0.01].index)

stepwise_selected_features_no_const = [f for f in stepwise_selected_features if f != 'const']

eliminated_rf = list(set(X_train_scaled.columns) - set(selected_features_rf))
eliminated_stepwise = list(set(selected_features_rf) - set(stepwise_selected_features_no_const))

print("Random Forest Selected Features:")
print(selected_features_rf)

print("\nRandom Forest Eliminated Features:")
print(eliminated_rf)

print("\nStepwise Regression Selected Features:")
print(stepwise_selected_features_no_const)

print("\nStepwise Regression Eliminated Features:")
print(eliminated_stepwise)

rf_selected_set = set(selected_features_rf)
stepwise_selected_set = set(stepwise_selected_features_no_const)

if rf_selected_set == stepwise_selected_set:
    print("\nThe selected features by Random Forest and Stepwise Regression are identical.")
else:
    print("\nThe selected features by Random Forest and Stepwise Regression are NOT identical.")

