import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.precision', 3)
pd.set_option('display.max_columns', None)
np.random.seed(5805)
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/Carseats.csv'
data = pd.read_csv(url)

#%%
print('1-a')
agg_sales = data.groupby(['ShelveLoc', 'US'])['Sales'].sum().reset_index()
sales_yes = agg_sales[agg_sales['US'] == 'Yes']
sales_no = agg_sales[agg_sales['US'] == 'No']

shelve_locs = sales_yes['ShelveLoc'].values
bar_width = 0.35
index = np.arange(len(shelve_locs))
plt.barh(index, sales_yes['Sales'], bar_width, label='US=Yes', color='b')
plt.barh(index + bar_width, sales_no['Sales'], bar_width, label='US=No', color='r')

plt.xlabel('Total Sales')
plt.ylabel('Shelve Location')
plt.title('Total Sales by Shelve Location and US')
plt.yticks(index + bar_width / 2, shelve_locs)
plt.legend()
plt.tight_layout()
plt.show()

total_sales_by_shelveloc = agg_sales.groupby('ShelveLoc')['Sales'].sum()
best_shelve_loc = total_sales_by_shelveloc.idxmax()
best_sales = total_sales_by_shelveloc.max()

print(f"The Shelve location with the highest sales is: {best_shelve_loc} with total sales of {best_sales:.3f}")
total_sales_us = agg_sales.groupby('US')['Sales'].sum()

if total_sales_us['Yes'] > total_sales_us['No']:
    print(f"The highest sales are inside the US with total sales of {total_sales_us['Yes']:.3f}.")
else:
    print(f"The highest sales are outside the US with total sales of {total_sales_us['No']:.3f}.")
#%%
print('1-b')
qualitative_features = ['ShelveLoc', 'Urban', 'US']
df_encoded = pd.get_dummies(data, columns=qualitative_features, drop_first=True)
df_encoded = df_encoded.astype(int)
print(df_encoded.head())

#%%
print('1-c')
X = df_encoded.drop(columns=['Sales'])
y = df_encoded['Sales']
y = pd.to_numeric(y, errors='coerce')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=5805)
encoded_features = [col for col in X.columns if '_' in col]
numeric_features = [col for col in X.columns if col not in encoded_features]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

print("First 5 rows of X_train: ")
print(X_train_scaled.head())
print("\nFirst 5 rows of X_test: ")
print(X_test_scaled.head())

#%%
print('2-a')
X_train_scaled = sm.add_constant(X_train_scaled)
features = X_train_scaled.columns.tolist()
elimination_results = []

while True:
    model = sm.OLS(y_train, X_train_scaled[features]).fit()
    print(model.summary())
    pvalues = model.pvalues.drop('const')
    max_pvalue = pvalues.max()
    if max_pvalue > 0.01:
        feature_to_remove = pvalues.idxmax()
        features.remove(feature_to_remove)
        elimination_results.append({
            'Eliminated Feature': feature_to_remove,
            'Adjusted R-squared': model.rsquared_adj,
            'AIC': model.aic,
            'BIC': model.bic,
            'Max P-value': max_pvalue
        })
    else:
        break

elimination_df = pd.DataFrame(elimination_results)
print(elimination_df)
eliminated_features = elimination_df['Eliminated Feature'].tolist()
features = [feature for feature in features if feature != 'const']
print('\nEliminated Features:', eliminated_features)
print('Final Selected Features:', features)

#%%
print('2-b')
X_train_scaled = sm.add_constant(X_train_scaled)
features = X_train_scaled.columns.tolist()
eliminated_results = []
threshold = 0.01

while True:
    model = sm.OLS(y_train, X_train_scaled[features]).fit()
    print(model.summary())
    pvalues = model.pvalues
    max_pvalue = pvalues.max()

    if max_pvalue > threshold:
        feature_to_remove = pvalues.idxmax()
        features.remove(feature_to_remove)
        eliminated_results.append({
            'Eliminated Feature': feature_to_remove,
            'Adjusted R-squared': model.rsquared_adj,
            'AIC': model.aic,
            'BIC': model.bic,
            'Max P-value': max_pvalue
        })
    else:
        break

elimination_df = pd.DataFrame(eliminated_results)
print(elimination_df)
features = [feature for feature in features if feature != 'const']
eliminated_features = elimination_df['Eliminated Feature'].tolist()
print('\nEliminated Features:', eliminated_features)
print('Final Selected Features:', features)

#%%
print('2-c')

X_train_scaled = sm.add_constant(X_train_scaled)
features_const = ['const'] + features
final_model = sm.OLS(y_train, X_train_scaled[features_const]).fit()
summary1 = final_model.summary().tables[0]
print(summary1)
summary_df = final_model.summary2().tables[1]
print(summary_df.round(3))

X_test_scaled = sm.add_constant(X_test_scaled)
y_prediction = final_model.predict(X_test_scaled[features_const])
print("\nPredicted values:")
print(y_prediction)
print("\nTest set values")
print(y_test)
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Sales', marker='o')
plt.plot(y_prediction.values, label='Predicted Sales', marker='o')
plt.xlabel('Index')
plt.ylabel('Sales')
plt.title('Actual Sales vs Predicted Sales')
plt.legend()
plt.tight_layout()
plt.show()

#%%
print('2-d')
mse = mean_squared_error(y_test.values, y_prediction.values)
print(f"Mean Squared Error on the test set: {mse:.3f}")

#%%
print('3-a')
pca = PCA()
pca.fit(X_train_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100
n_components = np.argmax(cumulative_variance >= 95) + 1
print(f"Number of principal components needed to explain at least 95% variance: {n_components}")


#%%
print('3-b')
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.grid(True)
plt.show()
#%%
print('3-c')
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.axhline(y=95, color ='r', linestyle='--')
plt.axvline(x=n_components, color ='r', linestyle='--')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.grid(True)
plt.show()

#%%
print('4-a')
rf = RandomForestRegressor(n_estimators=100, random_state=0)
X_train_scaled = X_train_scaled.loc[:,X_train_scaled.columns != 'const']
rf.fit(X_train_scaled, y_train)
importances  = rf.feature_importances_


feature_importance_df = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title('Feature Importance in Random Forest')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
print(feature_importance_df)

#%%
print('4-b')
X_train_scaled_with_const = sm.add_constant(X_train_scaled)
features = X_train_scaled_with_const.columns.tolist()
eliminated_features_reg = []
while True:
    model = sm.OLS(y_train, X_train_scaled_with_const[features]).fit()
    pvalues = model.pvalues.drop('const')
    max_pvalue = pvalues.max()
    if max_pvalue > 0.01:
        feature_to_remove = pvalues.idxmax()
        features.remove(feature_to_remove)
        eliminated_features_reg.append(feature_to_remove)
    else:
        break

final_selected_features_reg = [feature for feature in features if feature != 'const']
print(f"stepwise regression eliminated features: {eliminated_features_reg}")
print(f"stepwise regression final selected features: {final_selected_features_reg}")

rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_train_scaled, y_train)

importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

threshold = 0.05
selected_features_rf = feature_importance_df[feature_importance_df['Importance'] > threshold]['Feature'].tolist()
print(f"Random Forest Features: {selected_features_rf}")
print(f"Random Forest Eliminated Features:{feature_importance_df[feature_importance_df['Importance'] < threshold]['Feature'].tolist()}")
common_features = set(final_selected_features_reg).intersection(set(selected_features_rf))
diff_features = set(final_selected_features_reg).symmetric_difference(set(selected_features_rf))
print(f"Common Features: {common_features}")
print(f"Difference Features: {diff_features}")

#%%
print('4-c')
threshold = 0.05
selected_features_rf = feature_importance_df[feature_importance_df['Importance'] > threshold]['Feature'].tolist()

X_train_selected_rf = X_train_scaled[selected_features_rf]
X_train_selected_rf_const = sm.add_constant(X_train_selected_rf)
ols_model_rf = sm.OLS(y_train, X_train_selected_rf_const).fit()

print("OLS Regression Summary with Selected Features:")
print(ols_model_rf.summary())


#%%
print('4-d')
X_test_scaled_selected_rf = X_test_scaled[selected_features_rf]
X_test_scaled_selected_rf_const = sm.add_constant(X_test_scaled_selected_rf)
y_prediction_scaled = ols_model_rf.predict(X_test_scaled_selected_rf_const)

y_test = y_test.reset_index(drop=True)
y_prediction = y_prediction.reset_index(drop=True)
plt.figure(figsize=(10, 6))
plt.title('Actual Sales vs Predicted Sales')
plt.plot(y_test, label='Actual Sales', marker='o')
plt.plot(y_prediction, label='Predicted Sales', marker='*')
plt.xlabel('# of Samples')
plt.ylabel('Sales')
plt.legend()
plt.tight_layout()
plt.show()

print('\n Actual Sales Values:')
print(y_test)
print('\n Predicted Sales Values:')
print(y_prediction)

#%%
print('4-e')
mse_rf = mean_squared_error(y_test, y_prediction)
print(f"Mean Squared Error on the test set: {mse_rf:.3f}")

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
X_test_selected = X_test_scaled[final_selected_features_reg]
X_test_selected_const = sm.add_constant(X_test_selected)
predictions = final_model.get_prediction(X_test_selected_const)

prediction_summary = predictions.summary_frame(alpha=0.05)  # 95% confidence level
y_prediction = prediction_summary['mean']
y_prediction_lower = prediction_summary['obs_ci_lower']
y_prediction_upper = prediction_summary['obs_ci_upper']

plt.figure(figsize=(10, 6))
plt.plot(y_prediction.values, label='Predicted Sales', color='blue')
plt.fill_between(range(len(y_prediction)), y_prediction_lower, y_prediction_upper,
                 color='skyblue', alpha=0.5, label='95% Prediction Interval')

plt.xlabel('# of Samples')
plt.ylabel('Sales USD($)')
plt.title('Sales Prediction with Confidence Interval')
plt.legend()
plt.ylim(bottom=-2.5, top=16)
plt.tight_layout()
plt.show()

#%%
print('7')
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd

url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/Carseats.csv'
data = pd.read_csv(url)

#%%
print('7-a')
X = data[['Price']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)

pipeline = make_pipeline(PolynomialFeatures(), LinearRegression())
param_grid = {'polynomialfeatures__degree': np.arange(1, 16)}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

#%%
print('7-b')
optimal_degree_n = grid.best_params_['polynomialfeatures__degree']
print(f"The optimal polynomial degree (n): {optimal_degree_n}")

#%%
print('7-c')
degrees = param_grid['polynomialfeatures__degree']
mean_test_scores = grid.cv_results_['mean_test_score']
RMSE = np.sqrt(-mean_test_scores)

plt.figure(figsize=(10, 6))
plt.title('The RMSE vs The n Order')
plt.plot(degrees, RMSE, marker='o')
plt.xlabel('The n Order')
plt.ylabel('The RMSE')
plt.xticks(degrees)
plt.grid()
plt.show()

#%%
print('7-d')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)

final_model = make_pipeline(PolynomialFeatures(degree=optimal_degree_n), LinearRegression())
final_model.fit(X_train, y_train)
y_prediction = final_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test.values, label='Test Dataset', color='blue')
plt.plot(range(len(y_prediction)), y_prediction, label='Polynomial Regression', color='orange')
plt.xlabel('Observations')
plt.ylabel('Sales')
plt.title('Regression Model - Carseats Dataset')
plt.legend()
plt.tight_layout()
plt.grid()
plt.ylim(bottom=-2.5, top=16)
plt.show()

#%%
print('7-e')
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_prediction)
print(f"Mean Squared Error for the polynomial regression of order {optimal_degree_n}: {mse:.3f}")


