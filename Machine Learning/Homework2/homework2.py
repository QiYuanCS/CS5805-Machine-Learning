
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.precision', 3)
pd.set_option('display.max_columns', None)

url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/stock%20prices.csv'

#%%
df = pd.read_csv(url)
print('\n#1')
missing_number = df.isna().sum()
missing_features = missing_number[missing_number > 0]

print('\na')
print("\nThe missing features and the number of missing entries: ")
print(missing_features)

#%%
print('\nb')
df = pd.read_csv(url)
missing_number = df.isna().sum()
missing_features = missing_number[missing_number > 0]
df[missing_features.index] = df[missing_features.index].fillna(df[missing_features.index].mean())
missing_number_fill = df.isna().sum()
missing_features_fill = missing_number_fill[missing_number_fill > 0]
print("Replace the missing values with ‘mean’.")
print(missing_features_fill)
#%%
print('\nc')
df = pd.read_csv(url)
missing_number = df.isna().sum()
missing_features = missing_number[missing_number > 0]
df[missing_features.index] = df[missing_features.index].fillna(df[missing_features.index].mean())
missing_number_fill = df.isna().sum()
missing_features_fill = missing_number_fill[missing_number_fill > 0]
if missing_number_fill.empty:
    print("All missing values have been filled and cleaned.")
else:
    print("There are still missing values in the follow features: ")
    print(missing_number_fill)

#%%
print('\n#2')
print('\na')
unique_companies = df['symbol'].unique()
print(f'Number of Unique Companies: {len(unique_companies)}')
print('Unique companies:')
print(list(unique_companies))

print('\nb')
quantitative = df.select_dtypes(include=['float', 'int64']).columns
qualitative = df.select_dtypes(include=['object']).columns
print(f'Quantitative predictors : {quantitative}')
print(f'Qualitative predictors : {qualitative}')

print('\nc')
df_selected = df[df['symbol'].isin(['AAPL','GOOGL'])]
df_selected.loc[:,'date'] = pd.to_datetime(df_selected['date'])

plt.figure(figsize=(12, 8))

for company in ['AAPL','GOOGL']:
    company_data = df_selected[df_selected['symbol'] == company]
    plt.plot(company_data['date'], company_data['close'], label = company)

plt.title('Apple and Google Stock Closing Value Comparison')
plt.xlabel('date')
plt.ylabel('USD($)')
plt.xticks(rotation = 45)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

#%%
print('\n#3')
df = pd.read_csv(url)
missing_number = df.isna().sum()
missing_features = missing_number[missing_number > 0]
df[missing_features.index] = df[missing_features.index].fillna(df[missing_features.index].mean())
print('\nThe first 5 rows of cleaned data set')
print(df.head())
df_aggregated = df.groupby('symbol', as_index = False).sum(numeric_only=True)

print('\nNumbers of obejcts in the cleaned data set versus the aggregated data set')
print(f'The cleaned data set {len(df)} versus the aggregated data set {len(df_aggregated)}')

print('\nThe first 5 rows of the aggregated dataset:')
print(df_aggregated.head())



#%%
print('\n#4')
df_sliced = df[['symbol', 'close', 'volume']]
df_aggregated = df_sliced.groupby('symbol').agg(['mean', 'var'])

close_var = df_aggregated['close']['var']
max_variance_index = np.argmax(close_var)
max_variance_value = np.max(close_var)
max_variance_company = close_var.index[max_variance_index]

print(f'The company that has the maximum variance in the closing cost is {max_variance_company}\n'
      f'The maximum variance is {max_variance_value:.3f}')

#%%
print('\n#5')
df['date'] = pd.to_datetime(df['date'])

df_Google = df[(df['symbol'] == 'GOOGL') & (df['date'] > '2015-01-01')]

df_Google_close = df_Google[['date', 'close']]
print('The Google stock losing cost after 2015-01-01')
print(df_Google_close.head().to_string(formatters = {'close': '{:.3f}'.format}))


#%%
print('\n#6')
df['date'] = pd.to_datetime(df['date'])

df_Google = df[(df['symbol'] == 'GOOGL') & (df['date'] > '2015-01-01')].copy()

df_Google_close = df_Google[['date', 'close']]

df_Google_close = df_Google_close.assign(
    rolling_mean = df_Google_close['close'].rolling(window = 30, center = True).mean()
)
plt.figure(figsize=(12, 8))
plt.plot(df_Google_close['date'], df_Google_close['close'], label = 'close', color = 'blue')
plt.plot(df_Google_close['date'], df_Google_close['rolling_mean'], label = 'AVG_30', color = 'orange')

plt.title('Google closing stock price after jan 2015 versus Rolling Window')
plt.xlabel('date')
plt.ylabel('USD($)')
xticks = df_Google_close['date'][::100]
plt.xticks(xticks)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

missing_observations = df_Google_close['rolling_mean'].isna().sum()
print(f'Number of missing observations: {missing_observations}')

#%%
print('\n#7')
bin_labels = ['very low', 'low', 'normal','high','very high']
df_Google_close_bar = df_Google_close.drop(columns=['rolling_mean'])
df_Google_close_bar['price_category'] = pd.cut(df_Google_close_bar['close'], bins = 5, labels = bin_labels)


plt.figure(figsize=(10, 8))
colors = ['blue', 'orange', 'green', 'red', 'purple']
df_Google_close_bar['price_category'].value_counts().sort_index().plot(kind = 'bar', color = colors)
plt.title('Equal width discretization')
plt.xlabel('price_category')
plt.ylabel('count')
plt.xticks(rotation = 0)
plt.grid()
plt.tight_layout()
plt.show()

#print(df_Google_close_bar.head().to_string(formatters={'close': '{:.3f}'.format}))
print(df_Google_close_bar.to_string(formatters={'close': '{:.3f}'.format}))

#%%
print('\n#8')
plt.figure(figsize=(10, 8))
n, bins, patches = plt.hist(df_Google_close_bar['close'], bins = 5, edgecolor = 'black')

for i in range(len(patches)):
    patches[i].set_facecolor(colors[i])

plt.title('Histogram of close features (5 Bins)')
plt.xlabel('close')
plt.ylabel('Frequency')
plt.grid()
plt.xticks(rotation = 0)
plt.tight_layout()
plt.show()


#%%
print('\n#9')
bin_labels = ['very low', 'low', 'normal','high','very high']
df_Google_close_bar['price_category'] = pd.qcut(df_Google_close_bar['close'], q = 5, labels = bin_labels)
plt.figure(figsize=(10, 8))
colors = ['blue', 'orange', 'green', 'red', 'purple']
df_Google_close_bar['price_category'].value_counts().sort_index().plot( kind = 'bar', color = colors)

plt.title("\nEqual frequency discretization")
plt.xlabel('price_category')
plt.ylabel('count')
plt.xticks(rotation = 0)
plt.grid()
plt.tight_layout()
plt.show()

print(df_Google_close_bar.to_string(formatters={'close': '{:.3f}'.format}))
#print(df_Google_close_bar.head().to_string(formatters={'close': '{:.3f}'.format}))

#%%
print('\n#10')
df['date'] = pd.to_datetime(df['date'])
df_Google = df[(df['symbol'] == 'GOOGL') & (df['date'] > '2015-01-01')]

def covariance(x, y):
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    covariance = np.sum((x - mean_x) * (y - mean_y)) / (n - 1)
    return covariance

features_names = ['open', 'high', 'low', 'close', 'volume']
features = [df_Google[feature].values for feature in features_names]

n_features = len(features)
covariance_matrix = np.zeros((n_features, n_features))

for i in range(n_features):
    for j in range(n_features):
        covariance_matrix[i, j] = covariance(features[i], features[j])
        if i != j:
            covariance_matrix[j, i] = covariance_matrix[i, j]

covariance_df = pd.DataFrame(covariance_matrix, index = features_names, columns=features_names).round(3)
print("\n Covariance Matrix:")
print(covariance_df)

#%%
print('\n#11')
df['date'] = pd.to_datetime(df['date'])
df_Google = df[(df['symbol'] == 'GOOGL') & (df['date'] > '2015-01-01')]

df_cov = df_Google[['open', 'high', 'low', 'close', 'volume']]
covariance_matrix = df_cov.cov().round(3)

print("\n Covariance matrix:")
print(covariance_matrix)

correlation_matrix = df_cov.corr().round(3)
print('\n Correlation matrix')
print(correlation_matrix)