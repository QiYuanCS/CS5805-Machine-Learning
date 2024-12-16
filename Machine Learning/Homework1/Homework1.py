
import seaborn as sns
import  pandas as pd
from scipy.stats import gmean, hmean
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

print('\n#1')
datasets = sns.get_dataset_names()
print(datasets)

print('\n#2')

selected_datasets = ['diamonds', 'iris', 'tips', 'penguins', 'titanic']
for name in selected_datasets:
    data = sns.load_dataset(name)
    print(f"Dataset Name: {name}")
    print('\n Data information')
    print(data.head())
    print('\n')
    print(data.info())
#%%
print('\n#3')

titanic = sns.load_dataset('titanic')

print('The count, mean, std, min, 25%, 50%, 75% and max for the numerical features in the dataset.')
print(titanic.describe())

Categorical_nominal = ['survived', 'sex', 'embarked', 'who', 'adult_male', 'deck', 'alone']
Categorical_ordinal = ['pclass','class']
Numeric_ratio = ['age','sibsp', 'parch', 'fare']

print('\n nominal: ',Categorical_nominal)
print('\n ordinal: ',Categorical_ordinal)
print('\n ratio: ',Numeric_ratio)

missing_values = titanic.isnull().sum()
print('\nMissing observations inside the dataset')
print(titanic.isnull())
print(missing_values)

print('\n#4')

numerical_titanic = titanic[Numeric_ratio]
print('The first 5 rows of the original dataset')
print(titanic.head())

print('\n The new dataset with numerically selected features only')
print(numerical_titanic.head())

print('\n#5')

clean_numerical_titanic = numerical_titanic.dropna()
clean_missing_count = len(numerical_titanic) - len(clean_numerical_titanic)
eliminated_percent = (clean_missing_count / len(numerical_titanic)) * 100
print('The number of missing observations: ', clean_missing_count)
print(f'The Percentage of missing observations is eliminated to clean the dataset: {eliminated_percent:.2f}%')


print('\n#8')
arithmetic_mean = clean_numerical_titanic.mean()

geometric_mean  = clean_numerical_titanic.apply(gmean)

harmonic_mean   = clean_numerical_titanic.apply(hmean)

print('Arithmetic Mean: ', arithmetic_mean)
print('Geometric Mean: ', geometric_mean)
print('Harmonic Mean: ', harmonic_mean)

print('\n#9')

age_data = clean_numerical_titanic['age']

fare_data = clean_numerical_titanic['fare']

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.hist(age_data, bins = 20, color = 'red', alpha = 0.9)
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(fare_data, bins = 20, color = 'blue', alpha = 0.9)
plt.title('Histogram of Fare')
plt.xlabel('Fare')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

print('\n#10')
pd.plotting.scatter_matrix(clean_numerical_titanic,
                           c = clean_numerical_titanic['age'],
                           marker='o',
                           hist_kwds={'bins': 20},
                           s=6,
                           alpha=.9,
                           figsize=(15, 15),
                           )

plt.suptitle('Pairwise Bivariate Distributions of Dataset Features', fontsize=16, y=0.98)
plt.grid()
plt.tight_layout()
plt.show()