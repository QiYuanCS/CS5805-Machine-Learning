#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data
import yfinance as yf

yf.pdr_override()

pd.set_option('display.precision', 2)
start = "2000-01-01"
end = "2022-09-25"
#%%
print('#2')
from dataPreprocessing import DataPreprocessing
df = data.get_data_yahoo('AAPL', start = start, end = end)
print(df.head())
features = df[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']]

data_preprocessor = DataPreprocessing(features)

fig, axs = plt.subplots(2, 2, figsize = (15, 10))
data_preprocessor.show_original(axs[0, 0])

data_preprocessor.normalize()
data_preprocessor.show_normalized(axs[0, 1])

data_preprocessor.standardize()
data_preprocessor.show_standardized(axs[1, 0])

data_preprocessor.iqr()
data_preprocessor.show_iqr(axs[1, 1])

plt.tight_layout()
plt.show()

#%%
print('#3')
r_values = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4,5, 5, 5.5, 6]

theta = np.linspace(0, 2 * np.pi,2000)
x = np.cos(theta)
y = np.sin(theta)

plt.figure(figsize = (12, 12))

for r in r_values:
    norm = (np.abs(x)**r + np.abs(y)**r)**(1/r)
    plt.plot(x / norm, y / norm, label = f'$L_{{{r}}}$ norm')

plt.title('$L_r$ norm')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 'upper right')
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()

#%%
print('#6-a')
from tabulate import tabulate
np.random.seed(5805)
n = 1000
mean_u = 1
sigma_square = 2
sigma = np.sqrt(sigma_square)
x = np.random.normal(mean_u, sigma, n)

mean_u_epsilon = 2
sigma_square_epsilon = 3
sigma_epsilon = np.sqrt(sigma_square_epsilon)
epsilon = np.random.normal(mean_u_epsilon, sigma_epsilon, n)

y = x + epsilon
mean_x = np.mean(x)
mean_y = np.mean(y)

var_xx = np.sum((x - mean_x)**2) / (n - 1)
var_yy = np.sum((y - mean_y)**2) / (n - 1)

cov_xy = np.sum((x - mean_x) * (y - mean_y)) / (n - 1)

covariance_matrix = np.array([[var_xx, cov_xy],
                              [cov_xy, var_yy]])


headers = ['Variable', 'x', 'y']
table = [
    ['x', f'{covariance_matrix[0, 0]:.2f}', f'{covariance_matrix[0, 1]:.2f}'],
    ['y', f'{covariance_matrix[1, 0]:.2f}', f'{covariance_matrix[1, 1]:.2f}']
]
print('Estimated Covariance Matrix')
print(tabulate(table, headers, tablefmt='fancy_grid'))

#%%
print('#6-b')
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

headers = ['Eigenvalue', 'Eigenvector']
table = [
    [f'{eigenvalues[0]:.2f}', f'{eigenvalues[1]:.2f}'],
    ['Eigenvector related to Eigenvalue1 ', 'Eigenvector related to Eigenvalue2'],
    [f'{eigenvectors[0, 0]:.2f}', f'{eigenvectors[0, 1]:.2f}'],
     [f'{eigenvectors[1, 0]:.2f}', f'{eigenvectors[1, 1]:.2f}']
]
print('Estimated Eigenvalues and Eigenvectors of the covariance matrix')
print(tabulate(table, headers, tablefmt='fancy_grid'))

#%%
print('#6-c')
plt.figure(figsize=(12, 10))
plt.scatter(x, y, alpha = 0.7, label = 'Data Points')
origin = np.array([[mean_x], [mean_y]])
scale_factor = 2
for i in range(len(eigenvalues)):
    eigenvector = eigenvectors[:, i]
    eigenvalue = eigenvalues[i]
    vector = scale_factor * np.sqrt(eigenvalue) * eigenvector
    plt.quiver(mean_x, mean_y, vector[0], vector[1],
               angles = 'xy', scale_units='xy', scale=1,
                color=['red', 'green'][i], width=0.005,
                label=f'Eigenvalue {i + 1}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of x and y with Eigenvectors')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()

#%%
print('#6-d')
X_centered = np.column_stack((x - mean_x, y - mean_y))
U, Singular_values, V_transpose = np.linalg.svd(X_centered, full_matrices=False)
X_Transpose_X = X_centered.T @ X_centered

headers = ['Singular Value Index', 'Singular Value']
table = [[i+1, f'{Singular_values[i]:.2f}'] for i in range(len(Singular_values))]

print('Singular Values of the Centered Data Matrix')
print(tabulate(table, headers, tablefmt='fancy_grid'))

eigenvalues_X_Transpose_X = np.linalg.eigvals(X_Transpose_X)
eigenvalues_X_Transpose_X_sorted = np.sort(eigenvalues_X_Transpose_X)[::-1]
Singular_values_squared = Singular_values**2

headers = ['Index', 'Eigenvalue of X̃ᵀX̃', 'Squared Singular Value','Singular Value']
table = []
for i in range(len(eigenvalues_X_Transpose_X)):
    table.append([i+1, f'{eigenvalues_X_Transpose_X_sorted[i]:.2f}',
                 f'{Singular_values_squared[i]:.2f}',
                 f'{Singular_values[i]:.2f}'])

print('\nComparison between singular values and eigenvalues of X̃^T X̃')
print(tabulate(table, headers, tablefmt='fancy_grid'))

#%%
print('#6-e')
df = pd.DataFrame({'x':x, 'y':y})
correlation_matrix = df.corr()
print('Correlation Matrix using DataFrame.corr()')
print(correlation_matrix)

std_x = np.sqrt(var_xx)
std_y = np.sqrt(var_yy)
pearson_r = cov_xy / (std_x * std_y)
print(f'\nCalculated Pearson correlation coefficient: {pearson_r:.2f}')


#%%
print('#6-f')
def differencing(x_t, y_t):
    df = pd.DataFrame({
        'x(t)': x_t,
        'y(t)': y_t
    })
    df['Δy(t)'] = df['y(t)'].diff()
    df['Δ²y(t)'] = df['Δy(t)'].diff()
    df['Δ³y(t)'] = df['Δ²y(t)'].diff()

    return df

x_t = np.arange(-4, 5, 1)
y_t = x_t ** 3
df_differenced = differencing(x_t, y_t)
print(df_differenced)

plt.figure(figsize=(10, 6))
plt.plot(df_differenced['x(t)'], df_differenced['y(t)'], label='y(t) = x(t)^3', marker='o')
plt.plot(df_differenced['x(t)'], df_differenced['Δy(t)'], label='Δy(t)', marker='o')
plt.plot(df_differenced['x(t)'], df_differenced['Δ²y(t)'], label='Δ²y(t)', marker='o')
plt.plot(df_differenced['x(t)'], df_differenced['Δ³y(t)'], label='Δ³y(t)', marker='o')
plt.title('Original dataset and Differencing of y(t) = x(t)^3')
plt.xlabel('x(t)')
plt.ylabel('y(t) and Differences')
plt.grid()
plt.legend()
plt.show()
