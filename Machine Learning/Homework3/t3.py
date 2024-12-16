# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate  # For displaying tables
from prettytable import PrettyTable  # Alternative for displaying tables

# Part (a): Generate random variables and compute covariance matrix

# Set random seed for reproducibility
np.random.seed(5805)

# Sample size
n = 1000

# Generate x ~ N(µ=1, σ²=2)
mu_x = 1
sigma2_x = 2
sigma_x = np.sqrt(sigma2_x)
x = np.random.normal(mu_x, sigma_x, n)

# Generate ε ~ N(µ=2, σ²=3)
mu_epsilon = 2
sigma2_epsilon = 3
sigma_epsilon = np.sqrt(sigma2_epsilon)
epsilon = np.random.normal(mu_epsilon, sigma_epsilon, n)

# Generate y = x + ε
y = x + epsilon

# Construct feature matrix X = [x, y]
X = np.column_stack((x, y))

# Compute sample means
x_mean = np.mean(x)
y_mean = np.mean(y)

# Center the variables
x_centered = x - x_mean
y_centered = y - y_mean

# Compute variances and covariance using definitions
S_xx = np.sum(x_centered ** 2) / (n - 1)
S_yy = np.sum(y_centered ** 2) / (n - 1)
S_xy = np.sum(x_centered * y_centered) / (n - 1)

# Construct the covariance matrix
cov_matrix = np.array([[S_xx, S_xy],
                       [S_xy, S_yy]])

# Display the covariance matrix using PrettyTable
table = PrettyTable()
table.title = 'Estimated Covariance Matrix'
table.field_names = ['', 'x', 'y']
table.add_row(['x', f'{S_xx:.4f}', f'{S_xy:.4f}'])
table.add_row(['y', f'{S_xy:.4f}', f'{S_yy:.4f}'])
print(table)

# Justification of diagonal elements
print("\nJustification for Diagonal Elements:")
print(f"Sample variance of x (S_xx): {S_xx:.4f}, Theoretical variance: {sigma2_x}")
print(f"Sample variance of y (S_yy): {S_yy:.4f}, Theoretical variance of y: Var(y) = Var(x) + Var(ε) = {sigma2_x} + {sigma2_epsilon} = {sigma2_x + sigma2_epsilon}")

# Part (b): Calculate eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(cov_matrix)

# Display eigenvalues and eigenvectors in a table
eig_table = PrettyTable()
eig_table.title = 'Eigenvalues and Eigenvectors'
eig_table.field_names = ['Eigenvalue', 'Eigenvector']
for i in range(len(eigvals)):
    eig_table.add_row([f'{eigvals[i]:.4f}', f'[{eigvecs[0,i]:.4f}, {eigvecs[1,i]:.4f}]'])
print('\n', eig_table)

# Part (c): Scatter plot and plotting eigenvectors
plt.figure(figsize=(8,6))
plt.scatter(x, y, alpha=0.5, label='Data points')

# Plot eigenvectors
origin = np.array([[x_mean], [y_mean]])  # Mean of x and y

# Scale eigenvectors for visualization
scale = 2
for i in range(len(eigvals)):
    eigvec = eigvecs[:,i]
    plt.quiver(x_mean, y_mean, scale*eigvec[0], scale*eigvec[1], angles='xy', scale_units='xy', scale=1, color=['r','g'][i], label=f'Eigenvector {i+1}')

# Add labels, title, legend, and grid
plt.xlabel('Variable x')
plt.ylabel('Variable y')
plt.title('Scatter Plot of x and y with Eigenvectors')
plt.legend()
plt.grid(True)
plt.show()

# Explanation of eigenvectors
print("\nThe eigenvector corresponding to the maximum eigenvalue represents the direction of maximum variance in the data (first principal component).")
print("The eigenvector corresponding to the minimum eigenvalue represents the direction of minimum variance.")

# Decision on which feature to drop
print("\nIf we were to drop one feature, we might consider dropping 'x' because 'y' contains 'x' plus additional noise from 'ε', potentially capturing more variability.")

# Part (d): Calculate singular values of the centered feature matrix
# Center the feature matrix
X_centered = np.column_stack((x_centered, y_centered))

# Compute singular values using SVD
U, singular_values, Vt = np.linalg.svd(X_centered, full_matrices=False)

# Display singular values in a table
sv_table = PrettyTable()
sv_table.title = 'Singular Values of Centered Feature Matrix'
sv_table.field_names = ['Singular Value']
for sv in singular_values:
    sv_table.add_row([f'{sv:.4f}'])
print('\n', sv_table)

# Relationship between singular values and eigenvalues
print("\nRelationship between singular values and eigenvalues:")
print("Eigenvalues of the covariance matrix are proportional to the squared singular values divided by (n - 1).")
print("Computed Eigenvalues:")
for i in range(len(eigvals)):
    print(f"Eigenvalue {i+1}: {eigvals[i]:.4f}")
print("Squared Singular Values divided by (n - 1):")
for i in range(len(singular_values)):
    print(f"Value {i+1}: {(singular_values[i]**2)/(n - 1):.4f}")

# Part (e): Calculate correlation matrix and Pearson correlation coefficient
df = pd.DataFrame({'x': x, 'y': y})
corr_matrix = df.corr()

# Display the correlation matrix
corr_table = PrettyTable()
corr_table.title = 'Correlation Matrix'
corr_table.field_names = ['', 'x', 'y']
corr_table.add_row(['x', f'{corr_matrix.loc["x","x"]:.4f}', f'{corr_matrix.loc["x","y"]:.4f}'])
corr_table.add_row(['y', f'{corr_matrix.loc["y","x"]:.4f}', f'{corr_matrix.loc["y","y"]:.4f}'])
print('\n', corr_table)

# Sample Pearson correlation coefficient between x and y
r_xy = corr_matrix.loc['x', 'y']
print(f"\nSample Pearson correlation coefficient between x and y: {r_xy:.4f}")

# Relationship between covariance matrix and correlation coefficient matrix
print("\nThe correlation coefficient matrix is the covariance matrix normalized by the product of the standard deviations of the variables.")
print("r_xy = Cov(x, y) / (StdDev(x) * StdDev(y))")
computed_r_xy = S_xy / (np.sqrt(S_xx) * np.sqrt(S_yy))
print(f"Computed r_xy using covariance and variances: {computed_r_xy:.4f}")
print("This matches the value from the correlation matrix.")
