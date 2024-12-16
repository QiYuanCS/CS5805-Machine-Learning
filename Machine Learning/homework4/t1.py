import numpy as np
import matplotlib.pyplot as plt

# Create a general plot for f_X|Y(lambda1|lambda2) for different values of lambda2
lambda2_values = [0.2, 0.4, 0.6, 0.8]  # Different values of lambda2 to illustrate

plt.figure(figsize=(8, 6))

# Plot for different lambda2 values
for lambda2 in lambda2_values:
    lambda1_range = np.linspace(0, 1 - lambda2, 100)
    f_X_given_Y_general = 1 / (1 - lambda2) * np.ones_like(lambda1_range)

    plt.plot(lambda1_range, f_X_given_Y_general, label=r'$\lambda_2 = $' + str(lambda2), linewidth=2)

# Add labels, title, and legend
plt.xlabel(r'$\lambda_1$')
plt.ylabel(r'$f_{X|Y}(\lambda_1|\lambda_2)$')
plt.title(r'Plot of $f_{X|Y}(\lambda_1|\lambda_2)$ for different $\lambda_2$ values')

# Set limits and grid
plt.xlim([0, 1])
plt.ylim([0, 10])
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
