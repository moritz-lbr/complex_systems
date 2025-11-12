import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Step 1: Generate a non-Gaussian time series (Log-Normal distributed)
np.random.seed(42)
#non_gaussian_data = np.random.lognormal(mean=0, sigma=1, size=500)
non_gaussian_data = np.random.normal(size=500)

# Step 2: Rank-order transformation to a standard normal distribution
ranks = stats.rankdata(non_gaussian_data)  # Get ranks (1 to N)
quantiles = (ranks - 0.5) / len(ranks)  # Scale to (0,1) for percentiles
gaussian_mapped_data = stats.norm.ppf(quantiles)  # Map to standard normal

# Step 3: Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original distribution
axes[0].hist(non_gaussian_data, bins=30, color='c', alpha=0.7, density=True)
axes[0].set_title("Original Non-Gaussian Data")

# Transformed (Gaussian) distribution
axes[1].hist(gaussian_mapped_data, bins=30, color='m', alpha=0.7, density=True)
axes[1].set_title("Gaussianized Data (After Rank Transformation)")

plt.show()

# Verify with a Q-Q plot
stats.probplot(gaussian_mapped_data, dist="norm", plot=plt)
plt.title("Q-Q Plot of Transformed Data")
plt.show()
