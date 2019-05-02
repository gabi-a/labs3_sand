from scipy.stats import norm

# Probability of making $300 in 1 month
mu = 0.87
std = 4.33
print(f"1. {1-norm.cdf(3, mu, std):.3f}")
print(f"2. {norm.cdf(0, mu, std)-norm.cdf(-10, mu, std):.3f}")
print(f"3. {norm.cdf(-11, mu, std):.4f}")
print(f"4. {norm.cdf(-11, mu, std)**2:.4e}")
print(f"4. {norm.cdf(-11, mu, std)**-2:.4e}")
