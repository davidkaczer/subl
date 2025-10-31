import scipy.stats as stats
import numpy as np

# Data
n1 = 1000   # Sample size 1
x1 = 130     # Successes in sample 1

n2 = 1000  # Sample size 2
x2 = 162    # Successes in sample 2

print("=" * 60)
print("Two-Sample Bernoulli Test")
print("=" * 60)
print(f"Sample 1: {x1}/{n1} successes (p̂₁ = {x1/n1:.4f})")
print(f"Sample 2: {x2}/{n2} successes (p̂₂ = {x2/n2:.4f})")
print()

# Fisher's Exact Test (most appropriate for small counts)
# Create contingency table: [[successes_1, failures_1], [successes_2, failures_2]]
contingency_table = [[x1, n1 - x1], [x2, n2 - x2]]
print("Contingency Table:")
print(f"         Success  Failure")
print(f"Sample 1:   {x1:3d}     {n1-x1:3d}")
print(f"Sample 2:  {x2:3d}     {n2-x2:3d}")
print()

# Two-sided Fisher's exact test
odds_ratio, p_value_fisher = stats.fisher_exact(contingency_table, alternative='two-sided')
print("Fisher's Exact Test (two-sided):")
print(f"  Odds ratio: {odds_ratio:.4f}")
print(f"  P-value: {p_value_fisher:.4f}")
print()

# Alternative: Chi-square test (less appropriate with small counts)
chi2, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)
print("Chi-square Test (for comparison, less reliable with small counts):")
print(f"  Chi-square statistic: {chi2:.4f}")
print(f"  P-value: {p_value_chi2:.4f}")
print()

# Z-test for two proportions (another alternative)
p1 = x1 / n1
p2 = x2 / n2
p_pooled = (x1 + x2) / (n1 + n2)
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
z_stat = (p1 - p2) / se
p_value_z = 2 * stats.norm.cdf(-abs(z_stat))  # two-sided
print("Z-test for Two Proportions (for comparison):")
print(f"  Z-statistic: {z_stat:.4f}")
print(f"  P-value: {p_value_z:.4f}")
print()

# Interpretation
alpha = 0.05
print("=" * 60)
print("Interpretation:")
print("=" * 60)
if p_value_fisher < alpha:
    print(f"At α = {alpha}, we REJECT the null hypothesis.")
    print("The data provides evidence that the two proportions are different.")
else:
    print(f"At α = {alpha}, we FAIL TO REJECT the null hypothesis.")
    print("The data does not provide sufficient evidence that the proportions differ.")
