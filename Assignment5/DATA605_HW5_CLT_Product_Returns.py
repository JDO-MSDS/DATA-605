"""
DATA 605 - Week 9 - Homework 5
Central Limit Theorem: Planning for Product Returns

Scenario:
    An online retailer receives 300 orders during a promotion. Each order is
    independently returned with probability p = 0.12. The retailer can process
    at most 45 returns. We use the Central Limit Theorem (CLT) to study the
    number of returns and to estimate the risk of exceeding capacity.

Notation:
    S_n = number of returns among n orders  ->  S_n ~ Binomial(n, p)
"""

# =============================================================================
# Setup
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm

rng = np.random.default_rng(605)   # fixed random seed 
p = 0.12                           
N_ORDERS = 300                     
CAPACITY = 45                      
N_SIMS = 5000                      


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =============================================================================
# Part A - Expected Returns
# =============================================================================
header("PART A - Expected Returns (n = 300, p = 0.12)")

n = N_ORDERS
expected_returns = n * p                    
var_returns = n * p * (1 - p)               
sd_returns = np.sqrt(var_returns)          

print(f"E[S_300]   = n * p          = {n} * {p} = {expected_returns:.2f} returns")
print(f"Var(S_300) = n * p * (1-p)  = {n} * {p} * {1 - p:.2f} = {var_returns:.4f}")
print(f"SD(S_300)  = sqrt(Var)      = sqrt({var_returns:.4f}) = {sd_returns:.4f} returns")
print(f"Capacity ({CAPACITY}) is {CAPACITY - expected_returns:.0f} returns above the mean, "
      f"or {(CAPACITY - expected_returns) / sd_returns:.2f} standard deviations.")

# Interpretation (Part A):
#   - E[S_300] = 36: on average the retailer should plan for about 36 returns
#     from this 300-order promotion (12% of 300).
#   - SD(S_300) ~ 5.63: the actual number of returns will typically vary by about
#     5 to 6 returns around 36. Roughly, most promotions should land somewhere
#     between about 30 and 42 returns (mean +/- 1 SD), and nearly all between
#     about 25 and 47 (mean +/- 2 SD).
#   - The capacity of 45 is 9 returns above the mean, which is only about
#     1.6 SDs. That is not a large cushion, so going over capacity is unusual
#     but not rare. Part C puts a number on that risk.


# =============================================================================
# Part B - Demonstrate the Central Limit Theorem
# =============================================================================
header("PART B - CLT Simulation (5,000 cohorts per size)")

cohort_sizes = [10, 50, 300]
z_grid = np.linspace(-4, 4, 400)            # x-values for the N(0,1) curve

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

print(f"{'n':>5} {'np':>7} {'SD':>7} {'Sim mean Z':>11} {'Sim SD Z':>9} "
      f"{'Skewness Z':>11} {'Distinct Z values':>18}")

for ax, n_c in zip(axes, cohort_sizes):
    # 1 generate 5000 return counts from ninomial
    counts = rng.binomial(n_c, p, size=N_SIMS)

    # 2 standardize
    mu_c = n_c * p
    sd_c = np.sqrt(n_c * p * (1 - p))
    z = (counts - mu_c) / sd_c

    # summary of the standardized values 
    skew = np.mean((z - z.mean()) ** 3) / z.std() ** 3
    print(f"{n_c:>5} {mu_c:>7.2f} {sd_c:>7.3f} {z.mean():>11.4f} {z.std():>9.4f} "
          f"{skew:>11.4f} {len(np.unique(z)):>18}")

    # 3 histogram 
    edges = (np.arange(counts.min(), counts.max() + 2) - 0.5 - mu_c) / sd_c
    ax.hist(z, bins=edges, density=True, color="steelblue", alpha=0.6,
            edgecolor="white", label="Standardized simulated counts")
    ax.plot(z_grid, norm.pdf(z_grid), color="crimson", lw=2,
            label="Standard normal density N(0,1)")
    ax.set_title(f"n = {n_c}  (np = {mu_c:.1f}, SD = {sd_c:.2f})")
    ax.set_xlabel("Standardized return count  Z = (S_n - np) / sqrt(np(1-p))")
    ax.set_xlim(-4, 4)
    ax.legend(fontsize=8, loc="upper right")

axes[0].set_ylabel("Density")
fig.suptitle("CLT: Standardized Binomial Return Counts vs. Standard Normal "
             "(p = 0.12, 5,000 cohorts each)", fontsize=13)
fig.tight_layout()
fig.savefig("HW5_PartB_CLT_histograms.png", dpi=150)
print("\nFigure saved: HW5_PartB_CLT_histograms.png")

# Interpretation (Part B):
#   - In all three cases the simulated Z values have mean ~0 and SD ~1, which
#     is what standardizing guarantees. What changes with n is the SHAPE.
#   - n = 10: np = 1.2, so most cohorts have 0, 1, 2 or 3 returns. Z takes
#     only 7 distinct values, the histogram is a few wide blocks, and it is
#     clearly right-skewed (skewness ~0.66): counts cannot go below 0 (Z can't
#     go below about -1.2) but have a longer right tail. The normal curve is a
#     poor fit. A common rule of thumb asks for np >= 10 and n(1-p) >= 10, and
#     n = 10 is far from that.
#   - n = 50: np = 6. The shape is more bell-like and less skewed (skewness
#     ~0.33), but still lumpy (17 distinct values) and slightly right-skewed.
#     The normal curve is a rough approximation, still below the np >= 10
#     guideline.
#   - n = 300: np = 36 and n(1-p) = 264, both well above 10. The histogram has
#     many narrow bars (39 distinct values), is nearly symmetric (skewness
#     ~0.12), and follows the N(0,1) curve closely.
#   - Conclusion: as n increases, the standardized sum of independent Bernoulli
#     returns gets closer to a standard normal, exactly as the CLT predicts.
#     The normal approximation is reasonable for n = 300, marginal for n = 50,
#     and not reasonable for n = 10. This supports using it in Part C, where
#     n = 300.


# =============================================================================
# Part C - Estimate the Risk of Exceeding Capacity
# =============================================================================
header("PART C - Risk of Exceeding Capacity: P(S_300 > 45)")

# Exceeding capacity means 46 or more returns. With the continuity correction
# the boundary between 45 and 46 on the continuous normal scale is 45.5.
boundary = CAPACITY + 0.5
z_cc = (boundary - expected_returns) / sd_returns
p_approx = norm.sf(z_cc)                         # 1 - Phi(z)

# Exact binomial: P(S_300 > 45) = P(S_300 >= 46) = 1 - P(S_300 <= 45)
p_exact = binom.sf(CAPACITY, N_ORDERS, p)

abs_error = abs(p_approx - p_exact)

print(f"Continuity-corrected z = ({boundary} - {expected_returns:.0f}) / {sd_returns:.4f} "
      f"= {z_cc:.4f}")
print(f"Normal approximation   P(S_300 > 45) ~ 1 - Phi({z_cc:.4f}) = {p_approx:.6f} "
      f"({p_approx:.2%})")
print(f"Exact binomial         P(S_300 > 45) = binom.sf(45, 300, 0.12) = {p_exact:.6f} "
      f"({p_exact:.2%})")
print(f"Absolute error         |P_approx - P_exact| = {abs_error:.6f} "
      f"({abs_error * 100:.2f} percentage points)")

# --- Check against the 2% overflow-risk target ---
TARGET = 0.02
print(f"\nTarget: overflow risk no more than {TARGET:.0%}")
print(f"  Normal approximation meets target? {'YES' if p_approx <= TARGET else 'NO'} "
      f"({p_approx:.2%} vs {TARGET:.0%})")
print(f"  Exact binomial meets target?       {'YES' if p_exact <= TARGET else 'NO'} "
      f"({p_exact:.2%} vs {TARGET:.0%})")

# --- Recommendation: smallest capacity that meets the 2% target ---
# Search capacities c = 45, 46, ... until P(S_300 > c) <= 2%.
c_exact = CAPACITY
while binom.sf(c_exact, N_ORDERS, p) > TARGET:
    c_exact += 1
c_approx = CAPACITY
while norm.sf((c_approx + 0.5 - expected_returns) / sd_returns) > TARGET:
    c_approx += 1

print(f"\nSmallest capacity meeting the {TARGET:.0%} target:")
print(f"  Exact binomial:       {c_exact} returns  "
      f"(P(S_300 > {c_exact}) = {binom.sf(c_exact, N_ORDERS, p):.2%})")
print(f"  Normal approximation: {c_approx} returns  "
      f"(P(S_300 > {c_approx}) ~ "
      f"{norm.sf((c_approx + 0.5 - expected_returns) / sd_returns):.2%})")

# Interpretation (Part C):
#   - Normal approximation: P(S_300 > 45) ~ 4.57% (z ~ 1.688).
#     Exact binomial:       P(S_300 > 45) = 4.93%.
#     Absolute error ~ 0.0036 (0.36 percentage points). The CLT answer is very
#     close to the exact one, so the approximation works well at n = 300.
#   - The approximation is slightly LOWER than the exact value. The binomial
#     with p = 0.12 is still a little right-skewed (seen in Part B), so its
#     upper tail is a bit heavier than the symmetric normal curve assumes. The
#     approximation slightly understates the risk.
#   - Capacity 45 does NOT meet the 2% overflow-risk target. The exact risk is
#     about 4.9%, roughly 1 promotion in 20, which is about 2.5 times the
#     target. The approximate result (4.57%) also fails the target, so the
#     approximate and exact results support the same conclusion.
#   - Recommendation: raise capacity to at least 48 returns. Under the exact
#     binomial, P(S_300 > 48) ~ 1.60%, which meets the 2% target, and the
#     normal approximation also points to 48. If capacity must stay at 45, the
#     retailer should have a backup plan (temporary staff or overflow
#     processing) for the roughly 1-in-20 chance of going over.

plt.show()
