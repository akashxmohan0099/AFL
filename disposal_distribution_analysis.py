"""Step 1: Analyse the distribution of disposals in 2024 data."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
import config

df = pd.read_parquet(config.FEATURES_DIR / "feature_matrix.parquet")
df = df[(df["year"] == 2024) & (df["did_not_play"] == False)].copy()
di = df["DI"].values

mu = np.mean(di)
var = np.var(di)
sigma = np.std(di)
vr = var / mu

print(f"2024 Disposals (n={len(di)})")
print(f"  Mean:             {mu:.2f}")
print(f"  Median:           {np.median(di):.1f}")
print(f"  Std:              {sigma:.2f}")
print(f"  Variance:         {var:.2f}")
print(f"  Var/Mean ratio:   {vr:.2f}  (Poisson expects 1.0)")
print(f"  Skewness:         {stats.skew(di):.3f}")
print(f"  Kurtosis:         {stats.kurtosis(di):.3f}")
print(f"  Min:              {int(np.min(di))}")
print(f"  Max:              {int(np.max(di))}")
print(f"  P(DI=0):          {(di==0).mean():.3f}")

print(f"\n  DIAGNOSIS:")
if vr > 2:
    print(f"  Variance/Mean = {vr:.1f} >> 1.0 → OVERDISPERSED (Poisson is wrong)")
    print(f"  Poisson assumes Var=Mean, but actual Var is {vr:.1f}x the mean")
    print(f"  Gaussian or NegBin would be better fits")
else:
    print(f"  Variance/Mean = {vr:.1f} ≈ 1.0 → Poisson is reasonable")

# Fit NegBin: method of moments
# NegBin: mean = r*p/(1-p), var = r*p/(1-p)^2
# So var/mean = 1/(1-p), p = 1 - mean/var
# r = mean^2 / (var - mean)
p_nb = 1 - mu / var  # probability of failure
r_nb = mu ** 2 / (var - mu)  # number of successes
print(f"\n  NegBin fit (method of moments):")
print(f"    r = {r_nb:.2f}, p = {p_nb:.4f}")
print(f"    NegBin variance: {r_nb * (1 - p_nb) / p_nb**2:.2f} (target: {var:.2f})")

# Histogram
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

x = np.arange(0, 45)

# Actual distribution with overlays
axes[0].hist(di, bins=range(0, 46), density=True, alpha=0.7, color="steelblue", edgecolor="white")
# Poisson
poisson_pmf = stats.poisson.pmf(x, mu)
axes[0].plot(x, poisson_pmf, "r-", lw=2, label=f"Poisson(λ={mu:.1f})")
# Gaussian
gauss_pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
axes[0].plot(x, gauss_pdf, "g-", lw=2, label=f"Gaussian(μ={mu:.1f}, σ={sigma:.1f})")
# NegBin (scipy parameterization: nbinom(n, p) where n=r, p=success prob)
nb_pmf = stats.nbinom.pmf(x, r_nb, mu / var)  # p_success = mu/var
axes[0].plot(x, nb_pmf, "m-", lw=2, label=f"NegBin(r={r_nb:.1f})")
axes[0].legend(fontsize=9)
axes[0].set_title(f"Actual Disposals (2024)\nMean={mu:.1f}, Var={var:.1f}, Var/Mean={vr:.1f}")
axes[0].set_xlabel("Disposals")
axes[0].set_ylabel("Density")

# Q-Q plot vs Normal
axes[1].set_title("Q-Q Plot vs Normal")
stats.probplot(di, dist="norm", plot=axes[1])

# Variance vs Mean by disposal level
bins_edges = np.arange(0, 40, 5)
means_per_bin = []
vars_per_bin = []
for i in range(len(bins_edges) - 1):
    mask = (di >= bins_edges[i]) & (di < bins_edges[i + 1])
    if mask.sum() > 20:
        means_per_bin.append(di[mask].mean())
        vars_per_bin.append(di[mask].var())
axes[2].scatter(means_per_bin, vars_per_bin, s=80, c="steelblue", zorder=5)
axes[2].plot([0, 35], [0, 35], "r--", label="Var=Mean (Poisson)")
axes[2].plot([0, 35], [0, 35 * vr], "g--", alpha=0.5, label=f"Var={vr:.1f}×Mean (actual)")
axes[2].set_title("Variance vs Mean by Disposal Bin")
axes[2].set_xlabel("Bin Mean")
axes[2].set_ylabel("Bin Variance")
axes[2].legend()

plt.tight_layout()
plt.savefig("data/disposal_distribution_2024.png", dpi=150)
print(f"\nHistogram saved: data/disposal_distribution_2024.png")
