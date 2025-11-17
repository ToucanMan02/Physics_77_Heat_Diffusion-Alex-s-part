import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 0) GRID + INITIAL CONDITION
# ------------------------------------------------------
N = 50
dx = 1.0 / N

# Gaussian bump
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
xx, yy = np.meshgrid(x, y)
T0 = np.exp(-200 * ((xx - 0.5)**2 + (yy - 0.5)**2)).astype(float)


# ------------------------------------------------------
# 1) LAPLACIAN
# ------------------------------------------------------
def laplacian(T, dx):
    return (
        np.roll(T, 1, axis=0)
      + np.roll(T, -1, axis=0)
      + np.roll(T, 1, axis=1)
      + np.roll(T, -1, axis=1)
      - 4*T
    ) / dx**2


# ------------------------------------------------------
# 2) EXPLICIT HEAT SOLVER
# ------------------------------------------------------
def simulate_heat(T0, alpha, dx, dt, steps):
    T = T0.copy().astype(float)
    maxvals = []

    for _ in range(steps):
        T = T + alpha * dt * laplacian(T, dx)

        # If unstable, return inf
        if not np.isfinite(T).all():
            maxvals.append(np.inf)
            return T, maxvals

        maxvals.append(np.max(np.abs(T)))

    return T, maxvals


# ------------------------------------------------------
# 3) STABILITY TEST
# ------------------------------------------------------
dts = np.logspace(-8, -1, 40)
alphas = [a1, a2, a3]

 #//////////////CHANGE a1, a2, a3 according to Luke's explicit solver////////////// 

def stability_test_explicit(T0, dx, alphas, dts, steps=80):
    results = {}
    for alpha in alphas:
        max_last = [] 
        
       
        
        for dt in dts:
            _, maxvals = simulate_heat(T0, alpha, dx, dt, steps)
            max_last.append(maxvals[-1])
        results[alpha] = np.array(max_last, dtype=float)
    return results

stability_results = stability_test_explicit(T0, dx, alphas, dts)


# ------------------------------------------------------
# 4) FILTER ALL NON-FINITE VALUES (critical step!)
# ------------------------------------------------------
def safe_clip(a):
    """Replace nan and inf with a large finite number."""
    return np.where(np.isfinite(a), a, 1e8)

dts_clipped = safe_clip(dts)

for alpha in alphas:
    stability_results[alpha] = safe_clip(stability_results[alpha])


# ------------------------------------------------------
# 5) PLOT
# ------------------------------------------------------
plt.figure(figsize=(10, 6))

for alpha in alphas:
    yvals = stability_results[alpha]

    # Clip again for safety
    yvals = safe_clip(yvals)

    plt.plot(dts_clipped, yvals, marker='o', label=f"alpha={alpha}")

    # theoretical stability dt_crit
    dt_crit = dx**2 / (4 * alpha)

    # Protect against zero or negative or infinite dt_crit
    if np.isfinite(dt_crit) and dt_crit > 0:
        plt.axvline(dt_crit, ls='--', alpha=0.6, color='k')

# final failsafe
plt.xlim([dts_clipped.min(), dts_clipped.max()])
plt.ylim([1e-10, 1e7])  # keep finite

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Time step dt")
plt.ylabel("max |T| after simulation")
plt.title("Explicit Stability vs dt for Different α")
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.legend()
plt.show()
