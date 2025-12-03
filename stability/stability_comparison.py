import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# SETUP
# ============================================================================
N = 50
dx = 1.0 / N
dy = dx
alpha = 1.0  # Selected single alpha

# Initial Conditions (Gaussian Bump)
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
xx, yy = np.meshgrid(x, y)
T0 = np.exp(-200 * ((xx - 0.5)**2 + (yy - 0.5)**2)).astype(float)

# ============================================================================
# EXPLICIT METHOD
# ============================================================================
def laplacian(T, dx):
    return (
        np.roll(T, 1, axis=0) +
        np.roll(T, -1, axis=0) +
        np.roll(T, 1, axis=1) +
        np.roll(T, -1, axis=1) -
        4*T
    ) / dx**2

def explicit_heat_step(T, alpha, dx, dt, steps):
    T = T.copy().astype(float)
    maxvals = [np.max(np.abs(T))]
    
    for _ in range(steps):
        T = T + alpha * dt * laplacian(T, dx)
        if not np.isfinite(T).all():
            maxvals.append(np.inf)
            return T, maxvals, False
        maxvals.append(np.max(np.abs(T)))
    
    if maxvals[-1] > 10 * maxvals[0]:
        return T, maxvals, False
    return T, maxvals, True

# ============================================================================
# CRANK-NICOLSON (ADI) METHOD
# ============================================================================
def tri_disc(N, a):
    M = (np.diag(-a * np.ones(N-1), k=-1) +
         np.diag((1+2*a) * np.ones(N), k=0) +
         np.diag(-a * np.ones(N-1), k=1))
    M[0, 0] = 1+a; M[0, 1] = -a
    M[-1, -1] = 1+a; M[-1, -2] = -a
    return M

def crank_nicolson_step(T0, alpha, dx, dt, steps):
    N = T0.shape[0]
    h = dx
    k = alpha * dt / (2 * h**2)
    
    A = tri_disc(N, k)
    C = tri_disc(N, k)
    u = T0.copy().astype(float)
    maxvals = [np.max(np.abs(u))]
    
    for it in range(steps):
        u_padded = np.pad(u, 1, mode='edge')
        RHS1 = u + k * (u_padded[1:-1, 0:-2] - 2.0*u + u_padded[1:-1, 2:])
        try:
            u_star = np.linalg.solve(A, RHS1)
        except np.linalg.LinAlgError:
            return u, [np.inf], False

        u_padded = np.pad(u_star, 1, mode='edge')
        RHS2 = u_star + k * (u_padded[0:-2, 1:-1] - 2.0*u_star + u_padded[2:, 1:-1])
        try:
            u = np.linalg.solve(C, RHS2.T).T
        except np.linalg.LinAlgError:
            return u, [np.inf], False
        
        if not np.isfinite(u).all():
            return u, [np.inf], False
        maxvals.append(np.max(np.abs(u)))
            
    return u, maxvals, True

print(f"Running comparison for alpha = {alpha}")
dts = np.logspace(-6, -1, 50)
steps = 100

explicit_max = []
cn_max = []

for dt in dts:
    # Run Explicit
    _, vals_exp, _ = explicit_heat_step(T0, alpha, dx, dt, steps)
    explicit_max.append(vals_exp[-1] if np.isfinite(vals_exp[-1]) else 1e10)
    
    # Run CN
    _, vals_cn, _ = crank_nicolson_step(T0, alpha, dx, dt, steps)
    cn_max.append(vals_cn[-1] if np.isfinite(vals_cn[-1]) else 1e10)

# ============================================================================
# PLOTTING
# ============================================================================
plt.figure(figsize=(10, 6))

plt.plot(dts, explicit_max, 'o-', color='#e74c3c', label='Explicit (Forward Euler)', linewidth=2, markersize=5)

plt.plot(dts, cn_max, 's-', color='#3498db', label='Crank-Nicolson (ADI)', linewidth=2, markersize=5)

dt_crit = dx**2 / (4 * alpha)
plt.axvline(dt_crit, color='black', linestyle='--', alpha=0.7, label=f'Explicit Stability Limit\n(dt={dt_crit:.2e})')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time Step (dt)', fontsize=12, fontweight='bold')
plt.ylabel('Max Temperature after 100 steps', fontsize=12, fontweight='bold')
plt.title(f'Stability Comparison (alpha={alpha})', fontsize=14, fontweight='bold')
plt.ylim(1e-3, 10) 
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('single_alpha_comparison.png')
print("Plot saved as single_alpha_comparison.png")
plt.show()