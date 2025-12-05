import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

"""
Comprehensive Stability Analysis for all Heat Diffusion Methods in the Project
===============================================================================

This script analyzes the stability of:
1. Explicit (Forward Euler) method
2. Crank-Nicolson (ADI) implicit method

For each method, we test with various time steps and compare against theoretical
stability criteria.
"""

N = 50
dx = 1.0 / N
dy = dx  

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
xx, yy = np.meshgrid(x, y)
T0 = np.exp(-200 * ((xx - 0.5)**2 + (yy - 0.5)**2)).astype(float)

alphas = [0.5, 1.0, 2.0] 

time = 0.1


# ============EXPLICIT===============================

def laplacian(T, dx):
    """Compute 2D Laplacian using finite differences with periodic BC."""
    return (
        np.roll(T, 1, axis=0) +
        np.roll(T, -1, axis=0) +
        np.roll(T, 1, axis=1) +
        np.roll(T, -1, axis=1) -
        4*T
    ) / dx**2


def explicit_heat_step(T, alpha, dx, dt, steps):
    """
    Explicit (Forward Euler) method for heat equation.
    Stability condition: dt < dx²/(4*alpha) for 2D
    """
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


# =================CN ADI==================================
def tri_disc(N, a):
    """
    Creates the 1D implicit operator matrix (I - a*D_xx)
    with Neumann boundary conditions.
    """
    M = (np.diag(-a * np.ones(N-1), k=-1) +
         np.diag((1+2*a) * np.ones(N), k=0) +
         np.diag(-a * np.ones(N-1), k=1))
    
    M[0, 0] = 1+a
    M[0, 1] = -a
    M[-1, -1] = 1+a
    M[-1, -2] = -a
    
    return M


def crank_nicolson_step(T0, alpha, dx, dt, steps):
    """
    Crank-Nicolson (ADI) method for heat equation.
    Uses Alternating Direction Implicit (ADI) splitting with direct linear solvers.
    """
    N = T0.shape[0]
    h = dx
    k = alpha * dt / (2 * h**2)
    
    A = tri_disc(N, k)
    C = tri_disc(N, k)
    
    u = T0.copy().astype(float)
    maxvals = [np.max(np.abs(u))]
    
    for it in range(steps):
       
        u_padded = np.pad(u, 1, mode='edge')
        U_y_plus  = u_padded[1:-1, 2:]   
        U_y_minus = u_padded[1:-1, 0:-2] 
        
        RHS1 = u + k * (U_y_minus - 2.0*u + U_y_plus)

        try:
            u_star = np.linalg.solve(A, RHS1)
        except np.linalg.LinAlgError:
            maxvals.append(np.inf)
            return u, maxvals, False

        u_padded = np.pad(u_star, 1, mode='edge')
        U_x_plus  = u_padded[2:, 1:-1]   # u[i+1, j]
        U_x_minus = u_padded[0:-2, 1:-1] # u[i-1, j]
        
        RHS2 = u_star + k * (U_x_minus - 2.0*u_star + U_x_plus)

        try:
            u = np.linalg.solve(C, RHS2.T).T
        except np.linalg.LinAlgError:
            maxvals.append(np.inf)
            return u, maxvals, False
        
        if not np.isfinite(u).all():
            maxvals.append(np.inf)
            return u, maxvals, False
        
        maxvals.append(np.max(np.abs(u)))
    
    return u, maxvals, True


def test_stability(method_func, method_name, T0, dx, alphas, dts, steps=100):
    """Test stability for a given method across different parameters."""
    print(f"\nTesting {method_name}...")
    results = {}
    stability_map = {}
    
    for alpha in alphas:
        max_last = []
        is_stable = []
        i_stability = 0
        for dt in dts:
            _, maxvals, stable = method_func(T0, alpha, dx, dt, steps[i_stability])
            max_last.append(maxvals[-1] if np.isfinite(maxvals[-1]) else 1e10)
            is_stable.append(stable)
            i_stability += 1
        
        results[alpha] = np.array(max_last, dtype=float)
        stability_map[alpha] = is_stable
    
    return results, stability_map


print("="*80)
print("COMPREHENSIVE STABILITY ANALYSIS")
print("="*80)
print(f"Grid size: {N}x{N}")
print(f"Grid spacing: dx = dy = {dx:.4f}")
print(f"Thermal diffusivities (α): {alphas}")

dts = np.logspace(-5, -3, 50)
steps = (time // dts)
steps = steps.astype(int)
print(f'steps: {steps}')

explicit_results, explicit_stability = test_stability(
    explicit_heat_step, "Explicit (Forward Euler)", T0, dx, alphas, dts, steps
)



cn_results, cn_stability = test_stability(
    crank_nicolson_step, "Crank-Nicolson (ADI)", T0, dx, alphas, dts, steps
)


fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green

ax1 = fig.add_subplot(gs[0, 0])
for i, alpha in enumerate(alphas):
    yvals = explicit_results[alpha]
    yvals = np.where(np.isfinite(yvals), yvals, 1e10)
    
    ax1.plot(dts, yvals, marker='o', markersize=4, 
             label=f"α={alpha}", color=colors[i], linewidth=2)
    
    dt_crit = dx**2 / (4 * alpha)
    ax1.axvline(dt_crit, ls='--', alpha=0.7, color=colors[i], linewidth=2,
                label=f"α={alpha} critical: dt={dt_crit:.2e}")

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Time step (dt)", fontsize=12, fontweight='bold')
ax1.set_ylabel("max |T| after simulation", fontsize=12, fontweight='bold')
ax1.set_title("Explicit Method Stability Analysis", fontsize=14, fontweight='bold')
ax1.grid(True, which='both', ls='--', alpha=0.3)
ax1.legend(fontsize=9, loc='upper left')
ax1.set_ylim([1e-3, 1e10])

ax2 = fig.add_subplot(gs[0, 1])
for i, alpha in enumerate(alphas):
    yvals = cn_results[alpha]
    yvals = np.where(np.isfinite(yvals), yvals, 1e10)
    
    ax2.plot(dts, yvals, marker='o', markersize=4,
             label=f"α={alpha}", color=colors[i], linewidth=2)

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Time step (dt)", fontsize=12, fontweight='bold')
ax2.set_ylabel("max |T| after simulation", fontsize=12, fontweight='bold')
ax2.set_title("Crank-Nicolson (ADI) Stability Analysis\n(Unconditionally Stable)",
              fontsize=14, fontweight='bold')
ax2.grid(True, which='both', ls='--', alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_ylim([1e-3, 1e10])

ax3 = fig.add_subplot(gs[1, :])

for i, alpha in enumerate(alphas):
    explicit_stable = np.array(explicit_stability[alpha], dtype=int)
    ax3.plot(dts, explicit_stable + i*2.5 + 0.1, 'o-', 
             label=f"Explicit α={alpha}", color=colors[i], 
             markersize=6, linewidth=2, alpha=0.8)
    
    dt_crit = dx**2 / (4 * alpha)
    ax3.axvline(dt_crit, ls='--', alpha=0.5, color=colors[i], linewidth=2)
    
    cn_stable = np.array(cn_stability[alpha], dtype=int)
    ax3.plot(dts, cn_stable + i*2.5 - 0.1, 's-',
             label=f"Crank-Nicolson α={alpha}", color=colors[i],
             markersize=6, linewidth=2, alpha=0.4)

ax3.set_xscale("log")
ax3.set_xlabel("Time step (dt)", fontsize=12, fontweight='bold')
ax3.set_ylabel("Stable (1) vs Unstable (0)", fontsize=12, fontweight='bold')
ax3.set_title("Stability Regions: Explicit vs Crank-Nicolson", 
              fontsize=14, fontweight='bold')
ax3.grid(True, which='both', ls='--', alpha=0.3)
ax3.legend(fontsize=9, ncol=2)
ax3.set_ylim([-0.5, 7])

ax3.text(0.02, 0.98, 
         "Explicit Stability Criterion: dt < dx²/(4α)\n" +
         "Crank-Nicolson: Unconditionally stable",
         transform=ax3.transAxes, fontsize=11,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle("Comprehensive Stability Analysis: All Methods", 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('comprehensive_stability_analysis.png',
            dpi=300, bbox_inches='tight')
print("\n" + "="*80)
print("Plot saved as 'comprehensive_stability_analysis.png'")
print("="*80)


print("\n" + "="*80)
print("STABILITY SUMMARY")
print("="*80)
print(f"\nGrid spacing: dx = dy = {dx:.6f}")
print(f"Number of time steps tested: {len(dts)}")
print(f"Steps per simulation: {steps}")

print("\n" + "-"*80)
print("EXPLICIT METHOD (Forward Euler)")
print("-"*80)
print(f"{'Alpha':<10} {'dt_critical':<15} {'Stable Range':<30} {'Stability'}")
print("-"*80)

for alpha in alphas:
    dt_crit = dx**2 / (4 * alpha)
    stable_count = sum(explicit_stability[alpha])
    total_count = len(explicit_stability[alpha])
    stable_pct = 100 * stable_count / total_count
    
    stable_dts = dts[np.array(explicit_stability[alpha])]
    if len(stable_dts) > 0:
        stable_range = f"dt < {np.max(stable_dts):.2e}"
    else:
        stable_range = "None"
    
    print(f"{alpha:<10.1f} {dt_crit:<15.2e} {stable_range:<30} {stable_pct:.1f}% ({stable_count}/{total_count})")

print("\n" + "-"*80)
print("CRANK-NICOLSON (ADI) METHOD")
print("-"*80)
print(f"{'Alpha':<10} {'Theoretical':<20} {'Observed Stability'}")
print("-"*80)

for alpha in alphas:
    stable_count = sum(cn_stability[alpha])
    total_count = len(cn_stability[alpha])
    stable_pct = 100 * stable_count / total_count
    
    print(f"{alpha:<10.1f} {'Unconditionally Stable':<20} {stable_pct:.1f}% ({stable_count}/{total_count})")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print("1. Explicit Method: Stable only when dt < dx²/(4α)")
print("   - Larger α requires smaller dt for stability")
print("   - Violating this condition leads to exponential growth")
print("")
print("2. Crank-Nicolson (ADI): Unconditionally stable")
print("   - Can use much larger time steps")
print("   - Better for long-time simulations")
print("   - Requires solving linear systems (more computation per step)")
print("")
print("3. Trade-offs:")
print("   - Explicit: Simple, fast per step, but limited by stability")
print("   - Crank-Nicolson: More complex, but allows larger dt")
print("="*80)
plt.show()
