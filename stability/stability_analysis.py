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

# ============================================================================
# SETUP: Grid and Initial Conditions
# ============================================================================
N = 50
dx = 1.0 / N
dy = dx  # Square grid

# Create a Gaussian bump as initial condition
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
xx, yy = np.meshgrid(x, y)
T0 = np.exp(-200 * ((xx - 0.5)**2 + (yy - 0.5)**2)).astype(float)

# Test with different diffusion coefficients
alphas = [0.5, 1.0, 2.0]  # Different thermal diffusivity values


# ============================================================================
# METHOD 1: EXPLICIT (FORWARD EULER) METHOD
# ============================================================================
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
        
        # Check for instability
        if not np.isfinite(T).all():
            maxvals.append(np.inf)
            return T, maxvals, False  # Unstable
        
        maxvals.append(np.max(np.abs(T)))
    
    # Check if solution grew significantly (indication of instability)
    if maxvals[-1] > 10 * maxvals[0]:
        return T, maxvals, False
    
    return T, maxvals, True


# ============================================================================
# METHOD 2: CRANK-NICOLSON (ADI) IMPLICIT METHOD
# ============================================================================
def tri_disc(N, a):
    """
    Creates the 1D implicit operator matrix (I - a*D_xx)
    with Neumann boundary conditions.
    """
    M = (np.diag(-a * np.ones(N-1), k=-1) +
         np.diag((1+2*a) * np.ones(N), k=0) +
         np.diag(-a * np.ones(N-1), k=1))
    
    # Neumann BC: du/dx = 0 at boundaries
    M[0, 0] = 1+a
    M[0, 1] = -a
    M[-1, -1] = 1+a
    M[-1, -2] = -a
    
    return M


def crank_nicolson_step(T0, alpha, dx, dt, steps):
    """
    Crank-Nicolson (ADI) method for heat equation.
    Theoretically unconditionally stable.
    """
    N = T0.shape[0]
    h = dx
    k = alpha * dt / (2 * h**2)
    
    # Setup matrices
    A = tri_disc(N, k)
    C = tri_disc(N, k)
    
    try:
        A_inv = np.linalg.inv(A)
        C_inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return T0, [np.inf], False  # Matrix singular
    
    u = T0.copy()
    u_star = np.zeros((N, N), dtype=np.float64)
    maxvals = [np.max(np.abs(u))]
    
    for it in range(steps):
        # Pad for boundary handling
        u_padded = np.pad(u, 1, mode='edge')
        
        # Get neighbors
        S_north = u_padded[0:-2, 1:-1]
        S_south = u_padded[2:, 1:-1]
        S_west = u_padded[1:-1, 0:-2]
        S_east = u_padded[1:-1, 2:]
        S_center = (1 - 4*k) * u
        
        # Combine
        S = k * (S_north + S_south + S_west + S_east) + S_center
        
        # Implicit solve along x
        u_star[1:-1] = np.dot(S[1:-1], A_inv.T)
        
        # Boundary conditions
        u_star[0, 0] = k*(u[1, 0] - 2*u[0, 0] + u[0, 1]) + u[0, 0]
        u_star[0, -1] = k*(u[1, -1] - 2*u[0, -1] + u[0, -2]) + u[0, -1]
        u_star[-1, 0] = k*(u[-2, 0] - 2*u[-1, 0] + u[-1, 1]) + u[-1, 0]
        u_star[-1, -1] = k*(u[-2, -1] - 2*u[-1, -1] + u[-1, -2]) + u[-1, -1]
        u_star[0, 1:-1] = k*(-3*u[0, 1:-1] + u[1, 1:-1] + u[0, :-2] + u[0, 2:]) + u[0, 1:-1]
        u_star[-1, 1:-1] = k*(-3*u[-1, 1:-1] + u[-2, 1:-1] + u[-1, :-2] + u[-1, 2:]) + u[-1, 1:-1]
        
        # Implicit solve along y
        u = np.dot(C_inv, u_star.T).T
        
        # Check for instability
        if not np.isfinite(u).all():
            maxvals.append(np.inf)
            return u, maxvals, False
        
        maxvals.append(np.max(np.abs(u)))
    
    return u, maxvals, True


# ============================================================================
# STABILITY TESTING
# ============================================================================
def test_stability(method_func, method_name, T0, dx, alphas, dts, steps=100):
    """Test stability for a given method across different parameters."""
    print(f"\nTesting {method_name}...")
    results = {}
    stability_map = {}
    
    for alpha in alphas:
        max_last = []
        is_stable = []
        
        for dt in dts:
            _, maxvals, stable = method_func(T0, alpha, dx, dt, steps)
            max_last.append(maxvals[-1] if np.isfinite(maxvals[-1]) else 1e10)
            is_stable.append(stable)
        
        results[alpha] = np.array(max_last, dtype=float)
        stability_map[alpha] = is_stable
    
    return results, stability_map


# ============================================================================
# RUN STABILITY TESTS
# ============================================================================
print("="*80)
print("COMPREHENSIVE STABILITY ANALYSIS")
print("="*80)
print(f"Grid size: {N}x{N}")
print(f"Grid spacing: dx = dy = {dx:.4f}")
print(f"Thermal diffusivities (α): {alphas}")

# Range of time steps to test
dts = np.logspace(-8, -1, 50)
steps = 100

# Test explicit method
explicit_results, explicit_stability = test_stability(
    explicit_heat_step, "Explicit (Forward Euler)", T0, dx, alphas, dts, steps
)

# Test Crank-Nicolson method
cn_results, cn_stability = test_stability(
    crank_nicolson_step, "Crank-Nicolson (ADI)", T0, dx, alphas, dts, steps
)


# ============================================================================
# ANALYSIS AND PLOTTING
# ============================================================================
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Color scheme
colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green

# ============================================================================
# Plot 1: Explicit Method Stability
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])
for i, alpha in enumerate(alphas):
    yvals = explicit_results[alpha]
    yvals = np.where(np.isfinite(yvals), yvals, 1e10)
    
    ax1.plot(dts, yvals, marker='o', markersize=4, 
             label=f"α={alpha}", color=colors[i], linewidth=2)
    
    # Theoretical stability limit for explicit method: dt_crit = dx²/(4*alpha)
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

# ============================================================================
# Plot 2: Crank-Nicolson Method Stability
# ============================================================================
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

# ============================================================================
# Plot 3: Stability Regions Comparison
# ============================================================================
ax3 = fig.add_subplot(gs[1, :])

for i, alpha in enumerate(alphas):
    # Explicit stability
    explicit_stable = np.array(explicit_stability[alpha], dtype=int)
    ax3.plot(dts, explicit_stable + i*2.5 + 0.1, 'o-', 
             label=f"Explicit α={alpha}", color=colors[i], 
             markersize=6, linewidth=2, alpha=0.8)
    
    # Mark critical dt for explicit
    dt_crit = dx**2 / (4 * alpha)
    ax3.axvline(dt_crit, ls='--', alpha=0.5, color=colors[i], linewidth=2)
    
    # Crank-Nicolson stability
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

# Add theoretical annotations
ax3.text(0.02, 0.98, 
         "Explicit Stability Criterion: dt < dx²/(4α)\n" +
         "Crank-Nicolson: Unconditionally stable",
         transform=ax3.transAxes, fontsize=11,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle("Comprehensive Stability Analysis: All Methods", 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('/Users/jasonfan/Downloads/Physics_77_Heat_Diffusion/comprehensive_stability_analysis.png',
            dpi=300, bbox_inches='tight')
print("\n" + "="*80)
print("Plot saved as 'comprehensive_stability_analysis.png'")
print("="*80)


# ============================================================================
# PRINT SUMMARY TABLE
# ============================================================================
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
    
    # Find stable range
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
