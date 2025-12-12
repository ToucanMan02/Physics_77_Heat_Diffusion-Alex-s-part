import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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



plt.figure(figsize=(12, 4))
plt.suptitle("Initial State and Simulated Final State After 100ms", 
             fontsize=16, fontweight='bold', y=0.995)
T_explicit, _, _ = explicit_heat_step(T0, alphas[0], dx, dts[0], steps[0])
T_ADI, _, _ = crank_nicolson_step(T0, alphas[0], dx, dts[0], steps[0])

# 1. Initial condition
plt.subplot(1, 3, 1)
plt.imshow(T0, cmap='jet', extent=[0,1,0,1], origin='lower')
plt.title("Initial State $U_0$")
plt.colorbar(label='Temperature')

# 2. Explicit solution
plt.subplot(1, 3, 2)
plt.imshow(T_explicit, cmap='jet', extent=[0,1,0,1], origin='lower')
plt.title(f"Explicit Final State\n$\\Delta t = {dts[0]:.2e}$")
plt.colorbar(label='Temperature')

# 3. ADI solution
plt.subplot(1, 3, 3)
plt.imshow(T_ADI, cmap='jet', extent=[0,1,0,1], origin='lower')
plt.title(f"ADI Final State\n$\\Delta t = {dts[0]:.2e}$")
plt.colorbar(label='Temperature')

plt.tight_layout()
plt.savefig("comparison_explicit_adi_initial_stability.png")
print("Saved comparison_explicit_adi_initial_stability.png")

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