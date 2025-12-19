import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

Nx, Ny = 100, 100
Lx, Ly = 1.0, 1.0
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
h = dx
alpha = 2.1e-5 
T_ambient = 295.0

# --- 1. ANALYTICAL SOLUTION & INITIAL CONDITIONS (Uniform Material) ---
sigma0 = 0.05
def analytical_solution(X, Y, t, alpha, sigma0):
    numerator = sigma0**2
    denominator = sigma0**2 + 4 * alpha * t
    exponent = -((X - 0.5)**2 + (Y - 0.5)**2) / denominator
    U_exact = T_ambient + 10 * (numerator / denominator) * np.exp(exponent)
    return U_exact

U0 = analytical_solution(X, Y, t=1e-10, alpha=alpha, sigma0=sigma0)

# --- 2. DISCRETIZATION MATRIX FOR IMPLICIT SOLVER (ADI) ---
def tri_disc(N, a):
    M = (np.diag(-a * np.ones(N-1), k=-1) +
         np.diag((1 + 2*a) * np.ones(N), k=0) +
         np.diag(-a * np.ones(N-1), k=1))
    M[0, 0] = 1 + a
    M[0, 1] = -a
    M[-1, -1] = 1 + a
    M[-1, -2] = -a
    return M

# --- 3. NUMERICAL SOLVERS ---
def apply_neumann(U):
    U[:,0] = U[:,1]
    U[:,-1] = U[:,-2]
    U[0,:] = U[1,:]
    U[-1,:] = U[-2,:]
    return U

def solve_explicit(U_initial, dt, t_final, alpha, h):
    u = np.copy(U_initial)
    max_steps = int(t_final / dt)
    C = alpha * dt / h**2 
    
    if C > 0.25:
        return None 

    for _ in range(max_steps):
        laplacian = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]) / h**2
        u[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * alpha * laplacian
        u = apply_neumann(u)
    return u

def solve_implicit_adi(U_initial, dt, t_final, alpha, h):
    u = np.copy(U_initial)
    max_steps = int(t_final / dt)
    k = alpha * dt / (2 * h**2)
    A = tri_disc(Nx, k)
    
    for _ in range(max_steps):
        u_padded = np.pad(u, 1, mode='edge')
        U_y_plus = u_padded[1:-1, 2:]
        U_y_minus = u_padded[1:-1, 0:-2]
        RHS1 = u + k * (U_y_minus - 2.0 * u + U_y_plus)
        u_star = solve(A, RHS1) 

        u_star_padded = np.pad(u_star, 1, mode='edge')
        U_x_plus = u_star_padded[2:, 1:-1]
        U_x_minus = u_star_padded[0:-2, 1:-1]
        RHS2 = u_star + k * (U_x_minus - 2.0 * u_star + U_x_plus)
        u = solve(A, RHS2.T).T
    
    return u

# --- 4. ERROR ANALYSIS AND PLOTTING ---
def calculate_error(U_numerical, U_exact):
    return np.sqrt(np.mean((U_numerical - U_exact)**2))

def run_simulation_and_plot_error():
    t_final = 0.5
    # Range of time steps from small (accurate) to large (testing stability)
    dt_range = np.logspace(-3, 1, 30)
    
    U_exact_final = analytical_solution(X, Y, t_final, alpha, sigma0)
    
    error_explicit = []
    error_adi = []
    
    print("Starting error analysis...")
    
    for dt in dt_range:
        # 1. Explicit Solver
        U_explicit = solve_explicit(U0, dt, t_final, alpha, h)
        if U_explicit is not None:
            err_exp = calculate_error(U_explicit, U_exact_final)
            error_explicit.append(err_exp)
        else:
            error_explicit.append(np.nan)

        # 2. ADI Implicit Solver
        U_adi = solve_implicit_adi(U0, dt, t_final, alpha, h)
        err_adi = calculate_error(U_adi, U_exact_final)
        error_adi.append(err_adi)
    
    # --- Plotting the Results ---
    plt.figure(figsize=(9, 6))
    
    # Plotting error vs time step on a log-log scale
    plt.loglog(dt_range, error_explicit, 'o-', label='Explicit (Forward Euler)', color='red')
    plt.loglog(dt_range, error_adi, 's-', label='Implicit (ADI)', color='blue')
    
    # Adding the theoretical convergence line (First Order O(dt))
    # We find the first non-NaN error for ADI to start the slope
    first_valid_adi_idx = np.where(~np.isnan(error_adi))[0][0]
    dt_start = dt_range[first_valid_adi_idx]
    err_start = error_adi[first_valid_adi_idx]
    first_order_slope = err_start * (dt_range / dt_start)
    plt.loglog(dt_range, first_order_slope, 'k--', label=r'Theoretical $O(\Delta t)$ Slope', alpha=0.6)

    # Adding the stability limit for the explicit solver (CFL condition)
    dt_cfl = 0.25 * h**2 / alpha
    plt.axvline(dt_cfl, color='red', linestyle=':', label=r'Explicit Stability Limit ($\Delta t_{CFL}$)')
    
    plt.title('Time Step ($\Delta t$) vs. $L_2$ Error (Accuracy)')
    plt.xlabel('Time Step $\Delta t$ (s)')
    plt.ylabel(r'$L_2$ Error ($\sqrt{\sum (U_{num} - U_{exact})^2 / N^2}$)')
    plt.legend()
    plt.grid(which='both', linestyle='--', alpha=0.5)
    plt.savefig('error_vs_timestep_spaceship.png')
    
    print("\n--- Results ---")
    print(f"Explicit Stability Limit (dt_CFL): {dt_cfl:.2e}")
    print("Plot saved as error_vs_timestep_spaceship.png")

if __name__ == '__main__':
    run_simulation_and_plot_error()
