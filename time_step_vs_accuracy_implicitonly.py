import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

alpha = 0.8 
N = 100 
L = 1.0     
h = L / N   

sigma0 = 0.05 # Initial width of the Gaussian
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

def analytical_solution(X, Y, t, alpha, sigma0):
    """
    Analytical solution for 2D diffusion of a Gaussian heat spot centered at (0.5, 0.5).
    """
    numerator = sigma0**2
    denominator = sigma0**2 + 4 * alpha * t
    exponent = -((X - 0.5)**2 + (Y - 0.5)**2) / denominator
    U_exact = (numerator / denominator) * np.exp(exponent)
    return U_exact

U0 = analytical_solution(X, Y, t=0, alpha=alpha, sigma0=sigma0)

# --- 4. DISCRETIZATION MATRIX FOR IMPLICIT SOLVER (ADI) ---
def tri_disc(N, a):
    """
    Creates the 1D implicit operator matrix (I - a*D_xx) with Neumann BCs.
    Note: 'a' here is the numerical parameter k = alpha*dt/(2*h^2)
    """
    M = (np.diag(-a * np.ones(N-1), k=-1) +
         np.diag((1 + 2*a) * np.ones(N), k=0) +
         np.diag(-a * np.ones(N-1), k=1))
    
    # Neumann Boundary Conditions (Insulated)
    M[0, 0] = 1 + a
    M[0, 1] = -a
    M[-1, -1] = 1 + a
    M[-1, -2] = -a
    
    return M

def solve_explicit(U_initial, dt, t_final, alpha, h):

    u = np.copy(U_initial)
    t = 0.0
    
    # Stability parameter for explicit method: C = alpha * dt / h^2
    C = alpha * dt / h**2 
    if C > 0.25:
        print(f"Warning: Explicit method unstable for dt={dt:.2e}. C={C:.2f}")

    # Use integer time steps
    max_steps = int(t_final / dt)

    for _ in range(max_steps):
        # Laplacian calculation (5-point stencil)
        laplacian = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]) / h**2
        
        # Update step: U_new = U_old + dt * alpha * Laplacian
        u[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * alpha * laplacian
        
        # Neumann Boundary Conditions (Insulated Edges)
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
        
        t += dt
    
    return u

def solve_implicit_adi(U_initial, dt, t_final, alpha, h):
    u = np.copy(U_initial)
    t = 0.0
    
    k = alpha * dt / (2 * h**2)
    
    A = tri_disc(N, k)
    
    max_steps = int(t_final / dt)
    
    for _ in range(max_steps):
        
        u_padded = np.pad(u, 1, mode='edge')
        U_y_plus = u_padded[1:-1, 2:]     
        U_y_minus = u_padded[1:-1, 0:-2]   
        RHS1 = u + k * (U_y_minus - 2.0*u + U_y_plus)
        
        u_star = solve(A, RHS1) 

        u_star_padded = np.pad(u_star, 1, mode='edge')
        U_x_plus = u_star_padded[2:, 1:-1]   
        U_x_minus = u_star_padded[0:-2, 1:-1] 
        RHS2 = u_star + k * (U_x_minus - 2.0*u_star + U_x_plus)
        
        u = solve(A, RHS2.T).T
        
        t += dt
        
    return u

def calculate_error(U_numerical, U_exact):
    """
    Calculates the L2 norm of the error (Root Mean Square Error).
    """
    return np.sqrt(np.mean((U_numerical[-1] - U_exact[-1])**2))

def run_simulation_and_plot_error():
    t_final = 0.005 
    dt_range = np.logspace(-6, -4, 20) 

    U_exact_final = analytical_solution(X, Y, t_final, alpha, sigma0)
    
    error_explicit = []
    error_adi = []
    
    for dt in dt_range:
        print(f"Testing dt = {dt:.2e}...")
        
        try:
            U_explicit = solve_explicit(U0, dt, t_final, alpha, h)
            err_exp = calculate_error(U_explicit, U_exact_final)
            error_explicit.append(err_exp)
        except Exception as e:

            print(f"Explicit unstable at dt = {dt:.2e}")
            error_explicit.append(np.nan)

        U_adi = solve_implicit_adi(U0, dt, t_final, alpha, h)
        err_adi = calculate_error(U_adi, U_exact_final)
        error_adi.append(err_adi)
    
    plt.figure(figsize=(9, 6))
    
    #plt.loglog(dt_range, error_explicit, 'o-', label='Explicit (Forward Euler)', color='red')
    plt.loglog(dt_range, error_adi, 's-', label='Implicit (ADI)', color='blue')

    dt_cfl = 0.25 * h**2 / alpha
    plt.axvline(dt_cfl, color='red', linestyle=':', label=r'Explicit Stability Limit ($\Delta t_{CFL}$)')
    
    plt.title('Time Step ($\Delta t$) vs. $L_2$ Error (Accuracy)')
    plt.xlabel('Time Step $\Delta t$ (s)')
    plt.ylabel(r'$L_2$ Error ($\sqrt{\sum (U_{num} - U_{exact})^2 / N^2}$)')
    plt.legend()
    plt.grid(which='both', linestyle='--', alpha=0.5)
    
    plt.savefig('error_vs_timestep_no_explicit.png')
    print("\nPlot saved as error_vs_timestep_no_explicit.png")
    
    print(f"\nExplicit Stability Limit (dt_CFL) for N={N} grid: {dt_cfl:.2e}")
    
    plt.figure(figsize=(6, 6))
    plt.imshow(U_adi, cmap='jet', extent=[0, 1, 0, 1], origin='lower')
    plt.colorbar(label='Temperature')
    plt.title(f'Final State (ADI, $\Delta t$={dt_range[0]:.2e})')
    plt.savefig('adi_final_state.png')

if __name__ == '__main__':
    run_simulation_and_plot_error()
