import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

nature_rcParams = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (3.5, 3.5),
    'figure.dpi': 300,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'lines.markersize': 3,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'legend.frameon': False,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.format': 'png',
    'mathtext.fontset': 'dejavusans'
}
plt.rcParams.update(nature_rcParams)

alpha = 0.7
N = 100 
L = 1.0     
h = L / N   

sigma0 = 0.05 # Initial width of the Gaussian
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

def analytical_solution(X, Y, t, alpha, sigma0):
    numerator = sigma0**2
    denominator = sigma0**2 + 4 * alpha * t
    exponent = -((X - 0.5)**2 + (Y - 0.5)**2) / denominator
    U_exact = (numerator / denominator) * np.exp(exponent)
    return U_exact

U0 = analytical_solution(X, Y, t=0, alpha=alpha, sigma0=sigma0)

def tri_disc(N, a):
    M = (np.diag(-a * np.ones(N-1), k=-1) +
         np.diag((1 + 2*a) * np.ones(N), k=0) +
         np.diag(-a * np.ones(N-1), k=1))
    
    M[0, 0] = 1 + a
    M[0, 1] = -a
    M[-1, -1] = 1 + a
    M[-1, -2] = -a
    
    return M

def solve_explicit(U_initial, dt, t_final, alpha, h):

    u = np.copy(U_initial)
    t = 0.0
    
    C = alpha * dt / h**2 
    if C > 0.25:
        print(f"Warning: Explicit method unstable for dt={dt:.2e}. C={C:.2f}")

    max_steps = int(t_final / dt)

    for _ in range(max_steps):
  
        laplacian = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]) / h**2
        
        u[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * alpha * laplacian
        
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

dt_sweet_spot = (h**2) / (6 * alpha)

print(f"For alpha={alpha}, the most accurate timestep is {dt_sweet_spot:.2e}")

def calculate_error(U_numerical, U_exact):
    """
    Calculates the L2 norm of the error (Root Mean Square Error).
    """
    return np.sqrt(np.mean((U_numerical - U_exact)**2))

def run_simulation_and_plot_error():
    t_final = 0.005 
    
    # Calculate stability limit
    dt_cfl = 0.25 * h**2 / alpha
    
    # Set dt range to go up to the critical dt
    dt_range = np.logspace(-7, np.log10(dt_cfl), 20)

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
    
    plt.figure()
    
    plt.plot(dt_range, error_explicit, 'o-', label='Explicit (Forward Euler)', 
               color='#E64B35')
    plt.plot(dt_range, error_adi, 's-', label='Implicit (ADI)', 
               color='#4DBBD5')

    # Show stability limit as reference
    plt.axvline(dt_cfl, color='red', linestyle=':',
                label=r'Explicit Stability Limit ($\Delta t_{CFL}$)')
    
    # Zoom in more on the x-axis
    plt.xlim(0, 3.5e-5)
    
    # Calculate appropriate y-axis limits to show the difference - zoom in more
    plt.ylim(2.0e-4, 3.2e-4)
    
    plt.title('Time Step ($\Delta t$) vs. $L_2$ Error (Accuracy)')
    plt.xlabel('Time Step $\Delta t$ (s)')
    plt.ylabel(r'$L_2$ Error')
    plt.legend()
    
    ax = plt.gca()
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    ax.yaxis.get_offset_text().set_position((-0.1, 0))
    ax.yaxis.get_offset_text().set_horizontalalignment('right')
    
    ax.set_xticks(np.linspace(0, 3.5e-5, 6))
    ax.set_yticks(np.linspace(2.0e-4, 3.2e-4, 6))
    
    plt.savefig('error_vs_timestep.png')
    print("\nPlot saved as error_vs_timestep.png")
    
    print(f"\nExplicit Stability Limit (dt_CFL) for N={N} grid: {dt_cfl:.2e}")
    print(f"dt range used: {dt_range[0]:.2e} to {dt_range[-1]:.2e}")
    
    plt.figure()
    plt.imshow(U_adi, cmap='jet', extent=[0, 1, 0, 1], origin='lower')
    plt.colorbar(label='Temperature')
    plt.title(f'Final State (ADI, $\Delta t$={dt_range[0]:.2e})')
    plt.savefig('adi_final_state.png')

if __name__ == '__main__':
    run_simulation_and_plot_error()