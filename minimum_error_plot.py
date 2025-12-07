import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import factorized

style_rcParams = {
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

plt.rcParams.update(style_rcParams)

N = 50   
L = 1.0
h = L / N
sigma0 = 0.05
t_final = 0.002 

x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

def analytical_solution(X, Y, t, alpha, sigma0):
    numerator = sigma0**2
    denominator = sigma0**2 + 4 * alpha * t
    exponent = -((X - 0.5)**2 + (Y - 0.5)**2) / denominator
    return (numerator / denominator) * np.exp(exponent)

def solve_explicit(U0, dt, t_final, alpha, h):
    u = U0.copy()
    t = 0.0
    factor = dt * alpha / h**2
    max_steps = int(t_final / dt)
    
    c = u[1:-1, 1:-1]
    up = u[2:, 1:-1]
    down = u[:-2, 1:-1]
    left = u[1:-1, :-2]
    right = u[1:-1, 2:]
    
    for _ in range(max_steps):
        laplacian = (up + down + left + right - 4*c)
        c += factor * laplacian
        
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
        t += dt
    return u

def solve_implicit_adi_fast(U0, dt, t_final, alpha, h):
    u = U0.copy()
    t = 0.0
    N_grid = u.shape[0]
    k = alpha * dt / (2 * h**2)
    
    diag_main = (1 + 2*k) * np.ones(N_grid)
    diag_main[0] = 1 + k
    diag_main[-1] = 1 + k
    
    diag_upper = -k * np.ones(N_grid-1)
    diag_lower = -k * np.ones(N_grid-1)
    
    A_sparse = diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], format='csc')
    solve_A = factorized(A_sparse) 
    
    max_steps = int(t_final / dt)
    
    for _ in range(max_steps):
        u_padded = np.pad(u, 1, mode='edge')
        
        U_y_plus = u_padded[1:-1, 2:]
        U_y_minus = u_padded[1:-1, 0:-2]
        RHS1 = u + k * (U_y_minus - 2.0*u + U_y_plus)
        u_star = solve_A(RHS1)
        
        u_star_padded = np.pad(u_star, 1, mode='edge')
        U_x_plus = u_star_padded[2:, 1:-1]
        U_x_minus = u_star_padded[0:-2, 1:-1]
        RHS2 = u_star + k * (U_x_minus - 2.0*u_star + U_x_plus)
        u = solve_A(RHS2.T).T
        t += dt
        
    return u

def get_error(u_num, u_exact):
    return np.sqrt(np.mean((u_num - u_exact)**2))

def find_sweet_spots():
    alphas = np.linspace(0.1, 10, 50) 
    
    optimal_dt_explicit = []
    optimal_dt_implicit = []
    theoretical_dt = []
    
    print(f"Starting Sweep over {len(alphas)} alpha values...")
    print(f"Grid: {N}x{N}")
    
    for i, alpha in enumerate(alphas):
        U0 = analytical_solution(X, Y, 0, alpha, sigma0)
        U_final_exact = analytical_solution(X, Y, t_final, alpha, sigma0)
        
        dt_theory = h**2 / (6 * alpha)
        theoretical_dt.append(dt_theory)
        
        dt_limit = h**2 / (4 * alpha)
        dt_search = np.linspace(dt_theory * 0.5, dt_limit * 0.95, 30)
        
        errors_exp = []
        for dt in dt_search:
            try:
                res = solve_explicit(U0, dt, t_final, alpha, h)
                err = get_error(res, U_final_exact)
                errors_exp.append(err)
            except:
                errors_exp.append(np.inf)
        
        min_idx_exp = np.argmin(errors_exp)
        optimal_dt_explicit.append(dt_search[min_idx_exp])
        
        dt_search_imp = np.linspace(dt_theory * 0.5, dt_limit * 1.5, 30)
        errors_imp = []
        for dt in dt_search_imp:
            res = solve_implicit_adi_fast(U0, dt, t_final, alpha, h)
            err = get_error(res, U_final_exact)
            errors_imp.append(err)
            
        min_idx_imp = np.argmin(errors_imp)
        optimal_dt_implicit.append(dt_search_imp[min_idx_imp])
        
        print(f"Alpha {alpha:.2f}: Opt Explicit dt={optimal_dt_explicit[-1]:.2e}")

    plt.figure()
    plt.plot(alphas, theoretical_dt, 'k--', label=r'Theory ($r=1/6$)', alpha=0.6)
    plt.plot(alphas, optimal_dt_explicit, 'o-', color='#E64B35', 
             label='Explicit Min Error', markersize=4)
    plt.plot(alphas, optimal_dt_implicit, 's-', color='#4DBBD5', 
             label='Implicit Min Error', markersize=4)
    
    plt.title(r'Optimal Time Step ($\Delta t$) vs. Diffusivity ($\alpha$)')
    plt.xlabel(r'Diffusivity $\alpha$')
    plt.ylabel(r'Optimal $\Delta t$ (s)')
    plt.legend()
    
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig('sweet_spot_tracking.png')
    print("\nSaved sweet_spot_tracking.png")

if __name__ == '__main__':
    find_sweet_spots()