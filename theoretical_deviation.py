import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import factorized

# --- Plotting Style ---
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

N = 100
L = 1.0
h = L / N
sigma0 = 0.05
t_final = 0.005

x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

def analytical_solution(X, Y, t, alpha, sigma0):
    numerator = sigma0**2
    denominator = sigma0**2 + 4 * alpha * t
    exponent = -((X - 0.5)**2 + (Y - 0.5)**2) / denominator
    U_exact = (numerator / denominator) * np.exp(exponent)
    return U_exact

def solve_explicit(U_initial, dt, t_final, alpha, h):
    u = np.copy(U_initial)
    t = 0.0
    factor = dt * alpha / h**2
    max_steps = int(t_final / dt)
    
    s_mid = slice(1, -1)
    s_up = slice(2, None)
    s_down = slice(None, -2)
    
    for _ in range(max_steps):
        laplacian = (u[s_up, s_mid] + u[s_down, s_mid] + 
                     u[s_mid, s_up] + u[s_mid, s_down] - 4*u[s_mid, s_mid])
        u[s_mid, s_mid] += factor * laplacian
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
        t += dt
    return u

def solve_implicit_adi(U_initial, dt, t_final, alpha, h):
    u = np.copy(U_initial)
    t = 0.0
    N_grid = u.shape[0]
    k = alpha * dt / (2 * h**2)
    
    diag_main = (1 + 2*k) * np.ones(N_grid)
    diag_main[0] = 1 + k; diag_main[-1] = 1 + k
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

def calculate_error(U_numerical, U_exact):
    return np.sqrt(np.mean((U_numerical - U_exact)**2))

def run_deviation_analysis():
    alphas = np.linspace(0.1, 10.0, 50)
    samples_per_alpha = 100
    
    optimal_dt_explicit = []
    optimal_dt_implicit = []
    theoretical_dt = []
    
    print(f"Starting analysis for deviation plot (N={N})...")

    for i, alpha in enumerate(alphas):
        if i % 5 == 0: print(f"Processing alpha = {alpha:.2f}...")

        U0 = analytical_solution(X, Y, 0, alpha, sigma0)
        U_exact_final = analytical_solution(X, Y, t_final, alpha, sigma0)
        
        dt_theory = h**2 / (6 * alpha)
        theoretical_dt.append(dt_theory)
        
        dt_limit = h**2 / (4 * alpha)
        dt_search = np.linspace(dt_theory * 0.5, dt_limit * 0.98, samples_per_alpha)
        
        errors_exp = []
        for dt in dt_search:
            try:
                res = solve_explicit(U0, dt, t_final, alpha, h)
                err = calculate_error(res, U_exact_final)
                errors_exp.append(err)
            except:
                errors_exp.append(np.inf)
        optimal_dt_explicit.append(dt_search[np.argmin(errors_exp)])
        
        # Implicit
        errors_imp = []
        for dt in dt_search:
            res = solve_implicit_adi(U0, dt, t_final, alpha, h)
            err = calculate_error(res, U_exact_final)
            errors_imp.append(err)
        optimal_dt_implicit.append(dt_search[np.argmin(errors_imp)])

    print("Calculation complete. Plotting deviation...")

    
    diff_explicit = np.array(optimal_dt_explicit) - np.array(theoretical_dt)
    diff_implicit = np.array(optimal_dt_implicit) - np.array(theoretical_dt)

    
    plt.figure()
    
    
    plt.axhline(0, color='k', linestyle='--', label='Theory (Zero Deviation)', alpha=0.6)
    
    
    plt.plot(alphas, diff_explicit, 'o-', color='#E64B35', 
             label='Explicit Deviation', markersize=3, alpha=0.9)
    
    plt.plot(alphas, diff_implicit, 's-', color='#4DBBD5', 
             label='Implicit Deviation', markersize=3, alpha=0.9)
    
    plt.title(r'Deviation from Theoretical Optimal $\Delta t$')
    plt.xlabel(r'Diffusivity $\alpha$')
    plt.ylabel(r'$\Delta t_{actual} - \Delta t_{theory}$ (s)')
    plt.legend()
    
    ax = plt.gca()
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.yaxis.get_offset_text().set_position((-0.1, 0))
    ax.yaxis.get_offset_text().set_horizontalalignment('right')
    
    plt.savefig('sweet_spot_deviation.png')
    print("Plot saved as sweet_spot_deviation.png")

if __name__ == '__main__':
    run_deviation_analysis()