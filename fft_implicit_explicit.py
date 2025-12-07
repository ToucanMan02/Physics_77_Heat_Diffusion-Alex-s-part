import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import factorized
from scipy.fftpack import fftn, fftshift, fftfreq

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
    'ytick.major.size': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'legend.frameon': False,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'savefig.format': 'png',
    'mathtext.fontset': 'dejavusans'
}
plt.rcParams.update(style_rcParams)

# --- Simulation Parameters ---
N = 100
L = 1.0
h = L / N
alpha = 1.0       
sigma0 = 0.05
t_final = 0.001 # Shorter time so instability doesn't explode to NaN instantly

# 1. Safe Time Step (0.9x Limit)
dt_safe = (h**2 / (4 * alpha)) * 0.9

# 2. Unsafe Time Step (1.02x Limit) - To show what "explosion" looks like in FFT
dt_unsafe = (h**2 / (4 * alpha)) * 1.02

x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

def analytical_solution(X, Y, t, alpha, sigma0):
    numerator = sigma0**2
    denominator = sigma0**2 + 4 * alpha * t
    exponent = -((X - 0.5)**2 + (Y - 0.5)**2) / denominator
    U_exact = (numerator / denominator) * np.exp(exponent)
    return U_exact

# --- Solvers ---
def solve_explicit(U_initial, dt, t_final, alpha, h):
    u = np.copy(U_initial)
    t = 0.0
    factor = dt * alpha / h**2
    max_steps = int(t_final / dt)
    s_mid = slice(1, -1); s_up = slice(2, None); s_down = slice(None, -2)
    
    for _ in range(max_steps):
        laplacian = (u[s_up, s_mid] + u[s_down, s_mid] + 
                     u[s_mid, s_up] + u[s_mid, s_down] - 4*u[s_mid, s_mid])
        u[s_mid, s_mid] += factor * laplacian
        u[0, :] = u[1, :]; u[-1, :] = u[-2, :]; u[:, 0] = u[:, 1]; u[:, -1] = u[:, -2]
        t += dt
    return u

def solve_implicit_adi(U_initial, dt, t_final, alpha, h):
    u = np.copy(U_initial)
    t = 0.0
    N_grid = u.shape[0]
    k = alpha * dt / (2 * h**2)
    
    diag_main = (1 + 2*k) * np.ones(N_grid); diag_main[0] = 1 + k; diag_main[-1] = 1 + k
    diag_upper = -k * np.ones(N_grid-1); diag_lower = -k * np.ones(N_grid-1)
    
    A_sparse = diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], format='csc')
    solve_A = factorized(A_sparse) 
    max_steps = int(t_final / dt)
    
    for _ in range(max_steps):
        u_padded = np.pad(u, 1, mode='edge')
        U_y_plus = u_padded[1:-1, 2:]; U_y_minus = u_padded[1:-1, 0:-2] 
        RHS1 = u + k * (U_y_minus - 2.0*u + U_y_plus)
        u_star = solve_A(RHS1) 

        u_star_padded = np.pad(u_star, 1, mode='edge')
        U_x_plus = u_star_padded[2:, 1:-1]; U_x_minus = u_star_padded[0:-2, 1:-1] 
        RHS2 = u_star + k * (U_x_minus - 2.0*u_star + U_x_plus)
        u = solve_A(RHS2.T).T
        t += dt
    return u

# --- FFT Analysis ---
def perform_spectral_analysis(U_numerical, U_exact, L):
    N = U_numerical.shape[0]
    # FFT
    fft_num = fftshift(fftn(U_numerical))
    fft_exact = fftshift(fftn(U_exact))
    
    # Magnitudes
    mag_num = np.abs(fft_num)
    mag_exact = np.abs(fft_exact)
    
    # Ratio (Filter noise)
    mask = mag_exact > 1e-8 
    ratio = np.full_like(mag_num, np.nan) 
    ratio[mask] = mag_num[mask] / mag_exact[mask]
    
    # Wavenumbers
    kx = fftshift(fftfreq(N, d=L/N)) * 2 * np.pi # Multiply by 2pi to get proper k
    ky = fftshift(fftfreq(N, d=L/N)) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K_magnitude = np.sqrt(KX**2 + KY**2)
    
    return K_magnitude, ratio

def plot_spectral_fidelity_full(K_mag, ratio, method_name, color, filename):
    k_flat = K_mag.flatten()
    r_flat = ratio.flatten()
    valid = ~np.isnan(r_flat)
    k_flat = k_flat[valid]; r_flat = r_flat[valid]
    
    plt.figure()
    
    # Plot all modes with very small markers
    plt.scatter(k_flat, r_flat, s=1.0, alpha=0.3, c=color, label='Grid Modes', edgecolors='none')
    
    plt.axhline(1.0, color='k', linestyle='--', linewidth=1, label='Ideal Physics')
    
    plt.xlabel(r'Wavenumber magnitude $|k|$')
    plt.ylabel(r'Amplitude Ratio ($A_{num} / A_{exact}$)')
    plt.title(f'Spectral Fidelity: {method_name}')
    
    # ZOOM OUT SETTINGS
    plt.xlim(0, 320) # Go up to Nyquist limit (approx 314)
    plt.ylim(0.0, 1.2) # Show full drop to zero
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")

def run_fft_study():
    print(f"Running Full Spectrum Study (N={N})...")
    
    U0 = analytical_solution(X, Y, 0, alpha, sigma0)
    U_exact_end = analytical_solution(X, Y, t_final, alpha, sigma0)
    
    # 1. Implicit ADI (Safe)
    print("Solving Implicit...")
    U_imp = solve_implicit_adi(U0, dt_safe, t_final, alpha, h)
    K, r_imp = perform_spectral_analysis(U_imp, U_exact_end, L)
    plot_spectral_fidelity_full(K, r_imp, 'Implicit (ADI)', '#4DBBD5', 'spectrum_implicit_full.png')
    
    # 2. Explicit (Safe)
    print("Solving Explicit (Safe)...")
    U_exp = solve_explicit(U0, dt_safe, t_final, alpha, h)
    K, r_exp = perform_spectral_analysis(U_exp, U_exact_end, L)
    plot_spectral_fidelity_full(K, r_exp, 'Explicit (Safe)', '#E64B35', 'spectrum_explicit_safe.png')

    # 3. Explicit (Unsafe - Instability Demo)
    print("Solving Explicit (Unsafe/Unstable)...")
    try:
        # We might get overflows, so catch them
        np.seterr(all='ignore') 
        U_unsafe = solve_explicit(U0, dt_unsafe, t_final, alpha, h)
        K, r_unsafe = perform_spectral_analysis(U_unsafe, U_exact_end, L)
        
        # Plot with extended Y-axis to show explosion
        plt.figure()
        k_flat = K.flatten(); r_flat = r_unsafe.flatten()
        valid = ~np.isnan(r_flat)
        plt.scatter(k_flat[valid], r_flat[valid], s=1.0, alpha=0.3, c='#E64B35')
        plt.axhline(1.0, color='k', linestyle='--')
        plt.xlim(0, 320)
        plt.ylim(0.0, 2.0) # Show it shooting up
        plt.title('Instability Signature (Explicit Unsafe)')
        plt.xlabel(r'$|k|$'); plt.ylabel('Ratio')
        plt.savefig('spectrum_explicit_unsafe.png')
        print("Saved spectrum_explicit_unsafe.png")
    except:
        print("Unsafe simulation exploded too hard to plot.")

if __name__ == '__main__':
    run_fft_study()