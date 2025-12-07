import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import factorized
from scipy.fftpack import fftn, fftshift, fftfreq

# --- Plotting Style (Your Exact Style) ---
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
    'ytick.major.size': 0.5,  # Fixed key from your snippet
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
alpha = 1.0       # High diffusivity to test limits
sigma0 = 0.05
t_final = 0.005

# We pick a time step just *slightly* below the explicit limit
# This is where explicit is most accurate, but explicit damping errors are visible
dt = (h**2 / (4 * alpha)) * 0.9

x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

# --- Analytical Solution ---
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
    
    # Build Sparse Matrix
    diag_main = (1 + 2*k) * np.ones(N_grid)
    diag_main[0] = 1 + k; diag_main[-1] = 1 + k
    diag_upper = -k * np.ones(N_grid-1)
    diag_lower = -k * np.ones(N_grid-1)
    
    A_sparse = diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], format='csc')
    solve_A = factorized(A_sparse) 
    
    max_steps = int(t_final / dt)
    
    for _ in range(max_steps):
        # Step 1: Explicit X, Implicit Y
        u_padded = np.pad(u, 1, mode='edge')
        U_y_plus = u_padded[1:-1, 2:]     
        U_y_minus = u_padded[1:-1, 0:-2] 
        RHS1 = u + k * (U_y_minus - 2.0*u + U_y_plus)
        u_star = solve_A(RHS1) 

        # Step 2: Explicit Y, Implicit X
        u_star_padded = np.pad(u_star, 1, mode='edge')
        U_x_plus = u_star_padded[2:, 1:-1]   
        U_x_minus = u_star_padded[0:-2, 1:-1] 
        RHS2 = u_star + k * (U_x_minus - 2.0*u_star + U_x_plus)
        u = solve_A(RHS2.T).T
        t += dt
    return u

# --- FFT Analysis Functions ---
def perform_spectral_analysis(U_numerical, U_exact, L):
    """
    Compares the frequency content of the Numerical vs Exact solution.
    """
    N = U_numerical.shape[0]
    
    # 1. Compute 2D FFTs
    # We shift zero-frequency to the center for easier plotting
    fft_num = fftshift(fftn(U_numerical))
    fft_exact = fftshift(fftn(U_exact))
    
    # 2. Compute the Magnitude Spectrum
    mag_num = np.abs(fft_num)
    mag_exact = np.abs(fft_exact)
    
    # 3. Compute the Transfer Function Ratio (Numerical / Exact)
    # Avoid division by zero by masking very small values (noise floor)
    mask = mag_exact > 1e-5 
    ratio = np.full_like(mag_num, np.nan) # Fill with NaNs initially
    ratio[mask] = mag_num[mask] / mag_exact[mask]
    
    # 4. Get Wavenumbers (k) for plotting
    kx = fftshift(fftfreq(N, d=L/N))
    ky = fftshift(fftfreq(N, d=L/N))
    KX, KY = np.meshgrid(kx, ky)
    K_magnitude = np.sqrt(KX**2 + KY**2)
    
    return K_magnitude, ratio

def plot_spectral_fidelity(K_mag, ratio, method_name, color):
    # Flatten arrays to plot scatter of all modes
    k_flat = K_mag.flatten()
    r_flat = ratio.flatten()
    
    # Filter NaNs
    valid = ~np.isnan(r_flat)
    k_flat = k_flat[valid]
    r_flat = r_flat[valid]
    
    plt.figure()
    
    # Scatter plot of frequency response
    plt.scatter(k_flat, r_flat, s=2, alpha=0.4, c=color, label='Grid Modes')
    
    plt.axhline(1.0, color='k', linestyle='--', linewidth=1, label='Ideal Physics')
    
    plt.xlabel(r'Wavenumber magnitude $|k|$')
    plt.ylabel(r'Amplitude Ratio ($A_{num} / A_{exact}$)')
    plt.title(f'Spectral Fidelity: {method_name}')
    
    # Zoom in on the relevant area (Ratio around 1.0)
    plt.ylim(0.90, 1.10)
    plt.xlim(0, 30) # Focus on low-mid frequencies where energy is significant
    
    plt.legend()
    filename = f'spectral_fidelity_{method_name.split()[0].lower()}.png'
    plt.savefig(filename)
    print(f"Saved plot: {filename}")

# --- Main Execution ---
def run_fft_study():
    print(f"Running simulation with N={N}, alpha={alpha}...")
    
    U0 = analytical_solution(X, Y, 0, alpha, sigma0)
    U_exact_end = analytical_solution(X, Y, t_final, alpha, sigma0)
    
    # 1. Run Explicit
    print("Solving Explicit Method...")
    U_explicit = solve_explicit(U0, dt, t_final, alpha, h)
    
    print("Solving Implicit Method...")
    U_implicit = solve_implicit_adi(U0, dt, t_final, alpha, h)
    
    # 2. Perform Analysis
    print("Performing FFT Analysis (Explicit)...")
    K, ratio_exp = perform_spectral_analysis(U_explicit, U_exact_end, L)
    
    print("Performing FFT Analysis (Implicit)...")
    K, ratio_imp = perform_spectral_analysis(U_implicit, U_exact_end, L)
    
    # 3. Plot
    plot_spectral_fidelity(K, ratio_exp, 'Explicit (Forward Euler)', '#E64B35')
    plot_spectral_fidelity(K, ratio_imp, 'Implicit (ADI)', '#4DBBD5')
    
    print("\nDone! Check your folder for the PNG files.")

if __name__ == '__main__':
    run_fft_study()