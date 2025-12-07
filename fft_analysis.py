import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftn, fftshift, fftfreq

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


N = 100
L = 1.0
h = L / N
alpha = 1.0       
sigma0 = 0.05

t_final = 0.0005 


dt_unsafe = (h**2 / (4 * alpha)) * 1.05

x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

def analytical_solution(X, Y, t, alpha, sigma0):
    numerator = sigma0**2
    denominator = sigma0**2 + 4 * alpha * t
    exponent = -((X - 0.5)**2 + (Y - 0.5)**2) / denominator
    return (numerator / denominator) * np.exp(exponent)


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

def plot_stability_spectrum():
    print("Running Implicit Simulation...")
    U0 = analytical_solution(X, Y, 0, alpha, sigma0)
    U_exact = analytical_solution(X, Y, t_final, alpha, sigma0)
    
    
    U_implicit = solve_implicit_adi(U0, dt_unsafe, t_final, alpha, h)
    
    
    fft_implicit = fftshift(fftn(U_implicit))
    fft_exact = fftshift(fftn(U_exact))
    
    
    mag_implicit = np.abs(fft_implicit)
    mag_exact = np.abs(fft_exact)
    
    
    kx = fftshift(fftfreq(N, d=L/N)) * 2 * np.pi
    ky = fftshift(fftfreq(N, d=L/N)) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K_mag = np.sqrt(KX**2 + KY**2)
    
    
    k_flat = K_mag.flatten()
    e_implicit = mag_implicit.flatten()
    e_exact = mag_exact.flatten()
    
    
    plt.figure()
    
    
    mask_ex = e_exact > 1e-20
    plt.scatter(k_flat[mask_ex], np.log10(e_exact[mask_ex]), 
                s=1.0, color='black', alpha=0.3, label='Ideal Physics')
    
    
    
    mask_num = e_implicit > 1e-20
    plt.scatter(k_flat[mask_num], np.log10(e_implicit[mask_num]), 
                s=1.0, color='#4DBBD5', alpha=0.5, label='Implicit (ADI)')
    
    plt.xlabel(r'Wavenumber magnitude $|k|$')
    plt.ylabel(r'Log Amplitude ($\log_{10}|E|$)')
    plt.title('Implicit Stability Spectrum')
    
    plt.xlim(0, 320)
    
    plt.ylim(-15, 5) 
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('spectrum_log_implicit.png')
    print("Saved spectrum_log_implicit.png")

if __name__ == '__main__':
    plot_stability_spectrum()