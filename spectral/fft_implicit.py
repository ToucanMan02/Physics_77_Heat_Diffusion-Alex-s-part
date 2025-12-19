import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import factorized
from scipy.fftpack import fftn, fftshift, fftfreq

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
    'mathtext.fontset': 'dejavusans'
}
plt.rcParams.update(style_rcParams)

N = 100
L = 1.0
h = L / N
alpha = 1.0
t_final = 0.00005

np.random.seed(42)
U0 = np.random.randn(N, N)


def solve_implicit_adi(U_initial, dt, steps, alpha, h):
    u = np.copy(U_initial)
    N_grid = u.shape[0]
    k = alpha * dt / (2 * h**2)
    
    diag_main = (1 + 2*k) * np.ones(N_grid)
    diag_main[0] = 1 + k
    diag_main[-1] = 1 + k
    
    A_sparse = diags([-k*np.ones(N_grid-1), diag_main, -k*np.ones(N_grid-1)], 
                     [-1, 0, 1], format='csc')
    solve_A = factorized(A_sparse)
    
    for _ in range(steps):
        u_padded = np.pad(u, 1, mode='edge')
        RHS1 = u + k * (u_padded[1:-1, 0:-2] - 2.0*u + u_padded[1:-1, 2:])
        u_star = solve_A(RHS1)
        
        u_star_padded = np.pad(u_star, 1, mode='edge')
        RHS2 = u_star + k * (u_star_padded[0:-2, 1:-1] - 2.0*u_star + u_star_padded[2:, 1:-1])
        u = solve_A(RHS2.T).T
    
    return u


def get_exact_spectrum(U0, t, alpha, N, L):
    fft_0 = fftn(U0)
    kx = fftfreq(N, d=L/N) * 2 * np.pi
    KX, KY = np.meshgrid(kx, kx, indexing='ij')
    propagator = np.exp(-alpha * (KX**2 + KY**2) * t)
    return fft_0 * propagator


def run_analysis():
    dt_cfl = h**2 / (4 * alpha)
    dt = dt_cfl * 2.0
    steps = max(1, int(t_final / dt))
    actual_t = dt * steps
    
    U_num = solve_implicit_adi(U0, dt, steps, alpha, h)
    
    fft_exact = get_exact_spectrum(U0, actual_t, alpha, N, L)
    fft_num = fftn(U_num)
    
    mag_exact = fftshift(np.abs(fft_exact))
    mag_num = fftshift(np.abs(fft_num))
    
    kx = fftshift(fftfreq(N, d=L/N)) * 2 * np.pi
    KX, KY = np.meshgrid(kx, kx)
    K_mag = np.sqrt(KX**2 + KY**2)
    
    mask = mag_exact > 1e-15
    ratio = np.full_like(mag_num, np.nan)
    ratio[mask] = mag_num[mask] / mag_exact[mask]
    
    k_flat = K_mag.flatten()
    r_flat = ratio.flatten()
    valid = ~np.isnan(r_flat) & (r_flat < 3) & (r_flat > 0)
    k_v = k_flat[valid]
    r_v = r_flat[valid]
    
    n_bins = 40
    k_max = min(320, k_v.max() * 1.1)
    bin_edges = np.linspace(0, k_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    bin_means = []
    for i in range(n_bins):
        in_bin = (k_v >= bin_edges[i]) & (k_v < bin_edges[i+1])
        if np.sum(in_bin) > 3:
            bin_means.append(np.mean(r_v[in_bin]))
        else:
            bin_means.append(np.nan)
    bin_means = np.array(bin_means)
    
    plt.figure()
    
    plt.scatter(k_v, r_v, s=0.5, alpha=0.25, c='#4DBBD5', edgecolors='none', rasterized=True)
    
    valid_bins = ~np.isnan(bin_means)
    plt.plot(bin_centers[valid_bins], bin_means[valid_bins], '-', color='#00629B', 
             linewidth=1.2, label='Binned mean')
    
    plt.axhline(1.0, color='k', linestyle='--', linewidth=0.8, label='Ideal Physics')
    
    plt.xlabel(r'Wavenumber magnitude $|k|$')
    plt.ylabel(r'Amplitude Ratio ($A_{num}/A_{exact}$)')
    plt.title('Spectral Fidelity: Implicit (ADI)')
    
    plt.xlim(0, 320)
    plt.ylim(0.5, 1.6)
    plt.legend(loc='upper left')
    
    plt.savefig('spectrum_implicit_full.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    run_analysis()
