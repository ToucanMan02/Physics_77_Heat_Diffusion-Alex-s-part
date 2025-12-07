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
    'figure.figsize': (4, 3.5), 
    'figure.dpi': 300,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5,
    'legend.frameon': False,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'mathtext.fontset': 'dejavusans'
}
plt.rcParams.update(style_rcParams)


N = 128         
L = 1.0
h = L / N
alpha = 1.0     
t_final = 0.0001 
dt = (h**2 / (4 * alpha)) * 0.00010 




np.random.seed(42)
U0 = np.random.normal(0, 1, (N, N))


def get_exact_spectrum(U0, t, alpha):
    
    fft_0 = fftn(U0)
    
    
    kx = fftfreq(N, d=L/N) * 2 * np.pi
    ky = fftfreq(N, d=L/N) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    
    
    propagator = np.exp(-alpha * K2 * t)
    
    
    fft_exact = fft_0 * propagator
    return fft_exact


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
        
        u_padded = np.pad(u, 1, mode='wrap') 
        U_y_plus = u_padded[1:-1, 2:]     
        U_y_minus = u_padded[1:-1, 0:-2] 
        RHS1 = u + k * (U_y_minus - 2.0*u + U_y_plus)
        u_star = solve_A(RHS1) 

        
        u_star_padded = np.pad(u_star, 1, mode='wrap')
        U_x_plus = u_star_padded[2:, 1:-1]   
        U_x_minus = u_star_padded[0:-2, 1:-1] 
        RHS2 = u_star + k * (U_x_minus - 2.0*u_star + U_x_plus)
        u = solve_A(RHS2.T).T
        t += dt
    return u

def run_anisotropy_study():
    print("Running White Noise Anisotropy Test...")
    
    
    fft_exact = get_exact_spectrum(U0, t_final, alpha)
    
    
    U_num = solve_implicit_adi(U0, dt, t_final, alpha, h)
    fft_num = fftn(U_num)
    
    
    
    
    
    
    spec_exact = fftshift(np.abs(fft_exact))
    spec_num = fftshift(np.abs(fft_num))
    
    
    mask = spec_exact > 1e-10
    error_map = np.ones_like(spec_exact)
    error_map[mask] = spec_num[mask] / spec_exact[mask]
    
    
    plt.figure()
    
    
    k = fftshift(fftfreq(N, d=L/N)) * 2 * np.pi
    extent = [k.min(), k.max(), k.min(), k.max()]
    
    
    
    im = plt.imshow(error_map - 1.0, extent=extent, origin='lower', 
                    cmap='RdBu_r', vmin=-0.2, vmax=0.2)
    
    plt.colorbar(im, label='Relative Error (Num/Exact - 1)')
    plt.xlabel(r'$k_x$ (Horizontal Freq)')
    plt.ylabel(r'$k_y$ (Vertical Freq)')
    plt.title('The Geometry of Numerical Error (ADI)')
    
    
    theta = np.linspace(0, 2*np.pi, 100)
    for r in [50, 100, 150]:
        plt.plot(r*np.cos(theta), r*np.sin(theta), 'k--', linewidth=0.5, alpha=0.5)
        
    plt.tight_layout()
    plt.savefig('anisotropy_map.png')
    print("Saved anisotropy_map.png")

if __name__ == '__main__':
    run_anisotropy_study()