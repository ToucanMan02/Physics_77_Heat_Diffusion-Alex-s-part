import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import factorized
from scipy.fftpack import fftn, fftshift, fftfreq
from scipy.interpolate import RegularGridInterpolator

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
t_final = 0.0005

target_k_radius = 40 
alpha_complex = 1.0 + 0j 

x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

np.random.seed(42)
U0 = np.random.normal(0, 1, (N, N)) + 0j

def solve_complex_system(U_in, dt, steps, alpha_complex, h):
    u = np.copy(U_in).astype(np.complex128)
    N_grid = u.shape[0]
    k = alpha_complex * dt / (2 * h**2)
    
    diag_main = (1 + 2*k) * np.ones(N_grid); diag_main[0] = 1 + k; diag_main[-1] = 1 + k
    diag_upper = -k * np.ones(N_grid-1); diag_lower = -k * np.ones(N_grid-1)
    
    A_sparse = diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], format='csc', dtype=np.complex128)
    solve_A = factorized(A_sparse)
    
    for _ in range(steps):
        u_padded = np.pad(u, 1, mode='wrap')
        U_y_plus = u_padded[1:-1, 2:]; U_y_minus = u_padded[1:-1, 0:-2]
        RHS1 = u + k * (U_y_minus - 2.0*u + U_y_plus)
        u_star = solve_A(RHS1)

        u_star_padded = np.pad(u_star, 1, mode='wrap')
        U_x_plus = u_star_padded[2:, 1:-1]; U_x_minus = u_star_padded[0:-2, 1:-1]
        RHS2 = u_star + k * (U_x_minus - 2.0*u_star + U_x_plus)
        u = solve_A(RHS2.T).T
    return u

def get_exact_propagator(N, dt, alpha_complex):
    kx = fftfreq(N, d=L/N) * 2 * np.pi
    ky = fftfreq(N, d=L/N) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    return np.exp(-alpha_complex * K2 * dt)

def scan_angular_error_zoomed():
    print(f"Scanning at |k|={target_k_radius} to ensure signal survival...")
    
    dt = (h**2 / 4) * 100.0 
    steps = max(1, int(t_final / dt))
    
    U_num = solve_complex_system(U0, dt, steps, alpha_complex, h)
    
    fft_0 = fftn(U0)
    fft_exact = fft_0 * get_exact_propagator(N, steps * dt, alpha_complex)
    fft_num = fftn(U_num)
    
    spec_num = fftshift(np.abs(fft_num))
    spec_exact = fftshift(np.abs(fft_exact))
    
    mask = spec_exact > 1e-15 
    error_map = np.zeros_like(spec_num)
    error_map[mask] = np.abs(spec_num[mask] / spec_exact[mask] - 1.0)
    
    kx = fftshift(fftfreq(N, d=L/N)) * 2 * np.pi
    ky = fftshift(fftfreq(N, d=L/N)) * 2 * np.pi
    
    interp = RegularGridInterpolator((kx, ky), error_map, bounds_error=False, fill_value=0)
    
    thetas = np.linspace(0, 2*np.pi, 360)
    circle_kx = target_k_radius * np.cos(thetas)
    circle_ky = target_k_radius * np.sin(thetas)
    
    points = np.column_stack((circle_kx, circle_ky))
    angular_error = interp(points)
    
    print(f"Min Error: {np.min(angular_error):.2e}")
    print(f"Max Error: {np.max(angular_error):.2e}")

    plt.figure()
    plt.plot(np.degrees(thetas), angular_error, color='#E64B35', linewidth=1.5)
    plt.xlim(0, 360)
    
    y_min, y_max = np.min(angular_error), np.max(angular_error)
    margin = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 1e-10
    plt.ylim(y_min - margin, y_max + margin)
    
    plt.xlabel(r'Angle $\theta$ (Degrees)')
    plt.ylabel(f'Relative Error')
    plt.title(f'Angular Profile (Heat Mode, $|k|={target_k_radius}$)')
    plt.grid(True, alpha=0.3)
    
    for angle in [45, 135, 225, 315]:
        plt.axvline(angle, color='b', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('angular_error_zoomed.png')
    print("Saved angular_error_zoomed.png")

if __name__ == '__main__':
    scan_angular_error_zoomed()
