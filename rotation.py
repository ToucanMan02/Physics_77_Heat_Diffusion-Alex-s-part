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


def solve_complex_system(U_in, dt, steps, alpha_complex, h, y_first=True):
    """
    ADI solver with selectable sweep order.
    y_first=True:  Y-implicit first, then X-implicit → error peaks at 90°, 270° (Y direction)
    y_first=False: X-implicit first, then Y-implicit → error peaks at 0°, 180° (X direction)
    
    Key insight: The direction solved FIRST accumulates more splitting error,
    causing error to peak in that direction in Fourier space.
    """
    u = np.copy(U_in).astype(np.complex128)
    N_grid = u.shape[0]
    k = alpha_complex * dt / (2 * h**2)
    
    diag_main = (1 + 2*k) * np.ones(N_grid); diag_main[0] = 1 + k; diag_main[-1] = 1 + k
    diag_upper = -k * np.ones(N_grid-1); diag_lower = -k * np.ones(N_grid-1)
    
    A_sparse = diags([diag_lower, diag_main, diag_upper], [-1, 0, 1], format='csc', dtype=np.complex128)
    solve_A = factorized(A_sparse)
    
    for _ in range(steps):
        if y_first:
            u_padded = np.pad(u, 1, mode='wrap')
            U_x_plus  = u_padded[1:-1, 2:]    
            U_x_minus = u_padded[1:-1, 0:-2]
            RHS1 = u + k * (U_x_minus - 2.0*u + U_x_plus)
            
            u_star = solve_A(RHS1)  
            
            u_star_padded = np.pad(u_star, 1, mode='wrap')
            U_y_plus  = u_star_padded[2:, 1:-1]   
            U_y_minus = u_star_padded[0:-2, 1:-1]
            RHS2 = u_star + k * (U_y_minus - 2.0*u_star + U_y_plus)
            
            u = solve_A(RHS2.T).T
            
        else:
            u_padded = np.pad(u, 1, mode='wrap')
            U_y_plus  = u_padded[2:, 1:-1]    
            U_y_minus = u_padded[0:-2, 1:-1]
            RHS1 = u + k * (U_y_minus - 2.0*u + U_y_plus)
            
            u_star = solve_A(RHS1.T).T
             
            u_star_padded = np.pad(u_star, 1, mode='wrap')
            U_x_plus  = u_star_padded[1:-1, 2:]   
            U_x_minus = u_star_padded[1:-1, 0:-2]
            RHS2 = u_star + k * (U_x_minus - 2.0*u_star + U_x_plus)
            
            u = solve_A(RHS2)

    return u


def get_exact_propagator(N, dt, alpha_complex):
    kx = fftfreq(N, d=L/N) * 2 * np.pi
    ky = fftfreq(N, d=L/N) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    return np.exp(-alpha_complex * K2 * dt)

def scan_angular_error_zoomed():
    """Compare angular error profiles for X-first vs Y-first ADI sweep ordering."""
    print(f"Comparing ADI sweep orders at |k|={target_k_radius}...")
    
    dt = (h**2 / 4) * 100.0 
    steps = max(1, int(t_final / dt))
    
    fft_0 = fftn(U0)
    fft_exact = fft_0 * get_exact_propagator(N, steps * dt, alpha_complex)
    spec_exact = fftshift(np.abs(fft_exact))
    
    kx = fftshift(fftfreq(N, d=L/N)) * 2 * np.pi
    ky = fftshift(fftfreq(N, d=L/N)) * 2 * np.pi
    thetas = np.linspace(0, 2*np.pi, 360)
    circle_kx = target_k_radius * np.cos(thetas)
    circle_ky = target_k_radius * np.sin(thetas)
    points = np.column_stack((circle_kx, circle_ky))
    
    
    U_num_yfirst = solve_complex_system(U0, dt, steps, alpha_complex, h, y_first=True)
    fft_num_yfirst = fftn(U_num_yfirst)
    spec_num_yfirst = fftshift(np.abs(fft_num_yfirst))
    
    mask = spec_exact > 1e-15 
    error_map_yfirst = np.zeros_like(spec_num_yfirst)
    error_map_yfirst[mask] = np.abs(spec_num_yfirst[mask] / spec_exact[mask] - 1.0)
    
    interp_yfirst = RegularGridInterpolator((kx, ky), error_map_yfirst, bounds_error=False, fill_value=0)
    angular_error_yfirst = interp_yfirst(points)
    
    
    U_num_xfirst = solve_complex_system(U0, dt, steps, alpha_complex, h, y_first=False)
    fft_num_xfirst = fftn(U_num_xfirst)
    spec_num_xfirst = fftshift(np.abs(fft_num_xfirst))
    
    error_map_xfirst = np.zeros_like(spec_num_xfirst)
    error_map_xfirst[mask] = np.abs(spec_num_xfirst[mask] / spec_exact[mask] - 1.0)
    
    interp_xfirst = RegularGridInterpolator((kx, ky), error_map_xfirst, bounds_error=False, fill_value=0)
    angular_error_xfirst = interp_xfirst(points)
    
    print(f"Y-first (peaks 90°,270°): Min={np.min(angular_error_yfirst):.2e}, Max={np.max(angular_error_yfirst):.2e}")
    print(f"X-first (peaks 0°,180°):  Min={np.min(angular_error_xfirst):.2e}, Max={np.max(angular_error_xfirst):.2e}")

    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    
    
    ax = axes[0]
    ax.plot(np.degrees(thetas), angular_error_yfirst, color='#E64B35', linewidth=1.5)
    ax.set_xlim(0, 360)
    ax.set_xlabel(r'Angle $\theta$ (Degrees)')
    ax.set_ylabel('Relative Error')
    ax.set_title(r'(a) Y-implicit First $\rightarrow$ X-implicit')
    ax.grid(True, alpha=0.3)
    
    for angle in [90, 270]:
        ax.axvline(angle, color='#E64B35', linestyle='--', alpha=0.7, linewidth=1.5)
    for angle in [45, 135, 225, 315]:
        ax.axvline(angle, color='gray', linestyle=':', alpha=0.3)
    ax.annotate('Error peaks\nat 90°, 270°', xy=(90, ax.get_ylim()[1]*0.8 if ax.get_ylim()[1] > 0 else 100), 
                fontsize=8, color='#E64B35', ha='center')
    
    
    ax = axes[1]
    ax.plot(np.degrees(thetas), angular_error_xfirst, color='#4DBBD5', linewidth=1.5)
    ax.set_xlim(0, 360)
    ax.set_xlabel(r'Angle $\theta$ (Degrees)')
    ax.set_ylabel('Relative Error')
    ax.set_title(r'(b) X-implicit First $\rightarrow$ Y-implicit')
    ax.grid(True, alpha=0.3)
    
    for angle in [0, 180, 360]:
        ax.axvline(angle, color='#4DBBD5', linestyle='--', alpha=0.7, linewidth=1.5)
    for angle in [45, 135, 225, 315]:
        ax.axvline(angle, color='gray', linestyle=':', alpha=0.3)
    ax.annotate('Error peaks\nat 0°, 180°', xy=(270, ax.get_ylim()[1]*0.8 if ax.get_ylim()[1] > 0 else 200), 
                fontsize=8, color='#4DBBD5', ha='center')
    
    plt.tight_layout()
    plt.savefig('adi_sweep_order_comparison.png', dpi=300)
    print("Saved adi_sweep_order_comparison.png")
    
    plt.figure(figsize=(3.5, 3.5))
    plt.plot(np.degrees(thetas), angular_error_yfirst, color='#E64B35', linewidth=1.5)
    plt.xlim(0, 360)
    plt.xlabel(r'Angle $\theta$ (Degrees)')
    plt.ylabel('Relative Error')
    plt.title(f'Angular Profile (Y-first, $|k|={target_k_radius}$)')
    plt.grid(True, alpha=0.3)
    for angle in [90, 270]:
        plt.axvline(angle, color='b', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('angular_error_zoomed_y_first.png')
    print("Saved angular_error_zoomed_y_first.png")
    
    
    plt.figure(figsize=(3.5, 3.5))
    plt.plot(np.degrees(thetas), angular_error_xfirst, color='#4DBBD5', linewidth=1.5)
    plt.xlim(0, 360)
    plt.xlabel(r'Angle $\theta$ (Degrees)')
    plt.ylabel('Relative Error')
    plt.title(f'Angular Profile (X-first, $|k|={target_k_radius}$)')
    plt.grid(True, alpha=0.3)
    for angle in [0, 180]:
        plt.axvline(angle, color='b', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('angular_error_zoomed_x_first.png')
    print("Saved angular_error_zoomed_x_first.png")

def run_convergence_study():
    """
    Convergence study: verify O(dt²) for ADI method.
    Uses Richardson extrapolation: compare to fine-dt reference to isolate temporal error.
    Also shows angular anisotropy scaling.
    """
    print("="*60)
    print("CONVERGENCE STUDY: ADI Method (Richardson Extrapolation)")
    print("="*60)
    
    
    sigma0 = 0.05
    alpha_real = 1.0
    t_test = 0.002
    
    X_grid, Y_grid = np.meshgrid(np.linspace(0, L, N), np.linspace(0, L, N))
    
    def gaussian_exact(t):
        """Analytical solution for 2D Gaussian diffusion"""
        denom = sigma0**2 + 4 * alpha_real * t
        return (sigma0**2 / denom) * np.exp(-((X_grid - 0.5)**2 + (Y_grid - 0.5)**2) / denom)
    
    U0_gauss = gaussian_exact(0)
    
    
    dt_ref = t_test / 4000
    print(f"Computing reference solution with dt_ref = {dt_ref:.2e} (4000 steps)...")
    U_ref = np.real(solve_complex_system(U0_gauss + 0j, dt_ref, 4000, alpha_real + 0j, h))
    ref_norm = np.sqrt(np.mean(U_ref**2))
    fft_ref = fftn(U_ref)
    
    dt_base = h**2 / 4  
    
    
    step_counts = np.array([2000, 1000, 500, 200, 100, 50, 25, 16])
    dt_values = t_test / step_counts
    
    global_errors = []
    max_angular_errors = []
    anisotropy_amplitudes = []
    
    
    kx_grid = fftshift(fftfreq(N, d=L/N)) * 2 * np.pi
    ky_grid = fftshift(fftfreq(N, d=L/N)) * 2 * np.pi
    thetas = np.linspace(0, 2*np.pi, 360)
    circle_kx = target_k_radius * np.cos(thetas)
    circle_ky = target_k_radius * np.sin(thetas)
    points = np.column_stack((circle_kx, circle_ky))
    
    
    fft_exact_final = fft_ref
    
    print(f"\nGrid: N={N}, h={h:.4f}, CFL limit={dt_base:.2e}")
    print(f"Initial condition: Gaussian (σ₀={sigma0})")
    print(f"Target |k| for angular analysis: {target_k_radius}")
    print(f"Final time: t={t_test}")
    print("Comparing to fine-dt reference (isolates temporal error)")
    print("-"*60)
    
    for i, (dt, steps) in enumerate(zip(dt_values, step_counts)):
        
        U_num = np.real(solve_complex_system(U0_gauss + 0j, dt, steps, alpha_real + 0j, h))
        fft_num = fftn(U_num)
        
        
        l2_error = np.sqrt(np.mean((U_num - U_ref)**2))
        global_l2 = l2_error / ref_norm
        global_errors.append(global_l2)
        
        
        spec_num = fftshift(np.abs(fft_num))
        spec_ref = fftshift(np.abs(fft_ref))
        
        mask = spec_ref > 1e-12
        error_map = np.zeros_like(spec_num, dtype=float)
        error_map[mask] = np.abs(spec_num[mask] / spec_ref[mask] - 1.0)
        
        interp = RegularGridInterpolator((kx_grid, ky_grid), error_map, 
                                         bounds_error=False, fill_value=0)
        angular_error = interp(points)
        
        max_angular_errors.append(np.max(angular_error))
        anisotropy_amplitudes.append(np.max(angular_error) - np.min(angular_error))
        
        print(f"dt = {dt:.2e} ({dt/dt_base:5.1f}x CFL) | "
              f"steps={steps:4d} | L2={global_l2:.2e} | "
              f"max_angular={np.max(angular_error):.2e} | "
              f"anisotropy={anisotropy_amplitudes[-1]:.2e}")
    
    
    dt_values = np.array(dt_values)
    global_errors = np.array(global_errors)
    max_angular_errors = np.array(max_angular_errors)
    anisotropy_amplitudes = np.array(anisotropy_amplitudes)
    
    
    valid = ~np.isnan(global_errors) & (global_errors > 0)
    if np.sum(valid) >= 3:
        coeffs = np.polyfit(np.log(dt_values[valid]), np.log(global_errors[valid]), 1)
        order_global = coeffs[0]
        
        coeffs_aniso = np.polyfit(np.log(dt_values[valid]), 
                                   np.log(anisotropy_amplitudes[valid] + 1e-16), 1)
        order_aniso = coeffs_aniso[0]
    else:
        order_global = np.nan
        order_aniso = np.nan
    
    print("-"*60)
    print(f"Fitted convergence order (global L2): {order_global:.2f}")
    print(f"Fitted convergence order (anisotropy): {order_aniso:.2f}")
    print("Expected for Crank-Nicolson ADI: O(dt²) → order ≈ 2.0")
    print("="*60)
    
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    
    
    ax = axes[0]
    ax.loglog(dt_values, global_errors, 'o-', color='#4DBBD5', 
              linewidth=1.5, markersize=6, label='ADI (Numerical)')
    
    
    dt_plot = dt_values[valid]
    mid_idx = len(dt_plot)//2
    
    
    scale1 = global_errors[valid][mid_idx] / dt_plot[mid_idx]
    ax.loglog(dt_plot, scale1 * dt_plot, ':', color='#E64B35', 
              linewidth=1, alpha=0.7, label=r'$O(\Delta t)$ reference')
    
    
    scale2 = global_errors[valid][mid_idx] / dt_plot[mid_idx]**2
    ax.loglog(dt_plot, scale2 * dt_plot**2, '--', color='gray', 
              linewidth=1, label=r'$O(\Delta t^2)$ reference')
    
    ax.set_xlabel(r'Time step $\Delta t$ (s)')
    ax.set_ylabel(r'Relative $L_2$ Error')
    ax.set_title(f'(a) Convergence Order = {order_global:.2f}')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3, which='both')
    
    
    ax = axes[1]
    ax.loglog(dt_values, anisotropy_amplitudes, 's-', color='#E64B35', 
              linewidth=1.5, markersize=6, label='Anisotropy (max−min)')
    
    
    if not np.isnan(order_aniso):
        mid_idx_a = len(dt_plot)//2
        
        scale1_a = anisotropy_amplitudes[valid][mid_idx_a] / dt_plot[mid_idx_a]
        ax.loglog(dt_plot, scale1_a * dt_plot, ':', color='#4DBBD5', 
                  linewidth=1, alpha=0.7, label=r'$O(\Delta t)$ reference')
        
        scale2_a = anisotropy_amplitudes[valid][mid_idx_a] / dt_plot[mid_idx_a]**2
        ax.loglog(dt_plot, scale2_a * dt_plot**2, '--', color='gray', 
                  linewidth=1, label=r'$O(\Delta t^2)$ reference')
    
    ax.set_xlabel(r'Time step $\Delta t$ (s)')
    ax.set_ylabel(r'Angular Error Amplitude')
    ax.set_title(f'(b) Anisotropy Scaling = {order_aniso:.2f}')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3, which='both')
    
    
    ax = axes[2]
    n_dt = len(dt_values)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_dt))
    
    
    spec_ref_plot = fftshift(np.abs(fft_ref))
    
    
    selected_indices = [0, n_dt//3, 2*n_dt//3, n_dt-1]
    for idx in selected_indices:
        dt = dt_values[idx]
        steps = step_counts[idx]
        
        U_num = solve_complex_system(U0_gauss + 0j, dt, steps, alpha_real + 0j, h)
        fft_num = fftn(np.real(U_num))
        
        spec_num = fftshift(np.abs(fft_num))
        
        mask = spec_ref_plot > 1e-12
        error_map = np.zeros_like(spec_num, dtype=float)
        error_map[mask] = np.abs(spec_num[mask] / spec_ref_plot[mask] - 1.0)
        
        interp = RegularGridInterpolator((kx_grid, ky_grid), error_map, 
                                         bounds_error=False, fill_value=0)
        angular_error = interp(points)
        
        ax.plot(np.degrees(thetas), angular_error, color=colors[idx], 
                linewidth=1.2, label=f'{dt/dt_base:.1f}x CFL')
    
    ax.set_xlabel(r'Angle $\theta$ (Degrees)')
    ax.set_ylabel('Relative Error')
    ax.set_title(f'(c) Angular Profiles at $|k|={target_k_radius}$')
    ax.set_xlim(0, 360)
    ax.legend(fontsize=7, title=r'$\Delta t$')
    ax.grid(True, alpha=0.3)
    
    
    for angle in [45, 135, 225, 315]:
        ax.axvline(angle, color='k', linestyle=':', alpha=0.2, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('convergence_study.png', dpi=300)
    print("\nSaved convergence_study.png")
    
    return dt_values, global_errors, order_global


if __name__ == '__main__':
    
    scan_angular_error_zoomed()
    run_convergence_study()
