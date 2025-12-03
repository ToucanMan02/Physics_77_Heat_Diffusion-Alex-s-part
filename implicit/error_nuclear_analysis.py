import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.linalg import solve_banded

Nx, Ny = 80, 80  
Lx, Ly = 1.0, 1.0
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)


core_mask = np.zeros_like(X, dtype=bool)
rod_centers = np.linspace(0.25, 0.75, 4)
rod_size = 0.07 
for rx in rod_centers:
    for ry in rod_centers:
        rod_mask = (X >= rx - rod_size/2) & (X <= rx + rod_size/2) & \
                   (Y >= ry - rod_size/2) & (Y <= ry + rod_size/2)
        core_mask = core_mask | rod_mask

alpha = np.ones_like(X)
alpha_water = 2.0e-5   
alpha_fuel  = 1.0e-4   
alpha[:, :] = alpha_water
alpha[core_mask] = alpha_fuel

core_heating_strength = 800.0 
initial_temp = 300.0
T_final = 20  


max_alpha = np.max(alpha)
dt_cfl_limit = dx**2 / (4 * max_alpha)
print(f"Theoretical Explicit Stability Limit: {dt_cfl_limit:.5f} s")



def get_initial_u():
    return np.ones_like(X) * initial_temp


def step_explicit(u, dt):
  
    u_pad = np.pad(u, 1, mode='edge')
    d2x = (u_pad[1:-1, 2:] - 2*u_pad[1:-1, 1:-1] + u_pad[1:-1, :-2]) / dx**2
    d2y = (u_pad[2:, 1:-1] - 2*u_pad[1:-1, 1:-1] + u_pad[:-2, 1:-1]) / dy**2
    
    lap = d2x + d2y
    
 
    u_new = u + alpha * dt * lap
    u_new[core_mask] += core_heating_strength * dt
    
   
    u_new[0,:] = u_new[-1,:] = u_new[:,0] = u_new[:,-1] = 300.0
    return u_new


def solve_tridiagonal(alpha_slice, u_slice, dt, dx_sq):
    N = len(u_slice)
    r = alpha_slice * dt / (2 * dx_sq)
    
  
    ab = np.zeros((3, N))
    ab[0, 1:] = -r[:-1]
    ab[1, :]  = 1 + 2*r
    ab[2, :-1] = -r[1:]
    
    rhs = u_slice.copy()
    
  
    ab[0, 1] = 0; ab[1, 0] = 1; ab[2, 0] = 0; rhs[0] = 300.0
    ab[0, -1] = 0; ab[1, -1] = 1; ab[2, -2] = 0; rhs[-1] = 300.0
    
    return solve_banded((1, 1), ab, rhs)

def step_adi(u, dt):
    Ny, Nx = u.shape
    u_half = np.zeros_like(u)
    u_new  = np.zeros_like(u)
   
    heat_source = np.zeros_like(u)
    heat_source[core_mask] = core_heating_strength * (dt / 2.0)

    u_pad = np.pad(u, 1, mode='edge')
    d2y = (u_pad[2:, 1:-1] - 2*u_pad[1:-1, 1:-1] + u_pad[:-2, 1:-1]) / dy**2
    rhs_x = u + (alpha * (dt/2.0) * d2y) + heat_source
    
    for j in range(Ny):
        u_half[j, :] = solve_tridiagonal(alpha[j, :], rhs_x[j, :], dt/2.0, dx**2)
        
   
    u_half_pad = np.pad(u_half, 1, mode='edge')
    d2x = (u_half_pad[1:-1, 2:] - 2*u_half_pad[1:-1, 1:-1] + u_half_pad[1:-1, :-2]) / dx**2
    rhs_y = u_half + (alpha * (dt/2.0) * d2x) + heat_source
    
    for i in range(Nx):
        u_new[:, i] = solve_tridiagonal(alpha[:, i], rhs_y[:, i], dt/2.0, dy**2)
        
    return u_new



print("Generating Ground Truth (Reference Solution)...")

dt_ref = dt_cfl_limit / 20.0  
u_ref = get_initial_u()
t = 0
while t < T_final:
    u_ref = step_adi(u_ref, dt_ref)
    t += dt_ref

print(f"Reference computed. Max Temp: {np.max(u_ref):.2f} K")


dt_values = np.logspace(np.log10(dt_cfl_limit/5), np.log10(dt_cfl_limit*5), 15)
errors_explicit = []
errors_adi = []

print("Running Comparison Loop...")
for dt_val in dt_values:
  
    u_exp = get_initial_u()
    t = 0
    exploded = False
    
    try:
        while t < T_final:
            u_exp = step_explicit(u_exp, dt_val)
            t += dt_val
            if np.max(u_exp) > 1e6: 
                exploded = True
                break
    except:
        exploded = True

    if exploded or not np.isfinite(u_exp).all():
        errors_explicit.append(np.nan) 
    else:

        err = np.sqrt(np.mean((u_exp - u_ref)**2))
        errors_explicit.append(err)

    u_adi = get_initial_u()
    t = 0
    while t < T_final:
        u_adi = step_adi(u_adi, dt_val)
        t += dt_val
    
    err_adi = np.sqrt(np.mean((u_adi - u_ref)**2))
    errors_adi.append(err_adi)


plt.figure(figsize=(10, 7), dpi=100)
plt.rcParams.update({'font.size': 12})


plt.plot(dt_values, errors_explicit, 'ro-', label='Explicit (Forward Euler)', linewidth=2, markersize=6)

plt.plot(dt_values, errors_adi, 'bs-', label='Implicit (ADI)', linewidth=2, markersize=6)


plt.axvline(x=dt_cfl_limit, color='red', linestyle=':', linewidth=2, label='Explicit Stability Limit ($dt_{CFL}$)')


ref_x = dt_values
ref_y = errors_adi[0] * (ref_x / ref_x[0])**1 
plt.plot(ref_x, ref_y, 'k--', alpha=0.5, label='Theoretical $O(\Delta t)$ Slope')


plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time Step $\Delta t$ (s)')
plt.ylabel('$L_2$ Error (RMSE vs Reference)')
plt.title('Nuclear Reactor Simulation: Time Step vs. Accuracy')
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(loc='upper left')


plt.text(dt_cfl_limit * 1.1, np.nanmin(errors_adi)*2, "Explicit Method\nExplodes Here \u2192", color='red', fontsize=10, fontweight='bold')
plt.text(dt_cfl_limit * 0.9, np.nanmax(errors_adi), "Stable Region", color='green', ha='right', fontsize=10)

plt.tight_layout()
plt.savefig('nuclear_error_analysis.png')
print("Graph saved as 'nuclear_error_analysis.png'")
plt.show()
