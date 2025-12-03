import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# ==========================================
# 1. REACTOR SETUP (Geometry & Physics)
# ==========================================
print("Initializing Reactor Geometry...")

Nx, Ny = 120, 120
Lx, Ly = 1.0, 1.0
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Create Core Mask (Fuel Rods)
core_mask = np.zeros_like(X, dtype=bool)
rod_centers = np.linspace(0.25, 0.75, 4)
rod_size = 0.07

for rx in rod_centers:
    for ry in rod_centers:
        rod_mask = (X >= rx - rod_size/2) & (X <= rx + rod_size/2) & \
                   (Y >= ry - rod_size/2) & (Y <= ry + rod_size/2)
        core_mask = core_mask | rod_mask

# Variable Diffusivity Map (Alpha)
alpha_map = np.ones_like(X)
alpha_water = 2.0e-5
alpha_fuel = 1.0e-4
alpha_map[:, :] = alpha_water
alpha_map[core_mask] = alpha_fuel

# Heating Parameters
core_heating_strength = 800.0
initial_temp = 300.0

# Calculate Theoretical Limit based on the highest diffusivity (Fuel)
# Stability requires: dt <= dx^2 / (4 * max(alpha))
max_alpha = np.max(alpha_map)
dt_theoretical_limit = dx**2 / (4 * max_alpha)

print(f"Grid: {Nx}x{Ny}")
print(f"Max Alpha (Fuel): {max_alpha}")
print(f"Theoretical Critical dt: {dt_theoretical_limit:.6e} s")

# ==========================================
# 2. SOLVER DEFINITION (Explicit Method)
# ==========================================

def laplacian(U):
    """Compute 2D Laplacian using finite differences."""
    U_padded = np.pad(U, 1, mode='edge')
    d2x = (U_padded[1:-1, 2:] - 2*U_padded[1:-1, 1:-1] + U_padded[1:-1, :-2]) / dx**2
    d2y = (U_padded[2:, 1:-1] - 2*U_padded[1:-1, 1:-1] + U_padded[:-2, 1:-1]) / dy**2
    return d2x + d2y

def run_reactor_stability_test(dt, steps=200):
    """
    Runs the reactor simulation for a specific dt.
    Returns: (Final Max Temp, Is_Stable boolean)
    """
    u = np.ones_like(X) * initial_temp
    
    # Run simulation loop
    for _ in range(steps):
        lap = laplacian(u)
        
        # Explicit update with variable alpha and heating source
        # u_new = u + (alpha * lap * dt) + (heating * dt)
        delta_u = (alpha_map * dt * lap)
        delta_u[core_mask] += core_heating_strength * dt
        
        u += delta_u
        
        # Boundary Conditions (fixed at 300K)
        u[0,:] = u[-1,:] = u[:,0] = u[:,-1] = 300.0
        
        # Stability Check: Check for NaNs or Infinite values
        if not np.isfinite(u).all():
            return np.inf, False
            
        # Stability Check: Check for absurd non-physical growth (Numerical Explosion)
        # If temp exceeds 100,000K, it's definitely a math error, not a melt.
        if np.max(u) > 1e5: 
            return np.max(u), False

    return np.max(u), True

# ==========================================
# 3. STABILITY ANALYSIS LOOP
# ==========================================
print("\nRunning Stability Analysis on Reactor Core...")

# Generate test range: log scale around the theoretical limit
# We go from 0.1x limit to 5.0x limit
dts = np.logspace(np.log10(dt_theoretical_limit * 0.1), np.log10(dt_theoretical_limit * 5.0), 50)

final_temps = []
stability_status = []

for i, dt_test in enumerate(dts):
    temp, is_stable = run_reactor_stability_test(dt_test)
    final_temps.append(temp)
    stability_status.append(is_stable)
    
    if i % 10 == 0:
        print(f"Testing dt = {dt_test:.2e} | Stable: {is_stable}")

# ==========================================
# 4. VISUALIZATION
# ==========================================
fig = plt.figure(figsize=(14, 8), facecolor='#f0f0f0')
gs = GridSpec(2, 2, width_ratios=[1, 2])

# Plot A: The Reactor Geometry (Visual Context)
ax_geo = fig.add_subplot(gs[:, 0])
colors = [(0.0, "#000030"), (0.5, "#0000FF"), (1.0, "#00FFFF")]
reactor_cmap = LinearSegmentedColormap.from_list("reactor_blue", colors)

# Show the alpha map or geometry
im = ax_geo.imshow(alpha_map, cmap='Blues_r', extent=[0, 1, 0, 1], origin='lower')
ax_geo.set_title("Reactor Geometry\n(High Alpha regions in Dark)", fontweight='bold')
ax_geo.set_xlabel("L_x")
ax_geo.set_ylabel("L_y")

# Overlay Rods
for rx in rod_centers:
    for ry in rod_centers:
        rect = Rectangle((rx - rod_size/2, ry - rod_size/2), rod_size, rod_size,
                         linewidth=1, edgecolor='red', facecolor='none', alpha=0.7)
        ax_geo.add_patch(rect)
ax_geo.text(0.5, -0.15, "Red Outlines = Fuel Rods", ha='center', transform=ax_geo.transAxes, color='red')


# Plot B: Stability Curve
ax_stab = fig.add_subplot(gs[:, 1])

# Filter data for plotting (handle Infinity for log plot)
plot_temps = np.array(final_temps)
plot_temps = np.where(np.isfinite(plot_temps), plot_temps, 1e6) # Cap infinity for visual

# Plot lines
ax_stab.plot(dts, plot_temps, 'o-', color='#333333', label="Max Temp after 200 steps")

# Vertical line for Theoretical Limit
ax_stab.axvline(dt_theoretical_limit, color='red', linestyle='--', linewidth=2, 
                label=f"Theoretical Limit\n(dt={dt_theoretical_limit:.2e})")

# Highlight Stable vs Unstable regions
ax_stab.axvspan(min(dts), dt_theoretical_limit, color='green', alpha=0.1, label="Stable Region")
ax_stab.axvspan(dt_theoretical_limit, max(dts), color='red', alpha=0.1, label="Unstable Region")

ax_stab.set_xscale("log")
ax_stab.set_yscale("log")
ax_stab.set_xlabel("Time Step (dt)", fontweight='bold')
ax_stab.set_ylabel("Max Temperature (K)", fontweight='bold')
ax_stab.set_title("Reactor Stability Analysis", fontweight='bold', fontsize=14)
ax_stab.grid(True, which="both", ls="--", alpha=0.4)
ax_stab.legend(loc='upper left')

# Annotations
ax_stab.text(dt_theoretical_limit * 0.5, 500, "Physically Accurate\nSimulation", 
             color='green', ha='right', fontweight='bold')
ax_stab.text(dt_theoretical_limit * 1.5, 1e5, "Numerical Explosion\n(Math Error)", 
             color='red', ha='left', fontweight='bold')

plt.tight_layout()
plt.savefig('reactor_stability_analysis.png', dpi=150)
print("\nPlot saved as 'reactor_stability_analysis.png'")

# Print Summary Table
print("\n" + "="*60)
print("REACTOR STABILITY SUMMARY")
print("="*60)
print(f"{'dt (Time Step)':<20} {'Max Temp':<20} {'Status'}")
print("-" * 60)

# Sample a few points around the transition
indices = np.linspace(0, len(dts)-1, 10, dtype=int)
for i in indices:
    status = "STABLE" if stability_status[i] else "EXPLODED"
    temp_str = f"{final_temps[i]:.1f}" if stability_status[i] else "INF"
    prefix = ">>> " if (dts[i] > dt_theoretical_limit and stability_status[i]) else "    " # Mark anomalies
    print(f"{prefix}{dts[i]:<20.2e} {temp_str:<20} {status}")

print("="*60)
plt.show()