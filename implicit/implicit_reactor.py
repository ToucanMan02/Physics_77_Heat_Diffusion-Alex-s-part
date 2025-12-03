import numpy as np
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
from PIL import Image as PILImage
from IPython.display import Image
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.linalg import solve_banded

Nx, Ny = 120, 120 
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

u = np.ones_like(X) * 300.0      

core_heating_strength = 800.0 
cooling_failure_time = 40.0  

operating_temp = 600.0
zirconium_melt = 2100.0  
uranium_melt   = 3100.0  

dt = 0.05  
print(f"Using ADI Method. Time step dt = {dt}s (Significantly faster than Explicit)")

simulation_time = 100.0      
total_steps = int(simulation_time / dt)
steps_per_frame = 10 

frames = []
times = []
max_temps = []
status_log = []

colors = [(0.0,  "#000030"), (0.1,  "#0000FF"), (0.2,  "#00FFFF"), (0.4,  "#FFFF00"), (0.7,  "#FF4500"), (1.0,  "#FFFFFF")]
reactor_cmap = LinearSegmentedColormap.from_list("reactor_thermal", colors)



def solve_tridiagonal(alpha_slice, u_slice, dt, dx_sq, bc_type):
    """
    Solves the 1D implicit heat step: (I - r/2 * D2) u_new = RHS
    """
    N = len(u_slice)
    r = alpha_slice * dt / (2 * dx_sq)
    
 
    ab = np.zeros((3, N))
    
    ab[0, 1:] = -r[:-1]      
    ab[1, :]  = 1 + 2*r     
    ab[2, :-1] = -r[1:]      
    
    
    rhs = u_slice.copy()
    
  
    if bc_type == "dirichlet": 
       
        ab[0, 1] = 0         
        ab[1, 0] = 1         
        ab[2, 0] = 0        
        rhs[0]   = 300.0   
        
        ab[0, -1] = 0        
        ab[1, -1] = 1        
        ab[2, -2] = 0
        rhs[-1]   = 300.0
        
    elif bc_type == "neumann": 
        
        ab[0, 1] = 0
        ab[1, 0] = 1
        ab[2, 0] = -1  
        rhs[0]   = 0
        
       
        ab[0, -1] = 1 
        ab[1, -1] = -1
        ab[2, -2] = 0
        rhs[-1]   = 0

    return solve_banded((1, 1), ab, rhs)

def perform_adi_step(u_grid, dt, bc_type):
    """
    Peaceman-Rachford ADI Scheme
    Step 1: Implicit in X, Explicit in Y (t -> t + dt/2)
    Step 2: Implicit in Y, Explicit in X (t + dt/2 -> t + dt)
    """
    Ny, Nx = u_grid.shape
    u_half = np.zeros_like(u_grid)
    u_new  = np.zeros_like(u_grid)
    
 
    heat_source = np.zeros_like(u_grid)
    heat_source[core_mask] = core_heating_strength * (dt / 2.0)


    u_padded = np.pad(u_grid, 1, mode='edge')
    d2y = (u_padded[2:, 1:-1] - 2*u_padded[1:-1, 1:-1] + u_padded[:-2, 1:-1]) / dy**2
    

    rhs_x_step = u_grid + (alpha * (dt / 2.0) * d2y) + heat_source
    

    for j in range(Ny):
        u_half[j, :] = solve_tridiagonal(alpha[j, :], rhs_x_step[j, :], dt/2.0, dx**2, bc_type)


    u_half_padded = np.pad(u_half, 1, mode='edge')
    d2x = (u_half_padded[1:-1, 2:] - 2*u_half_padded[1:-1, 1:-1] + u_half_padded[1:-1, :-2]) / dx**2
    
    rhs_y_step = u_half + (alpha * (dt / 2.0) * d2x) + heat_source
    

    for i in range(Nx):
        u_new[:, i] = solve_tridiagonal(alpha[:, i], rhs_y_step[:, i], dt/2.0, dy**2, bc_type)
        
    return u_new


print(f"Starting ADI Simulation...")

t = 0.0
current_status = "STARTUP"
meltdown_captured = False
bc_mode = "dirichlet"

for step in range(total_steps):

  
    if t < cooling_failure_time:
        bc_mode = "dirichlet"
        if np.max(u) > 550:
            current_status = "NORMAL OPERATION (600 K)"
        else:
            current_status = "STARTUP / HEATING"
    else:
        bc_mode = "neumann"
   
        if np.max(u) > uranium_melt:
            current_status = "!!! CORE MELTDOWN !!!"
            
   
            if not meltdown_captured:
                print(f"Meltdown threshold reached at t={t:.2f}s. Saving snapshot...")
                fig_snap, ax_snap = plt.subplots(figsize=(6, 6), dpi=150)
                bg_color = "#101010" 
                fig_snap.patch.set_facecolor(bg_color)
                ax_snap.set_facecolor(bg_color)
                im_snap = ax_snap.imshow(u, cmap=reactor_cmap, origin="lower",
                                         extent=[0, 1, 0, 1], vmin=300, vmax=3500)
                for rx in rod_centers:
                    for ry in rod_centers:
                        rect = Rectangle((rx - rod_size/2, ry - rod_size/2), rod_size, rod_size,
                                         linewidth=1, edgecolor='white', facecolor='none', alpha=0.5)
                        ax_snap.add_patch(rect)
                for spine in ax_snap.spines.values():
                    spine.set_edgecolor('#606060')
                    spine.set_linewidth(5)
                ax_snap.text(0.03, 0.94, "!!! CORE MELTDOWN !!!", color="red", weight="bold", size=10,
                        transform=ax_snap.transAxes, bbox=dict(facecolor='black', alpha=0.8, edgecolor='none'))
                ax_snap.set_title(f"CRITICAL FAILURE EVENT\nTime: {t:.2f}s | Temp: {np.max(u):.0f} K", color='white')
                ax_snap.set_xticks([])
                ax_snap.set_yticks([])
                fig_snap.savefig("meltdown_event_adi.png", bbox_inches='tight', facecolor=bg_color)
                plt.close(fig_snap)
                print("Snapshot saved: meltdown_event_adi.png")
                meltdown_captured = True 

        elif np.max(u) > zirconium_melt:
            current_status = "CLADDING FAILURE"
        else:
            current_status = "LOSS OF COOLANT (LOCA)"


    u = perform_adi_step(u, dt, bc_mode)
    t += dt

   
    if step % steps_per_frame == 0:
        frames.append(u.copy())
        times.append(t)
        max_temps.append(np.max(u))
        status_log.append(current_status)
        
        if len(frames) % 10 == 0:
            print(f"Time: {t:.1f}s | Max Temp: {np.max(u):.0f} K | Status: {current_status}")

print("Rendering high-contrast visualization...")


gif_frames = []
for i, frame in enumerate(frames):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    bg_color = "#101010" 
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    if "MELTDOWN" in status_log[i]: text_color = "#FF0000"
    elif "CLADDING" in status_log[i]: text_color = "#FFA500" 
    else: text_color = "#00FFFF" 

    im = ax.imshow(frame, cmap=reactor_cmap, origin="lower",
        extent=[0, 1, 0, 1], vmin=300, vmax=3500)
    
    for rx in rod_centers:
        for ry in rod_centers:
            edge_c = 'white' if max_temps[i] > zirconium_melt else '#404040'
            rect = Rectangle((rx - rod_size/2, ry - rod_size/2), rod_size, rod_size,
                             linewidth=1, edgecolor=edge_c, facecolor='none', alpha=0.5)
            ax.add_patch(rect)

    for spine in ax.spines.values():
        spine.set_edgecolor('#606060')
        spine.set_linewidth(5)

    ax.text(0.03, 0.94, status_log[i], color=text_color, weight="bold", size=9,
            transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.8, edgecolor='none'))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (K)", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax.set_title(f"PWR Core (ADI Method)\nTime: {times[i]:.1f}s | Peak Temp: {max_temps[i]:.0f} K", color='white')
    ax.set_xticks([])
    ax.set_yticks([])

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=bg_color)
    buf.seek(0)
    img = np.array(PILImage.open(buf))
    gif_frames.append(img)
    plt.close(fig)
    buf.close()

last_frame = gif_frames[-1]
for _ in range(30):
    gif_frames.append(last_frame)

imageio.mimsave("reactor_adi.gif", gif_frames, fps=10)
print("Saved: reactor_adi.gif")
Image(filename="reactor_adi.gif")
