import numpy as np
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
from PIL import Image as PILImage
from IPython.display import Image
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


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
alpha_water = 2.0e-5   # Water alpha diffusivity
alpha_fuel  = 1.0e-4   # Fuel alpha  diffusivity
alpha[:, :] = alpha_water
alpha[core_mask] = alpha_fuel


u = np.ones_like(X) * 300.0      

core_heating_strength = 800.0 
cooling_failure_time = 40.0  

operating_temp = 600.0
zirconium_melt = 2100.0  
uranium_melt   = 3100.0  

dt = 0.15 * min(dx, dy)**2 / np.max(alpha)

simulation_time = 100.0      
total_steps = int(simulation_time / dt)
steps_per_frame = 60         

frames = []
times = []
max_temps = []
status_log = []

colors = [(0.0,  "#000030"), (0.1,  "#0000FF"), (0.2,  "#00FFFF"), (0.4,  "#FFFF00"), (0.7,  "#FF4500"), (1.0,  "#FFFFFF")]
reactor_cmap = LinearSegmentedColormap.from_list("reactor_thermal", colors)

def laplacian(U):
    U_padded = np.pad(U, 1, mode='edge')
    d2x = (U_padded[1:-1, 2:] - 2*U_padded[1:-1, 1:-1] + U_padded[1:-1, :-2]) / dx**2
    d2y = (U_padded[2:, 1:-1] - 2*U_padded[1:-1, 1:-1] + U_padded[:-2, 1:-1]) / dy**2
    return d2x + d2y

print(f"Starting Calibrated Simulation...")
print(f"Target Normal Temp: ~{operating_temp} K")
print(f"Target Meltdown Temp: >{uranium_melt} K")

t = 0.0
current_status = "STARTUP"

meltdown_captured = False

for step in range(total_steps):

    if t < cooling_failure_time:

        u[0,:] = u[-1,:] = u[:,0] = u[:,-1] = 300.0
        

        if np.max(u) > 550:
            current_status = "NORMAL OPERATION (600 K)"
        else:
            current_status = "STARTUP / HEATING"
            
    else:

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
            
            # Save the file
            fig_snap.savefig("meltdown_event.png", bbox_inches='tight', facecolor=bg_color)
            plt.close(fig_snap)
            
            print("Snapshot saved: meltdown_event.png")
            meltdown_captured = True 

            
        elif np.max(u) > zirconium_melt:
            current_status = "CLADDING FAILURE"
        else:
            current_status = "LOSS OF COOLANT (LOCA)"


    lap = laplacian(u)
    u_new = u + alpha * dt * lap
    u_new[core_mask] += core_heating_strength * dt
    u = u_new
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

    # fun title color
    if "MELTDOWN" in status_log[i]:
        text_color = "#FF0000"
    elif "CLADDING" in status_log[i]:
        text_color = "#FFA500" 
    else:
        text_color = "#00FFFF" 

    im = ax.imshow(
        frame, cmap=reactor_cmap, origin="lower",
        extent=[0, 1, 0, 1], vmin=300, vmax=3500 
    )
    
    for rx in rod_centers:
        for ry in rod_centers:
            # Color of outline changes if rod is melting
            edge_c = 'white' if max_temps[i] > zirconium_melt else '#404040'
            rect = Rectangle((rx - rod_size/2, ry - rod_size/2), rod_size, rod_size,
                             linewidth=1, edgecolor=edge_c, facecolor='none', alpha=0.5)
            ax.add_patch(rect)

    # Overlay Tank Walls
    for spine in ax.spines.values():
        spine.set_edgecolor('#606060')
        spine.set_linewidth(5)


    ax.text(0.03, 0.94, status_log[i], color=text_color, weight="bold", size=9,
            transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.8, edgecolor='none'))

  
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (K)", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax.set_title(f"PWR Core Simulation\nTime: {times[i]:.1f}s | Peak Temp: {max_temps[i]:.0f} K", color='white')
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

imageio.mimsave("reactor_accurate.gif", gif_frames, fps=4)
print("Saved: reactor_accurate.gif")
Image(filename="reactor_accurate.gif")
