import numpy as np
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
from PIL import Image as PILImage
from IPython.display import Image


Nx, Ny = 200, 200
Lx, Ly = 1.0, 1.0
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)


core_mask = (X >= 0.35) & (X <= 0.65) & (Y >= 0.35) & (Y <= 0.65)


alpha = np.ones_like(X)
alpha_outside = 1.0e-6
alpha_core    = 4.0e-5

alpha[:, :] = alpha_outside
alpha[core_mask] = alpha_core


u = np.ones_like(X) * 300.0      
boundary_temp = 300.0            # fixed boundary temp
u[core_mask] = 500.0             


core_heating_strength = 2000  # can adjust


dt = 0.24 * min(dx, dy)**2 / np.max(alpha) # max alpha to avoid stability issues

steps_per_frame = 200
Nframes = 200

frames = []
times = []
max_temps = []


def laplacian(U):
    return (
        (np.roll(U,1,axis=0)-2*U+np.roll(U,-1,axis=0))/dy**2 +
        (np.roll(U,1,axis=1)-2*U+np.roll(U,-1,axis=1))/dx**2
    )


def apply_dirichlet(U):
    U[0,:]  = boundary_temp
    U[-1,:] = boundary_temp
    U[:,0]  = boundary_temp
    U[:,-1] = boundary_temp
    return U


print(f"Starting simulation with dt={dt:.6f} seconds")
print(f"Total frames: {Nframes}, steps per frame: {steps_per_frame}")

t = 0.0
for frame in range(Nframes):
    if frame % 20 == 0:
        print(f"Computing frame {frame}/{Nframes}...")

    for _ in range(steps_per_frame):
        lap = laplacian(u)

        u_new = u + alpha * dt * lap
        u_new[core_mask] += core_heating_strength * dt

        u_new = apply_dirichlet(u_new)

        u = u_new
        t += dt

    frames.append(u.copy())
    times.append(t)
    max_temps.append(np.max(u))

print(f"Simulation complete. Final time: {t:.2f} seconds")


print("Generating GIF frames...")
gif_frames = []
for i, frame in enumerate(frames):
    if i % 20 == 0:
        print(f"Rendering frame {i}/{len(frames)}...")
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    im = ax.imshow(
        frame,
        cmap="inferno",
        origin="lower",
        extent=[0, 1, 0, 1],
        vmin=300,  
        vmax=np.max(max_temps)
    )
    cbar = fig.colorbar(im, ax=ax, label="Temperature (K)")
    ax.set_title(f"Nuclear Reactor Heat Diffusion\nTime: {times[i]:.1f}s, Max T: {max_temps[i]:.1f}K")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = np.array(PILImage.open(buf))
    
    gif_frames.append(img)
    plt.close(fig)
    buf.close()

print("Saving GIF...")
imageio.mimsave("reactor.gif", gif_frames, fps=20)
print("Saved: reactor.gif")

print(f"\nSimulation Statistics:")
print(f"  Max temperature reached: {np.max(max_temps):.1f} K")
print(f"  Final max temperature: {max_temps[-1]:.1f} K")
print(f"  Total simulation time: {times[-1]:.2f} seconds")

Image(filename="reactor.gif")
