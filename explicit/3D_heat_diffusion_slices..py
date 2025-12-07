import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os


nx = ny = 60
nz = 10

alpha = 0.2      
dx = dy = dz = 1
dt = 0.05       

steps = 180  
save_every = 2   

rows, cols = 2, 5


x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
z = np.linspace(-1, 1, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

sigma = 0.18
u = 10.0 * np.exp(-(X**2 + Y**2 + Z**2) / (2*sigma**2))

def laplacian(u):
    lap = np.zeros_like(u)
    
    lap[1:-1,1:-1,1:-1] = (
        u[2:,1:-1,1:-1] + u[:-2,1:-1,1:-1] +
        u[1:-1,2:,1:-1] + u[1:-1,:-2,1:-1] +
        u[1:-1,1:-1,2:] + u[1:-1,1:-1,:-2]
        - 6*u[1:-1,1:-1,1:-1]
    )

    # Neumann
    lap[0,:,:]  = lap[1,:,:]
    lap[-1,:,:] = lap[-2,:,:]
    lap[:,0,:]  = lap[:,1,:]
    lap[:,-1,:] = lap[:,-2,:]
    lap[:,:,0]  = lap[:,:,1]
    lap[:,:,-1] = lap[:,:,-2]
    
    return lap


tmp_dir = "frames_diffusion_colorbar"
os.makedirs(tmp_dir, exist_ok=True)

frames = []


for step in range(steps + 1):

    if step % save_every == 0:
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5))
        axes = axes.flatten()

        # Normalize for consistent colorbar scaling
        u_norm = (u - u.min()) / (u.max() - u.min() + 1e-8)

        # Plot slices
        for i in range(nz):
            ax = axes[i]
            img = ax.imshow(u_norm[:, :, i], cmap="inferno",
                            origin="lower", vmin=0, vmax=1)
            ax.set_title(f"Slice {i}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        cbar = fig.colorbar(img, ax=axes.tolist(), fraction=0.02, pad=0.02)
        cbar.set_label("Normalized Temperature", rotation=270, labelpad=15)

        fig.suptitle(f"3D Heat Diffusion — t={step*dt:.2f}", fontsize=16)
        
        fname = f"{tmp_dir}/frame_{step:04d}.png"
        fig.savefig(fname, dpi=140, bbox_inches="tight")
        plt.close()

        frames.append(imageio.imread(fname))

    if step < steps:
        u = u + dt * alpha * laplacian(u)


gif_path = "3D_heat_diffusion_slices.gif"
imageio.mimsave(gif_path, frames, duration=0.12)

gif_path
