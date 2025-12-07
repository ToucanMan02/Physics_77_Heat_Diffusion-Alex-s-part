

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

alpha = 0.1  

letter = "❤️"  


def letter_m_ini(x, y, N, temp_inside, temp_outside, upscale=4):

    
    Nx = N * upscale
    Ny = N * upscale

    xx = np.linspace(0, 1, Nx)
    yy = np.linspace(0, 1, Ny)
    xv, yv = np.meshgrid(xx, yy) 

    
    tp = TextPath((0, 0), letter, size=1)
    bbox = tp.get_extents() 
    

    
    scale = 0.8 / max(bbox.width, bbox.height)
    tp = tp.transformed(transforms.Affine2D().scale(scale))

    
    bbox = tp.get_extents() 

    
    shift_x = 0.5 - (bbox.xmin + bbox.width / 2)
    shift_y = 0.5 - (bbox.ymin + bbox.height / 2)

    
    tp = tp.transformed(transforms.Affine2D().translate(shift_x, shift_y))

    
    pts = np.vstack([xv.ravel(), yv.ravel()]).T 
    mask_hr = tp.contains_points(pts).reshape((Ny, Nx))

    
    mask_lr = mask_hr.reshape(N, upscale, N, upscale).mean(axis=(1, 3))

    
    phi = np.where(mask_lr >= 0.5, temp_inside, temp_outside)

    
    phi = gaussian_filter(phi, sigma=1.0)

    return phi.astype(np.float32) 



def tri_disc(N, a):
    M = (np.diag(-a * np.ones(N-1), k=-1) + 
         np.diag((1+2*a) * np.ones(N), k=0) +
         np.diag(-a * np.ones(N-1), k=1))

    
    M[0, 0] = 1 + a
    M[0, 1] = -a
    M[-1, -1] = 1 + a
    M[-1, -2] = -a
    return M


print("Processing...")


N = 300              
dt = 1e-4            
x = np.linspace(0, 1, N) 
y = np.linspace(0, 1, N)

h = 1/N              
k = alpha * dt/(2*h**2)  


T_font = 1.0 
T_surrounding = -1.0 


u0 = letter_m_ini(x, y, N, T_font, T_surrounding, upscale=4) 
u = np.copy(u0)

print("Generated centered initial condition.") 


A = np.array(tri_disc(N, k), dtype=np.float32)
C = np.array(tri_disc(N, k), dtype=np.float32)
A_inv = np.linalg.inv(A)
C_inv = np.linalg.inv(C)


u_pred = [np.copy(u)] 
u_star = np.zeros((N, N), dtype=np.float32) 
max_iter = 100 

print("Starting iteration session...")


for it in range(max_iter-1): 

    u_padded = np.pad(u, 1, mode='edge')

    
    S_north = u_padded[0:-2, 1:-1]
    S_south = u_padded[2:, 1:-1]
    S_west  = u_padded[1:-1, 0:-2]
    S_east  = u_padded[1:-1, 2:]
    S_center = (1 - 4*k) * u

    
    S = k * (S_north + S_south + S_west + S_east) + S_center

    
    u_star[1:-1] = np.dot(S[1:-1], A_inv.T)

    
    u_star[0,0]     = k*(u[1,0]-2*u[0,0]+u[0,1])+u[0,0]
    u_star[0,-1]    = k*(u[1,-1]-2*u[0,-1]+u[0,-2])+u[0,-1]
    u_star[-1,0]    = k*(u[-2,0]-2*u[-1,0]+u[-1,1])+u[-1,0]
    u_star[-1,-1]   = k*(u[-2,-1]-2*u[-1,-1]+u[-1,-2])+u[-1,-1]
    u_star[0,1:-1]  = k*(-3*u[0,1:-1]+u[1,1:-1]+u[0,:-2]+u[0,2:])+u[0,1:-1]
    u_star[-1,1:-1] = k*(-3*u[-1,1:-1]+u[-2,1:-1]+u[-1,:-2]+u[-1,2:])+u[-1,1:-1]

    
    u = np.dot(C_inv, u_star.T).T

    
    u_pred.append(np.copy(u))

print("Simulation complete. Creating animation...")


fig, ax = plt.subplots(figsize=(8, 8), dpi=250) 


im = ax.imshow(
    u_pred[0],
    interpolation='bicubic',   
    cmap='nipy_spectral',
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='equal'
)

im.set_clim(T_surrounding, T_font) 
fig.colorbar(im, ax=ax, label='Temperature')
ax.axis('off') 
title = ax.set_title(f"Time: 0.0000 (Iteration 0)") 


def animate(i):
    im.set_array(u_pred[i]) 
    title.set_text(f"Time: {i*dt:.4f} (Iteration {i})")
    return im, title


anim = FuncAnimation(fig, animate, frames=max_iter, interval=150000, blit=True)

try:
    
    anim.save('heat_diffusion.gif', writer='pillow', fps=20, dpi=250) 
    print("Animation saved as heat_diffusion.gif")
except Exception as e:
    print("Error saving GIF:", e)

plt.close(figc)


from IPython.display import Image
Image(filename="heat_diffusion.gif")
