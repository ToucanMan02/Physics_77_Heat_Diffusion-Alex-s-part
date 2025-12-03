import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
from IPython.display import Image

def D_of_r(r):
    """Smooth radial diffusivity model from core→mantle→crust."""
    return 1e-6 + 1e-5 * np.exp(-3*r) + 5e-6 * (r**2)

N = 80
L = 1.0
x = np.linspace(-1,1,N)
y = np.linspace(-1,1,N)
dx = x[1]-x[0]
dy = y[1]-y[0]
X,Y = np.meshgrid(x,y)
r = np.sqrt(X**2 + Y**2)
alpha = D_of_r(r)
sigma = 0.15
u0 = np.exp(-((X**2 + Y**2)/(2*sigma**2)))

dt = 0.2 * min(dx,dy)**2 / np.max(alpha)
steps = 101
print("dt =",dt," steps =",steps)

def explicit_evolve(u0, alpha, steps):
    u = u0.copy()
    states = [u.copy()]
    for n in range(steps):
        u_pad = np.pad(u,1,mode='edge')
        ddx = (u_pad[2:,1:-1] - 2*u_pad[1:-1,1:-1] + u_pad[:-2,1:-1]) / dx**2
        ddy = (u_pad[1:-1,2:] - 2*u_pad[1:-1,1:-1] + u_pad[1:-1,:-2]) / dy**2
        u = u + dt*alpha*(ddx+ddy)
        states.append(u.copy())
    return np.array(states)

def implicit_like(u0, alpha, steps):
    u = u0.copy()
    states = [u.copy()]
    ax = dt/(2*dx**2)
    ay = dt/(2*dy**2)
    A = np.eye(N)* (1+2*ax) + np.eye(N,k=1)*(-ax) + np.eye(N,k=-1)*(-ax)
    C = np.eye(N)* (1+2*ay) + np.eye(N,k=1)*(-ay) + np.eye(N,k=-1)*(-ay)
    Ainv = np.linalg.inv(A)
    Cinv = np.linalg.inv(C)
    for n in range(steps):
        u_pad = np.pad(u,1,mode='edge')
        ddx = (u_pad[2:,1:-1] - 2*u_pad[1:-1,1:-1] + u_pad[:-2,1:-1]) / dx**2
        ddy = (u_pad[1:-1,2:] - 2*u_pad[1:-1,1:-1] + u_pad[1:-1,:-2]) / dy**2
        S = u + dt*alpha*(ddx+ddy)
        # ADI X-sweep then Y-sweep
        for j in range(N):
            S[j,:] = Ainv @ S[j,:]
        for i in range(N):
            S[:,i] = Cinv @ S[:,i]
        u = S.copy()
        states.append(u.copy())
    return np.array(states)

u_exp = explicit_evolve(u0, alpha, steps)
u_imp = implicit_like(u0, alpha, steps)

print("explicit shape:",u_exp.shape)
print("implicit shape:",u_imp.shape)

error_states = u_imp - u_exp

frames = []
for i in range(steps):
    fig, ax = plt.subplots(figsize=(5,5), dpi=100)
    im = ax.imshow(error_states[i], cmap='bwr', 
                   vmin=-np.max(np.abs(error_states)), 
                   vmax=np.max(np.abs(error_states)), 
                   origin='lower')
    ax.set_title(f"Error field — frame {i}")
    plt.colorbar(im, ax=ax)
    
    fig.canvas.draw()
    
    # Fix: calculate actual dimensions from buffer size
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    
    # The buffer size tells us the actual pixel dimensions
    buf_size = buf.size
    actual_pixels = buf_size // 4  # 4 channels (RGBA)
    actual_h = int(np.sqrt(actual_pixels * h / w))
    actual_w = int(np.sqrt(actual_pixels * w / h))
    
    frame = buf.reshape(actual_h, actual_w, 4)
    frames.append(frame)
    plt.close(fig)

imageio.mimsave("error_evolution.gif", frames, fps=20)
print("Saved: error_evolution.gif")
Image(filename="error_evolution.gif")