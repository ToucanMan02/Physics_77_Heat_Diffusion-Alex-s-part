import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
from IPython.display import Image

Nx, Ny = 200, 200
Lx, Ly = 1.0, 1.0
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

body_mask = (X >= 0.3) & (X <= 0.7) & (Y >= 0.1) & (Y <= 0.9)

left_slope  = (0.95 - 0.9) / (0.5 - 0.3)
right_slope = (0.95 - 0.9) / (0.5 - 0.7)

tri_left  = Y >= (0.9 + left_slope  * (X - 0.3))
tri_right = Y >= (0.9 + right_slope * (X - 0.7))
tri_mask  = tri_left & tri_right & (Y <= 0.95)

rocket_mask = body_mask | tri_mask

astro_cx, astro_cy = 0.5, 0.5
astro_r = 0.08
astronaut_mask = ((X - astro_cx)**2 + (Y - astro_cy)**2) <= astro_r**2
astronaut_mask &= body_mask

alpha = np.zeros_like(X)
alpha_air  = 2.1e-5
alpha_hull = 8.0e-4
alpha_body = 1.4e-7

alpha[rocket_mask] = alpha_air
alpha[astronaut_mask] = alpha_body

hull_mask = np.zeros_like(rocket_mask)
for dy_i, dx_i in [(1,0),(-1,0),(0,1),(0,-1),(2,0),(-2,0),(0,2),(0,-2)]:
    neigh = np.roll(np.roll(rocket_mask, dy_i, axis=0), dx_i, axis=1)
    hull_mask |= rocket_mask & (~neigh)

alpha[hull_mask] = alpha_hull
heating_mask = hull_mask.copy()

u = np.ones_like(X) * 295.0
u[astronaut_mask] = 310.0
heating_strength = 350

dt_small = 0.24 * min(dx, dy)**2 / np.max(alpha)
steps_per_frame = 200                               
Nframes = 200                                    

total_iterations = steps_per_frame * Nframes
frames = []
times = []

t = 0.0
for frame in range(Nframes):

    # take multiple tiny stable steps
    for s in range(steps_per_frame):
        w = u.copy()

        #trying to make this work boundary
        w[:,0] = w[:,1]
        w[:,-1] = w[:,-2]
        w[0,:] = w[1,:]
        w[-1,:] = w[-2,:]

        ddx = (w[2:, 1:-1] - 2*w[1:-1, 1:-1] + w[:-2, 1:-1]) / dx**2
        ddy = (w[1:-1, 2:] - 2*w[1:-1, 1:-1] + w[1:-1, :-2]) / dy**2
        u[1:-1, 1:-1] = w[1:-1, 1:-1] + dt_small * alpha[1:-1, 1:-1] * (ddx + ddy)

        u[heating_mask] += heating_strength * dt_small

        #trying to make this work boundary
        u[:,0] = u[:,1]
        u[:,-1] = u[:,-2]
        u[0,:] = u[1,:]
        u[-1,:] = u[-2,:]

        t += dt_small

    frames.append(u.copy())
    times.append(t)

gif_frames = []
for frame, T in zip(frames, times):

    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(
        frame,
        cmap="inferno",
        origin="lower",
        extent=[0,1,0,1]
    )
    fig.colorbar(im, ax=ax, label="Temperature (K)")

    ax.set_title(f"t = {T:.3f} s")
    ax.axis("off")

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    gif_frames.append(img)
    plt.close(fig)

imageio.mimsave("rocket.gif", gif_frames, fps=20)
Image(filename="rocket.gif")
