import numpy as np
import matplotlib.pyplot as plt

nx = ny = 200
L = 1.0
dx = dy = L / nx

x = np.linspace(0, L, nx)
y = np.linspace(0, L, ny)
X, Y = np.meshgrid(x, y)


alpha_min = 1e-6
alpha_max = 1.0
alpha = alpha_min + (alpha_max - alpha_min) * X   # 0 → 1 gradient

T_surround = -0.5
sigma = 0.05

gauss = np.exp(-(((X - 0.5)**2 + (Y - 0.5)**2)/(2*sigma**2)))
T_peak = 1.0

u = T_surround + (T_peak - T_surround) * gauss  # 2D array same as FiPy CellVariable
u = u.astype(float)



dt = 0.24 * min(dx,dy)**2 / np.max(alpha)    # CFL explicit stability
print("dt =", dt)

steps = 3000
plot_every = 300


def apply_neumann(arr):
    arr[:,0]  = arr[:,1]
    arr[:,-1] = arr[:,-2]
    arr[0,:]  = arr[1,:]
    arr[-1,:] = arr[-2,:]
    return arr


for n in range(steps):

    u_old = u.copy()
    u_old = apply_neumann(u_old)

    # Laplacian
    ddx = (u_old[2:,1:-1] - 2*u_old[1:-1,1:-1] + u_old[:-2,1:-1]) / dx**2
    ddy = (u_old[1:-1,2:] - 2*u_old[1:-1,1:-1] + u_old[1:-1,:-2]) / dy**2
    lap = ddx + ddy

    # Explicit update: u^{n+1} = u^n + dt * alpha * ∇²u
    u[1:-1,1:-1] = u_old[1:-1,1:-1] + dt * alpha[1:-1,1:-1] * lap

    u = apply_neumann(u)

    if n % plot_every == 0:
        plt.clf()
        plt.title(f"t = {n*dt:.4f} s")
        plt.imshow(u, cmap="jet", origin="lower", extent=[0,1,0,1])
        plt.colorbar()
        plt.pause(0.01)

plt.show()
