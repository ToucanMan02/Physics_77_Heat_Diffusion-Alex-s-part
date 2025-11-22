import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid2D, DiffusionTerm, TransientTerm

# a mesh grid
nx = ny = 100
L = 1.0
dx = dy = L / nx
mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)

x, y = mesh.cellCenters  

D = CellVariable(mesh=mesh, value=1e-6 + (1 - 1e-6) * x)

'''
so the thing above sets a mesh grid with varying alpha (linear increase from 1e-6 to 1 (linear increase)
'''

T_surround = -0.5
sigma = 0.05 # this is the size of gaussian in the middle
gauss = np.exp(-(((x - 0.5)**2 + (y - 0.5)**2) / (2 * sigma**2)))
# scale Gaussian so that center reaches 1.0 while background is -0.5
T_peak = 1.0
T = CellVariable(mesh=mesh, value=T_surround + (T_peak - T_surround) * gauss, name="Temperature")


eq = TransientTerm() == DiffusionTerm(coeff=D)

dt = 1
steps = 100
for _ in range(steps):
    eq.solve(var=T, dt=dt)

temp_arr = np.array(T.value).reshape((ny, nx))
normed = (temp_arr - temp_arr.min()) / (temp_arr.max() - temp_arr.min())
fig, ax = plt.subplots(figsize=(6, 6))
img = ax.imshow(normed, origin="lower", extent=[0, 1, 0, 1], cmap="jet",
                vmin=0.0, vmax=1.0)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(f"T after {steps} steps (dt={dt})")
cbar = fig.colorbar(img, ax=ax, ticks=[0.0, 0.5, 1.0])
cbar.set_label("Temperature")
plt.tight_layout()
plt.savefig("fipy_variable_alpha.png", dpi=300)
print("Saved fipy_variable_alpha.png")
print(f"Time passed is {dt} times {steps} = {dt*steps}")
