import numpy as np
import matplotlib.pyplot as plt

# FiPy may not be installed in every environment; import guardedly
try:
    from fipy import CellVariable, Grid2D, DiffusionTerm, TransientTerm
except ImportError as exc:
    raise ImportError(
        "FiPy is required for this script. Install via 'pip install fipy'.") from exc

"""fipy_variable_alpha.py
Simulates 2-D heat diffusion with spatially varying thermal diffusivity using FiPy
and saves a figure comparable to the provided fipy_variable_alpha.png screenshot.

Domain: unit square [0,1]×[0,1] with zero-flux (Neumann) boundaries.
α(x,y) = 1e-6 + (1 - 1e-6) * x  (linear gradient from left to right)
Initial condition: Gaussian peak centered at (0.5, 0.5) with σ = 0.05.
Time integration: explicit TransientTerm with variable diffusion for 100 steps
of Δt = 1e-4.
"""

# -----------------------------------------------------------------------------
# Mesh & Variables
# -----------------------------------------------------------------------------

nx = ny = 100
L = 1.0
dx = dy = L / nx
mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)

x, y = mesh.cellCenters  # FiPy vectors

# Spatially varying diffusivity (CellVariable)
D = CellVariable(mesh=mesh, value=1e-6 + (1 - 1e-6) * x)

# Temperature field
T_surround = -0.5
sigma = 0.05
gauss = np.exp(-(((x - 0.5)**2 + (y - 0.5)**2) / (2 * sigma**2)))  # peak 1 at center
# Scale Gaussian so that center reaches 1.0 while background is -0.5
T_peak = 1.0
T = CellVariable(mesh=mesh, value=T_surround + (T_peak - T_surround) * gauss, name="Temperature")

# -----------------------------------------------------------------------------
# Equation & Time stepping
# -----------------------------------------------------------------------------

eq = TransientTerm() == DiffusionTerm(coeff=D)

dt = 1e-3
steps = 100
for _ in range(steps):
    eq.solve(var=T, dt=dt)

# -----------------------------------------------------------------------------
# Plot result
# -----------------------------------------------------------------------------

temp_arr = np.array(T.value).reshape((ny, nx))
# Normalize to 0–1 for consistent colour mapping
normed = (temp_arr - temp_arr.min()) / (temp_arr.max() - temp_arr.min())
fig, ax = plt.subplots(figsize=(6, 6))
img = ax.imshow(normed, origin="lower", extent=[0, 1, 0, 1], cmap="jet",
                vmin=0.0, vmax=1.0)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(f"T after {steps} steps (dt={dt})")
# Colour-bar with real temperature values
cbar = fig.colorbar(img, ax=ax, ticks=[0.0, 0.5, 1.0])
cbar.ax.set_yticklabels(["0.0", "0.5", "1.0"])
cbar.set_label("Normalized Temperature (0–1)")
plt.tight_layout()
plt.savefig("fipy_variable_alpha.png", dpi=300)
print("Saved fipy_variable_alpha.png")
print(f"Time passed is {dt} times {steps} = {dt*steps}")

