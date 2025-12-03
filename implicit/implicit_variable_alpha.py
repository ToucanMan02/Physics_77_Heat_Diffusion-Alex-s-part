import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Corrected Setup (variable α; retain original ADI structure)
# -----------------------------------------------------------------------------
N = 100
L = 1.0

dt = 1e-4
h = 1.0 / N  # grid spacing consistent with original implicit.py

# Create meshgrid first so every node knows its (x,y)
x = np.linspace(0, L, N, dtype=np.float32)
y = np.linspace(0, L, N, dtype=np.float32)
xv, yv = np.meshgrid(x, y, indexing="xy")

# α increases linearly from left (≈0) to right (1).
alpha_mat = (xv / L).astype(np.float32) + 1e-6  # add epsilon to avoid zero-k
alpha_mat = np.clip(alpha_mat, 0.0, 1.0)

k_mat = alpha_mat * dt / (2 * h ** 2)

# -----------------------------------------------------------------------------
# Initial temperature: Gaussian bump in centre
# -----------------------------------------------------------------------------
u0 = np.exp(-200 * ((xv - 0.5) ** 2 + (yv - 0.5) ** 2)).astype(np.float32)

# -----------------------------------------------------------------------------
# Helper: classic tri-diagonal builder (same as original implicit.py)
# -----------------------------------------------------------------------------

def tri_disc(N, a):
    """Creates the 1-D implicit operator matrix (I − a D_xx) with Neumann BC."""
    M = (
        np.diag(-a * np.ones(N - 1), k=-1)
        + np.diag((1 + 2 * a) * np.ones(N), k=0)
        + np.diag(-a * np.ones(N - 1), k=1)
    )
    M[0, 0] = 1 + a
    M[0, 1] = -a
    M[-1, -1] = 1 + a
    M[-1, -2] = -a
    return M

# -----------------------------------------------------------------------------
# Pre-compute row/column inverses with local k (approximate but preserves structure)
# -----------------------------------------------------------------------------
A_inv_rows = [np.linalg.inv(tri_disc(N, k_mat[i, 0])) for i in range(N)]  # k constant along row
C_inv_cols = [np.linalg.inv(tri_disc(N, k_mat[0, j])) for j in range(N)]  # k constant along column

# -----------------------------------------------------------------------------
# Plot helper (unchanged except filename)
# -----------------------------------------------------------------------------

def resplot(x, y, u_pred, dt, max_iter):
    fig = plt.figure(figsize=(8, 2))
    fig.subplots_adjust(wspace=0.4)

    steps = [0, int(0.25 * max_iter), int(0.5 * max_iter), max_iter - 1]
    titles = ["$u_0$", f"$u_{steps[1]}$", f"$u_{steps[2]}$", f"$u_{steps[3]}$"]

    for idx, step in enumerate(steps):
        ax = plt.subplot(1, 4, idx + 1)
        im = ax.imshow(
            u_pred[step],
            interpolation="nearest",
            cmap="jet",
            extent=[0, L, 0, L],
            origin="lower",
        )
        ax.set_title(titles[idx], fontsize=14)
        ax.text(0.5, -0.12, f"$t = {step * dt:.3f}$", transform=ax.transAxes,
                ha="center", va="top", fontsize=11)
        ax.axis("off")
        im.set_clim(-1, 1)

    fig.colorbar(im, ax=fig.axes, shrink=0.8, pad=0.02)
    plt.savefig("implicit/variable_alpha_result.png")
    print("Plot saved to implicit/variable_alpha_result.png")

# -----------------------------------------------------------------------------
# ADI Time-stepping (minimal edits: replace scalar k with k_mat; row/col inverses)
# -----------------------------------------------------------------------------
print("Running variable-α ADI (simple) …")

max_iter = 100
u = u0.copy()
u_pred = [u.copy()]
u_star = np.zeros_like(u)

for it in range(max_iter - 1):
    u_padded = np.pad(u, 1, mode="edge")

    S_north = u_padded[:-2, 1:-1]
    S_south = u_padded[2:, 1:-1]
    S_west = u_padded[1:-1, :-2]
    S_east = u_padded[1:-1, 2:]
    S_center = (1 - 4 * k_mat) * u

    S = k_mat * (S_north + S_south + S_west + S_east) + S_center

    # Implicit solve along x (row by row)
    for i in range(1, N - 1):
        u_star[i, :] = S[i, :] @ A_inv_rows[i].T

    # Boundary rows
    u_star[0, 0] = k_mat[0, 0] * (u[1, 0] - 2 * u[0, 0] + u[0, 1]) + u[0, 0]
    u_star[0, -1] = k_mat[0, -1] * (u[1, -1] - 2 * u[0, -1] + u[0, -2]) + u[0, -1]
    u_star[-1, 0] = k_mat[-1, 0] * (u[-2, 0] - 2 * u[-1, 0] + u[-1, 1]) + u[-1, 0]
    u_star[-1, -1] = k_mat[-1, -1] * (u[-2, -1] - 2 * u[-1, -1] + u[-1, -2]) + u[-1, -1]

    u_star[0, 1:-1] = k_mat[0, 1:-1] * (
        -3 * u[0, 1:-1] + u[1, 1:-1] + u[0, :-2] + u[0, 2:]
    ) + u[0, 1:-1]
    u_star[-1, 1:-1] = k_mat[-1, 1:-1] * (
        -3 * u[-1, 1:-1] + u[-2, 1:-1] + u[-1, :-2] + u[-1, 2:]
    ) + u[-1, 1:-1]

    # Implicit solve along y (column by column)
    for j in range(N):
        u[:, j] = C_inv_cols[j] @ u_star[:, j]

    u_pred.append(u.copy())

print("Simulation complete.")
resplot(x, y, u_pred, dt, max_iter)