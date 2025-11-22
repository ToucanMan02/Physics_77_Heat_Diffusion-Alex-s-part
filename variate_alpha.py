import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable, Grid2D, DiffusionTerm, TransientTerm

# a mesh grid
nx = ny = 100
L = 1.0
dx = dy = L / nx
mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)
