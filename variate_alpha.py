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
