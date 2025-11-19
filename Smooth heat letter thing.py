#First part

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

alpha = 0.1  # heat diffusion constant (controls how quickly the heat spreads)

letter = "WOOOOAAAAAHHAHAHA"  # the text that will be turned into a temperature mask


def letter_m_ini(x, y, N, temp_inside, temp_outside, upscale=4):

    #Create supersampled grid, essentially make big and really detailed then shrink down and appoximate to make smooth
    Nx = N * upscale
    Ny = N * upscale

    xx = np.linspace(0, 1, Nx)
    yy = np.linspace(0, 1, Ny)
    xv, yv = np.meshgrid(xx, yy) # a meshgrid is basically like a pixel map with coords (xx,yy)

    #Build a TextPath, bassically takes the text imput makes it into vectors and uses Bézier curves to estimate the shape (thats why some emojis dont work)
    tp = TextPath((0, 0), letter, size=1)
    bbox = tp.get_extents() #find the bounds of the shape via rectangle pre-scaling
    

    #Scale text so it fits inside the unit square with margin
    scale = 0.8 / max(bbox.width, bbox.height)
    tp = tp.transformed(transforms.Affine2D().scale(scale))

    # this bbox finds the rect of the entire screen to then find the center
    bbox = tp.get_extents() 

    # based on previous comment, finds the center of the screen
    shift_x = 0.5 - (bbox.xmin + bbox.width / 2)
    shift_y = 0.5 - (bbox.ymin + bbox.height / 2)

    #move to center of the screen
    tp = tp.transformed(transforms.Affine2D().translate(shift_x, shift_y))

    #get pixel coords as one datatype as opposed to (xx,yy)
    pts = np.vstack([xv.ravel(), yv.ravel()]).T # .ravel turns the 2D array into a 1D one, then make it 2xN matrix
    mask_hr = tp.contains_points(pts).reshape((Ny, Nx))#contains_points() requires Nx2 matrix (thats why we did vstack), this returns a boolean for each pixel in the text we provided

    # make small, make smooth
    mask_lr = mask_hr.reshape(N, upscale, N, upscale).mean(axis=(1, 3))

    # more inside -> hotter
    phi = np.where(mask_lr >= 0.5, temp_inside, temp_outside)

    # more smooth, gaussian_filter is gets the value of a pixel by getting a weighted average of the surrounding pixels with impact on color being inversly prop to distance
    phi = gaussian_filter(phi, sigma=1.0)

    return phi.astype(np.float32) #float 32 to save memory as by default it is 64



def tri_disc(N, a):
    M = (np.diag(-a * np.ones(N-1), k=-1) + #M is the the tridiagnolized matrix used to solve PDEs the values come from the 1D heat equation
         np.diag((1+2*a) * np.ones(N), k=0) +#we dont do the 2D equation because doing 2 1Ds (corresponidng for the y and x) is a lot less computationally expensive
         np.diag(-a * np.ones(N-1), k=1))

    # du/dx = 0 is computed for each corner in the matrix as those correspond to the the edges of the display
    M[0, 0] = 1 + a
    M[0, 1] = -a
    M[-1, -1] = 1 + a
    M[-1, -2] = -a
    return M


print("Processing...")


N = 300              # the dimentions for the screen (they are both equal to N)
dt = 1e-4            # what you would think, time step
x = np.linspace(0, 1, N) # coords for the pixels that will actually be on the screen
y = np.linspace(0, 1, N)

h = 1/N              # distance between each pixel
k = alpha * dt/(2*h**2)  #coefficient with the same symbol from the heat equation

# Temperature values
T_font = 1.0 #temp of the letters
T_surrounding = -1.0 #temp of the background (these numbers are good for the heat equation, math math more simple)


u0 = letter_m_ini(x, y, N, T_font, T_surrounding, upscale=4) #setting up initial conditions
u = np.copy(u0)

print("Generated centered initial condition.") #verification

# set up of the matrixes required for the solving each frame
A = np.array(tri_disc(N, k), dtype=np.float32)
C = np.array(tri_disc(N, k), dtype=np.float32)
A_inv = np.linalg.inv(A)
C_inv = np.linalg.inv(C)

# Prepare simulation buffers
u_pred = [np.copy(u)] #initial condition frame
u_star = np.zeros((N, N), dtype=np.float32) #the variable that stores the energy as it evolves
max_iter = 100 #number of itterations kinda like total_time = dt * max_iter

print("Starting iteration session...")


for it in range(max_iter-1): #max_iter iterations

    u_padded = np.pad(u, 1, mode='edge')# really makes sure no energy leaves or enters

    # get the temps of the surroundings all the way to the boarders
    S_north = u_padded[0:-2, 1:-1]
    S_south = u_padded[2:, 1:-1]
    S_west  = u_padded[1:-1, 0:-2]
    S_east  = u_padded[1:-1, 2:]
    S_center = (1 - 4*k) * u

    # Combine each surrounding components temps into S (S = du caused by the surrounding particles per half dt) we do half dt becasue the equations require it (specifically the ADI)
    S = k * (S_north + S_south + S_west + S_east) + S_center

    # Implicitly solve along x for each fixed y, second partion of u with respect to x, from equation ut =uxx +uyy
    u_star[1:-1] = np.dot(S[1:-1], A_inv.T)

    # ensuring du/dt=0 at the corners and edges (essentially doing that one guy whos name starts with an N's boundry condition)
    u_star[0,0]     = k*(u[1,0]-2*u[0,0]+u[0,1])+u[0,0]
    u_star[0,-1]    = k*(u[1,-1]-2*u[0,-1]+u[0,-2])+u[0,-1]
    u_star[-1,0]    = k*(u[-2,0]-2*u[-1,0]+u[-1,1])+u[-1,0]
    u_star[-1,-1]   = k*(u[-2,-1]-2*u[-1,-1]+u[-1,-2])+u[-1,-1]
    u_star[0,1:-1]  = k*(-3*u[0,1:-1]+u[1,1:-1]+u[0,:-2]+u[0,2:])+u[0,1:-1]
    u_star[-1,1:-1] = k*(-3*u[-1,1:-1]+u[-2,1:-1]+u[-1,:-2]+u[-1,2:])+u[-1,1:-1]

    # Solve uyy for the ut equation
    u = np.dot(C_inv, u_star.T).T

    # add frame
    u_pred.append(np.copy(u))

print("Simulation complete. Creating animation...")


fig, ax = plt.subplots(figsize=(8, 8), dpi=250) #create figure (dpi = dots per inch [I think could be very wrong, pretty sure its something per inch])

# Display first frame
im = ax.imshow(
    u_pred[0],
    interpolation='bicubic',   # smooths out appearance
    cmap='nipy_spectral',#this is the color (jet is good)
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='equal'
)

im.set_clim(T_surrounding, T_font) #set temp colors
fig.colorbar(im, ax=ax, label='Temperature')
ax.axis('off') #remove axis, not relevant for this simulation
title = ax.set_title(f"Time: 0.0000 (Iteration 0)") #show time t


def animate(i):
    im.set_array(u_pred[i]) #update the image
    title.set_text(f"Time: {i*dt:.4f} (Iteration {i})")#update the time
    return im, title

# make the animation now that we have all the frames
anim = FuncAnimation(fig, animate, frames=max_iter, interval=150000, blit=True)

try:
    # save the animation
    anim.save('heat_diffusion.gif', writer='pillow', fps=20, dpi=250) #fps controls speed
    print("Animation saved as heat_diffusion.gif")
except Exception as e:
    print("Error saving GIF:", e)

plt.close(fig)

#part 2
from IPython.display import Image
Image(filename="heat_diffusion.gif")
