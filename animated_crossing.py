# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 11:35:23 2023

Functions init_faces and init_vert taken from https://stackoverflow.com/questions/56901394/animate-matplotlibs-poly3dcollection
by Luca(https://stackoverflow.com/users/4690023/luca) and ImportanceOfBeingErnest(https://stackoverflow.com/users/4124317/importanceofbeingernest)

@author: Christoph M. Schmidt
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from typing import Tuple
from pypaperutils.design import TUDcolors

from cyclistsocialforce.vizualisation import Arrow2D, BicycleDrawing2D
from cyclistsocialforce.vehicle import InvPendulumBicycle, Bicycle
from cyclistsocialforce.intersection import SocialForceIntersection


def calc_full_forcefields(X: np.ndarray, Y:np.ndarray, psi: float, 
                          bikes: Bicycle) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Aggregate the force fileds of the given bicycles in all locations (X,Y)

    Parameters
    ----------
    X : np.ndarray
        X-locations to be evaluated.
    Y : np.ndarray
        Y-locations to be evaluated.
    psi : float
        Ego-orientation for evaluation of the force fields.
    bikes : Bicycle
        Bicycles, who's repulsive forces should be evaluated.

    Returns
    -------
    Fx : np.ndarray
        X-component of the resulting force field.
    Fy : np.ndarray
        Y-component of the resulting force field.

    '''
    
    Fx = np.zeros_like(X, dtype=float)
    Fy = np.zeros_like(X, dtype=float)

    for b in bikes:
        Fxb, Fyb = b.calcRepulsiveForce(X,Y, psi)
        Fx += Fxb
        Fy += Fyb
    return Fx, Fy

def init_faces(N):
    f = []
    for r in range(N-1):
        for c in range(N-1):
            v0 = r*N+c
            f.append([v0, v0+1, v0+N+1, v0+N])
    return np.array(f) 

def init_vert(N):
    v = np.meshgrid(range(N),range(N),[0])
    return np.dstack(v).reshape(-1,3)

def update(frame, pc_2):
    intersection.step()
    intersection.step()
    v_new = []
    for b in intersection.vehicles:
        v_new += drawing.calc_keypoints(b.s)
    pc_1.set_verts(v_new)
    
    Fx, Fy = calc_full_forcefields(X, Y, intersection.vehicles[0].s[2], 
                          (intersection.vehicles[1], intersection.vehicles[2]))
    F = (np.sqrt(Fx**2+Fy**2))
    #v[:,2] = F+5
         
    pc_2[0].remove()
    pc_2[0] = ax.plot_surface(X, Y, F, alpha=0.5 ,cmap='RdPu', vmin=0, vmax=10)
    
    #cmap = plt.get_cmap("jet")
    #v_new = v[f]
    #facecolors = pc_2.get_facecolors()
    #for i in range(len(facecolors)):
    #    facecolors[i] = cmap(int(256*(max(F))/2))
    #facecolors = [cmap(int(256*(v_new[i,0,2])/5)) for i in range(v_new.shape[0])]
    
    #print(len(facecolors))
    #print(len(v[f]))
    
    #pc_2.set_facecolors(facecolors)
    #pc_2.set_verts(v[f]) 
    
    print("was here")
    
    return pc_1, pc_2,


# create actual bicycles for animation
bike1 = InvPendulumBicycle((-23+17, 0, 0, 5, 0, 0), 
                        userId='a', 
                        saveForces=True)
bike1.params.v_desired_default = 4.5
bike2 = InvPendulumBicycle((0+15, -20, np.pi/2, 5, 0, 0), 
                        userId='b', 
                        saveForces=True)
bike2.params.v_desired_default = 5.
bike3 = InvPendulumBicycle((-2+15, -20, np.pi/2, 5, 0, 0), 
                        userId='c', 
                        saveForces=True)
bike3.params.v_desired_default = 5.

bike1.setDestinations((35, 64, 65),(0,0,0))
bike2.setDestinations((15,15,15),( 20, 49, 50))
bike3.setDestinations((13,13,13),( 20, 49, 50))

# Create Intersection object to manage the bikes
intersection = SocialForceIntersection((bike1, bike2,bike3), 
                                       use_traci=False, 
                                       animate=False)

# create dummy bicycle for drawing
dummy = InvPendulumBicycle((0,0,0,5,0,0))
fig2 = plt.figure()
ax2  = fig2.add_subplot(projection="3d")
drawing = BicycleDrawing2D(ax2, dummy, proj_3d=True, animated=False)

# create figure
xlim=(5,20)
ylim=(-10,5)

N = 500

fig = plt.figure(figsize=(15,20))
ax  = fig.add_subplot(projection="3d")
fig.subplots_adjust(top=1.1, bottom=-.1)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_zlim(0,10)
ax.set_aspect('equal', adjustable='box')

# calculate vertices of all three bikes
v_1 = drawing.calc_keypoints(bike1.s)
v_2 = drawing.calc_keypoints(bike2.s)
v_3 = drawing.calc_keypoints(bike3.s)
v_b = v_1+v_2+v_3

#bike colors
colors = TUDcolors()
color_bike = colors.get("cyaan")
color_rider = colors.get("donkerblauw")
color_wheels = 'black'
color_head = colors.get('cyaan')
facecolors = ((color_wheels, color_wheels, color_bike, color_bike, color_rider, color_rider, color_rider, color_head, (0.,0.,0.)))

# create 3D Poly collection
pc_1 = art3d.Poly3DCollection(v_b, facecolors=facecolors+facecolors+facecolors)
ax.add_collection(pc_1)

#calculate force field vertices
x = np.linspace(xlim[0], xlim[1], N)
y = np.linspace(ylim[0], ylim[1], N)
X,Y = np.meshgrid(x,y)
Z = np.zeros((X.size,1))+3
v_f = np.concatenate((X.flatten()[:,np.newaxis], Y.flatten()[:,np.newaxis],Z), axis=1)
v = v_f.astype(float)

#v = init_vert(N)
f = init_faces(N)

#pc_2 = art3d.Poly3DCollection(v[f], alpha = 0.5)
pc_2 = [ax.plot_surface(X, Y, np.zeros_like(X), cmap='magma')]
#ax.add_collection(pc_2[0])

ani = FuncAnimation(fig, update, fargs=(pc_2,), frames=np.linspace(0, 1, 700),
                    blit=False, repeat=True)
#ani.save('animated_crossing_3d.gif', fps=int(1/0.01)) 
