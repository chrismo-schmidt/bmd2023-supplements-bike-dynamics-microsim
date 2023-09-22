# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:45:28 2023

Plots the figure "Repulsive force field of a cyclist"

@author: christophschmi
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from cyclistsocialforce.vehicle import UnStableBicycle
from pypaperutils.design import figure_for_latex
from pypaperutils.io import export_to_pgf

#output directory
outdir = './figures/'

def config_matplotlib_for_latex(save=True):
    ''' Presets of matplotlib for beautiul latex plots
    
    Parameters
    ----------
    save : boolean, optional
        Flag indicating if 2D bicycles or inverted pendulum bicycles should be
        simulated. True = inv. pendulum. False = 2D bicycles. 
    '''
    
    if save:
        matplotlib.use("pgf")
    else:
        matplotlib.use("Qt5Agg")

    matplotlib.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 8,
        "axes.titlesize": 10,
        "font.size": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8})
    
    plt.close("all")
    
def plot_force_direction(ax, X, Y, Fx, Fy):
    '''Create a quiver plot that shows the direction of the force plots. 
    
    UNUSED.

    Parameters
    ----------
    ax : axes
        Axes to plot in.
    X : TYPE
        X-location where the force field is evaluated.
    Y : TYPE
        Y-location where the force field is evaluated.
    Fx : TYPE
        X-component of the force at the locations X.
    Fy : TYPE
        Y-component of the force at the locations Y.

    Returns
    -------
    None.

    '''
    Fx = Fx[::10,::10]
    Fy = Fy[::10,::10]
    X = X[::10,::10]
    Y = Y[::10,::10]
    
    Fx = 2*Fx/(np.sqrt(Fx**2+Fy**2))
    Fy = 2*Fy/(np.sqrt(Fx**2+Fy**2))
    
    ax.quiver(X,Y, Fx, Fy, scale=50)

def plot_force_magnitude(ax, X, Y, Fx, Fy):
    '''Create a contour plot that shows the magnitude of the force. 

    Parameters
    ----------
    ax : axes
        Axes to plot in.
    X : TYPE
        X-location where the force field is evaluated.
    Y : TYPE
        Y-location where the force field is evaluated.
    Fx : TYPE
        X-component of the force at the locations X.
    Fy : TYPE
        Y-component of the force at the locations Y.

    Returns
    -------
    QuadContourSet
        Filled contour showing the magnitude
    QuadContourSet
        Single contour highlighting where the repulsive force equal

    '''
    F = np.sqrt(Fx**2+Fy**2)  
    c = ax.contourf(X, Y, F, levels=np.arange(0,5.5,1), extend="max")   
    cl = ax.contour(c, levels=np.array((c.levels[-1],)), colors='r')
    return c, cl #clw
    
def main():
    '''
    Plot and export a figure showing the magnitude and directions of 
    the repulsive force fields for different locations and relative 
    orientations.

    Returns
    -------
    None.

    '''
    
    save = False
    
    config_matplotlib_for_latex(save)
    
    b = UnStableBicycle((0,0,0,5,0,0))
    
    lnspx = np.arange(-10, 40, 0.1)
    lnspy = np.arange(-10,10, 0.1)
    X, Y = np.meshgrid(lnspx, lnspy)
    psis = np.array([0,np.pi/4,np.pi/2])
    
    fig = figure_for_latex(4.5)
    axes = fig.subplots(1, 3, sharey=True, sharex=True)
    
    titles = (r"\textbf{Parallell interactions}"+"\n"+
              r"\small{$\psi_{a,b} = 0$, $\psi_{a,b} = \pm \pi$}", \
              r"\textbf{45 \textdegree~interactions}"+"\n"+
              r"\small{$\psi_{a,b} = \pm \frac{1}{4}\pi$"+
              r" $\psi_{a,b} = \pm \frac{3}{4}\pi$}", \
              r"\textbf{Perpendicular interactions}"+"\n"+
              r"\small{$\psi_{a,b} = \pm \frac{1}{2}\pi$}")
    
    for ax, psi, title in zip(axes.flatten(), psis, titles):
        Fx, Fy = b.calcRepulsiveForce(X, Y, psi)
        c,cl = plot_force_magnitude(ax, X, Y, Fx, Fy)
        plot_force_direction(ax, X, Y, Fx, Fy)
        #b.makeBikeDrawing(ax, animated=False)
        ax.set_xlim(-5,15)
        ax.set_ylim(-5,5)
        ax.set_aspect("equal")
        ax.set_title(title, y=1.1)
        
    cbar = fig.colorbar(c, 
                        ax=axes, 
                        shrink=0.6, 
                        location='right',
                        aspect=10,
                        pad=0.01,
                        ticks=[0,1.25,2.5,3.25,5])
    cbar.ax.set_yticklabels(("0", "", r"$\frac{v_d}{2}$", "", r"$v_d$"))
    #cbar.add_lines(clw)
    cbar.add_lines(cl)
    
    fig.supxlabel(r'$x_{a,b}$ [m]', y=0.01, fontsize=8)
    fig.supylabel(r'$y_{a,b}$ [m]', fontsize=8)
    
    export_to_pgf(fig, os.path.join(outdir, "repulsive_force_fields"), 
                  save=save)
    
if __name__ == "__main__":
    main()