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

def config_matplotlib_for_latex(save=True):
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
    Fx = Fx[::10,::10]
    Fy = Fy[::10,::10]
    X = X[::10,::10]
    Y = Y[::10,::10]
    
    Fx = 2*Fx/(np.sqrt(Fx**2+Fy**2))
    Fy = 2*Fy/(np.sqrt(Fx**2+Fy**2))
    
    ax.quiver(X,Y, Fx, Fy, scale=50)

def plot_force_magnitude(ax, X, Y, Fx, Fy):
    F = np.sqrt(Fx**2+Fy**2)  
    c = ax.contourf(X, Y, F, levels=np.arange(0,5.5,1), extend="max")   
    #clw = ax.contour(c, levels=np.array((c.levels)), colors="silver", linewidths=0.2)
    cl = ax.contour(c, levels=np.array((c.levels[-1],)), colors='r')
    return c, cl #clw
    
def figure_for_latex(height, width=18.47987, num=1):
    cm = 1/2.54
    return plt.figure(layout='constrained', num=2,figsize=(width*cm, height*cm))
    
def export_to_pgf(fig, filename, dirname=None, save=True): 
    if save:
        if dirname is not None:
            path = os.path.join(filename, dirname) + '.pgf'
            if not os.path.exists(dirname):
                os.makedirs()  
        else:
            path = filename + '.pgf'
        fig.savefig(path)  
    
def main():
    
    save = True
    
    config_matplotlib_for_latex(save)
    
    b = UnStableBicycle((0,0,0,5,0,0))
    
    lnspx = np.arange(-10, 40, 0.1)
    lnspy = np.arange(-10,10, 0.1)
    X, Y = np.meshgrid(lnspx, lnspy)
    psis = np.array([0,np.pi/4,np.pi/2])
    
    fig = figure_for_latex(4.5)
    axes = fig.subplots(1, 3, sharey=True, sharex=True)
    
    titles = (r"\textbf{Parallell interactions}"+"\n"+r"\small{$\psi_{a,b} = 0$, $\psi_{a,b} = \pm \pi$}", \
              r"\textbf{45 \textdegree~interactions}"+"\n"+r"\small{$\psi_{a,b} = \pm \frac{1}{4}\pi$, $\psi_{a,b} = \pm \frac{3}{4}\pi$}", \
              r"\textbf{Perpendicular interactions}"+"\n"+r"\small{$\psi_{a,b} = \pm \frac{1}{2}\pi$}")
    
    for ax, psi, title in zip(axes.flatten(), psis, titles):
        Fx, Fy = b.calcRepulsiveForce(X, Y, psi)
        c,cl = plot_force_magnitude(ax, X, Y, Fx, Fy)
        plot_force_direction(ax, X, Y, Fx, Fy)
        #b.makeBikeDrawing(ax, animated=False)
        ax.set_xlim(-5,15)
        ax.set_ylim(-5,5)
        ax.set_aspect("equal")
        ax.set_title(title, y=1.1)
        plt.show()
        
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
    
    export_to_pgf(fig, "repulsive_force_fields", save=save)
    
    
    plt.figure(4)
    plot_force_direction(ax, X, Y, Fx, Fy)
    
if __name__ == "__main__":
    main()