# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 17:28:59 2023

Plots the figure "Stability limits"

@author: christophschmi
"""

import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import os

from cyclistsocialforce.vehicle import UnStableBicycleParameters

TUDcolors = ((0./255, 166./255, 214./255),
             # (12./255, 35./255, 64./255),
             (0./255, 184./255, 200./255),
             (0./255, 118./255, 194./255), 
             (111./255, 29./255, 119./255),
             (239./255, 96./255, 163./255),
             (165./255, 0./255, 52./255) ,
             (224./255, 60./255, 49./255),
             (237./255, 104./255, 66./255),
             (255, 184, 28), 
             (108./255, 194./255, 74./255),
             (0./255, 155./255, 119./255))

def unstableSpeed_inner(k0,k1, c, g, a, b):   
    x = k0
    y = c*g*(a+b)
    z = c*g*(a+b)*k1
    
    v1 = (-y+np.sqrt(y**2-4*x*z))/(2*x)
    v2 = (-y-np.sqrt(y**2-4*x*z))/(2*x)   
    return v1, v2

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

def figure_for_latex(height, width=18.47987, num=1):
    cm = 1/2.54
    return plt.figure(layout='constrained', num=num,figsize=(width*cm, height*cm))
    
def export_to_pgf(fig, filename, dirname=None, save=True): 
    if save:
        if dirname is not None:
            path = os.path.join(filename, dirname) + '.pgf'
            if not os.path.exists(dirname):
                os.makedirs()  
        else:
            path = filename + '.pgf'
        fig.savefig(path)  

params = UnStableBicycleParameters()

#Stabilty limits inner loop 
v = np.linspace(0.1,10,100)

#derived parameters
tau2 = params.l2/v
K = (v**2)/(params.g*(params.l1+params.l2))


def main():
    
    save = False
    
    config_matplotlib_for_latex(save)
    
    v = np.linspace(0.1,10,100)
    
    tau1_squared = (params.ixx+params.m*params.h**2)/(params.m*params.g*params.h)
    tau2 = params.l2/v
    tau3 = (params.l1+params.l2)/v
    K = (v**2)/(params.g*(params.l1+params.l2))
    I = 0.07
    k0 = -600
    k1 = 0.2
    c=50
    
    v1, v2 = unstableSpeed_inner(k0,k1, c, params.g, params.l1, params.l2)
    Kd = k0/(v+k1)
    
    fig = figure_for_latex(4, width = 9, num=1)
    
    plt.plot(v, -I/(tau2*K), color = "k", label=r"upper limit (I)")
    plt.text(0.3, -120, "(I)")
    plt.text(2.2, -190, "(II)")
    plt.text(4, -250, r"$K_D(v)$", color=TUDcolors[0])
    plt.text(1.2, -900, r"$v_\mathrm{min}\approx0.98~\frac{\mathrm{m}}{\mathrm{s}}$", color=TUDcolors[6])
    plt.plot(v, -c/(K), color = "k", label=r"upper limit (II)")
    plt.plot(v, Kd, color=TUDcolors[0], label=r"K_D(v)")
    plt.plot((v2, v2), (-1000,100), color=TUDcolors[6], linestyle='dashed')
    plt.xlabel(r"$v$ $\textstyle\left[\frac{\mathrm{m}}{\mathrm{s}}\right]$")
    plt.ylabel(r"$K_D$")
    plt.xlim(0,10)
    plt.ylim(-1000,100)

    plt.show()
    
    fig2 = figure_for_latex(4, num=2)  
    axes = fig2.subplots(1,2, sharex=True)
    
    Kimax = (c*tau3+K*Kd*tau3)/(tau1_squared*Kd)
    Ki = 0.2*v2*(-1/v+1/v2)
    axes[0].plot(v,Kimax, color="k")
    axes[0].plot(v, np.zeros_like(v), color="k")
    axes[0].plot(v, Ki, color=TUDcolors[0])
    axes[0].plot((v2, v2), (-1000,100), color=TUDcolors[6], linestyle='dashed')
    #for kp in np.arange(0.29,0.309,0.005):
    #    axes[0].plot(v, c/I * kp + 1/(tau1_squared*params.g)*v - (c*params.l2)/(tau1_squared**2*params.g), color="k")
    axes[0].set_ylim(-2,2)
    axes[0].set_xlim(0,10)
    axes[0].set_xlabel(r"$v$ $\textstyle\left[\frac{\mathrm{m}}{\mathrm{s}}\right]$")
    axes[0].set_ylabel(r"$K_I$")
    axes[0].text(2.4,1,"(IV)")
    #axes[0].text(5.2,-2,r"(V), $K_P=0.305$", fontsize = 8)
    #axes[0].text(6.8,-4.1,r"(V), $K_P=0.300$" , fontsize = 8)
    axes[0].text(3,-0.5,r"(III)")
    axes[0].text(5,0.5,r"$K_I(v)$", color=TUDcolors[0])
    axes[0].text(1.2, -1.5, r"$v_\mathrm{min}\approx0.98~\frac{\mathrm{m}}{\mathrm{s}}$", color=TUDcolors[6])
     
    Kpmax = (I*tau3+K*Kd*tau3*tau2)/tau1_squared*Kd
    axes[1].plot(v,Kpmax, color="k")
    axes[1].plot(v, np.zeros_like(v), color="k")
    axes[1].plot(v, -I/(tau1_squared*params.g*c)*v+I*params.l2/(tau1_squared**2*params.g)+I/c * Ki, color="k")
    axes[1].plot(v, np.ones_like(v)*0.25, color=TUDcolors[0])
    axes[1].plot((v2, v2), (-1000,100), color=TUDcolors[6], linestyle='dashed')
    axes[1].set_xlim(0,10)
    axes[1].set_ylim(-0.05,0.4)
    axes[1].set_xlabel(r"$v$ $\textstyle\left[\frac{\mathrm{m}}{\mathrm{s}}\right]$")
    axes[1].set_ylabel(r"$K_P$")
    axes[1].text(8,0.02,r"(I)")
    axes[1].text(4,0.33,r"(V)")
    axes[1].text(5,0.18,r"$K_P(v)$", color=TUDcolors[0])
    axes[1].text(1.2, 0.1, r"$v_\mathrm{min}\approx0.98~\frac{\mathrm{m}}{\mathrm{s}}$", color=TUDcolors[6])
    
    
    export_to_pgf(fig, "stability-limits-inner-loop", save=save)
    export_to_pgf(fig2, "stability-limits-outer-loop", save=save)
if __name__ == "__main__":
    main()