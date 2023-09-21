# -*- coding: utf-8 -*-
'''
Created on Tue Aug 15 17:28:59 2023

Plots the figures on inner and outer stability limits. 

Requires cyclistsocialforce>=1.0.0
Requires pypaperutils (https://github.com/chrismo-schmidt/pypaperutils)

@author: christophschmi
'''

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from cyclistsocialforce.vehicle import UnStableBicycleParameters
from pypaperutils.design import TUDcolors, figure_for_latex
from pypaperutils.io import export_to_pgf

#get TUD colors
tudcolors = TUDcolors()
red = tudcolors.get('rood')
cyan = tudcolors.get('cyaan')
black = 'black'

#output directory
outdir = './figures/'

def unstableSpeed_inner(k0,k1, c, g, a, b):  
    '''Calculate the unstable speed w.r.t the inner loop (p.7 in the paper.)

    Parameters
    ----------
    k0 : float
        k0 component of the adaptive gain.
    k1 : float
        k0 component of the adaptive gain.
    c : float
        Damping coefficient of the steer column. 
    g : float
        Graviational constant.
    a : float
        Distance between front wheel and bicycle center.
    b : TYPE
        Distance between rear wheel and bicycle center.

    Returns
    -------
    v1 : float
        1st solution
    v2 : TYPE
        2nd solution
    '''
    x = k0
    y = c*g*(a+b)
    z = c*g*(a+b)*k1
    
    v1 = (-y+np.sqrt(y**2-4*x*z))/(2*x)
    v2 = (-y-np.sqrt(y**2-4*x*z))/(2*x)   
    return v1, v2

def config_matplotlib_for_latex(save=True):
    ''' Presets of matplotlib for beautiul latex plots
    
    Parameters
    ----------
    save : boolean, optional
        Flag indicating if 2D bicycles or inverted pendulum bicycles should be
        simulated. True = inv. pendulum. False = 2D bicycles. 
    '''
    
    if save:
        matplotlib.use('pgf')
    else:
        matplotlib.use('Qt5Agg')

    matplotlib.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'axes.labelsize': 8,
        'axes.titlesize': 10,
        'font.size': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8})
    
    plt.close('all')

def main():
    '''Create plots of the stability limits of the system using Routh
    '''
    
    save = True
    
    config_matplotlib_for_latex(save)
    
    #model parameters
    v = np.linspace(0.1,10,100)
    
    params = UnStableBicycleParameters()
    tau1_squared = (params.ixx+params.m*params.h**2)/ \
                   (params.m*params.g*params.h)
    tau2 = params.l2/v
    tau3 = (params.l1+params.l2)/v
    K = (v**2)/(params.g*(params.l1+params.l2))
    I = 0.07
    k0 = -600
    k1 = 0.2
    c=50
    
    v1, v2 = unstableSpeed_inner(k0,k1, c, params.g, params.l1, params.l2)
    Kd = k0/(v+k1)
    
    #inner loop
    fig = figure_for_latex(4, width = 9)
    
    plt.plot(v, -I/(tau2*K), color = black, label=r'upper limit (I)')
    plt.text(0.3, -120, '(I)')
    plt.text(2.2, -190, '(II)')
    plt.text(4, -250, r'$K_D(v)$', color=cyan)
    plt.text(1.2, -900, 
             r'$v_\mathrm{min}\approx0.98~\frac{\mathrm{m}}{\mathrm{s}}$', 
             color=red)
    plt.plot(v, -c/(K), color = black, label=r'upper limit (II)')
    plt.plot(v, Kd, color=cyan, label=r'K_D(v)')
    plt.plot((v2, v2), (-1000,100), color=red, linestyle='dashed')
    plt.xlabel(r'$v$ $\textstyle\left[\frac{\mathrm{m}}{\mathrm{s}}\right]$')
    plt.ylabel(r'$K_D$')
    plt.xlim(0,10)
    plt.ylim(-1000,100)
    
    
    #outer loop
    fig2 = figure_for_latex(4)  
    axes = fig2.subplots(1,2, sharex=True)
    
    #Ki
    Kimax = (c*tau3+K*Kd*tau3)/(tau1_squared*Kd)
    Ki = 0.2*v2*(-1/v+1/v2)
    axes[0].plot(v,Kimax, color=black)
    axes[0].plot(v, np.zeros_like(v), color=black)
    axes[0].plot(v, Ki, color=cyan)
    axes[0].plot((v2, v2), (-1000,100), color=red, linestyle='dashed')
    axes[0].set_ylim(-2,2)
    axes[0].set_xlim(0,10)
    axes[0].set_xlabel(
        r'$v$ $\textstyle\left[\frac{\mathrm{m}}{\mathrm{s}}\right]$')
    axes[0].set_ylabel(r'$K_I$')
    axes[0].text(2.4,1,'(IV)')
    axes[0].text(3,-0.5,r'(III)')
    axes[0].text(5,0.5,r'$K_I(v)$', color=cyan)
    axes[0].text(1.2, -1.5, 
                 r'$v_\mathrm{min}\approx0.98~\frac{\mathrm{m}}{\mathrm{s}}$', 
                 color=red)
     
    #Kp
    Kpmax = (I*tau3+K*Kd*tau3*tau2)/tau1_squared*Kd
    axes[1].plot(v,Kpmax, color=black)
    axes[1].plot(v, np.zeros_like(v), color=black)
    axes[1].plot(v, -I/(tau1_squared*params.g*c)*v+
                    I*params.l2/(tau1_squared**2*params.g)+I/c * Ki, 
                color=black)
    axes[1].plot(v, np.ones_like(v)*0.25, color=cyan)
    axes[1].plot((v2, v2), (-1000,100), color=red, linestyle='dashed')
    axes[1].set_xlim(0,10)
    axes[1].set_ylim(-0.05,0.4)
    axes[1].set_xlabel(
        r'$v$ $\textstyle\left[\frac{\mathrm{m}}{\mathrm{s}}\right]$')
    axes[1].set_ylabel(r'$K_P$')
    axes[1].text(8,0.02,r'(I)')
    axes[1].text(4,0.33,r'(V)')
    axes[1].text(5,0.18,r'$K_P(v)$', color=cyan)
    axes[1].text(1.2, 0.1, 
                 r'$v_\mathrm{min}\approx0.98~\frac{\mathrm{m}}{\mathrm{s}}$', 
                 color=red)
    
    #save
    export_to_pgf(fig, os.path.join(outdir,'stability-limits-inner-loop'), 
                  save=save)
    export_to_pgf(fig2, os.path.join(outdir,'stability-limits-outer-loop'), 
                  save=save)
    
if __name__ == '__main__':
    main()