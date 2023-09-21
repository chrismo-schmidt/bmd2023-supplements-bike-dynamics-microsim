# -*- coding: utf-8 -*-
'''
Created on Fri Aug 18 09:15:33 2023

Experiments and plots for example scenarios of cyclist social forces with
inverted pendulum bicycles. 

Requires cyclistsocialforce>=1.0.0
Requires pypaperutils (https://github.com/chrismo-schmidt/pypaperutils)
Requires pytrafficutils (https://github.com/chrismo-schmidt/pytrafficutils)

@author: Christoph M. Schmidt
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import matplotlib



from time import time

#own packages
from cyclistsocialforce.vehicle import UnStableBicycle, StableBicycle
from cyclistsocialforce.intersection import SocialForceIntersection
from pypaperutils.design import TUDcolors, figure_for_latex
from pypaperutils.measure import draw_distance_to_line, distance_to_line
from pypaperutils.io import export_to_pgf
from pytrafficutils.ssm import pet

#get TUD colors
tudcolors = TUDcolors()
red = tudcolors.get('rood')
cyan = tudcolors.get('cyaan')
black = 'black'
gray = 'gray'

#output directory
outdir = os.getcwd()

    
def scenario_passing(fig, axes, unstable):
    '''PASSING TEST SCENARIO
    
    Two cyclists are heading towards each other with a minimal lateral offset
    and need to perform a small evasive maneuver to pass each other
    

    Parameters
    ----------
    fig : figure
        Figure for plotting.
    axes : axes
        Axes in fig for plotting.
    unstable : boolen
        Flag indicating if 2D bicycles or inverted pendulum bicycles should be
        simulated. True = inv. pendulum. False = 2D bicycles. 

    Returns
    -------
    None.

    '''    
    
    plt.rcParams['text.usetex'] = False
    
    if unstable:
        bike1 = UnStableBicycle((0, 0.1, 0 , 3, 0, 0), 
                                userId='a', 
                                saveForces=True)
        bike2 = UnStableBicycle((30, -0.1, np.pi, 5, 0, 0), 
                                userId='b', 
                                saveForces=True)
    else:
        bike1 = StableBicycle((0, 0.1, 0 , 3, 0, 0), 
                              userId='a', 
                              saveForces=True)
        bike2 = StableBicycle((30, -0.1, np.pi, 5, 0, 0), 
                              userId='b', 
                              saveForces=True)

    # A single destination for bike 1
    bike1.setDestinations((30, 31, 32, 33),(.1, .1, .1, .1))
    bike2.setDestinations((0, -1, -2, -3),(-0.1, -0.1, -0.1, -0.1))

    # A social force intersection to manage the two bicycles. Deactivate Traci
    # to run the simulation without SUMO. Activate animation for some nice 
    # graphics. 
    intersection = SocialForceIntersection((bike1, bike2), use_traci=False, 
                                           animate=unstable, axes=axes[0])

    fig, fig_bg, ax = init_animation(fig, axes[0])
    
    #save_output
    name='scenario-passing'
    snapshots=np.array((3.))
    
    # Run the simulation
    tsim = (2.5,7)
    run(intersection, fig, fig_bg, tsim, snapshots,name)
    
    if unstable:
        color=cyan
    else:
        color=red

    axes[1].plot(bike1.traj[0][0:int(tsim[1]/bike1.params.ts)], 
                 bike1.traj[1][0:int(tsim[1]/bike1.params.ts)], 
                 color=color, linewidth=1, marker='4', markevery=0.3)
    axes[1].plot(bike2.traj[0][0:int(tsim[1]/bike1.params.ts)], 
                 bike2.traj[1][0:int(tsim[1]/bike1.params.ts)],
                 color=color, linewidth=1, marker='3', markevery=0.4)

        
def scenario_overtaking(fig,axes, unstable):
    '''OVERTAKING TEST SCENARIO
    
    Two cyclists are following each other with with different speeds.
    The rear cyclist needs to perform a small evasive maneuver to pass.
    

    Parameters
    ----------
    fig : figure
        Figure for plotting.
    axes : axes
        Axes in fig for plotting.
    unstable : boolen
        Flag indicating if 2D bicycles or inverted pendulum bicycles should be
        simulated. True = inv. pendulum. False = 2D bicycles. 

    Returns
    -------
    None.

    ''' 
    
    
    plt.rcParams['text.usetex'] = False
    
    if unstable:
        bike1 = UnStableBicycle((-5, .1, 0, 3, 0, 0), 
                                userId='a', 
                                saveForces=True)
        bike1.params.v0 = 3
        bike2 = UnStableBicycle((-20, 0, 0, 6, 0, 0), 
                                userId='b', 
                                saveForces=True)
        bike2.params.v0 = 6
    else:
        bike1 = StableBicycle((-5, 0.1, 0, 3, 0, 0), 
                              userId='a', 
                              saveForces=True)
        bike1.params.v0 = 3
        bike2 = StableBicycle((-20, 0, 0, 6, 0, 0), 
                              userId='b', 
                              saveForces=True)
        bike2.params.v0 = 6
        

    # A single destination for bike 1
    bike1.setDestinations(( 30, 49, 50),(.1, .1, .1))
    bike2.setDestinations(( 30, 49, 50),(0,0,0))

    # A social force intersection to manage the two bicycles. Deactivate Traci
    # to run the simulation without SUMO. Activate animation for some nice 
    # graphics. 
    intersection = SocialForceIntersection((bike1, bike2), use_traci=False, 
                                           animate=unstable, axes=axes[0])
    fig, fig_bg, ax = init_animation(fig, axes[0])
    
    #save_output
    name='scenario-overtaking'
    snapshots=np.array((6.))
    
    # Run the simulation
    tsim = (6,12)
    run(intersection, fig, fig_bg, tsim, snapshots, name)
    
    if unstable:
        color=cyan
    else:
        color=red
        
    axes[1].plot(bike1.traj[0][0:int(tsim[1]/bike1.params.ts)], 
                 bike1.traj[1][0:int(tsim[1]/bike1.params.ts)],
                 color=color, linewidth=1, marker='4', markevery=0.4)
    axes[1].plot(bike2.traj[0][0:int(tsim[1]/bike1.params.ts)], 
                 bike2.traj[1][0:int(tsim[1]/bike1.params.ts)],
                 color=color, linewidth=1, marker='4', markevery=0.3)

        
def scenario_crossing(fig,axes, unstable):
    '''CROSSING TEST SCENARIO
    
    Two cyclists are riding in y-direction, another cyclist is riding in 
    x direction. Their indended paths cross and they have to perform evasive
    maneuvers to prevent collision. 
    
    Calculates the PET between cyclists with encroaching trajectories. 
    

    Parameters
    ----------
    fig : figure
        Figure for plotting.
    axes : axes
        Axes in fig for plotting.
    unstable : boolen
        Flag indicating if 2D bicycles or inverted pendulum bicycles should be
        simulated. True = inv. pendulum. False = 2D bicycles. 

    Returns
    -------
    None.

    '''
    
    plt.rcParams['text.usetex'] = False
    
    if unstable:
        bike1 = UnStableBicycle((-23+17, 0, 0, 5, 0, 0), 
                                userId='a', 
                                saveForces=True)
        bike1.params.v0 = 4.5
        bike2 = UnStableBicycle((0+15, -20, np.pi/2, 5, 0, 0), 
                                userId='b', 
                                saveForces=True)
        bike2.params.v0 = 5
        bike3 = UnStableBicycle((-2+15, -20, np.pi/2, 5, 0, 0), 
                                userId='c', 
                                saveForces=True)
        bike3.params.v0 = 5
    else:
        bike1 = StableBicycle((-23+17, 0, 0, 5, 0, 0), 
                              userId='a', 
                              saveForces=True)
        bike1.params.v0 = 4.5
        bike2 = StableBicycle((0+15, -20, np.pi/2, 5, 0, 0), 
                              userId='b', 
                              saveForces=True)
        bike2.params.v0 = 5
        bike3 = StableBicycle((-2+15, -20, np.pi/2, 5, 0, 0), 
                              userId='c', 
                              saveForces=True)
        bike3.params.v0 = 5

    # A single destination for bike 1
    bike1.setDestinations((35, 64, 65),(0,0,0))
    bike2.setDestinations((15,15,15),( 20, 49, 50))
    bike3.setDestinations((13,13,13),( 20, 49, 50))

    # A social force intersection to manage the two bicycles. Deactivate Traci
    # to run the simulation without SUMO. Activate animation for some nice 
    # graphics. 
    intersection = SocialForceIntersection((bike1, bike2,bike3), 
                                           use_traci=False, 
                                           animate=unstable, 
                                           axes=axes[0])
    fig, fig_bg, ax = init_animation(fig, axes[0])
    
    # Run the simulation
    tsim = (4.8,12) 
       
    #save_output
    name='scenario-cossing'
    snapshots=np.array((6.))
    
    run(intersection, fig, fig_bg, tsim, snapshots, name)
    
    if unstable:
        color=cyan
        label='inv. pendulum'
    else:
        color=red
        label='2D model'

    T = int(tsim[1]/bike1.params.ts)
    
    axes[1].plot(bike1.traj[0,0:T], bike1.traj[1,0:T], 
                 color=color, linewidth=1,marker='4',markevery=0.4, 
                 label=label)
    axes[1].plot(bike2.traj[0,0:T], bike2.traj[1,0:T],
                 color=color, linewidth=1, marker='2',markevery=0.4)
    axes[1].plot(bike3.traj[0,0:T], bike3.traj[1,0:T],
                 color=color, linewidth=1, marker='2',markevery=0.3)
    
    #Calculate the minimum distance in space-time
    d12 = np.sqrt(np.sum((bike1.traj[0:2,0:T]-bike2.traj[0:2,0:T])**2,0))
    d13 = np.sqrt(np.sum((bike1.traj[0:2,0:T]-bike3.traj[0:2,0:T])**2,0)) 
    d23 = np.sqrt(np.sum((bike2.traj[0:2,0:T]-bike3.traj[0:2,0:T])**2,0)) 
    
    imin12 = np.argmin(d12)
    imin13 = np.argmin(d13)
    imin23 = np.argmin(d23)
    
    print('Minimum distances:')
    print(f'b1-b2: {d12[imin12]:.2f} m')
    print(f'b1-b3: {d13[imin13]:.2f} m')
    print(f'b2-b3: {d23[imin23]:.2f} m')

    #Calculate the post-encroachment time
    pet12, p121, p122 = pet(bike1.traj[0:2,0:int(tsim[1]/bike1.params.ts)],
                      bike2.traj[0:2,0:int(tsim[1]/bike2.params.ts)],
                      bike1.params.ts)
    pet13, p131, p132 = pet(bike1.traj[0:2,0:int(tsim[1]/bike1.params.ts)],
                      bike3.traj[0:2,0:int(tsim[1]/bike3.params.ts)],
                       bike1.params.ts)
    pet23, p231, p232 = pet(bike2.traj[0:2,0:int(tsim[1]/bike2.params.ts)],
                      bike3.traj[0:2,0:int(tsim[1]/bike3.params.ts)],
                       bike1.params.ts)
    
    print('PET:')
    print(f'b1-b2: {pet12:.2f} s')
    print(f'b1-b3: {pet13:.2f} s')
    print(f'b2-b3: {pet23:.2f} s')
    
    return bike1.traj[1,0:T], np.arange(0,tsim[1], bike1.params.ts)        


def plot_lateral_deviation(dev1, dev2, t, save=True):
    '''Plot the lateral deviation of two road users from straight line travel
    in x-direction. 
    
    Parameters
    ----------
    dev1 : array
        Lateral deviation of the first road user
    dev2 : array
        Lateral deviation of the second road user
    t : array
        Time stamps corresponding to the lateral deviations. 
    save : boolean, optional
        Flag indicating if the plot should be written to pfg or not. 
        
    Returns
    -------
    figure
        Figure of the plot
        
    '''
    
    config_matplotlib_for_latex(save=save)
    
    fig = figure_for_latex(3, width=12)
    ax2 = fig.add_subplot(1,1,1)

    ax2.plot(t, dev1, color=cyan, label = 'inv. pendulum')
    ax2.plot(t, dev2, color=red, label = '2D model')
    
    ip0 = np.argmin(dev1)
    ip1 = np.argmin(dev2)
    p0 = (t[ip0], dev1[ip0])
    p1 = (t[ip1], dev2[ip1])
    p2 = (t[ip1]+1, dev2[ip1])
    
    ax2.set_xlabel(r'$t$  [s]')
    ax2.set_ylabel('deviation  [m]')
    draw_distance_to_line(ax2, p0, p1, p2, -2)
    ax2.plot((4.5,t[ip1]),(dev2[ip1], dev2[ip1]), linestyle='--', color=black, linewidth=.5)
    ax2.text(2,-1.1, f'{dev1[ip0]-dev2[ip1]:.2f} m')
    
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    
    export_to_pgf(fig, 'crossing_deviation', dirname=outdir, save=save)
    
    return fig

def scenario_stepresponse(save=False):
    '''STEP RESPONSE TEST SCENARIO
    
    Test a single bicycle experiencing a sudden change in the desired
    destination, equalling the step response of the dynamic model. This only 
    tests the dynamic model and path planning, not thesocial force.
    
    Compares the step response for a 2D and an inverted pendulum bicycle. 
    
    Parameters
    ----------
    save : boolean, optional
        Flag indicating if the plot should be written saved or not. 
    '''
        
    tsim = (4.,20.,1.)    
    bike1 = UnStableBicycle((1, 3, 0 , 5, 0, 0), userId='b1', saveForces=True)
    bike2 = StableBicycle((1, 3, 0 , 5, 0, 0), userId='b1', saveForces=True)

    bike1.params.v0 = 5
    bike2.params.v0 = 5
    
    bike2.setDestinations([1000,],[1000,])
     
    Fx = 5.
    Fy = 0.
    bike1.force = (Fx,Fy)
    bike2.force = (Fx,Fy)

    phi_d = np.zeros(int(tsim[1]/bike1.params.ts))
    v_d = np.zeros(int(tsim[1]/bike1.params.ts))
    dstate = np.zeros(int(tsim[1]/bike1.params.ts))
    xideal = np.zeros_like(phi_d)
    yideal = np.zeros_like(phi_d)
    xidealprev = 1
    yidealprev = 3

    for i in range(0,int(tsim[1]/bike1.params.ts)):
        phi_d[i] = np.arctan2(Fy,Fx)
        v_d[i] = np.sqrt(Fx**2+Fy**2)
        dstate[i] = 1*bike1.znav[0]+2*bike1.znav[1]+3*bike1.znav[2]
        
        if i*bike1.params.ts == tsim[2]:
            Fx = 5*(5./np.sqrt(3**2+5**2))
            Fy = 5*(3./np.sqrt(3**2+5**2))
            bike1.force = (Fx,Fy)
            bike2.force = (Fx,Fy)

        bike1.step(Fx, Fy)
        bike2.step(Fx, Fy)
        
        xideal[i] = xidealprev+Fx*bike1.params.ts
        yideal[i] = yidealprev+Fy*bike1.params.ts
        xidealprev = xideal[i]
        yidealprev = yideal[i]
    
    psi = np.zeros((int(tsim[1]/bike1.params.ts,)))
    psi[int(tsim[2]/bike1.params.ts):] = np.arctan2(Fy, Fx)
        
    make_step_response_figure(bike1, bike2, (xideal, yideal), psi, tsim[1], 
                              save=save)
        
def make_step_response_figure(bu, bs, trajideal, psi, tsim, save=False):
    '''Plot for the STEP RESPONSE TEST SCENARIO
    
    Plots the results of the STEP RESPONSE TEST SCENARIO. 
    
    Parameters
    ----------
    bu : bike
        Inverted pendulum bicycle object. 
    bs : bike
        2D bicycle object
    trajideal : array
        Desired direction step input.
    psi : array
        Force direction corresponding o the desired direction step input.
    tsim : float
        Duration of the simulaiton
    save : boolean, optional
        Flag indicating if the plot should be written saved or not. 
    '''
    
    if save:
        matplotlib.use('pgf')
    else:
        matplotlib.use('Qt5Agg')

    fig = figure_for_latex(9)

    gs0 = fig.add_gridspec(3, 2)

    #main trajectory plot
    ax1 = fig.add_subplot(gs0[:, 0])
    ax1.set_xlim(0,50)
    ax1.set_ylim(0,50)
    ax1.set_aspect('equal')
    ax1.set_xlabel(r'$x$  [m]')
    ax1.set_ylabel(r'$y$  [m]')
    ax1.set_title('Trajectories')

    #inset zoomed on the moment of the step
    ax10 = zoomed_inset_axes(ax1, zoom=10, loc='upper center', 
                             bbox_to_anchor=(190, 310))
    ax10.set_xlim(5.5,9)
    ax10.set_ylim(2.5,4)
    ax10.set_aspect('equal')
    ax10.set_title('Countersteer')
    

    #bicycle state plots
    ax2 = fig.add_subplot(gs0[0, 1])
    ax3 = fig.add_subplot(gs0[1, 1], sharex=ax2)
    ax4 = fig.add_subplot(gs0[2, 1])

    ax2.set_xlim(1,10)
    ax2.set_xticks([0,5,10,15])
    ax2.set_xticklabels([])
    ax2.set_ylabel(r'yaw $\psi_{\rm{a}}$ in \textdegree')
    ax2.set_title('Bicycle States')

    ax3.set_ylabel(r'steer $\delta_{\mathrm{a}}$ in \textdegree')
    ax3.set_yticks([-15,0,15])

    ax4.set_ylabel(r'lean $\theta_{\mathrm{a}}$ in \textdegree')
    ax4.set_xlim(1,10)
    ax4.set_xticks([0,5,10,15])
    ax4.set_yticks([0,5,10,15])
    ax4.set_xlabel(r'time $t$  [s]')
    
    #distance measurements
    idmin = np.argmin(bu.traj[1][0:int(tsim/bu.params.ts)])
    ydmin = bu.traj[1][idmin]
    xdmin = bu.traj[0][idmin]
    
    t = np.arange(0,tsim,bs.params.ts)
    ax1.plot(trajideal[0], trajideal[1], 
             color=gray, linewidth=1, label='desired')
    ax1.plot(bs.traj[0][0:int(tsim/bs.params.ts)], 
             bs.traj[1][0:int(tsim/bs.params.ts)],
             color=red, linewidth=1.5,label='2D model')
    ax1.plot(bu.traj[0][0:int(tsim/bu.params.ts)], 
             bu.traj[1][0:int(tsim/bu.params.ts)],
             color=cyan, linewidth=1.5,label='inv. pendulum')
    
    dmax = 0
    idmax= 0
    for i in range(int(2/bu.params.ts), int(tsim/bu.params.ts)):
        d, pc = distance_to_line((bu.traj[0][i],bu.traj[1][i]), 
                                    (trajideal[0][int(3/bu.params.ts)],
                                     trajideal[1][int(3/bu.params.ts)]), 
                                    (trajideal[0][int(15/bu.params.ts)],
                                     trajideal[1][int(15/bu.params.ts)]))
        if d > dmax:
            dmax = d
            idmax = i
            
    dmax2, pc = draw_distance_to_line(ax1,
                                      (bu.traj[0][idmax], bu.traj[1][idmax]), 
                                      (trajideal[0][int(3/bu.params.ts)],
                                       trajideal[1][int(3/bu.params.ts)]), 
                                      (trajideal[0][int(15/bu.params.ts)],
                                       trajideal[1][int(15/bu.params.ts)]),
                                      20, width=1)
    ax1.text(33,16, f'{dmax:.2f} m')   
    ax1.plot((31,32.8),(16.3,16.5), linewidth=.5,  color=black)
    
    
    ax10.plot((4,8), (3,3), color=black, linestyle='--', linewidth = .5)
    ax10.plot((8.1,8.5), (2.9,3.1), color=black, linewidth = .5)
    ax10.plot(trajideal[0], trajideal[1], 
              color=gray, linewidth=1, label='desired')        
    ax10.plot(bs.traj[0][0:int(tsim/bs.params.ts)], 
              bs.traj[1][0:int(tsim/bs.params.ts)], 
              color=red, linewidth=2,label='2D model')
    ax10.plot(bu.traj[0][0:int(tsim/bu.params.ts)], 
              bu.traj[1][0:int(tsim/bu.params.ts)],
              color=cyan, linewidth=2,label='inv. pendulum')
    dmax_cs, pc = draw_distance_to_line(ax10, (xdmin,ydmin), 
                                        (1,3), (10,3), 0, width=0.1)
    ax10.text(8.1,3.2, f'{dmax_cs:.2f} m')
    ax1.legend(loc='lower right')

    ax2.plot(t, (360/(2*np.pi)) * psi, 
             color=gray,label='2D model', linewidth=1)
    ax2.plot(t, (360/(2*np.pi)) * bu.traj[2][0:int(tsim/bu.params.ts)], 
             color=cyan)
    ax2.plot(t, (360/(2*np.pi)) * bs.traj[2][0:int(tsim/bu.params.ts)], 
             color=red)
    ax3.plot(t, (360/(2*np.pi)) * bu.traj[4][0:int(tsim/bu.params.ts)], 
             color=cyan)
    ax3.plot(t, (360/(2*np.pi)) * bs.traj[4][0:int(tsim/bu.params.ts)], 
             color=red)
    ax4.plot(t, (360/(2*np.pi)) * bu.traj[5][0:int(tsim/bu.params.ts)], 
             color=cyan)

    pp, p1, p2 = mark_inset(ax1, ax10, loc1=3, loc2=4, linewidth=0.5, zorder=3)
    p2.remove()
    
    export_to_pgf(fig, 'step_response', dirname=outdir, save=save)
        
def scenario_parcours(fig, axes, unstable):
    '''PARCOURS TEST SCENARIO
    
    Test a single bicycle experiencing riding a parcours of intermediate
    destinations offset laterally from straight travel to form a sinusodial 
    shape. 
    
    Parameters
    ----------
    fig : figure
        Figure for plotting.
    axes : axes
        Axes in fig for plotting.
    unstable : boolen
        Flag indicating if 2D bicycles or inverted pendulum bicycles should be
        simulated. True = inv. pendulum. False = 2D bicycles. 

    Returns
    -------
    None.

    '''   
    plt.rcParams['text.usetex'] = False
    
    if unstable:
        bike1 = UnStableBicycle((1-5, 0, 0 , 6, 0, 0), 
                                userId='a', saveForces=True)
    else:
        bike1 = StableBicycle((1-5, 0, 0 , 6, 0, 0), 
                              userId='a', saveForces=True)

    # Destinations
    x = np.array((5,15,25,35,45,55,56))-5#(5,10,15,20,25)
    y = np.array((8,5,10,5,8,8,8))-8#(8,7,9,8,0)
    stop = (0, 0, 0, 0, 0, 1,0)
    bike1.setDestinations(x,y,stop)

    # A social force intersection to manage the two bicycles. Deactivate Traci
    # to run the simulation without SUMO. Activate animation for some nice 
    # graphics. 
    
    intersection = SocialForceIntersection((bike1,), use_traci=False, 
                                           animate=unstable, axes=axes[0])

    fig, fig_bg, ax = init_animation(fig, axes[0])
    
    #save_output
    name='scenario-parcours'
    snapshots=np.array((3.))
    
    # Run the simulation
    tsim = (2.5,7)
    run(intersection, fig, fig_bg, tsim, snapshots,name)
    
    
    if unstable:
        color=cyan
    else:
        color=red

    axes[1].plot(bike1.traj[0][0:int(tsim[1]/bike1.params.ts)], 
                 bike1.traj[1][0:int(tsim[1]/bike1.params.ts)],
                 color=color, linewidth=1, marker='4',markevery=0.4)
    

def run(intersection, fig, fig_bg, tsim, snapshots=(-1,), 
        savename='test-scenario'):
    '''Run the simulation for the test scenarios.
    
    Replaces the use of the Scenario class in cyclistsocialforce, which only
    works toghether with SUMO.
    
    Uses blitting for faster animation. 
    (https://matplotlib.org/stable/users/explain/animations/blitting.html)
    
    Parameters
    ----------
    intersection : intersection
        Intersection object to be simulated.
    fig : figure
        Figure for animation.
    fig_bg : array
        Static elements of the figure for blitting. 
    tsim : array
        Duration of the simulation in s, where tsim[0] is the duration of the 
        animation and tsim[1] the total duration of the simulaiton.
        tsim[0] !< tsim[1].
    '''
    ts = 0.01
    imax = int(tsim[0]/ts)       

    i = 0
    
    if intersection.animate:
        while i < imax:       
            fig.canvas.restore_region(fig_bg)        
            intersection.step()  
        
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()
         
            i = i + 1
            
    intersection.endAnimation()
    
    imax = int(tsim[1]/ts)
    while i < imax:
        intersection.step()  
        i = i + 1

def init_animation(fig, ax):
    ''' Initialize the animation of the demo. 
    
    Replaces the use of the Scenario class in cyclistsocialforce, which only
    works toghether with SUMO.
    
    Uses blitting for faster animation. 
    (https://matplotlib.org/stable/users/explain/animations/blitting.html)
    
    Parameters
    ----------
    fig : figure
        Figure for animation.
    ax : axes
        Axes in fig for animation. 

    Returns
    -------
    fig : figure handle
        The figure the animation will be shown in 
    fig_bg : image
        The background of the figure for blitting 
    ax : axes object
        The axes the animation will be shown in 

    '''
    plt.sca(ax)
    ax.set_aspect('equal')#, adjustable='box')
    #figManager = plt.get_current_fig_manager()
    #figManager.resize(500, 500)
    plt.show(block=False)
    plt.pause(.1)
    fig_bg = fig.canvas.copy_from_bbox(fig.bbox)
    fig.canvas.blit(fig.bbox)
    
    return fig, fig_bg, ax

def make_test_scenario_figure():
    '''Prepare a figure with 6 axes objects to display the results of the 
    test scenarios. 
    '''

    fig = figure_for_latex(9)
    
    heights = np.array((7, 4, 4, 11))
    
    gs = fig.add_gridspec(4, 2, height_ratios=heights/np.sum(heights))
    layout=((0,0),(0,1),(1,0),(1,1),(2,0),(2,1),(3,0),(3,1))
    i=0
    ax=None
    for g in gs:
        ax = fig.add_subplot(g)
        if layout[i][1]==1:
            if layout[i][0]==2:
                ax.text(-4.5,-2,r'$y$  [m]', rotation=90, fontsize=8 )
            #ax.set_yticklabels([])
            ax.yaxis.set_label_position('right')
        
        if layout[i][0]==0:
            ax.set_xticklabels([])
            ax.set_ylim(-3.5,3.5)
            ax.set_yticks((-2, 0, 2))
        elif layout[i][0]<3:    
            ax.set_xticklabels([])
            ax.set_ylim(-2,2)
            ax.set_yticks((-2,0,2))
        else:
            ax.set_ylim(-3,8)
            ax.set_yticks((-2,0,2,4,6,8))
            ax.set_xlabel(r'$x$  [m]')
        ax.set_xlim(0,30)   
        ax.set_aspect('equal')
        
        i+=1
    #fig.supxlabel(r'$x$  [m]')
    
    fig.supylabel(r'$y$  [m]')
    
    return fig, fig.axes

def config_matplotlib_for_latex(save=True):
    """ Presets of matplotlib for beautiul latex plots
    
    Parameters
    ----------
    save : boolean, optional
        Flag indicating if 2D bicycles or inverted pendulum bicycles should be
        simulated. True = inv. pendulum. False = 2D bicycles. 
    
    """
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
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.labelsize':8})

def main():
    """ Main function running the Test scenarios
    """
    
    #set to true to save output
    save = False
    
    #selectors
    run_step_response = True
    run_interaction_tests = True
  
    #disable to see results of the other
    if run_step_response:
        config_matplotlib_for_latex(save)
        scenario_stepresponse(save=save)
    
    #disable to see results of the other
    if run_interaction_tests:
        config_matplotlib_for_latex(False)
        fig, axes = make_test_scenario_figure()
        scenario_parcours(fig, axes[0:2], True)
        scenario_parcours(fig, axes[0:2], False)
        scenario_passing(fig, axes[2:4], True)
        scenario_passing(fig, axes[2:4], False)
        scenario_overtaking(fig, axes[4:6], True)
        scenario_overtaking(fig, axes[4:6], False)
        dev1,t = scenario_crossing(fig, axes[6:8], True)
        dev2,t = scenario_crossing(fig, axes[6:8], False)

        plt.figure(fig.number)
        axes[6].plot((-10,-11), (-10,-11), color=gray, linewidth=1, linestyle='dashed',label= 'planned path')
        axes[6].plot((-10,-11), (-10,-11), color=cyan, linewidth=1,label= 'trajectory')
        axes[6].legend(loc='upper right', fontsize=8)
        axes[7].legend(loc='upper right', fontsize=8)
        axes[1].set_ylabel('Parcours')
        axes[3].set_ylabel('Passing')
        axes[5].set_ylabel('Overtaking')
        axes[7].set_ylabel('Encroaching')
        axes[0].set_title('Simulation snapshots')
        axes[1].set_title('Simulated trajectories')

        if save:
            plt.figure(fig.number)
            plt.savefig('test-scenarios.pdf', dpi=300, format='pdf')
                   
        fig_crossing = plot_lateral_deviation(dev1, dev2, t, save=save)
   
# Entry point
if __name__ == '__main__':
    main()
    