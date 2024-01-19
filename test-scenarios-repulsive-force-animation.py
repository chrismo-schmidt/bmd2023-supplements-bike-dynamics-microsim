# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 09:15:33 2023

test_scenarios.py

Runs the test scenarios and plots Figure 8: "Yaw angle step response at 
constant speed.", Figure 9: "Test scenarios of the cyclist social force model 
with bicycle dynamics.", and Figure 10: "Deviation of cyclist a in the 
encroaching scenario from it’s undisturbed path.", for Schmidt et al. (2023).
Calculates the Post Encroachment times in the encroaching test scenario for 
Table 1.   

Schmidt, C., Dabiri, A., Schulte, F., Happee, R. & Moore, J. (2023). Essential 
Bicycle Dynamics for Microscopic Traffic Simulation: An Example Using the 
Social Force Model [preprint]. The Evolving Scholar - BMD 2023, 5th Edition. 
https://doi.org/10.59490/65037d08763775ba4854da53

Requires cyclistsocialforce>=1.1.0
Requires pypaperutils (https://github.com/chrismo-schmidt/pypaperutils)
Requires pytrafficutils (https://github.com/chrismo-schmidt/pytrafficutils)

Usage: $ python stability_limits.py

@author: Christoph M. Schmidt, TU Delft. 
"""


import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
from time import sleep

# own packages
from cyclistsocialforce.vehicle import InvPendulumBicycle, TwoDBicycle, Bicycle
from cyclistsocialforce.parameters import InvPendulumBicycleParameters
from cyclistsocialforce.intersection import SocialForceIntersection
from cyclistsocialforce.vizualisation import BicycleDrawing2D
from pypaperutils.design import TUDcolors

import cv2 as cv


# get TUD colors
tudcolors = TUDcolors()
red = tudcolors.get("rood")
cyan = tudcolors.get("cyaan")
black = "black"
gray = "gray"

# output directory
outdir = "./figures/"


def scenario_errorcase(fig, axes, unstable):
    """PARCOURS TEST SCENARIO

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

    """
    plt.rcParams["text.usetex"] = False

    name = "error"

    if unstable:
        params1 = InvPendulumBicycleParameters(v_desired_default=3.0)
        params2 = InvPendulumBicycleParameters(v_desired_default=3.0)

        bike1 = InvPendulumBicycle(
            (1 - 5, 0, 0, 6, 0, 0), userId="a", saveForces=True, params=params1
        )

        bike2 = InvPendulumBicycle(
            (18, -24, np.pi / 2, 6, 0, 0),
            userId="b",
            saveForces=True,
            params=params2,
        )

    else:
        bike1 = TwoDBicycle((1 - 5, 0, 0, 6, 0), userId="a", saveForces=True)

    # Destinations
    x = np.array((5, 15, 25, 35, 45, 55, 56)) - 5  # (5,10,15,20,25)
    y = np.array((8, 5, 10, 5, 8, 8, 8)) - 8  # (8,7,9,8,0)
    stop = (0, 0, 0, 0, 0, 0, 0)
    bike1.setDestinations(x, y, stop)

    # Destinations
    y = np.array((5, 15, 25, 35, 45, 55, 56)) - 20  # (5,10,15,20,25)
    x = np.array((8, 5, 10, 5, 8, 8, 8)) + 10  # (8,7,9,8,0)
    stop = (0, 0, 0, 0, 0, 0, 0)
    bike2.setDestinations(x, y, stop)

    # A social force intersection to manage the two bicycles. Deactivate Traci
    # to run the simulation without SUMO. Activate animation for some nice
    # graphics.

    intersection = SocialForceIntersection(
        (bike1, bike2),
        activate_sumo_cosimulation=False,
        animate=unstable,
        axes=axes[0],
    )

    fig, fig_bg, ax = init_animation(fig, axes[0])

    # Run the simulation
    tsim = 14
    run(
        intersection,
        name,
        fig,
        fig_bg,
        tsim,
        bikes_for_forcefield_plot=(1, (0,)),
    )

    axes = bike1.plot_states(t_end=14)
    axes2 = bike2.plot_states(t_end=14)


def scenario_passing(fig, axes, unstable):
    """PASSING TEST SCENARIO

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

    """

    name = "passing"

    plt.rcParams["text.usetex"] = False

    if unstable:
        bike1 = InvPendulumBicycle(
            (0, 0.1, 0, 3, 0, 0), userId="a", saveForces=True
        )
        bike2 = InvPendulumBicycle(
            (30, -0.1, np.pi, 5, 0, 0), userId="b", saveForces=True
        )
    else:
        bike1 = TwoDBicycle((0, 0.1, 0, 3, 0), userId="a", saveForces=True)
        bike2 = TwoDBicycle(
            (30, -0.1, np.pi, 5, 0), userId="b", saveForces=True
        )

    # A single destination for bike 1
    bike1.setDestinations((30, 31, 32, 33), (0.1, 0.1, 0.1, 0.1))
    bike2.setDestinations((0, -1, -2, -3), (-0.1, -0.1, -0.1, -0.1))

    # A social force intersection to manage the two bicycles. Deactivate Traci
    # to run the simulation without SUMO. Activate animation for some nice
    # graphics.
    intersection = SocialForceIntersection(
        (bike1, bike2),
        activate_sumo_cosimulation=False,
        animate=unstable,
        axes=axes[0],
    )

    fig, fig_bg, ax = init_animation(fig, axes[0])

    # Run the simulation
    tsim = 7
    run(
        intersection,
        name,
        fig,
        fig_bg,
        tsim,
        bikes_for_forcefield_plot=(1, (0,)),
    )


def scenario_overtaking(fig, axes, unstable):
    """OVERTAKING TEST SCENARIO

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

    """

    name = "overtaking"

    plt.rcParams["text.usetex"] = False

    if unstable:
        bike1 = InvPendulumBicycle(
            (-5, 0.1, 0, 3, 0, 0), userId="a", saveForces=True
        )
        bike1.params.v_desired_default = 3.0
        bike2 = InvPendulumBicycle(
            (-20, 0, 0, 6, 0, 0), userId="b", saveForces=True
        )
        bike2.params.v_desired_default = 6.0
    else:
        bike1 = TwoDBicycle((-5, 0.1, 0, 3, 0), userId="a", saveForces=True)
        bike1.params.v_desired_default = 3.0
        bike2 = TwoDBicycle((-20, 0, 0, 6, 0), userId="b", saveForces=True)
        bike2.params.v_desired_default = 6.0

    # A single destination for bike 1
    bike1.setDestinations((30, 49, 50), (0.1, 0.1, 0.1))
    bike2.setDestinations((30, 49, 50), (0, 0, 0))

    # A social force intersection to manage the two bicycles. Deactivate Traci
    # to run the simulation without SUMO. Activate animation for some nice
    # graphics.
    intersection = SocialForceIntersection(
        (bike1, bike2),
        activate_sumo_cosimulation=False,
        animate=unstable,
        axes=axes[0],
    )
    fig, fig_bg, ax = init_animation(fig, axes[0])

    # Run the simulation
    tsim = 12
    run(
        intersection,
        name,
        fig,
        fig_bg,
        tsim,
        bikes_for_forcefield_plot=(0, (1,)),
    )


def scenario_crossing(fig, axes, unstable):
    """CROSSING TEST SCENARIO

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

    """

    name = "crossing"

    plt.rcParams["text.usetex"] = False

    if unstable:
        bike1 = InvPendulumBicycle(
            (-23 + 17, 0, 0, 5, 0, 0), userId="a", saveForces=True
        )
        bike1.params.v_desired_default = 4.5
        bike2 = InvPendulumBicycle(
            (0 + 15, -20, np.pi / 2, 5, 0, 0), userId="b", saveForces=True
        )
        bike2.params.v_desired_default = 5.0
        bike3 = InvPendulumBicycle(
            (-2 + 15, -20, np.pi / 2, 5, 0, 0), userId="c", saveForces=True
        )
        bike3.params.v_desired_default = 5.0
    else:
        bike1 = TwoDBicycle(
            (-23 + 17, 0, 0, 5, 0), userId="a", saveForces=True
        )
        bike1.params.v_desired_default = 4.5
        bike2 = TwoDBicycle(
            (0 + 15, -20, np.pi / 2, 5, 0), userId="b", saveForces=True
        )
        bike2.params.v_desired_default = 5.0
        bike3 = TwoDBicycle(
            (-2 + 15, -20, np.pi / 2, 5, 0), userId="c", saveForces=True
        )
        bike3.params.v_desired_default = 5.0

    # A single destination for bike 1
    bike1.setDestinations((35, 64, 65), (0, 0, 0))
    bike2.setDestinations((15, 15, 15), (20, 49, 50))
    bike3.setDestinations((13, 13, 13), (20, 49, 50))

    # A social force intersection to manage the two bicycles. Deactivate Traci
    # to run the simulation without SUMO. Activate animation for some nice
    # graphics.
    intersection = SocialForceIntersection(
        (bike1, bike2, bike3),
        activate_sumo_cosimulation=False,
        animate=unstable,
        axes=axes[0],
    )
    fig, fig_bg, ax = init_animation(fig, axes[0])

    # Run the simulation
    tsim = 9

    run(
        intersection,
        name,
        fig,
        fig_bg,
        tsim,
        bikes_for_forcefield_plot=(2, (0, 1)),
    )


def scenario_parcours(fig, axes, unstable):
    """PARCOURS TEST SCENARIO

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

    """

    name = "parcours"

    plt.rcParams["text.usetex"] = False

    if unstable:
        bike1 = InvPendulumBicycle(
            (1 - 5, 0, 0, 6, 0, 0), userId="a", saveForces=True
        )
    else:
        bike1 = TwoDBicycle((1 - 5, 0, 0, 6, 0), userId="a", saveForces=True)

    # Destinations
    x = np.array((5, 15, 25, 35, 45, 55, 56)) - 5  # (5,10,15,20,25)
    y = np.array((8, 5, 10, 5, 8, 8, 8)) - 8  # (8,7,9,8,0)
    stop = (0, 0, 0, 0, 0, 1, 0)
    bike1.setDestinations(x, y, stop)

    # A social force intersection to manage the two bicycles. Deactivate Traci
    # to run the simulation without SUMO. Activate animation for some nice
    intersection = SocialForceIntersection(
        (bike1,), activate_sumo_cosimulation=False, animate=True, axes=axes[0]
    )

    fig, fig_bg, ax = init_animation(fig, axes[0])

    # Run the simulation
    tsim = 9
    run(intersection, name, fig, fig_bg, tsim)


def calc_full_forcefields(
    X: np.ndarray, Y: np.ndarray, psi: float, bikes: Bicycle
) -> Tuple[np.ndarray, np.ndarray]:
    """
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

    """

    Fx = np.zeros_like(X, dtype=float)
    Fy = np.zeros_like(X, dtype=float)

    for b in bikes:
        Fxb, Fyb = b.calcRepulsiveForce(X, Y, psi)
        Fx += Fxb
        Fy += Fyb
    return Fx, Fy


def run(intersection, name, fig, fig_bg, tsim, bikes_for_forcefield_plot=[]):
    """Run the simulation for the test scenarios.

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
    """

    imax = int(tsim / intersection.vehicles[0].params.t_s)

    i = 0

    filename_temp = "C:/Users/christophschmi/Desktop/temp.png"
    filename = f"C:/Users/christophschmi/Desktop/scenario_{name}_animation.avi"

    if os.path.isfile(filename):
        raise Exception(f"{filename} already exists!")

    t_s = intersection.vehicles[0].params.t_s
    f = 1 / t_s
    divider = 2
    subsampling = 0
    while f > 50.0:
        subsampling += 1
        t_s = t_s * divider
        f = 1 / t_s

    testdata = np.array(fig.canvas.renderer.buffer_rgba())
    framesize = testdata.shape
    output = cv.VideoWriter(
        filename,
        cv.VideoWriter_fourcc("M", "J", "P", "G"),
        f,
        frameSize=(framesize[0], framesize[1]),
    )

    print(
        f'Simulating "scenario_{name}" to {filename} with {f} fps (t_s={t_s}):'
    )

    contour = 0
    xlim = intersection.ax.get_xlim()
    ylim = intersection.ax.get_ylim()
    x = np.linspace(xlim[0], xlim[1], num=500)
    y = np.linspace(ylim[0], ylim[1], num=500)
    X, Y = np.meshgrid(x, y)

    while i < imax:
        fig.canvas.restore_region(fig_bg)

        intersection.step()

        if i % divider**subsampling == 0:
            intersection.endAnimation()

            if (
                len(bikes_for_forcefield_plot) > 0
                and len(bikes_for_forcefield_plot[1]) > 0
            ):
                bikes = [
                    intersection.vehicles[i]
                    for i in bikes_for_forcefield_plot[1]
                ]
                psi = intersection.vehicles[bikes_for_forcefield_plot[0]].s[2]
                # Fx, Fy = calc_full_forcefields(X, Y, psi, bikes)
                # contour = intersection.ax.contourf(
                #    X,
                #    Y,
                #    cmap="RdPu",
                #    levels=15,
                #    zorder=1,
                #    vmin=0,
                #    vmax=10,
                # )
                # im = intersection.ax.imshow(np.sqrt(Fx**2+Fy**2), cmap='RdPu', zorder=1, vmin=0, vmax=10, extent=
                #                            [xlim[0], xlim[1], ylim[0], ylim[1]], origin='lower')

            print(f"\rFrame {i}/{imax-1}", end="")

            fig.canvas.draw()

            # filename_temp = f'C:/Users/christophschmi/Desktop/temp{i}.png'
            # fig.savefig(filename_temp)
            # frame = imageio.imread(filename_temp)

            # data = np.array(fig.canvas.renderer.buffer_rgba())
            # data = data[:, :, :3]
            # data = data[:, :, [2, 1, 0]]
            # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # imageio.imwrite(filename_temp, data)

            # writer.append_data(data)

            # output.write(data)

            if (
                len(bikes_for_forcefield_plot) > 0
                and len(bikes_for_forcefield_plot[1]) > 0
            ):
                # contour.remove()
                pass

            intersection.restartAnimation()

        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()

        i = i + 1

    print("Writing .avi file...")
    output.release()

    print("Done!")

    # os.remove(filename_temp)
    # data[0].save(filename, save_all=True, append_images=data[1:], duration=t_s, loop=0)


def init_animation(fig, ax):
    """Initialize the animation of the demo.

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

    """
    plt.sca(ax)
    ax.set_aspect("equal")  # , adjustable='box')
    # figManager = plt.get_current_fig_manager()
    # figManager.resize(500, 500)
    plt.show(block=False)
    plt.pause(0.1)
    fig_bg = fig.canvas.copy_from_bbox(fig.bbox)
    fig.canvas.blit(fig.bbox)

    return fig, fig_bg, ax


def main():
    """Main function running the Test scenarios"""

    # set to true to save output
    write_parcours = False
    write_passing = False
    write_overtaking = False
    write_crossing = False
    write_error = True

    plt.close("all")

    # parcours animation1
    if write_parcours:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_ylim(-5, 5)
        ax.set_xlim(0, 30)
        ax.set_aspect("equal")
        sleep(2)
        scenario_parcours(fig, (ax, None), True)

    if write_passing:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_ylim(-5, 5)
        ax.set_xlim(-0, 30)
        ax.set_aspect("equal")
        scenario_passing(fig, (ax, None), True)

    if write_overtaking:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_ylim(-5, 5)
        ax.set_xlim(-0, 30)
        ax.set_aspect("equal")
        scenario_overtaking(fig, (ax, None), True)

    if write_crossing:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_ylim(-5, 15)
        ax.set_xlim(-0, 30)
        ax.set_aspect("equal")
        scenario_crossing(fig, (ax, None), True)

    if write_error:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_ylim(-15, 15)
        ax.set_xlim(-0, 30)
        ax.set_aspect("equal")
        scenario_errorcase(fig, (ax, None), True)


# Entry point
if __name__ == "__main__":
    main()
