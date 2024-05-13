# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:45:28 2023

repulsive-force-fields.py

Plots Figure 3: "Repulsive force fields of a cyclist a located at (0, 0) for 
different relative orientations ψa,b and positions (xa,b, ya,b) of a cyclist b. 
Colors indicate the magnitude of the force field as multiples of the desired 
velocity vd of b. The red line marks where the repulsive force equals the 
maximum magnitude of b’s destination force. The repulsive force direction 
experienced by b is perpendicular to the contour lines and indicated by black 
arrows.", for Schmidt et al. (2023).

Schmidt, C., Dabiri, A., Schulte, F., Happee, R. & Moore, J. (2023). Essential 
Bicycle Dynamics for Microscopic Traffic Simulation: An Example Using the 
Social Force Model [preprint]. The Evolving Scholar - BMD 2023, 5th Edition. 
https://doi.org/10.59490/65037d08763775ba4854da53


Requires cyclistsocialforce>=1.1.0
Requires pypaperutils (https://github.com/chrismo-schmidt/pypaperutils)

Usage: $ python repulsive_force_fields.py

@author: Christoph M. Schmidt, TU Delft. 
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

from cyclistsocialforce.vehicle import InvPendulumBicycle
from cyclistsocialforce.vizualisation import BicycleDrawing2D
from pypaperutils.design import figure_for_latex, config_matplotlib_for_latex
from pypaperutils.io import export_to_pgf

# output directory
outdir = "./figures/"


def plot_force_direction(ax, X, Y, Fx, Fy):
    """Create a quiver plot that shows the direction of the force plots.

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

    """
    Fx = Fx[::10, ::10]
    Fy = Fy[::10, ::10]
    X = X[::10, ::10]
    Y = Y[::10, ::10]

    Fx = 2 * Fx / (np.sqrt(Fx**2 + Fy**2))
    Fy = 2 * Fy / (np.sqrt(Fx**2 + Fy**2))

    ax.quiver(X, Y, Fx, Fy, scale=50, color="white")


def plot_force_magnitude(ax, X, Y, Fx, Fy):
    """Create a contour plot that shows the magnitude of the force.

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
        Countours separating the filled regions of the contour plot
    QuadContourSet
        Single contour highlighting where the repulsive force equals the destination force

    """
    F = np.sqrt(Fx**2 + Fy**2)
    c = ax.contourf(X, Y, F, levels=np.arange(0, 5.5, 1), extend="max")
    cl = ax.contour(
        X,
        Y,
        F,
        levels=np.arange(0, 5.5, 1),
        extend="max",
        colors="white",
        linewidths=0.1,
    )
    cr = ax.contour(
        c,
        levels=np.array((c.levels[-1],)),
        colors="r",
    )
    return c, cl, cr  # clw


def main():
    """
    Plot and export a figure showing the magnitude and directions of
    the repulsive force fields for different locations and relative
    orientations.

    Returns
    -------
    None.

    """

    save = False
    draw_bike = False

    config_matplotlib_for_latex(save)

    b = InvPendulumBicycle((0, 0, 0, 5, 0, 0))

    lnspx = np.arange(-10, 40, 0.1)
    lnspy = np.arange(-10, 10, 0.1)
    X, Y = np.meshgrid(lnspx, lnspy)
    psis = np.array([0, np.pi / 4, np.pi / 2])

    fig = figure_for_latex(4.1)
    axes = fig.subplots(1, 3, sharey=True, sharex=True)

    titles = (
        r"\textbf{Parallel interactions}"
        + "\n"
        + r"\footnotesize{$\psi_{a,b} = 0$, $\psi_{a,b} = \pm \pi$}",
        r"\textbf{45 \textdegree~interactions}"
        + "\n"
        + r"\footnotesize{$\psi_{a,b} = \pm \frac{1}{4}\pi$"
        + r" $\psi_{a,b} = \pm \frac{3}{4}\pi$}",
        r"\textbf{Perpendicular interactions}"
        + "\n"
        + r"\footnotesize{$\psi_{a,b} = \pm \frac{1}{2}\pi$}",
    )

    for ax, psi, title in zip(axes.flatten(), psis, titles):
        Fx, Fy = b.calcRepulsiveForce(X, Y, psi)
        c, cl, cr = plot_force_magnitude(ax, X, Y, Fx, Fy)
        plot_force_direction(ax, X, Y, Fx, Fy)
        if draw_bike:
            bdrawing = BicycleDrawing2D(ax, b)
            bdrawing.show_roll_indicator = False
            bdrawing.fcolors = ["0"] * len(bdrawing.fcolors)
            bdrawing.p.set(facecolor=bdrawing.fcolors)
            bdrawing.update(b)
        ax.set_xlim(-5, 15)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.set_title(title, y=1)

    cbar = fig.colorbar(
        c,
        ax=axes,
        shrink=0.6,
        location="right",
        aspect=10,
        pad=0.01,
        ticks=[0, 2.5, 5],
    )
    cbar.ax.set_yticklabels(
        ("0", r"$\frac{v_\mathrm{d}}{2}$", r"$v_\mathrm{d}$")
    )

    cbar.add_lines(cr)
    fig.supxlabel(r"$x_{a,b}$ [m]", y=-0.01, x=0.51, fontsize=8)
    fig.supylabel(r"$y_{a,b}$ [m]", fontsize=8)

    export_to_pgf(
        fig, os.path.join(outdir, "repulsive_force_fields"), save=save
    )


if __name__ == "__main__":
    main()
