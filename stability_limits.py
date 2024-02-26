# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 17:28:59 2023

stability_limits.py

Plots Figure 5: "Stability limits of the inner loop for the gain KD(v).", and
Figure 6: "Stability limits of the outer loop for the gains KP (v) and KI (v) 
with ki0 = 0.2.", for Schmidt et al. (2023).

Schmidt, C., Dabiri, A., Schulte, F., Happee, R. & Moore, J. (2023). Essential 
Bicycle Dynamics for Microscopic Traffic Simulation: An Example Using the 
Social Force Model [preprint]. The Evolving Scholar - BMD 2023, 5th Edition. 
https://doi.org/10.59490/65037d08763775ba4854da53


Requires cyclistsocialforce>=1.1.0
Requires pypaperutils (https://github.com/chrismo-schmidt/pypaperutils)

Usage: $ python stability_limits.py

@author: Christoph M. Schmidt, TU Delft. 
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from cyclistsocialforce.parameters import InvPendulumBicycleParameters
from pypaperutils.design import TUDcolors, figure_for_latex
from pypaperutils.io import export_to_pgf

# get TUD colors
tudcolors = TUDcolors()
red = "red"
cyan = tudcolors.get("cyaan")
black = "black"

# output directory
outdir = "./figures/review/"


def config_matplotlib_for_latex(save=True):
    """Presets of matplotlib for beautiul latex plots

    Parameters
    ----------
    save : boolean, optional
        Flag indicating if 2D bicycles or inverted pendulum bicycles should be
        simulated. True = inv. pendulum. False = 2D bicycles.
    """

    if save:
        matplotlib.use("pgf")
    else:
        matplotlib.use("Qt5Agg")

    matplotlib.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "axes.labelsize": 8,
            "axes.titlesize": 10,
            "font.size": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    plt.close("all")


def main():
    """Create plots of the stability limits of the system using Routh"""

    save = False

    config_matplotlib_for_latex(save)

    # model parameters
    v = np.linspace(0.1, 10, 100)

    params = InvPendulumBicycleParameters()

    K, K_tau_2 = params.timevarying_combined_params(v)
    tau2 = K_tau_2 / K
    tau3 = (params.l_1 + params.l_2) / v

    v_min_stable = params.min_stable_speed_inner()
    K_r2 = params.r2_adaptive_gain(v)
    Kd = K_r2[2]
    K_r1 = params.r1_adaptive_gain(v)
    Ki = K_r1[1]
    Kp = K_r1[0]

    # inner loop
    fig = figure_for_latex(3.5, width=9)

    plt.plot(
        v,
        -params.i_steer_vertvert / (tau2 * K),
        color=black,
        label=r"upper limit (I)",
    )
    plt.text(0.3, -150, "(I)")
    plt.text(2.2, -220, "(II)")
    plt.text(4, -280, r"$K_\mathrm{D}(v)$", color=cyan)
    plt.text(
        1.2,
        -900,
        r"$v_\mathrm{min}\approx0.98~\frac{\mathrm{m}}{\mathrm{s}}$",
        color=red,
    )
    plt.plot(v, -params.c_steer / (K), color=black, label=r"upper limit (II)")
    plt.plot(v, Kd, color=cyan, label=r"K_\mathrm{D}(v)")
    plt.plot(
        (v_min_stable, v_min_stable),
        (-1000, 100),
        color=red,
        linestyle="dashed",
    )
    plt.xlabel(r"$v$ $\textstyle\left[\frac{\mathrm{m}}{\mathrm{s}}\right]$")
    plt.gca().xaxis.set_label_coords(0.5, -0.2)
    plt.ylabel(r"$K_\mathrm{D}$")
    plt.xlim(0, 10)
    plt.ylim(-1000, 100)

    # outer loop
    fig2 = figure_for_latex(3.5)
    axes = fig2.subplots(1, 2, sharex=True)

    # Ki
    Kimax = (params.c_steer * tau3 + K * Kd * tau3) / (
        params.tau_1_squared * Kd
    )
    axes[0].plot(v, Kimax, color=black)
    axes[0].plot(v, np.zeros_like(v), color=black)
    axes[0].plot(v, Ki, color=cyan)
    axes[0].plot(
        (v_min_stable, v_min_stable),
        (-1000, 100),
        color=red,
        linestyle="dashed",
    )
    axes[0].set_ylim(-2, 2)
    axes[0].set_xlim(0, 10)
    axes[0].set_xlabel(
        r"$v$ $\textstyle\left[\frac{\mathrm{m}}{\mathrm{s}}\right]$"
    )
    axes[0].xaxis.set_label_coords(0.5, -0.2)
    axes[0].set_ylabel(r"$K_\mathrm{I}$")
    axes[0].text(2.4, 1, "(IV)")
    axes[0].text(3, -0.6, r"(III)")
    axes[0].text(5, 0.5, r"$K_\mathrm{I}(v)$", color=cyan)
    axes[0].text(
        1.2,
        -1.5,
        r"$v_\mathrm{min}\approx0.98~\frac{\mathrm{m}}{\mathrm{s}}$",
        color=red,
    )

    # Kp
    Kpmax = (params.i_steer_vertvert * tau3 + K * Kd * tau3 * tau2) / (
        params.tau_1_squared * Kd
    )
    axes[1].plot(v, Kpmax, color=black)
    axes[1].plot(v, np.zeros_like(v), color=black)
    axes[1].plot(
        v,
        -params.i_steer_vertvert
        / (params.tau_1_squared * params.g * params.c_steer)
        * v
        + params.l_2 / (params.tau_1_squared * params.g)
        + params.i_steer_vertvert / params.c_steer * Ki,
        color=black,
    )
    axes[1].plot(v, np.ones_like(v) * Kp, color=cyan)
    axes[1].plot(
        (v_min_stable, v_min_stable),
        (-1000, 100),
        color=red,
        linestyle="dashed",
    )
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(-0.05, 0.5)
    axes[1].set_xlabel(
        r"$v$ $\textstyle\left[\frac{\mathrm{m}}{\mathrm{s}}\right]$"
    )
    axes[1].xaxis.set_label_coords(0.5, -0.2)
    axes[1].set_ylabel(r"$K_\mathrm{P}$")
    axes[1].text(8, 0.02, r"(I)")
    axes[1].text(4, 0.33, r"(V)")
    axes[1].text(5, 0.18, r"$K_\mathrm{P}(v)$", color=cyan)
    axes[1].text(
        1.2,
        0.1,
        r"$v_\mathrm{min}\approx0.98~\frac{\mathrm{m}}{\mathrm{s}}$",
        color=red,
    )

    # save
    export_to_pgf(
        fig, os.path.join(outdir, "stability-limits-inner-loop"), save=save
    )
    export_to_pgf(
        fig2, os.path.join(outdir, "stability-limits-outer-loop"), save=save
    )


if __name__ == "__main__":
    main()
