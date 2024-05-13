# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 17:28:59 2023

stability_limits_symbolic.py

Used to verify the symbolic expressions of Schmidt et al. (2023)
Plots Figure 5 and 6 for Schmidt et al. (2023).

Schmidt, C., Dabiri, A., Schulte, F., Happee, R. & Moore, J. (2023). Essential 
Bicycle Dynamics for Microscopic Traffic Simulation: An Example Using the 
Social Force Model [preprint]. The Evolving Scholar - BMD 2023, 5th Edition. 
https://doi.org/10.59490/65037d08763775ba4854da53


Requires cyclistsocialforce>=1.1.0
Requires pypaperutils (https://github.com/chrismo-schmidt/pypaperutils)

Usage: $ python stability_limits_symbolic.py

@author: Christoph M. Schmidt, TU Delft
@author: Jason K. Moore, TU Delft (see PR#9 in cyclistsocialforce)
"""

import warnings
import os

import numpy as np

import sympy as sm
import sympy.physics.control as cn
import matplotlib
import matplotlib.pyplot as plt

from cyclistsocialforce.parameters import InvPendulumBicycleParameters
from pypaperutils.design import (
    TUDcolors,
    figure_for_latex,
    config_matplotlib_for_latex,
)
from pypaperutils.io import (
    export_to_pgf,
)

plt.close("all")

tudcolors = TUDcolors()

outdir = "./figures/review/"

analyse_inner_loop = True
analyse_outer_loop = True

expand_to_physical_vars = False
calc_routh_inner = True
calc_routh_outer = True
print_coeffs = False
solve_routh_parameters = True

plot_paper_format = True
plot_save = True
plot_stepwidth_limits = 0.01


def determine_stability_point(tab, controlvals, paramvals, dynparams=None):
    if dynparams != None:
        tab = tab.subs(dynparams)
    tab = tab.subs(paramvals)

    n = len(controlvals[list(controlvals.keys())[0]])
    for k in controlvals:
        assert (
            len(controlvals[k]) == n
        ), f"Provide the same number of values for all control parameters"

    stability = [False] * n
    for i in range(n):
        c = {k: controlvals[k][i] for k in controlvals}
        tab_i = tab.subs(c)
        stability[i] = np.all(np.array(tab_i[0, :]) >= 0) & np.all(
            np.array(tab_i[:, 0]) >= 0
        )

    return stability


def determine_stability_limits(
    eqs, controlvars, controlvals, paramvals, dynparams, color, eqnames, axes
):
    assert len(controlvars) == len(
        axes
    ), "Provide an axes object for each control variable!"

    assert len(eqs) == len(eqnames), "Provide a name for each equation!"

    conds = []
    for e, en in zip(eqs, eqnames):
        conds.append([])
        for cvar, ax in zip(controlvars, axes):
            c = sm.solve(e, cvar)
            conds[-1].append(c)

            evaluate_condition(
                c,
                paramvals,
                controlvals,
                dynparams,
                ax,
                en,
                color,
            )

    return conds


def evaluate_condition(
    cond,
    paramvals,
    controlvals,
    dynparams=None,
    ax=None,
    label="label",
    color="k",
):
    for cc in cond:
        if dynparams != None:
            cc = cc.subs(dynparams).simplify()
        cc = cc.subs(paramvals)

        stability_limit = np.full((len(controlvals[v]),), np.nan)
        warn = False
        i = 0

        for vv, kpv, kiv, kdv in zip(
            controlvals[v],
            controlvals[KP],
            controlvals[KI],
            controlvals[KD],
        ):
            limit_i = cc.subs({v: vv, KP: kpv, KD: kdv, KI: kiv})
            if limit_i.is_real:
                limit_i = float(limit_i)
            else:
                limit_i = None
                warn = True

            stability_limit[i] = limit_i
            i += 1

        if warn:
            warnings.warn(
                f"Condition {label} has complex solutions! \
                    Truncating complex solutions for plotting.",
                RuntimeWarning,
            )

        if ax != None:
            ax.plot(
                vvals,
                stability_limit,
                color=color,
                label=label,
            )


def get_coefficients(G, var, print_to_console=True, name="system"):
    den = sm.poly(G.doit().simplify().expand().den, var)
    num = sm.poly(G.doit().simplify().expand().num, var)

    if print_to_console:
        print(f"{name} has the coefficients:")

        i = 0
        for c in num.coeffs():
            print(f"b_{i} = ", end="")
            sm.pprint(c)
            i += 1

        i = 0
        for c in den.coeffs():
            print(f"A_{i} = ", end="")
            sm.pprint(c)
            i += 1

    return den.coeffs(), num.coeffs()


def routh_table(polynomial, var, name="system"):
    p = sm.poly(polynomial, var)
    n = p.degree()
    coeffs = p.coeffs()
    table = sm.zeros(n + 1, n + 1)
    first = True
    row1, row2 = [], []
    for c in coeffs:
        if first:
            row1.append(c)
            first = False
        elif not first:
            row2.append(c)
            first = True
    for i, v in enumerate(row1):
        table[0, i] = v
    for i, v in enumerate(row2):
        table[1, i] = v
    for j in range(2, n + 1):
        for i in range(n):
            table[j, i] = (
                table[j - 1, 0] * table[j - 2, i + 1]
                - table[j - 2, 0] * table[j - 1, i + 1]
            ) / table[j - 1, 0]
            table[j, i] = sm.simplify(table[j, i])
    return table


if solve_routh_parameters:
    calc_routh_outer = True
    # expand_to_physical_vars = True

KD, KP, KI, v = sm.symbols("K_D, K_P, K_I, v", real=True)
Ib, Is, c, m, h, g, l2, l1, l = sm.symbols(
    "I_b, I_s, c, m, h, g, l2, l1, l", real=True, nonnegative=True
)
s = sm.symbols("s")

if expand_to_physical_vars:
    tau1_sq = (Ib + m * h**2) / m / g / h
    tau2 = l2 / v
    tau3 = l / v
    K = v**2 / g / l
else:
    Is, c = sm.symbols("I_s, c", real=True, nonnegative=True)
    tau1_sq = sm.symbols("tau_1^2")
    tau2 = sm.symbols("tau_2")
    tau3 = sm.symbols("tau_3")
    K = sm.symbols("K")

# Build systems
Gtheta = cn.TransferFunction(-K * (tau2 * s + 1), tau1_sq * s**2 - 1, s)
Gdelta = cn.TransferFunction(1, Is * s**2 + c * s, s)
Gpsi = cn.TransferFunction(1, tau3 * s, s)
Pcont = cn.TransferFunction(KP, 1, s)
Icont = cn.TransferFunction(KI, s, s)
Dcont = cn.TransferFunction(KD * s, 1, s)
PIcont = cn.Parallel(Pcont, Icont)
Gunity = cn.TransferFunction(1, 1, s)

# Build closed inner loop transfer function
Ginner_theta = cn.Feedback(
    Dcont * Gdelta * Gtheta, cn.TransferFunction(1, 1, s)
)
a_inner, b_inner = get_coefficients(
    Ginner_theta, s, print_coeffs, "G_inner_theta"
)

# ----- STABILITY OF THE OUTER LOOP -----

# ----- Transfer Functions -----

# Build closed outer loop transfer function
Ginner_delta = cn.Feedback(Dcont * Gdelta, Gtheta)
# NOTE : Shouldn't have to call .doit() on Ginner for this to work.
Gouter = cn.Feedback(PIcont * Ginner_delta.doit() * Gpsi, Gunity)
a_outer, b_outer = get_coefficients(Gouter, s, print_coeffs, "G_outer")

# ----- Routh Tables -----

if calc_routh_inner:
    char_eq = Ginner_theta.doit().simplify().expand().den
    tab_inner = routh_table(char_eq, s, "inner loop")


if calc_routh_outer:
    char_eq = Gouter.doit().simplify().expand().den

    tab_outer = routh_table(-1 * char_eq, s, "outer loop")

# ----- Stability Conditions -----

if solve_routh_parameters:
    # ----- Parameters and Values -----
    # get bicycle parameters
    params = InvPendulumBicycleParameters()

    # create a param value dict for physical bike parameters
    paramvals = {
        Ib: params.i_bike_longlong,
        Is: params.i_steer_vertvert,
        c: params.c_steer,
        m: params.m,
        h: params.h,
        g: params.g,
        l2: params.l_2,
        l: params.l,
    }

    # create a param dict for combined dynamic bike parameters
    dynparams = {
        tau1_sq: (Ib + m * h**2) / m / g / h,
        tau2: l2 / v,
        tau3: l / v,
        K: v**2 / g / l,
    }

    # create gain variable and gain value dicts including speed range
    vvals = np.arange(0.1, 10, plot_stepwidth_limits)
    controlvars_inner = (KD,)
    controlvars_outer = (KI, KP)
    controlvals = {
        v: vvals,
        KP: [params.r1_adaptive_gain(vv)[0] for vv in vvals],
        KI: [params.r1_adaptive_gain(vv)[1] for vv in vvals],
        KD: [params.r2_adaptive_gain(vv)[2] for vv in vvals],
    }

    # ----- plot inner loop conditions -----

    if analyse_inner_loop:
        if plot_paper_format:
            config_matplotlib_for_latex(plot_save)
            fig_inner = figure_for_latex(3.5, width=9)  # 3.5)
            axes_inner = fig_inner.subplots(1, 1)
        else:
            fig_inner, axes_inner = plt.subplots(1, 1)

        # plot gain curves
        axes_inner.plot(
            controlvals[v],
            controlvals[KD],
            color=tudcolors.get("cyaan"),
            label="Kd(v)",
            zorder=200,
        )

        # plot stability limits
        color = "k"
        eqnames = [f"a{i}" for i in range(len(a_inner))] + [
            "a0",
            "a1",
            "b1",
            "a3",
        ]
        conds_inner = determine_stability_limits(
            a_inner + list(tab_inner[:, 0]),
            controlvars_inner,
            controlvals,
            paramvals,
            dynparams,
            color,
            eqnames,
            [axes_inner],
        )

        # plot minimum stable speed

        # plot stability area as scatter plot of stable samples
        nVvals = 30
        nKDvals = 15
        Vvals, KDvals = np.meshgrid(
            np.linspace(1, 10, nVvals), np.linspace(-1000, 100, nKDvals)
        )

        Vvals = Vvals.flatten()
        KDvals = KDvals.flatten()

        cvals_inner = controlvals.copy()
        cvals_inner[v] = Vvals
        cvals_inner[KD] = KDvals
        del cvals_inner[KI]
        del cvals_inner[KP]

        stability_inner = determine_stability_point(
            tab_inner, cvals_inner, paramvals, dynparams=dynparams
        )

        plt.scatter(
            Vvals[stability_inner],
            KDvals[stability_inner],
            s=0.5,
            marker=".",
            color=(0.8, 0.8, 0.8),
            zorder=50,
        )

        # format plots
        if plot_paper_format:
            axes_inner.set_ylabel("$K_\mathrm{D}$")
            # axes_inner.set_xlabel("$v$")
            axes_inner.text(4.8, -1300, "$v$")
            axes_inner.text(0.1, -170, "$\mathrm{A}_2, \mathrm{B}_1$")
            axes_inner.text(1, -750, "$\mathrm{A}_3$")
            axes_inner.text(
                4,
                -300,
                "$K_\mathrm{D}(v)$",
                color=tudcolors.get("cyaan"),
                backgroundcolor="white",
                zorder=100,
            )
        else:
            axes_inner.legend()
            axes_inner.set_ylabel("KD")
            axes_inner.set_xlabel("v")
            axes_inner.set_title("Stability limitis of Kd")
        axes_inner.set_ylim(-1000, 100)
        axes_inner.set_xlim(0, 10)

        export_to_pgf(
            fig_inner,
            os.path.join(outdir, "stability-limits-inner-loop"),
            save=plot_save,
        )

    # ----- plot outer loop conditions -----

    if analyse_outer_loop:
        fig_outer = figure_for_latex(3.5)  # 3.5)
        axes_outer = fig_outer.subplots(1, 2, sharex=True)

        # plot gain curves
        axes_outer[0].plot(
            controlvals[v],
            controlvals[KI],
            color=tudcolors.get("cyaan"),
            label="Ki(v)",
            zorder=200,
        )
        axes_outer[1].plot(
            controlvals[v],
            controlvals[KP],
            color=tudcolors.get("cyaan"),
            label="Kp(v)",
            zorder=200,
        )

        # plot stability limits
        color = "k"
        eqnames = [f"a{i}" for i in range(6)] + [
            "a0",
            "a1",
            "b1",
            "c1",
            "d1",
            "e1",
        ]
        conds_outer = determine_stability_limits(
            a_outer + list(tab_outer[:, 0]),
            controlvars_outer,
            controlvals,
            paramvals,
            dynparams,
            color,
            eqnames,
            axes_outer,
        )

        # plot/print minimum stable speed
        # vmin = []

        # for c in conds_outer:
        #    for cc, k in zip(c, controlvars_outer):
        #        for ccc in cc:
        #            condv = sm.Eq(ccc, k)
        #            condv = condv.subs(dynparams)
        #            condv = condv.subs(paramvals)
        #            condv = condv.subs({KD: params.k_d0_r2 / (v + params.k_d1_r2)})
        #            condv = condv.subs({KP: params.k_p_r1, KI: params.k_i0_r1})
        #
        #            # if condv != sm.false:
        #            vmin_c = sm.solve(condv, v)
        #            # if len(vmin_c) > 0:
        #            vmin += [round(float(vi), 3) for vi in vmin_c if vi > 0]

        # solve for the intersec between d1 and KI(v) / KP(v)\
        vins = []
        for c in conds_outer:
            for cc, k, ax in zip(c, controlvars_outer, axes_outer):
                for ccc in cc:
                    condv = ccc - k
                    condv = condv.subs(dynparams)
                    condv = condv.subs(paramvals)
                    condv = condv.subs(
                        {KD: params.k_d0_r2 / (v + params.k_d1_r2)}
                    )
                    condv = condv.subs({KP: params.k_p_r1, KI: params.k_i0_r1})

                    try:
                        vins_candidate = sm.nsolve(condv, v, 1)
                        if not vins_candidate.is_real:
                            continue
                        vins.append(round(float(vins_candidate), 5))
                    except ValueError:
                        continue
        if len(vins) > 0:
            vins = np.unique(vins)
            print(f"Intersections gain curves and stab. limits:")
            for f in vins:
                print(f"    {f:.2f} m/s")
            vins = vins[vins < 10]
            if len(vins > 0):
                vmin = np.amax(vins)
                print(f"Minimum stable speed:")
                print(f"    {vmin:.2f} m/s")
                axes_outer[0].plot(
                    (vmin, vmin),
                    (-5, 5),
                    color="red",
                    linestyle="--",
                    zorder=300,
                )
                axes_outer[1].plot(
                    (vmin, vmin),
                    (-5, 5),
                    color="red",
                    linestyle="--",
                    zorder=300,
                )

        # plot stability area as scatter plot of stable samples
        nVvals = 30
        nKIvals = 15
        nKPvals = 15
        Vvals, KIvals = np.meshgrid(
            np.linspace(1, 10, nVvals), np.linspace(-2.0, 2.0, nKIvals)
        )
        Vvals, KPvals = np.meshgrid(
            np.linspace(1, 10, nVvals), np.linspace(-0.1, 0.5, nKPvals)
        )

        Vvals = Vvals.flatten()
        KPvals = KPvals.flatten()
        KIvals = KIvals.flatten()

        KPvals_adaptive = np.zeros_like(Vvals)
        KIvals_adaptive = np.zeros_like(Vvals)
        KDvals_adaptive = np.zeros_like(Vvals)
        for i in range(len(KDvals)):
            KPvals_adaptive[i] = params.r1_adaptive_gain(Vvals[i])[0]
            KIvals_adaptive[i] = params.r1_adaptive_gain(Vvals[i])[1]
            KDvals_adaptive[i] = params.r2_adaptive_gain(Vvals[i])[2]

        cvals_outer = controlvals.copy()
        cvals_outer[v] = Vvals
        cvals_outer[KD] = KDvals_adaptive

        cvals_outer[KI] = KIvals
        cvals_outer[KP] = KPvals_adaptive
        stability_outer_ki = determine_stability_point(
            tab_outer, cvals_outer, paramvals, dynparams=dynparams
        )

        cvals_outer[KI] = KIvals_adaptive
        cvals_outer[KP] = KPvals
        stability_outer_kp = determine_stability_point(
            tab_outer, cvals_outer, paramvals, dynparams=dynparams
        )

        axes_outer[0].scatter(
            Vvals[stability_outer_ki],
            KIvals[stability_outer_ki],
            s=0.5,
            marker=".",
            color=(0.6, 0.6, 0.6),
        )

        axes_outer[1].scatter(
            Vvals[stability_outer_kp],
            KPvals[stability_outer_kp],
            s=0.5,
            marker=".",
            color=(0.6, 0.6, 0.6),
        )

        # plot condition names and format plots

        axes_outer[0].set_ylim(-1, 2)
        axes_outer[1].set_ylim(-0.05, 0.6)
        axes_outer[0].set_xlim(0, 10)
        axes_outer[1].set_xlim(0, 10)

        if plot_paper_format:
            axes_outer[0].text(
                2.5,
                -0.6,
                r"$v_\mathrm{min}\approx2.26~\frac{\mathrm{m}}{\mathrm{s}}$",
                color="red",
            )
            axes_outer[0].text(1.5, 1.2, "$\mathrm{A}_3$")
            axes_outer[0].text(3.7, 1.4, "$\mathrm{C}_1$")
            axes_outer[0].text(5, 1.15, "$\mathrm{D}_1$")
            axes_outer[0].text(8, -0.4, "$\mathrm{A}_5$")
            axes_outer[0].text(
                7,
                0.5,
                "$K_\mathrm{I}$",
                color=tudcolors.get("cyaan"),
                backgroundcolor="white",
                zorder=100,
            )

            axes_outer[1].text(8, 0.51, "$\mathrm{A}_2, \mathrm{B}_1$")
            axes_outer[1].text(1, 0.35, "$\mathrm{C}_1$")
            axes_outer[1].plot(
                (1.3, 1.5),
                (0.3, 0.2),
                linewidth=0.5,
                color="black",
            )
            axes_outer[1].plot(
                (0.85, 0.35),
                (0.35, 0.32),
                linewidth=0.5,
                color="black",
            )

            axes_outer[1].text(4, 0.135, "$\mathrm{D}_1$")
            axes_outer[1].plot(
                (4, 4 - 0.5),
                (0.15 + 0.04, 0.33),
                linewidth=0.5,
                color="black",
            )
            axes_outer[1].plot(
                (4, 4 - 0.5),
                (0.13, 0.06),
                linewidth=0.5,
                color="black",
            )
            axes_outer[1].text(1.6, 0.04, "$\mathrm{A}_4$")
            axes_outer[1].text(
                7,
                0.15,
                "$K_\mathrm{P}$",
                color=tudcolors.get("cyaan"),
                backgroundcolor="white",
                zorder=100,
            )
            axes_outer[1].text(
                2.5,
                0.515,
                r"$v_\mathrm{min}\approx2.26~\frac{\mathrm{m}}{\mathrm{s}}$",
                color="red",
            )
            axes_outer[0].set_ylabel("$K_\mathrm{I}$")
            axes_outer[1].set_ylabel("$K_\mathrm{P}$")
            # axes_outer[0].set_xlabel("$v$")
            # axes_outer[1].set_xlabel("$v$")
            axes_outer[0].text(5, -1.9, "$v$")
            axes_outer[1].text(5, -0.25, "$v$")

        else:
            axes_outer[0].legend()
            axes_outer[1].legend()
            axes_outer[0].set_ylabel("KI")
            axes_outer[1].set_ylabel("KP")
            axes_outer[0].set_title("Stability limitis of Ki")
            axes_outer[1].set_title("Stability limitis of Kp")

        export_to_pgf(
            fig_outer,
            os.path.join(outdir, "stability-limits-outer-loop"),
            save=plot_save,
        )
