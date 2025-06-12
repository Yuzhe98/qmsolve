import os
import sys

print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))

import inspect  # for check()
import re  # for check()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pandas as pd
import time

from scipy.interpolate import interp1d
from qmsolve.util.constants import kg, J, m, s, earth_radius


def GiveDateandTime():
    timestr = time.strftime("%Y%m%d_%H%M%S")
    # timestr = 'session_'+timestr
    return timestr


def check(arg):
    """
    Print information of input arg

    Example
    -------
    import numpy as np

    a = np.zeros((2, 4))

    check(a)

    a+=1

    check(a)

    check(len(a))

    TERMINAL OUTPUT:

    d:\Yu0702\casper-gradient-code\\testofcheckpoint.py @45 a : ndarray(array([[0., 0., 0., 0.], [0., 0., 0., 0.]])) [shape=(2, 4)]

    d:\Yu0702\casper-gradient-code\\testofcheckpoint.py @47 a : ndarray(array([[1., 1., 1., 1.], [1., 1., 1., 1.]])) [shape=(2, 4)]

    d:\Yu0702\casper-gradient-code\\testofcheckpoint.py @48 len(a) : int(2)

    d:\Yu0702\casper-gradient-code\\testofcheckpoint.py @49 a.shape : tuple((2, 4)) [len=2]

    Copyright info:
    ------
    Adopted from https://gist.github.com/HaleTom/125f0c0b0a1fb4fbf4311e6aa763844b

    Author: Tom Hale

    Original comment: Print the line and filename, function call, the class, str representation and some other info
                    Inspired by https://stackoverflow.com/a/8856387/5353461
    """
    frame = inspect.currentframe()
    callerframeinfo = inspect.getframeinfo(frame.f_back)
    try:
        context = inspect.getframeinfo(frame.f_back).code_context
        caller_lines = "".join([line.strip() for line in context])
        m = re.search(r"check\s*\((.+?)\)$", caller_lines)
        if m:
            caller_lines = m.group(1)
            position = (
                str(callerframeinfo.filename) + " line " + str(callerframeinfo.lineno)
            )

            # Add additional info such as array shape or string length
            additional = ""
            if hasattr(arg, "shape"):
                additional += "[shape={}]".format(arg.shape)
            elif hasattr(arg, "__len__"):  # shape includes length information
                additional += "[len={}]".format(len(arg))

            # Use str() representation if it is printable
            str_arg = str(arg)
            str_arg = str_arg if str_arg.isprintable() else repr(arg)

            print(position, "" + caller_lines + " : ", end="")
            print(arg.__class__.__name__ + "(" + str_arg + ")", additional)
        else:
            print("check: couldn't find caller context")
    finally:
        del frame
        del callerframeinfo


def loadPEMdata():
    # Load the data
    data = pd.read_csv(
        "examples\\time dependent solver examples\\PEM_Parametric_Earth_Models_data.txt",
        comment="#",
        sep=r"\s+",
        header=None,
        names=[
            "radius_m",
            "density_kg_m3",
            "vpv",
            "vsv",
            "Q_kappa",
            "Q_miu",
            "vph",
            "vsh",
            "eta",
        ],
    )

    # Sort just in case
    data = data.sort_values(by="radius_m")

    return data


def earth_grav_potential_infty():
    """
    Returns a function Phi(r[m]) [J/kg], valid both inside and outside Earth.
    Uses PREM-like model for interior, point-mass approximation for exterior.
    """
    # Load the data (assumed to return a DataFrame with 'radius_m' and 'density_kg_m3')
    data = loadPEMdata()

    # Extract radius and density
    r = data["radius_m"].values
    rho = data["density_kg_m3"].values

    # Compute shell thickness
    dr = np.gradient(r)

    # Shell volume and mass
    dV = 4 * np.pi * r**2 * dr
    dm = rho * dV

    # Cumulative mass
    M_r = np.cumsum(dm)
    M_total = M_r[-1]

    # Gravitational constant
    G = 6.67430e-11  # m³/kg/s²

    # Extend to radii beyond Earth's surface
    r_max = r[-1]
    r_outside = np.linspace(r_max, 10 * r_max, 200)  # from surface to 10 Earth radii
    Phi_outside = -G * M_total / r_outside

    # Compute gravitational potential inside Earth
    Phi_inside = np.zeros_like(r)
    Phi_inside[1:] = 2 * Phi_outside[0] + G * M_r[1:] / r[1:]
    Phi_inside[0] = 2 * Phi_outside[0] + G * M_r[1] / r[1]  # avoid zero-division

    # Combine inside and outside
    r_full = np.concatenate([r, r_outside])
    Phi_full = np.concatenate([Phi_inside, Phi_outside])
    # Phi_full -= np.amin(Phi_full)

    # Enforce symmetry: add negative r values
    r_sym = np.concatenate([-r_full[::-1], r_full])  # mirror and append
    Phi_sym = np.concatenate([Phi_full[::-1], Phi_full])  # symmetric values

    # Optional: sort to ensure increasing r (for interp1d)
    sorted_indices = np.argsort(r_sym)
    r_sym_sorted = r_sym[sorted_indices]
    Phi_sym_sorted = Phi_sym[sorted_indices]

    # Interpolation function: now Phi_func(-r) = Phi_func(r)
    Phi_func = interp1d(
        r_sym_sorted,
        Phi_sym_sorted,
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )

    return Phi_func


def earth_grav_potential_earth_center():
    """
    Returns a function Phi(r[m]) [J/kg], valid both inside and outside Earth.
    Uses PREM-like model for interior, point-mass approximation for exterior.
    """
    # Load the data (assumed to return a DataFrame with 'radius_m' and 'density_kg_m3')
    data = loadPEMdata()

    # Extract radius and density
    r = data["radius_m"].values
    rho = data["density_kg_m3"].values

    # Compute shell thickness
    dr = np.gradient(r)

    # Shell volume and mass
    dV = 4 * np.pi * r**2 * dr
    dm = rho * dV

    # Cumulative mass
    M_r = np.cumsum(dm)
    M_total = M_r[-1]

    # Gravitational constant
    G = 6.67430e-11  # m³/kg/s²

    # Extend to radii beyond Earth's surface
    r_max = r[-1]
    r_outside = np.linspace(r_max, 10 * r_max, 200)  # from surface to 10 Earth radii
    Phi_outside = -G * M_total / r_outside

    # Compute gravitational potential inside Earth
    Phi_inside = np.zeros_like(r)
    Phi_inside[1:] = 2 * Phi_outside[0] + G * M_r[1:] / r[1:]
    Phi_inside[0] = 2 * Phi_outside[0] + G * M_r[1] / r[1]  # avoid zero-division

    # Combine inside and outside
    r_full = np.concatenate([r, r_outside])
    Phi_full = np.concatenate([Phi_inside, Phi_outside])
    # Phi_full -= np.amin(Phi_full)

    # Enforce symmetry: add negative r values
    r_sym = np.concatenate([-r_full[::-1], r_full])  # mirror and append
    Phi_sym = np.concatenate([Phi_full[::-1], Phi_full])  # symmetric values

    # Optional: sort to ensure increasing r (for interp1d)
    sorted_indices = np.argsort(r_sym)
    r_sym_sorted = r_sym[sorted_indices]
    Phi_sym_sorted = Phi_sym[sorted_indices]
    Phi_sym_sorted -= np.amin(Phi_sym_sorted)

    # Interpolation function: now Phi_func(-r) = Phi_func(r)
    Phi_func = interp1d(
        r_sym_sorted,
        Phi_sym_sorted,
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )

    return Phi_func


def earth_grav_potential_earth_center_au():
    """
    Returns a function Phi(r[a.u.]) [a.u.], valid both inside and outside Earth.
    Uses PREM-like model for interior, point-mass approximation for exterior.
    """
    # Load the data (assumed to return a DataFrame with 'radius_m' and 'density_kg_m3')
    data = loadPEMdata()

    # Extract radius and density
    r = data["radius_m"].values
    rho = data["density_kg_m3"].values

    # Compute shell thickness
    dr = np.gradient(r)

    # Shell volume and mass
    dV = 4 * np.pi * r**2 * dr
    dm = rho * dV

    # Cumulative mass
    M_r = np.cumsum(dm)
    M_total = M_r[-1]

    # Gravitational constant
    G = 6.67430e-11  # m³/kg/s²

    # Extend to radii beyond Earth's surface
    r_max = r[-1]
    r_outside = np.linspace(r_max, 10 * r_max, 200)  # from surface to 10 Earth radii
    Phi_outside = -G * M_total / r_outside

    # Compute gravitational potential inside Earth
    Phi_inside = np.zeros_like(r)
    Phi_inside[1:] = 2 * Phi_outside[0] + G * M_r[1:] / r[1:]
    Phi_inside[0] = 2 * Phi_outside[0] + G * M_r[1] / r[1]  # avoid zero-division

    # Combine inside and outside
    r_full = np.concatenate([r, r_outside])
    Phi_full = np.concatenate([Phi_inside, Phi_outside])
    # Phi_full -= np.amin(Phi_full)

    # Enforce symmetry: add negative r values
    r_sym = np.concatenate([-r_full[::-1], r_full])  # mirror and append
    Phi_sym = np.concatenate([Phi_full[::-1], Phi_full])  # symmetric values

    # Optional: sort to ensure increasing r (for interp1d)
    sorted_indices = np.argsort(r_sym)
    r_sym_sorted = r_sym[sorted_indices]
    Phi_sym_sorted = Phi_sym[sorted_indices]
    Phi_sym_sorted -= np.amin(Phi_sym_sorted)

    # Interpolation function: now Phi_func(-r) = Phi_func(r)
    Phi_func = interp1d(
        r_sym_sorted * m,
        Phi_sym_sorted * (J / kg),
        kind="linear",
        fill_value="extrapolate",
        bounds_error=False,
    )
    # check(np.amax(r_sym_sorted) / m)
    return Phi_func


def plot_dual_harmonic_oscillator():
    extent = 10 * earth_radius
    radii = np.linspace(-extent / 2, extent / 2, num=1000)
    Phi_func = earth_grav_potential_earth_center_au()

    # plot style
    plt.rc("font", size=6)  # font size for all figures
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams['mathtext.fontset'] = 'dejavuserif'

    # Make math text match Times New Roman
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["mathtext.rm"] = "Times New Roman"

    # plt.style.use('seaborn-dark')  # to specify different styles
    # print(plt.style.available)  # if you want to know available styles

    cm = 1 / 2.56  # convert cm to inch

    fig = plt.figure(figsize=(8.5 * cm, 6 * cm), dpi=300)  # initialize a figure

    gs = gridspec.GridSpec(nrows=1, ncols=1)  # create grid for multiple figures

    ax00 = fig.add_subplot(gs[0, 0])
    ax00.plot(
        radii / earth_radius,
        Phi_func(np.abs(radii)),
        label="Obtained by earth density and calculation",
        # color="darkorange",
        # linestyle="dashed",
    )
    # ax00.plot(
    #     radii / earth_radius,
    #     dual_harmonic_oscillator(radii),
    #     label="dual_harmonic_oscillator",
    # )
    # ax00.set_ylim(0, 15)
    ax00.set_xlabel("Radius (earth radius)")
    ax00.set_ylabel("Grav. Pot. (atomic units)")

    ax00.legend()
    # ax10.gca().invert_xaxis()  # Optional: so Earth's center is on the left

    fig.suptitle("")
    fig.tight_layout()
    # plt.savefig("figures/Earth_Profiles_(PEM_Data).png", transparent=False)
    plt.show()

    return


def dual_harmonic_oscillator(x):
    x = np.asarray(x)  # ensure array input
    # radius = np.abs(x)
    ma_au = 8.09329979249468e-15  # 1-MHz axion mass
    m_particle = ma_au
    factor = 1e0
    omega = 2 * np.pi * 9.8e-21 * factor
    k_earth = omega**2  #
    x1 = earth_radius * 3450 / 6371.0
    check(x1)
    check(k_earth)
    check(omega)
    check(earth_radius)
    factor1 = 1
    result = np.where(
        np.abs(x) < x1,
        0.5 * k_earth * x**2,
        0.5 * k_earth * factor1 * x**2 + 0.5 * k_earth * (1 - factor1) * x1**2,
    )
    return result


if __name__ == "__main__":
    # extent = 10 * earth_radius
    # radii = np.linspace(-extent / 2, extent / 2, num=10)
    # # check(dual_harmonic_oscillator(radii))
    # result = np.where(
    #     np.abs(radii) < earth_radius,
    #     0.5,
    #     0.5 + 0.5,
    # )
    # check(result)
    # check(dual_harmonic_oscillator(radii))
    plot_dual_harmonic_oscillator()
