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
from qmsolve.util.constants import kg, J, m, s


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
    check(np.amax(r_sym_sorted) / m)
    return Phi_func


if __name__ == "__main__":
    Phi_func = earth_grav_potential_earth_center_au()
    print(Phi_func(0))
    print(f"{Phi_func(6e6 * m):.3e}")
