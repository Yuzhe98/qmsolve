# This file:
# examples\time dependent solver examples\1D_harmonic_oscillator_copy.py


import os
import sys

print(os.path.abspath(os.curdir))
# os.chdir("..")  # go to parent folder
# print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))
# os.chdir("qmsolve")
# sys.path.insert(0, os.path.abspath(os.curdir))
# print(os.path.abspath(os.curdir))

import numpy as np

from qmsolve.hamiltonian import Hamiltonian
from qmsolve.particle_system.single_particle import SingleParticle
from qmsolve.time_dependent_solver.time_dependent_solver import TimeSimulation
from qmsolve.visualization.visualization import init_visualization
from qmsolve.util.constants import femtoseconds, m_e, Å, m, s, earth_radius

from functioncache import check, earth_grav_potential_earth_center_au

# =========================================================================================================#
# First, we define the Hamiltonian of a single particle confined in an harmonic oscillator potential.
# =========================================================================================================#

earth_pot_au = 1.2536582952711697e-05

# earth_radius = 6371.0e3 * m
ma_au = 8.09329979249468e-15
#
# scaling = earth_radius
extent = 6 * earth_radius

# m_particle = m_e / np.sqrt(scaling)
# m_particle = m_e / 1e26
# m_particle = m_e / 1e22
# m_particle = m_e / 1e20
# m_particle = m_e / 1e18

m_particle = ma_au

check(m_particle)
# ma in a.u. =  8.09329979249468e-15
check(ma_au)

# period = 1e-6 * s
# half_period = period / 2.0

# omega = 2 * np.pi / period

# k_earth = 1.7298013435612702e-39
# check(k_earth)

# k_harmonic = m_particle * omega**2
# check(k_harmonic)

play_period = 2  # (s) playtime of 1/2 period

# omega = (2 * earth_pot_au / (m_particle * earth_radius**2)) ** 0.5 * 10**(-0.0)

factor = 1e3
omega = 2 * np.pi * 4.780345599341959e-21 * factor
# omega = 2 * np.pi * 4.780345599341959e-21
check(omega)
# omega should be 2 * pi * 4.780345599341959e-21
period = 2 * np.pi / omega
check(period / s)
half_period = period / 2.0

k_earth = m_particle * omega**2
check(0.5 * k_earth * (earth_radius) ** 2)


total_time = period * 5
num_step_time = 20000
dt = total_time / num_step_time
store_steps = int(num_step_time // 4)


# interaction potential
def harmonic_oscillator(particle):
    # m = m_particle
    # # period = period
    # w = 2 * np.pi / period
    # k = m * w**2
    # return 0.5 * k_harmonic * (particle.x) ** 2
    return 0.5 * k_earth * (particle.x) ** 2

Phi_func = earth_grav_potential_earth_center_au()
# interaction potential
def earth_grav_pot(particle):
    # m = m_particle
    # # period = period
    # w = 2 * np.pi / period
    # k = m * w**2
    # return 0.5 * k_harmonic * (particle.x) ** 2
    return m_particle * Phi_func(particle.x) * factor


# build the Hamiltonian of the system
H = Hamiltonian(
    particles=SingleParticle(m=m_particle),
    # potential=harmonic_oscillator,
    potential=earth_grav_pot,
    spatial_ndim=1,
    N=10000,
    extent=extent,
)

# =========================================================================================================#
# Define the wavefunction at t = 0  (initial condition)
# =========================================================================================================#


def initial_wavefunction(particle):
    # This wavefunction correspond to a gaussian wavepacket with a mean X momentum equal to p_x0
    σ = earth_radius * 0.02
    v0 = 0e1 * m / s
    p_x0 = m_particle * v0
    return (
        1.0
        / (2 * np.pi * σ**2) ** 0.25
        * np.exp(-1 / (4 * σ**2) * ((particle.x - earth_radius / 3) ** 2))
        * np.exp(p_x0 * particle.x * 1j)
    )


# =========================================================================================================#
# Set and run the simulation
# =========================================================================================================#


# set the time dependent simulation
sim = TimeSimulation(hamiltonian=H, method="split-step")
# sim = TimeSimulation(hamiltonian=H, method="crank-nicolson")

sim.run(
    initial_wavefunction,
    total_time=total_time,
    dt=dt,
    store_steps=store_steps,
)

# =========================================================================================================#
# Finally, we visualize the time dependent simulation
# =========================================================================================================#

visualization = init_visualization(sim)
visualization.animate_cus_units(
    xlim=[-extent / 2, extent / 2],
    animation_duration=total_time,
    save_animation=False,
    fps=24,
    playspeed=half_period / play_period,
)

# for visualizing a single frame, use plot method instead of animate:
# visualization.plot_axion_earth(t=0 * half_period, xlim=[-extent / 2, extent / 2])
# visualization.plot_axion_earth(t=0.00125 * half_period, xlim=[-extent / 2, extent / 2])
# visualization.plot_axion_earth(t=0.25 * half_period, xlim=[-extent / 2, extent / 2])
# visualization.plot_axion_earth(t=0.375 * half_period, xlim=[-extent / 2, extent / 2])
# visualization.plot_axion_earth(t=0.5 * half_period, xlim=[-extent / 2, extent / 2])

# visualization.plot_axion_earth(t=half_period, xlim=[-extent / 2, extent / 2])
