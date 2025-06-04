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
from qmsolve.util.constants import femtoseconds, m_e, Å, m

# =========================================================================================================#
# First, we define the Hamiltonian of a single particle confined in an harmonic oscillator potential.
# =========================================================================================================#

#
scaling = 1e6
extent = scaling * 30 * Å

# m_particle = m_e / np.sqrt(scaling)
m_particle = m_e / 10 / scaling

period = (m_e / m_particle) ** 0.5 * 0.62 * femtoseconds
half_period = period / 2.0

play_period = 2  # (s) playtime of 1/2 period


# interaction potential
def harmonic_oscillator(particle):
    m = m_particle
    # period = period
    w = 2 * np.pi / period
    k = m * w**2
    return 0.5 * k * (particle.x) ** 2


# build the Hamiltonian of the system
H = Hamiltonian(
    particles=SingleParticle(m=m_particle),
    potential=harmonic_oscillator,
    spatial_ndim=1,
    N=5000,
    extent=extent,
)

# =========================================================================================================#
# Define the wavefunction at t = 0  (initial condition)
# =========================================================================================================#


# def initial_wavefunction(particle):
#     # This wavefunction correspond to a gaussian wavepacket with a mean X momentum equal to p_x0
#     # not normalized!!!!!
#     σ = scaling**0.75 * 0.7 * Å
#     v0 = 0 * 60 * Å / femtoseconds
#     p_x0 = m_particle * v0
#     return np.exp(
#         -1 / (4 * σ**2) * ((particle.x) ** 2) / np.sqrt(2 * np.pi * σ**2)
#     ) * np.exp(p_x0 * particle.x * 1j)


def initial_wavefunction(particle):
    # This wavefunction correspond to a gaussian wavepacket with a mean X momentum equal to p_x0
    σ = scaling * 0.7 * Å
    v0 = 0 * 60 * Å / femtoseconds
    p_x0 = m_particle * v0
    return (
        1.0
        / (2 * np.pi * σ**2) ** 0.25
        * np.exp(-1 / (4 * σ**2) * ((particle.x - extent / 6) ** 2))
        * np.exp(p_x0 * particle.x * 1j)
    )


# =========================================================================================================#
# Set and run the simulation
# =========================================================================================================#

total_time = half_period * 5.2
# set the time dependent simulation
sim = TimeSimulation(hamiltonian=H, method="split-step")
# sim = TimeSimulation(hamiltonian=H, method="crank-nicolson")

sim.run(
    initial_wavefunction,
    total_time=total_time,
    dt=total_time / (1600.0),
    store_steps=800,
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
visualization.plot(t=0 * femtoseconds, xlim=[-extent / 2, extent / 2])
visualization.plot(t=half_period, xlim=[-extent / 2, extent / 2])
# visualization.plot(t=0.35 * femtoseconds, xlim=[-extent / 2, extent / 2])
