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
import matplotlib.pyplot as plt

from qmsolve.hamiltonian import Hamiltonian
from qmsolve.particle_system.single_particle import SingleParticle
from qmsolve.time_dependent_solver.time_dependent_solver import TimeSimulation
from qmsolve.visualization.visualization import init_visualization
from qmsolve.util.constants import femtoseconds, m_e, Å, m

# =========================================================================================================#
# First, we define the Hamiltonian of a single particle confined in an harmonic oscillator potential.
# =========================================================================================================#

#
scaling = 1000
extent = scaling * 30 * Å

# m_particle = m_e / np.sqrt(scaling)
m_particle = m_e / scaling

period = (m_e / m_particle) ** 0.5 * 0.62 * femtoseconds
half_period = period / 2.0

play_period = 4  # (s) playtime of 1/2 period


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


def initial_wavefunction(x):
    # This wavefunction correspond to a gaussian wavepacket with a mean X momentum equal to p_x0
    σ = scaling**0.75 * 0.7 * Å
    v0 = 0 * 60 * Å / femtoseconds
    p_x0 = m_particle * v0
    return (
        1.0
        / (2 * np.pi * σ**2) ** 0.25
        * np.exp(-1 / (4 * σ**2) * ((x) ** 2))
        * np.exp(p_x0 * x * 1j)
    )


# --- x-range ---
x = np.linspace(-extent / 2, extent / 2, 1000)
ψ = initial_wavefunction(x)

print(np.sum(np.abs(ψ) ** 2) * np.abs(x[0] - x[1]))
# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(x / Å, ψ.real, label="Re[ψ(x)]")
plt.plot(x / Å, ψ.imag, label="Im[ψ(x)]")
plt.plot(x / Å, np.abs(ψ), label="|ψ(x)|", linestyle="--")
plt.xlabel("x (Å)")
plt.ylabel("Wavefunction")
plt.title("Initial Wavefunction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
