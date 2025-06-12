# examples\eigenstate solver examples\1D_grav_pot_eigen.py
import os
import sys

print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))

import numpy as np

# from qmsolve import Hamiltonian,  SingleParticle, init_visualization, Å, eV
from qmsolve.hamiltonian import Hamiltonian
from qmsolve.particle_system.single_particle import SingleParticle
from qmsolve.visualization.visualization import init_visualization
from qmsolve.util.constants import femtoseconds, m_e, Å, m, eV, earth_radius

from qmsolve.util.functioncache import check, earth_grav_potential_earth_center_au, dual_harmonic_oscillator

earth_pot_au = 1.2536582952711697e-05

# earth_radius = 6371.0e3 * m
ma_au = 8.09329979249468e-15
#
# scaling = earth_radius
extent = 16 * earth_radius

# m_particle = m_e / np.sqrt(scaling)
# m_particle = m_e / 1e26
# m_particle = m_e / 1e22
# m_particle = m_e / 1e20
# m_particle = m_e / 1e18

m_particle = ma_au

check(m_particle)
# ma in a.u. =  8.09329979249468e-15
check(ma_au)


Phi_func = earth_grav_potential_earth_center_au()

factor = 1e0

# interaction potential
def earth_grav_pot(particle):
    return m_particle * Phi_func(particle.x) * factor**2


def dual_harmonic_oscillator_pot(particle):
    return m_particle * dual_harmonic_oscillator(particle.x) * factor**2


# =========================================================================================================#
# We define the Hamiltonian of a single particle confined in an harmonic oscillator potential.
# Then, we compute its eigenstates.
# =========================================================================================================#


# interaction potential
def harmonic_oscillator(particle):
    k = 100 * eV / Å**2
    return 0.5 * k * particle.x**2


# define the Hamiltonian
H = Hamiltonian(
    particles=SingleParticle(m=m_particle),
    potential=earth_grav_pot,
    # potential=dual_harmonic_oscillator_pot,
    spatial_ndim=1,
    N=int(2**14),
    extent=extent,
)

# Diagonalize the Hamiltonian and compute the eigenstates
eigenstates = H.solve(max_states=10)

print(eigenstates.energies)  # the printed energies are expressed in eV

# Visualize the Eigenstates
visualization = init_visualization(eigenstates)
visualization.slider_plot_earth()  # interactive slider

# (Optional: Visualize a specific eigenstate)
# visualization.plot_eigenstate(0)
