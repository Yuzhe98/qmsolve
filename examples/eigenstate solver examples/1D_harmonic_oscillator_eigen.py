import os
import sys

print(os.path.abspath(os.curdir))
sys.path.insert(0, os.path.abspath(os.curdir))

import numpy as np
# from qmsolve import Hamiltonian,  SingleParticle, init_visualization, Å, eV
from qmsolve.hamiltonian import Hamiltonian
from qmsolve.particle_system.single_particle import SingleParticle
from qmsolve.time_dependent_solver.time_dependent_solver import TimeSimulation
from qmsolve.visualization.visualization import init_visualization
from qmsolve.util.constants import femtoseconds, m_e, Å, m, eV

# =========================================================================================================#
# We define the Hamiltonian of a single particle confined in an harmonic oscillator potential.
# Then, we compute its eigenstates.
# =========================================================================================================#


# interaction potential
def harmonic_oscillator(particle):
	k = 100 * eV / Å**2
	return 0.5 * k * particle.x**2


# define the Hamiltonian
H = Hamiltonian(particles = SingleParticle(), 
				potential = harmonic_oscillator, 
				spatial_ndim = 1, N = 512, extent = 20*Å)

# Diagonalize the Hamiltonian and compute the eigenstates
eigenstates = H.solve(max_states = 10)

print(eigenstates.energies) # the printed energies are expressed in eV

# Visualize the Eigenstates
visualization = init_visualization(eigenstates)
visualization.slider_plot() #interactive slider

# (Optional: Visualize a specific eigenstate)
# visualization.plot_eigenstate(0)
