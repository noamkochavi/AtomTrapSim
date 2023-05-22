# external imports
from scipy.constants import c
import numpy as np

# internal imports
from math_functions import uniform_random_direction
from constants import *


class Particle:
    """
    Object representing a moving particle
    """
    def __init__(self, mass, excited_energy, start_coords=None, start_v=None, start_energy=None):
        """
        Create a new particle
        :param mass: mass of the particle
        :param excited_energy: energy required to excite particle
        :param start_coords: coordinates of the particle at the beginning of the simulation
        :param start_v: velocity of the particle at the beginning of the simulation
        :param start_energy: energy of the particle at the beginning of the simulation
        """
        self.mass = mass
        self.coords = start_coords if start_coords is not None else np.zeros(3)
        self.v = start_v if start_v is not None else np.zeros(3)
        self.energy = start_energy if start_energy is not None else 0
        self.excited_energy = excited_energy
        self.coord_cache = np.zeros((CACHE_SIZE, 3), float)

    @property
    def momentum(self):
        return self.mass * self.v

    @momentum.setter
    def momentum(self, value):
        self.v = value / self.mass

    @property
    def is_excited(self):
        return self.energy == self.excited_energy

    def step(self):
        """
        Move the particle according to the current configuration and TIME_RESOLUTION
        """
        emiss_dir = None
        if self.energy == self.excited_energy:
            # TODO: add probability + random direction
            emiss_dir = uniform_random_direction()
            self.momentum -= emiss_dir * self.energy / c  # TODO: check formula - p=E/c
            self.energy = 0
        self.coord_cache = np.roll(self.coord_cache, 1, 0)
        self.coord_cache[0] = self.coords
        self.coords = self.coords + self.v * TIME_RESOLUTION
        return emiss_dir
