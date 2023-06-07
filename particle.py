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

    def __init__(self, mass, excited_energy, excited_lifetime, start_coords=None, start_v=None):
        """
        Create a new particle
        :param mass: mass of the particle (kg)
        :param excited_energy: energy required to excite particle (J)
        :param excited_lifetime: how long the particle stays excited (s)
        :param start_coords: coordinates of the particle at the beginning of the simulation
        :param start_v: velocity of the particle at the beginning of the simulation (m/s)
        """
        self.mass = mass
        self.coords = start_coords if start_coords is not None else np.zeros(3)
        self.v = start_v if start_v is not None else np.zeros(3)
        self.excited_energy = excited_energy
        self.excited_lifetime = excited_lifetime
        self.excited = False
        self.excited_time = -1
        self.coord_cache = np.zeros((CACHE_SIZE, 3), float)

    @property
    def momentum(self):
        return self.mass * self.v

    @momentum.setter
    def momentum(self, value):
        self.v = value / self.mass

    @property
    def emit_time(self):
        """
        Time of expected emission
        """
        if self.excited:
            return self.excited_time + self.excited_lifetime
        return -1

    def absorb(self, momentum, time):
        """
        Absorb a photon
        :param momentum: momentum of the exciting laser
        :param time: time of excitement
        """
        self.momentum += momentum
        self.excited = True
        self.excited_time = time

    def emit(self):
        """
        Emit a photon at a random uniform direction
        :return: direction vector of emission
        """
        emiss_dir = uniform_random_direction()
        self.momentum -= emiss_dir * self.excited_energy / c
        self.excited = False
        self.excited_time = -1
        return emiss_dir

    def step(self, time, dt):
        """
        Move the particle according to the current configuration and time `dt`
        :param time: current time of the simulation
        :param dt: timespan of the step
        """
        prev_v = self.v
        v_change_t = 0
        emit_time = self.emit_time
        emiss = None
        if self.excited and (time >= emit_time):
            emiss_dir = self.emit()
            v_change_t = dt - (time - emit_time)
            emiss = (self.coords + prev_v * v_change_t, emiss_dir)
        self.coord_cache = np.roll(self.coord_cache, 1, 0)
        self.coord_cache[0] = self.coords
        self.coords = self.coords + prev_v * v_change_t + self.v * (dt - v_change_t)
        return emiss
