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
        self.trigger_excited = False
        self.trigger_excited_time = -1
        self.trigger_excited_momentum = -1
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

    def trigger_absorb(self, time, momentum):
        """
        Cue to absorb a photon during step
        :param time: time of excitement
        :param momentum: momentum of the exciting laser
        """
        self.trigger_excited = True
        self.trigger_excited_time = time
        self.trigger_excited_momentum = momentum

    def emit(self, rng):
        """
        Emit a photon at a random uniform direction
        :return: direction vector of emission
        """
        emiss_dir = uniform_random_direction(rng)
        self.momentum -= emiss_dir * self.excited_energy / c
        self.excited = False
        self.excited_time = -1
        return emiss_dir

    def step(self, time, dt, rng):
        """
        Move the particle according to the current configuration and time `dt`
        :param time: current time of the simulation
        :param dt: timespan of the step
        :param rng: RNG to use for the random functions
        """
        prev_v = self.v
        emit_v_change_rel_t = 0
        absorb_v_change_rel_t = dt
        absorb_momentum_change = 0
        emit_time = self.emit_time
        emiss = None

        if self.excited and (time >= emit_time):
            emiss_dir = self.emit(rng)
            emit_v_change_rel_t = dt - (time - emit_time)
            emiss = (self.coords + prev_v * emit_v_change_rel_t, emiss_dir)
        if self.trigger_excited:
            absorb_v_change_rel_t = dt - (time - self.trigger_excited_time)
            absorb_momentum_change = self.trigger_excited_momentum
            self.excited = True
            self.excited_time = self.trigger_excited_time
            self.trigger_excited = False
            self.trigger_excited_time = -1
            self.trigger_excited_momentum = -1

        self.coord_cache = np.roll(self.coord_cache, 1, 0)
        self.coord_cache[0] = self.coords

        # Apply movement before possible absorption
        self.coords = self.coords + prev_v * emit_v_change_rel_t + self.v * (absorb_v_change_rel_t - emit_v_change_rel_t)

        # Apply movement after possible absorption
        self.momentum += absorb_momentum_change
        self.coords = self.coords + self.v * (dt - absorb_v_change_rel_t)
        return emiss
