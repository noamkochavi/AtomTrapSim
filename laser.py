# external imports
from scipy.constants import hbar, c, pi
import numpy as np

# internal imports
from constants import *


class PulsingLaser:
    """
    Object representing a pulsing laser that applies momentum on all particles
    in the simulation in a set time frequency
    """
    def __init__(self, direction, k, n_pulses, t_on, t_off, time_offset=0):
        """
        Create a new pulsing laser
        :param direction: np.array representing the directional vector of the laser (normalized)
        :param k: angular wave-number of the laser
        :param t_on: Number of simulation frames for which to activate a single pulse
        :param t_off: Number of simulation frames between pulses
        :param time_offset: Number of simulation frames before the first pulse
        """
        self.direction = direction / np.linalg.norm(direction)
        self.k = k
        self.n_pulses = n_pulses
        self.t_on = t_on
        self.t_off = t_off
        self.time_offset = time_offset
        self.active = False

    @property
    def end_time(self):
        """
        Time at which the laser deactivates
        """
        return self.time_offset + (self.t_on + self.t_off) * self.n_pulses

    def _delta(self, t):
        """
        Delta function, represents the status of the laser at the given time
        :param t: given time
        :return: 1 if on, 0 if off
        """
        return int((self.time_offset <= t <= self.end_time)
                   and
                   ((t - self.time_offset) % (self.t_on + self.t_on) < self.t_on))

    @property
    def momentum(self):
        """
        The momentum that the photons of the laser add to an atom at interaction
        """
        return self.direction * hbar * self.k

    @property
    def energy(self):
        """
        The energy of a photon in the laser beam
        """
        return hbar * self.k * c

    def apply(self, particles, time, dt):
        """
        Apply external force on the given particle
        :param particles: a list of Particle objects
        :param time: current time frame of the simulation
        :param dt: time between this run and the previous
        """
        x = np.linspace(time-dt, time, INTEGRATION_RESOLUTION, endpoint=False)
        y = np.vectorize(self._delta)(x)
        on_time = sum(y) * dt / INTEGRATION_RESOLUTION
        del x
        del y

        if on_time > 0:
            for p in particles:
                # IMPORTANT! assumes dt*(scattering rate) << 1
                prob_sc = dt * NATURAL_LINEWIDTH * pi
                rand_n = np.random.rand()
                if rand_n <= prob_sc:
                    # TODO: randomize time of absorption out of dt?
                    p.absorb(self.momentum, time)
        self.active = bool(self._delta(time))
