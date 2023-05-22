# external imports
from scipy.constants import hbar, c
import numpy as np

# internal imports
from particle import Particle


class PulsingLaser:
    """
    Object representing a pulsing laser that applies momentum on all particles
    in the simulation in a set time frequency
    """
    def __init__(self, direction, k, t_on, t_off, time_offset=0):
        """
        Create a new pulsing laser
        :param direction: np.array representing the directional vector of the laser (normalized)
        :param k: angular wavenumber of the laser
        :param t_on: Number of simulation frames for which to activate a single pulse
        :param t_off: Number of simulation frames between pulses
        :param time_offset: Number of simulation frames before the first pulse
        """
        self.direction = direction / np.linalg.norm(direction)
        self.k = k
        self.t_on = t_on
        self.t_off = t_off
        self.time_offset = time_offset
        self.active = False

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

    def apply(self, particles, time):
        """
        Apply external force on the given particle
        :param particles: a list of Particle objects
        :param time: current time frame of the simulation
        """
        if time < self.time_offset:
            self.active = False
            return
        if (time - self.time_offset) % (self.t_off + self.t_on) < self.t_on:
            for p in particles:
                # TODO: add probabilities
                p.momentum += self.momentum
                if p.energy == 0 and p.excited_energy == self.energy:
                    p.energy = self.energy
            self.active = True
            return
        self.active = False
