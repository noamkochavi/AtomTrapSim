# external imports
import numpy as np

# internal imports
from constants import *


class Sim:
    """
    Generator object representing the entire simulation.
    Yields coordinates of particles in discrete moments in time
    """
    def __init__(self, dt, particles, lenses, lasers=None):
        """
        Create a new simulation
        :param dt: time between simulation frames
        :param particles: list of Particle objects
        :param lenses: list of Lens objects
        :param lasers: list of objects from laser.py
        """
        self.time = 0
        self.dt = dt
        self.particles = particles
        self.lasers = lasers if lasers is not None else []
        self.lenses = lenses

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        self.time += self.dt

        # Stop when process ends or when there are no particles left
        if len(self.particles) == 0 or len(self.lasers) == 0:
            raise StopIteration()

        # Apply interactions between particles
        if len(self.particles) > 1:
            # TODO: add interactions between particles to simulation
            pass

        # Apply particle-laser interactions
        for laser in self.lasers:
            laser.apply(self.particles, self.time, self.dt)

        # Advance particles in the simulation
        emissions = []
        for p in self.particles:
            emiss = p.step(self.time, self.dt)
            if emiss:
                emissions.append(emiss)

        # Record all emissions
        # TODO: count emission events, how many caught by lens out of those
        for em_src, em_dir in emissions:
            for lens in self.lenses:
                lens.record(em_src, em_dir)

        # Remove irrelevant particles
        parts_copy = [x for x in self.particles]
        for p in parts_copy:
            if (p.coords < [X_MIN, Y_MIN, Z_MIN]).any() or (p.coords > [X_MAX, Y_MAX, Z_MAX]).any():
                self.particles.remove(p)

        # Remove irrelevant lasers
        las_copy = [x for x in self.lasers]
        for l in las_copy:
            if self.time >= l.end_time:
                self.lasers.remove(l)

        return self
