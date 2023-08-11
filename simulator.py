# external imports
from datetime import datetime
import numpy as np

# internal imports
from constants import *


class Sim:
    """
    Generator object representing the entire simulation.
    Yields coordinates of particles in discrete moments in time
    """
    def __init__(self, dt, particles, lenses, lasers, terminate_time=None, fast=True, destruct_particles=True, debug=False, seed=None):
        """
        Create a new simulation
        :param id: identification of the particle (for data collecting)
        :param dt: time between simulation frames
        :param particles: list of Particle objects
        :param lenses: list of Lens objects
        :param lasers: list of objects from laser.py
        :param terminate_time: time at which the simulation is terminated (in seconds).
                               If None, the simulation terminates at the absence of Particles and/or Lasers
        :param fast: True for faster, less precise mode.
                     Assumes all the time settings for the lasers are divisible by dt
        :param destruct_particles: if True, destructs particles that leave the focus area
        :param debug: logs events in LOGGING_FILE when True
        :param seed: seed for the simulation's RNG. None for default
        """
        self.time = 0
        self.dt = dt
        self.particles = particles
        self.lasers = lasers if lasers is not None else []
        self.lenses = lenses
        self.fast = fast
        self.destruct_particles = destruct_particles
        self.debug = debug
        self.seed = seed
        self.__rng = np.random.default_rng(seed)
        self.__logging_file = None

        if terminate_time is None:
            self.terminate_time = max([l.end_time for l in self.lasers])
        else:
            self.terminate_time = terminate_time

        if debug:
            self.__logging_file = open(LOGGING_FILE, "w")
            self.__logging_file.write("sim_time,action,info\n")

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        self.time += self.dt

        # Stop when process ends or when there are no particles left
        if self.time > self.terminate_time or len(self.particles) == 0 or len(self.lasers) == 0:
            if self.debug:
                self.__logging_file.write(f",".join([str(self.time),
                                                     SIM_END, ""]) + "\n")
                self.__logging_file.close()
            raise StopIteration()

        # Apply interactions between particles
        if len(self.particles) > 1:
            # TODO: add interactions between particles to simulation
            pass

        # Apply particle-laser interactions
        for laser in self.lasers:
            laser.apply(self.particles, self.time, self.dt, self.fast, self.__rng)

        # Advance particles in the simulation
        emissions = []
        for p in self.particles:
            emiss = p.step(self.time, self.dt, self.__rng)
            if emiss:
                emissions.append(emiss)
                if self.debug:
                    self.__logging_file.write(f",".join([str(self.time),
                         EMISS_EVENT,
                         f"particle_id={p.id}&source_coords={' '.join([format(n,'.3e') for n in emiss[0]])}&direction={' '.join([format(n,'.3e') for n in emiss[1]])}"]) + "\n")

        # Record all emissions
        # TODO: count emission events, how many caught by lens out of those
        for em_src, em_dir in emissions:
            # TODO: error in capturing?
            for lens in self.lenses:
                lens.record(em_src, em_dir)

        # Remove irrelevant particles
        if self.destruct_particles:
            parts_copy = [x for x in self.particles]
            for p in parts_copy:
                if (p.coords < [X_MIN, Y_MIN, Z_MIN]).any() or (p.coords > [X_MAX, Y_MAX, Z_MAX]).any():
                    if self.debug:
                        self.__logging_file.write(f",".join([str(self.time),
                             DELETE_PARTICLE,
                             f"particle_id={p.id}&coords={' '.join([format(n,'.3e') for n in p.coords])}"]) + "\n")
                    self.particles.remove(p)

        # Remove irrelevant lasers
        las_copy = [x for x in self.lasers]
        for l in las_copy:
            if self.time >= l.end_time:
                if self.debug:
                    self.__logging_file.write(f",".join([str(self.time),
                         DELETE_LASER,
                         f"direction={' '.join([format(n,'.3e') for n in l.direction])}"]) + "\n")
                self.lasers.remove(l)

        return self
