# external imports
import numpy as np

# internal imports
from math_functions import line_zplane_intersect
from constants import *


class Lens:
    def __init__(self, image_dim, focus_area, z_loc):
        """
        Create a new lens that records photon emissions
        :param image_dim: length and width in pixels of the resulting image
        :param focus_area: real area represented by the image
        :param z_loc: relative z-axis location of the lens (directed at the xy plane)
        """
        self.image = np.zeros((image_dim, image_dim), dtype="int")
        self.focus_area = focus_area
        self.z_loc = z_loc

    @property
    def pix_len(self):
        """
        Side length of the area represented by a pixel, in meters
        """
        return np.sqrt(self.focus_area) / len(self.image)

    @property
    def max_x(self):
        """
        Maximum x-axis location of the lens plane in the simulation
        """
        return (self.image.shape[0] * self.pix_len) / 2

    @property
    def min_x(self):
        """
        Minimum x-axis location of the lens plane in the simulation
        """
        return -self.max_x

    @property
    def max_y(self):
        """
        Maximum y-axis location of the lens plane in the simulation
        """
        return (self.image.shape[1] * self.pix_len) / 2

    @property
    def min_y(self):
        """
        Minimum y-axis location of the lens plane in the simulation
        """
        return -self.max_y

    def record(self, source_coords, direction_vector):
        """
        Record a photon, released from `source_coords` with momentum `momentum_vector`, in the image
        :param source_coords: coordinates from which the photon was emitted
        :param direction_vector: direction vector of the photon emission
        """
        if np.sign(self.z_loc - source_coords[2]) == np.sign(direction_vector[2]):
            # if photon is emitted in the z-axis direction of the lens (and not the opposite)
            inter_coords = line_zplane_intersect(self.z_loc, source_coords, direction_vector)
            if self.min_x <= inter_coords[0] <= self.max_x and \
                    self.min_y <= inter_coords[1] <= self.max_y and \
                    self.min_x <= source_coords[0] <= self.max_x and \
                    self.min_y <= source_coords[1] <= self.max_y:
                # If photon hits lens, record
                pix_coords = ((source_coords + [self.max_x, self.max_y, 0]) / self.pix_len).astype(int)
                self.image[pix_coords[1], pix_coords[0]] += 1
