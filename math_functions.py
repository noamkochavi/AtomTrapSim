# external imports
import numpy as np


def line_zplane_intersect(plane_z, line_origin, line_dir):
    """
    Calculate the intersection point of a line, starting at `line_origin` in direction `line_dir`, with plane z=`plane_z`
    :param plane_z: z coordinate of the plane
    :param line_origin: starting point (x,y,z) of the line (np.array)
    :param line_dir: direction vector (x,y,z) of the line (np.array)
    :return: (x,y,z) of intersection point (np.array)
    """
    t = (plane_z - line_origin[2]) / line_dir[2]  # z of intersection = z of plane
    return line_origin + t * line_dir  # line = p0+t*v


def uniform_random_direction(rng, dim=3):
    """
    Generate a uniformly random direction vector in 3d space
    """
    vec = rng.random(dim) - 0.5
    return vec / np.linalg.norm(vec)


def gauss2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                                    + c*((y-yo)**2)))
    return g.ravel()


def gauss2d_simple(xy, amplitude, xo, yo, sigma):
    x, y = xy
    g = amplitude*np.exp(-((x-xo)**2+(y-yo)**2)/(2*sigma**2))
    return g.ravel()
