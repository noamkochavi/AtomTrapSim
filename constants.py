# external imports
import numpy as np

# constants
TIME_RESOLUTION = 5e-9  # sec
NATURAL_LINEWIDTH = 6.035e6  # Hz
LASER_PULSE_TIME = 1e-6  # sec
PULSES_PER_LASER = 40
LASER_K = 1  # rad # TODO: check value
LENS_PIXELS_DIM = 30  # pix
LENS_FOCUS_AREA = 1e-2  # m^2
INTEGRATION_RESOLUTION = 10000

X_MIN = -10
X_MAX = 10
Y_MIN = -10
Y_MAX = 10
Z_MIN = -10
Z_MAX = 10

LEFT = np.array([-1, 0, 0])
RIGHT = np.array([1, 0, 0])
UP = np.array([0, 1, 0])
DOWN = np.array([0, -1, 0])
IN = np.array([0, 0, -1])
OUT = np.array([0, 0, 1])

CACHE_SIZE = 5
