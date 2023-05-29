# external imports
import numpy as np

# constants
TIME_RESOLUTION = 5e-9  # sec
NATURAL_LINEWIDTH = 6.035e6  # Hz
LASER_PULSE_TIME = 1e-6  # sec
PULSES_PER_LASER = 40
LASER_WAVELENGTH = 1.128e-02  # TODO: check value
LASER_K = 2 * np.pi / LASER_WAVELENGTH  # rad # TODO: check value
PARTICLE_MASS = 6.64e-26  # kg (potassium-40)
LENS_PIXELS_DIM = 30  # pix
LENS_FOCUS_AREA = 1e-9  # m^2
INTEGRATION_RESOLUTION = 10000

X_MIN = -LENS_FOCUS_AREA
X_MAX = LENS_FOCUS_AREA
Y_MIN = -LENS_FOCUS_AREA
Y_MAX = LENS_FOCUS_AREA
Z_MIN = -LENS_FOCUS_AREA
Z_MAX = LENS_FOCUS_AREA
X_TICKS = 10
Y_TICKS = 10

LEFT = np.array([-1, 0, 0])
RIGHT = np.array([1, 0, 0])
UP = np.array([0, 1, 0])
DOWN = np.array([0, -1, 0])
IN = np.array([0, 0, -1])
OUT = np.array([0, 0, 1])

CACHE_SIZE = 5
