# external imports
import numpy as np

# constants
TIME_RESOLUTION = 5e-9  # sec
NATURAL_LINEWIDTH = 6.035e6  # Hz
LASER_PULSE_TIME = 1e-6  # sec
PULSES_PER_LASER = 40
TERMINATE_TIME = 80e-6  # sec
LASER_WAVELENGTH = 766.7e-9  # m
EXCITED_LIFETIME = 26.37e-9  # s
LASER_K = 2 * np.pi / LASER_WAVELENGTH  # rad
PARTICLE_MASS = 6.64e-26  # kg (potassium-40)
LENS_PIXELS_DIM = 30  # pix
LENS_FOCUS_SIDE_LENGTH = 30e-6  # m
OUTPUT_IMAGE_MAX_VAL = 50  # photons per pixel
ADJ_OUTPUT_MAX_VAL = 3800  # photons per pixel
MAX_X0 = 0e-6  # m
MAX_V0 = 0  # m/s
DIST_CENTER_X = -0.48e-6  # m
DIST_CENTER_Y = 0.16e-6  # m
DIST_CENTER = np.array([DIST_CENTER_X, DIST_CENTER_Y, 0])
QUANTUM_EFFICIENCY = .85  # %  # TODO: Estimated from graph in manual, get exact QE. Andor iXon Ultra 888
AMPLIFICATION = 950 # TODO: According to Yanay. How to use?
INTEGRATION_RESOLUTION = 10000

X_MIN = -LENS_FOCUS_SIDE_LENGTH / 2
X_MAX = LENS_FOCUS_SIDE_LENGTH / 2
Y_MIN = -LENS_FOCUS_SIDE_LENGTH / 2
Y_MAX = LENS_FOCUS_SIDE_LENGTH / 2
Z_MIN = -LENS_FOCUS_SIDE_LENGTH / 2  # TODO: Calculate Depth of Field, possibly better precision
Z_MAX = LENS_FOCUS_SIDE_LENGTH / 2
X_TICKS = 10
Y_TICKS = 10

LEFT = np.array([-1, 0, 0])
RIGHT = np.array([1, 0, 0])
UP = np.array([0, 1, 0])
DOWN = np.array([0, -1, 0])
IN = np.array([0, 0, -1])
OUT = np.array([0, 0, 1])

CACHE_SIZE = 5
LOGGING_FILE = "sim_log.csv"
SIM_END = "SIM_END"
EMISS_EVENT = "EMISS_EVENT"
DELETE_PARTICLE = "DELETE_PARTICLE"
DELETE_LASER = "DELETE_LASER"
