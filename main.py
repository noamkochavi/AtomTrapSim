# external imports
import matplotlib.pyplot as plt
import numpy as np
import time

import scipy.integrate

# internal imports
from laser import PulsingLaser
from particle import Particle
from simulator import Sim
from constants import *
from lens import Lens


# code
v0 = np.array([0.1, 0, 0])
left_laser = PulsingLaser(direction=RIGHT, k=LASER_K,
                          t_on=LASER_PULSE_TIME, t_off=LASER_PULSE_TIME, time_offset=0)
right_laser = PulsingLaser(direction=LEFT, k=LASER_K,
                           t_on=LASER_PULSE_TIME, t_off=LASER_PULSE_TIME, time_offset=LASER_PULSE_TIME)
p = Particle(mass=1, excited_energy=left_laser.energy, start_v=v0)
ph_lens = Lens(shape=(LENS_PIXELS_H, LENS_PIXELS_W), pixel_dim=1, z_loc=Z_MAX)
sim = Sim(particles=[p], lenses=[ph_lens], lasers=[left_laser, right_laser])

cache = np.zeros((len(sim.particles), CACHE_SIZE, 3), float)
for sim_image in sim:
    plt.grid()
    plt.xlim([X_MIN, X_MAX])
    plt.ylim([Y_MIN, Y_MAX])
    plt.xticks(np.arange(X_MIN, X_MAX + 1, 1))
    plt.yticks(np.arange(Y_MIN, Y_MAX + 1, 1))

    for l in sim_image.lasers:
        if l.active:
            plt.arrow(0, 0, l.direction[0] * 10, l.direction[1] * 10, color="r", width=0.25)

    for part in sim_image.particles:
        x, y, z = part.coords
        plt.plot([x], [y], "bo")
        plt.plot([x] + [xc[0] for xc in part.coord_cache], [y] + [xc[1] for xc in part.coord_cache], "k")

    plt.show(block=False)
    time.sleep(0.1)
    plt.gcf().canvas.flush_events()

plt.imshow(ph_lens.image)
plt.show()
