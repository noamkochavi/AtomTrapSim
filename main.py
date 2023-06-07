# external imports
import matplotlib.pyplot as plt
import threading
import logging


# internal imports
from laser import PulsingLaser
from particle import Particle
from simulator import Sim
from constants import *
from lens import Lens

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO,
                    datefmt="%H:%M:%S")


# simulation function
# TODO: Use exact python variables
def run_sim(idx):
    logging.info(f"run_sim({idx}): start")
    v0 = np.array([1e-7, 0, 0])
    left_laser = PulsingLaser(direction=RIGHT, k=LASER_K,
                              n_pulses=PULSES_PER_LASER, t_on=LASER_PULSE_TIME, t_off=LASER_PULSE_TIME, time_offset=0)
    right_laser = PulsingLaser(direction=LEFT, k=LASER_K,
                               n_pulses=PULSES_PER_LASER, t_on=LASER_PULSE_TIME, t_off=LASER_PULSE_TIME, time_offset=LASER_PULSE_TIME)
    p = Particle(mass=PARTICLE_MASS, excited_energy=left_laser.energy, excited_lifetime=EXCITED_LIFETIME, start_v=v0)
    ph_lens = Lens(image_dim=LENS_PIXELS_DIM, focus_area=LENS_FOCUS_SIDE_LENGTH ** 2, z_loc=Z_MAX)
    sim = Sim(dt=TIME_RESOLUTION, particles=[p], lenses=[ph_lens], lasers=[left_laser, right_laser])

    last_time = sim.time - 1
    for sim_image in sim:
        if DEBUG and (last_time + 1e-6 < sim.time):
            last_time = sim.time
            plt.grid(zorder=-1)
            plt.xlim([X_MIN, X_MAX])
            plt.ylim([Y_MIN, Y_MAX])
            tick_len = (X_MAX - X_MIN)/X_TICKS
            tick_arr = np.arange(X_MIN, X_MAX + tick_len, tick_len)
            plt.xticks(tick_arr)
            plt.yticks(tick_arr)

            for l in sim_image.lasers:
                if l.active:
                    x, y = np.meshgrid(tick_arr + tick_len / 2, tick_arr + tick_len / 2)
                    plt.quiver(x, y, l.direction[0], l.direction[1], color="r")

            for part in sim_image.particles:
                x, y, z = part.coords
                plt.plot([x], [y], "bo")
                plt.plot([x] + [xc[0] for xc in part.coord_cache], [y] + [xc[1] for xc in part.coord_cache], "k")

            plt.show(block=False)
            plt.gcf().canvas.flush_events()

    plt.imshow(ph_lens.image, origin="lower")
    plt.savefig(f"results\\res_{idx}.png")

    logging.info(f"run_sim({idx}): end")


# run trials
n_threads = 10
threads = [threading.Thread(target=run_sim, args=(i,)) for i in range(n_threads)]
for t in threads:
    t.start()
for t in threads:
    t.join()
