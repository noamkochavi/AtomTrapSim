# external imports
from numpy.random import SeedSequence
import matplotlib.pyplot as plt
import multiprocessing
from glob import glob
import pandas as pd
import logging
import os


# internal imports
from math_functions import uniform_random_direction
from laser import PulsingLaser
from particle import Particle
from simulator import Sim
from constants import *
from lens import Lens

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO,
                    datefmt="%H:%M:%S")


# simulation function
# TODO: Use exact python variables
def run_sim(args, debug=False):
    idx, num_part, seed, loc_data, destruct_particles = args
    rng = np.random.default_rng(seed)
    ssk = seed.spawn_key[0] if seed is not None else None
    logging.info(f"run_sim(idx={idx}, n_p={num_part}): start. seed={ssk}")

    left_laser = PulsingLaser(direction=RIGHT, k=LASER_K, n_pulses=PULSES_PER_LASER,
                              t_on=LASER_PULSE_TIME, t_off=LASER_PULSE_TIME,
                              time_offset=-0.5*LASER_PULSE_TIME)
    right_laser = PulsingLaser(direction=LEFT, k=LASER_K, n_pulses=PULSES_PER_LASER,
                               t_on=LASER_PULSE_TIME, t_off=LASER_PULSE_TIME,
                               time_offset=0.5*LASER_PULSE_TIME)
    p = [Particle(id=i, mass=PARTICLE_MASS,
                  excited_energy=left_laser.energy, excited_lifetime=EXCITED_LIFETIME,
                  start_coords=np.append(uniform_random_direction(rng, dim=2) * rng.random() * MAX_X0, 0),
                  start_v=np.append(uniform_random_direction(rng, dim=2) * rng.random() * MAX_V0, 0))
         for i in range(num_part)]
    ph_lens = Lens(image_dim=LENS_PIXELS_DIM,
                   focus_area=LENS_FOCUS_SIDE_LENGTH ** 2,
                   z_loc=Z_MAX)
    sim = Sim(dt=TIME_RESOLUTION,
              particles=p, lenses=[ph_lens], lasers=[left_laser, right_laser], terminate_time=TERMINATE_TIME,
              seed=seed, destruct_particles=destruct_particles, debug=debug)

    last_time = sim.time - 1
    loc_dicts = []
    for sim_image in sim:
        if last_time + 5e-7 < sim.time:
            last_time = sim.time
            if debug:
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
            if loc_data:
                for part in sim_image.particles:
                    loc_dicts.append({"t": sim.time, "id": part.id,
                                      "x": part.coords[0], "y": part.coords[1], "z": part.coords[2],
                                      "r": np.linalg.norm(part.coords)})

    if loc_data:
        pd.DataFrame(loc_dicts).to_csv(f"results\\loc_{idx}_noa_{num_part}.csv", index=False)

    plt.imshow(ph_lens.image, origin="lower", vmin=0, vmax=IMAGE_COUNT_LIMIT)
    plt.axis("off")
    plt.savefig(f"results\\res_{idx}_noa_{num_part}.png", bbox_inches='tight', pad_inches=0)

    df = pd.DataFrame(ph_lens.image)
    df.to_csv(f"results\\res_{idx}_noa_{num_part}.csv", index=False, header=False)

    logging.info(f"run_sim(idx={idx}, n_p={num_part}): end")

# TODO: Quantum efficiency & Amplification for camera


def debug_trial():
    run_sim((-1, 1, None, True, True), True)


def run_trials(n_images_per_noa, noas, loc_data=False, destruct_particles=True):
    noas = list(noas)
    n_images = n_images_per_noa * len(noas)
    ss = SeedSequence()
    seeds = ss.spawn(n_images)
    pool = multiprocessing.Pool()
    args = zip(range(n_images), np.sort(noas * n_images_per_noa), seeds, [loc_data]*n_images, [destruct_particles] * n_images)
    pool.map(run_sim, args)


def average_results(res_dir):
    ps = glob(os.path.join(res_dir, "res*noa*.csv"))
    noas = {int(os.path.splitext(os.path.basename(p))[0].split("_")[3]) for p in ps}
    for n in noas:
        ps = glob(os.path.join(res_dir, f"res*noa_{n}.csv"))
        sum_arr = np.zeros((LENS_PIXELS_DIM, LENS_PIXELS_DIM))
        for p in ps:
            df = pd.read_csv(p, header=None)
            sum_arr += df.to_numpy()
        sum_arr /= len(ps)

        df = pd.DataFrame(sum_arr)
        df.to_csv(os.path.join(res_dir, f"mean_noa_{n}.csv"), index=False, header=False)

        plt.imshow(sum_arr, origin="lower", vmin=0, vmax=IMAGE_COUNT_LIMIT)
        plt.axis("off")
        plt.savefig(os.path.join(res_dir, f"mean_noa_{n}.png"), bbox_inches='tight', pad_inches=0)


def main():
    debug_trial()
    # run_trials(500, range(3, 6), loc_data=True, destruct_particles=False)
    # average_results("results_05aug23_huge_halfoffset")


if __name__ == "__main__":
    main()
