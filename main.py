# external imports
from numpy.random import SeedSequence
from scipy.ndimage import gaussian_filter
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
    dir_name, idx, num_part, seed, loc_data, destruct_particles = args
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
                  start_coords=DIST_CENTER + np.append(uniform_random_direction(rng, dim=2) * rng.random() * MAX_X0, 0),
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
        if last_time + 1e-6 < sim.time:
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
        pd.DataFrame(loc_dicts).to_csv(os.path.join("results", dir_name, "loc", f"{idx}_noa_{num_part}.csv"), index=False)

    plt.imshow(ph_lens.image, origin="lower", vmin=0, vmax=IMAGE_COUNT_LIMIT)
    plt.axis("off")
    plt.savefig(os.path.join("results", dir_name, "res", f"{idx}_noa_{num_part}.png"), bbox_inches='tight', pad_inches=0)

    df = pd.DataFrame(ph_lens.image)
    df.to_csv(os.path.join("results", dir_name, "res", f"{idx}_noa_{num_part}.csv"), index=False, header=False)

    logging.info(f"run_sim(idx={idx}, n_p={num_part}): end")

# TODO: Amplification for camera


def debug_trial(dir_name):
    os.mkdir(os.path.join("results", dir_name))
    os.mkdir(os.path.join("results", dir_name, "res"))
    os.mkdir(os.path.join("results", dir_name, "loc"))
    run_sim((dir_name, -1, 1, None, True, True), True)


def run_trials(dir_name, n_images_per_noa, noas, loc_data=False, destruct_particles=True):
    os.mkdir(os.path.join("results", dir_name))
    os.mkdir(os.path.join("results", dir_name, "res"))
    if loc_data:
        os.mkdir(os.path.join("results", dir_name, "loc"))

    noas = list(noas)
    n_images = n_images_per_noa * len(noas)
    ss = SeedSequence()
    seeds = ss.spawn(n_images)
    pool = multiprocessing.Pool()
    args = zip([dir_name]*n_images, range(n_images), np.sort(noas * n_images_per_noa), seeds, [loc_data]*n_images, [destruct_particles] * n_images)
    pool.map(run_sim, args)


def fuzz_results(res_dir, sigma):
    fuzzdir = os.path.join("results", res_dir, "fuzzres")
    os.mkdir(fuzzdir)

    ps = glob(os.path.join("results", res_dir, "res", "*noa*.csv"))
    for p in ps:
        filename = os.path.splitext(os.path.basename(p))[0]

        df = pd.read_csv(p, header=None)
        fuzzdata = gaussian_filter(df.to_numpy(dtype="float"), sigma=sigma)

        pd.DataFrame(fuzzdata).to_csv(os.path.join(fuzzdir, filename + ".csv"), index=False, header=False)

        plt.imshow(fuzzdata, origin="lower", vmin=0, vmax=IMAGE_COUNT_LIMIT)
        plt.axis("off")
        plt.savefig(os.path.join(fuzzdir, filename + ".png"), bbox_inches='tight',
                    pad_inches=0)


def average_results(res_dir, fuzz=False):
    prefix = ["", "fuzz"][fuzz]
    os.mkdir(os.path.join("results", res_dir, prefix + "mean"))
    ps = glob(os.path.join("results", res_dir, prefix + "res", "*noa*.csv"))
    noas = {int(os.path.splitext(os.path.basename(p))[0].split("_")[2]) for p in ps}
    for n in noas:
        ps = glob(os.path.join("results", res_dir, prefix + "res", f"*noa_{n}.csv"))
        sum_arr = np.zeros((LENS_PIXELS_DIM, LENS_PIXELS_DIM))
        for p in ps:
            df = pd.read_csv(p, header=None)
            sum_arr += df.to_numpy()
        sum_arr /= len(ps)

        df = pd.DataFrame(sum_arr)
        df.to_csv(os.path.join("results", res_dir, prefix+"mean", f"noa_{n}.csv"), index=False, header=False)

        plt.imshow(sum_arr, origin="lower", vmin=0, vmax=IMAGE_COUNT_LIMIT)
        plt.axis("off")
        plt.savefig(os.path.join("results", res_dir, prefix+"mean", f"noa_{n}.png"), bbox_inches='tight', pad_inches=0)

    ps = glob(os.path.join("results", res_dir, prefix + "res", f"*noa*.csv"))
    sum_arr = np.zeros((LENS_PIXELS_DIM, LENS_PIXELS_DIM))
    for p in ps:
        df = pd.read_csv(p, header=None)
        sum_arr += df.to_numpy()
    sum_arr /= len(ps)

    df = pd.DataFrame(sum_arr)
    df.to_csv(os.path.join("results", res_dir, prefix + "mean", f"all.csv"), index=False, header=False)

    plt.imshow(sum_arr, origin="lower", vmin=0, vmax=IMAGE_COUNT_LIMIT)
    plt.axis("off")
    plt.savefig(os.path.join("results", res_dir, prefix + "mean", f"all.png"), bbox_inches='tight', pad_inches=0)


def main():
    # debug_trial("debug")
    # run_trials("311010_offcenter", 300, range(1, 5), loc_data=False, destruct_particles=True)
    # fuzz_results("311010_offcenter", sigma=1)
    average_results("311010_offcenter")
    average_results("311010_offcenter", fuzz=True)


if __name__ == "__main__":
    main()
