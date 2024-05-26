# external imports
from scipy.ndimage import gaussian_filter
from numpy.random import SeedSequence
import matplotlib.pyplot as plt
from scipy import stats
import multiprocessing
from enum import Enum
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


# Enum of image types for averaging function
class AvgMode(Enum):
    Regular = 0
    Fuzzed = 1
    Amplified = 2
    Adjusted = 3


# Enum of noise types
class NoiseType(Enum):
    Constant = 0
    BetaDist = 1


def run_sim(args, debug=False):
    """Main simulation function"""

    # Initiate arguments
    dir_name, idx, num_part, seed, loc_data, destruct_particles = args
    rng = np.random.default_rng(seed)
    ssk = seed.spawn_key[0] if seed is not None else None
    logging.info(f"run_sim(idx={idx}, n_p={num_part}): start. seed={ssk}")

    # Configure simulated lab - lasers, particles and lens
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

    # Create the simulation
    sim = Sim(dt=TIME_RESOLUTION,
              particles=p, lenses=[ph_lens], lasers=[left_laser, right_laser], terminate_time=TERMINATE_TIME,
              seed=seed, destruct_particles=destruct_particles, debug=debug)

    # Iterate over simulation timeframes
    last_time = sim.time - 1
    loc_dicts = []
    for sim_image in sim:
        if last_time + 1e-6 < sim.time:
            last_time = sim.time

            # When debugging, plot status of simulation every 1e-6 sec
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

            # Save location data of the particles every 1e-6 sec
            if loc_data:
                for part in sim_image.particles:
                    loc_dicts.append({"t": sim.time, "id": part.id,
                                      "x": part.coords[0], "y": part.coords[1], "z": part.coords[2],
                                      "r": np.linalg.norm(part.coords)})

    # Save location data
    if loc_data:
        pd.DataFrame(loc_dicts).to_csv(os.path.join("results", dir_name, "loc", f"{idx}_noa_{num_part}.csv"), index=False)

    # Save image recorded in lens (png)
    plt.imshow(ph_lens.image, origin="lower", vmin=0, vmax=OUTPUT_IMAGE_MAX_VAL)
    plt.axis("off")
    plt.savefig(os.path.join("results", dir_name, "res", f"{idx}_noa_{num_part}.png"), bbox_inches='tight', pad_inches=0)

    # Save image recorded in lens (csv)
    df = pd.DataFrame(ph_lens.image)
    df.to_csv(os.path.join("results", dir_name, "res", f"{idx}_noa_{num_part}.csv"), index=False, header=False)

    logging.info(f"run_sim(idx={idx}, n_p={num_part}): end")

# TODO: Amplification for camera


def debug_trial(dir_name):
    """Run a single trial in debug mode, save location data"""
    os.mkdir(os.path.join("results", dir_name))
    os.mkdir(os.path.join("results", dir_name, "res"))
    os.mkdir(os.path.join("results", dir_name, "loc"))
    run_sim((dir_name, -1, 1, None, True, True), True)


def run_trials(dir_name, n_images_per_noa, noas, loc_data=False, destruct_particles=True):
    """Run `n_images_per_noa` trial for every noa (number of atoms) value in the list `noas`"""
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


def fuzz_result(args):
    """Fuzz a single result - apply gaussian filter with given `sigma` on given result image"""
    fuzzdir, path, sigma = args
    filename = os.path.splitext(os.path.basename(path))[0]
    logging.info(f"fuzz_result({filename}): start")

    df = pd.read_csv(path, header=None)
    fuzzdata = gaussian_filter(df.to_numpy(dtype="float"), sigma=sigma)

    pd.DataFrame(fuzzdata).to_csv(os.path.join(fuzzdir, filename + ".csv"), index=False, header=False)

    plt.imshow(fuzzdata, origin="lower", vmin=0, vmax=OUTPUT_IMAGE_MAX_VAL)
    plt.axis("off")
    plt.savefig(os.path.join(fuzzdir, filename + ".png"), bbox_inches='tight',
                pad_inches=0)
    logging.info(f"fuzz_result({filename}): end")


def fuzz_results(res_dir, sigma):
    """Fuzz all results in directory - apply gaussian filter with given `sigma` on all results"""
    fuzzdir = os.path.join("results", res_dir, "fuzzres")
    os.mkdir(fuzzdir)

    ps = glob(os.path.join("results", res_dir, "res", "*noa*.csv"))
    pool = multiprocessing.Pool()
    args = zip([fuzzdir] * len(ps), ps, [sigma] * len(ps))
    pool.map(fuzz_result, args)


def amp_result(args):
    """Amplify a single result - multiply by the factor function and add offset to given result image"""
    basedir, path, factor_params, off = args
    filename = os.path.splitext(os.path.basename(path))[0]
    noa = int(filename.split('_')[-1])
    logging.info(f"amp_result({filename}): start")

    factor_func = lambda N, k, m, n: k*np.exp(-m*N)+n
    df = pd.read_csv(path, header=None)
    adj_data = factor_func(noa, *factor_params) * df.to_numpy(dtype="float") + off

    pd.DataFrame(adj_data).to_csv(os.path.join(basedir, filename + ".csv"), index=False, header=False)

    plt.imshow(adj_data, origin="lower", vmin=0, vmax=ADJ_OUTPUT_MAX_VAL)
    plt.axis("off")
    plt.savefig(os.path.join(basedir, filename + ".png"), bbox_inches='tight',
                pad_inches=0)
    logging.info(f"amp_result({filename}): end")


def amp_results(res_dir, factor_params, off):
    """Amplify all results in given directory - multiply by the factor function and add offset to given result images"""
    basedir = os.path.join("results", res_dir, "ampres")
    os.mkdir(basedir)

    ps = glob(os.path.join("results", res_dir, "fuzzres", "*noa*.csv"))

    args = zip([basedir] * len(ps), ps, [factor_params] * len(ps), [off] * len(ps))
    pool = multiprocessing.Pool()
    pool.map(amp_result, args)


def noise_result(args):
    """Add given noise to a single result"""
    basedir, path, noise = args
    filename = os.path.splitext(os.path.basename(path))[0]
    logging.info(f"noise_result({filename}): start")

    df = pd.read_csv(path, header=None)
    adj_data = df.to_numpy(dtype="float") + noise

    pd.DataFrame(adj_data).to_csv(os.path.join(basedir, filename + ".csv"), index=False, header=False)

    plt.imshow(adj_data, origin="lower", vmin=0, vmax=ADJ_OUTPUT_MAX_VAL)
    plt.axis("off")
    plt.savefig(os.path.join(basedir, filename + ".png"), bbox_inches='tight',
                pad_inches=0)
    logging.info(f"noise_result({filename}): end")


def noise_results(res_dir, noise_path, noise_type):
    """Add noise to all results in given directory. By NoiseType, can add a constant noise
    or a randomly Beta-distributed noise (according to given beta distribution parameters)"""
    basedir = os.path.join("results", res_dir, "adjres")
    os.mkdir(basedir)

    ps = glob(os.path.join("results", res_dir, "ampres", "*noa*.csv"))

    if noise_type == NoiseType.Constant:
        noise = pd.read_csv(noise_path, header=None).to_numpy()
        noises = [noise] * len(ps)
    elif noise_type == NoiseType.BetaDist:
        noise_beta_df = pd.read_csv(noise_path)
        noises = np.zeros((len(ps), LENS_PIXELS_DIM, LENS_PIXELS_DIM))
        for i in range(LENS_PIXELS_DIM):
            for j in range(LENS_PIXELS_DIM):
                _, _, f_alpha, f_beta, f_loc, f_scale = noise_beta_df[(noise_beta_df['i'] == i) & (noise_beta_df['j'] == j)] \
                                                        .to_numpy()[0]
                noises[:, i, j] = stats.beta.rvs(f_alpha, f_beta, loc=f_loc, scale=f_scale, size=len(ps))
        noises = list(noises)
    else:
        raise ValueError("Non-supported noise_type value. Use the NoiseType enum")

    args = zip([basedir] * len(ps), ps, noises)
    pool = multiprocessing.Pool()
    pool.map(noise_result, args)


def average_results(res_dir, mode=AvgMode.Regular):
    """Average all results in the directory with matching NoA values"""
    prefix = ["", "fuzz", "amp", "adj"][mode.value]
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

        plt.imshow(sum_arr, origin="lower", vmin=0,
                   vmax=[OUTPUT_IMAGE_MAX_VAL, ADJ_OUTPUT_MAX_VAL][mode == AvgMode.Adjusted])
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

    plt.imshow(sum_arr, origin="lower", vmin=0,
               vmax=[OUTPUT_IMAGE_MAX_VAL, ADJ_OUTPUT_MAX_VAL][mode == AvgMode.Adjusted])
    plt.axis("off")
    plt.savefig(os.path.join("results", res_dir, prefix + "mean", f"all.png"), bbox_inches='tight', pad_inches=0)


# Usage example
def main():
    # Run simulation
    # debug_trial("debug")
    run_trials("240102_large", 1670, range(1, 7), loc_data=False, destruct_particles=True)

    # Post-processing
    fuzz_results("240102_large", sigma=1)
    average_results("240102_large")
    average_results("240102_large", mode=AvgMode.Fuzzed)
    amp_results("240102_large", (46.54, 0.28, 36.74), 178.6)
    average_results("240102_large", mode=AvgMode.Amplified)
    noise_results("240102_large", "noise_beta_dists.csv", NoiseType.BetaDist)
    average_results("240102_large", mode=AvgMode.Adjusted)


if __name__ == "__main__":
    main()
