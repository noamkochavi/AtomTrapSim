# external imports
import numpy as np
from numpy.random import SeedSequence
import matplotlib.pyplot as plt
import scipy.optimize as opt
import multiprocessing
from glob import glob
from PIL import Image
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
    idx, num_part, seed = args
    rng = np.random.default_rng(seed)
    ssk = seed.spawn_key[0] if seed is not None else None
    logging.info(f"run_sim(idx={idx}, n_p={num_part}): start. seed={ssk}")

    left_laser = PulsingLaser(direction=RIGHT, k=LASER_K, n_pulses=PULSES_PER_LASER,
                              t_on=LASER_PULSE_TIME, t_off=LASER_PULSE_TIME,
                              time_offset=0)
    right_laser = PulsingLaser(direction=LEFT, k=LASER_K, n_pulses=PULSES_PER_LASER,
                               t_on=LASER_PULSE_TIME, t_off=LASER_PULSE_TIME,
                               time_offset=LASER_PULSE_TIME)
    p = [Particle(mass=PARTICLE_MASS,
                  excited_energy=left_laser.energy, excited_lifetime=EXCITED_LIFETIME,
                  start_coords=np.append(uniform_random_direction(rng, dim=2) * rng.random() * 0.5e-6, 0),
                  start_v=np.append(uniform_random_direction(rng, dim=2) * rng.random() * 1e-7, 0))
         for i in range(num_part)]
    ph_lens = Lens(image_dim=LENS_PIXELS_DIM,
                   focus_area=LENS_FOCUS_SIDE_LENGTH ** 2,
                   z_loc=Z_MAX)
    sim = Sim(dt=TIME_RESOLUTION,
              particles=p, lenses=[ph_lens], lasers=[left_laser, right_laser],
              seed=seed)

    last_time = sim.time - 1
    for sim_image in sim:
        if debug and (last_time + 1e-6 < sim.time):
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

    plt.imshow(ph_lens.image, origin="lower", vmin=0, vmax=IMAGE_COUNT_LIMIT)
    plt.axis("off")
    plt.savefig(f"results\\res_{idx}_noa_{num_part}.png", bbox_inches='tight', pad_inches=0)

    df = pd.DataFrame(ph_lens.image)
    df.to_csv(f"results\\res_{idx}_noa_{num_part}.csv", index=False, header=False)

    logging.info(f"run_sim(idx={idx}, n_p={num_part}): end")

# TODO: Quantum efficiency & Amplification for camera


def debug_trial():
    run_sim((-1, 1, None), True)


def run_trials(n_images_per_noa, noas):
    noas = list(noas)
    n_images = n_images_per_noa * len(noas)
    ss = SeedSequence()
    seeds = ss.spawn(n_images)
    pool = multiprocessing.Pool()
    args = zip(range(n_images), np.sort(noas * n_images_per_noa), seeds)
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


def fit_results(res_dir):
    ps = glob(os.path.join(res_dir, "mean_noa*.csv"))
    for p in ps:
        arr = pd.read_csv(p, header=None).to_numpy()
        x = np.arange(LENS_PIXELS_DIM)
        y = np.arange(LENS_PIXELS_DIM)
        x, y = np.meshgrid(x, y)

        # noinspection PyTupleAssignmentBalance
        popt, pcov = opt.curve_fit(gauss2d, (x, y), arr.ravel(),
                                   p0=(
                                       np.max(arr),
                                       LENS_PIXELS_DIM // 2 - 1,
                                       LENS_PIXELS_DIM // 2 - 1,
                                       2, 2, 0, 0
                                      ))

        data_fitted = gauss2d((x, y), *popt)

        # fig, ax = plt.subplots(1, 1)
        # ax.imshow(arr, origin='lower',
        #           extent=(x.min(), x.max(), y.min(), y.max()), vmin=0, vmax=IMAGE_COUNT_LIMIT)
        # ax.contour(x, y, data_fitted.reshape(LENS_PIXELS_DIM, LENS_PIXELS_DIM), 2, colors='w')
        # plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(arr.ravel()), 0.5, 0.5, arr.ravel(), alpha=0.3, shade=True)
        ax.contour(x, y, data_fitted.reshape(LENS_PIXELS_DIM, LENS_PIXELS_DIM), 10, linewidths=3, cmap="coolwarm")
        ax.text(15, 25, np.max(arr), rf"$\sigma_x = {popt[3]:.3f}$"\
                    "\n"\
                    rf"$\sigma_y = {popt[4]:.3f}$"\
                    "\n"\
                    rf"$A = {popt[0]:.3f}$")
        plt.show()


def main():
    # debug_trial()
    run_trials(200, range(1, 6))
    # average_results("results_19jun23_rinit")
    # fit_results("results_19jun23_rinit")


if __name__ == "__main__":
    main()
