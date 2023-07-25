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
from math_functions import uniform_random_direction, gauss2d_simple
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
    idx, num_part, seed, loc_data = args
    rng = np.random.default_rng(seed)
    ssk = seed.spawn_key[0] if seed is not None else None
    logging.info(f"run_sim(idx={idx}, n_p={num_part}): start. seed={ssk}")

    left_laser = PulsingLaser(direction=RIGHT, k=LASER_K, n_pulses=PULSES_PER_LASER,
                              t_on=LASER_PULSE_TIME, t_off=LASER_PULSE_TIME,
                              time_offset=0)
    right_laser = PulsingLaser(direction=LEFT, k=LASER_K, n_pulses=PULSES_PER_LASER,
                               t_on=LASER_PULSE_TIME, t_off=LASER_PULSE_TIME,
                               time_offset=LASER_PULSE_TIME)
    p = [Particle(id=i, mass=PARTICLE_MASS,
                  excited_energy=left_laser.energy, excited_lifetime=EXCITED_LIFETIME,
                  start_coords=np.append(uniform_random_direction(rng, dim=2) * rng.random() * 0.5e-6, 0),
                  start_v=np.append(uniform_random_direction(rng, dim=2) * rng.random() * MAX_V0, 0))
         for i in range(num_part)]
    ph_lens = Lens(image_dim=LENS_PIXELS_DIM,
                   focus_area=LENS_FOCUS_SIDE_LENGTH ** 2,
                   z_loc=Z_MAX)
    sim = Sim(dt=TIME_RESOLUTION,
              particles=p, lenses=[ph_lens], lasers=[left_laser, right_laser],
              seed=seed)

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
        pd.DataFrame(loc_dicts).to_csv(f"results\\loc_{idx}_noa_{num_part}.csv", index=False)

    plt.imshow(ph_lens.image, origin="lower", vmin=0, vmax=IMAGE_COUNT_LIMIT)
    plt.axis("off")
    plt.savefig(f"results\\res_{idx}_noa_{num_part}.png", bbox_inches='tight', pad_inches=0)

    df = pd.DataFrame(ph_lens.image)
    df.to_csv(f"results\\res_{idx}_noa_{num_part}.csv", index=False, header=False)

    logging.info(f"run_sim(idx={idx}, n_p={num_part}): end")

# TODO: Quantum efficiency & Amplification for camera


def debug_trial():
    run_sim((-1, 1, None), True)


def run_trials(n_images_per_noa, noas, loc_data=False):
    noas = list(noas)
    n_images = n_images_per_noa * len(noas)
    ss = SeedSequence()
    seeds = ss.spawn(n_images)
    pool = multiprocessing.Pool()
    args = zip(range(n_images), np.sort(noas * n_images_per_noa), seeds, [loc_data]*n_images)
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


def average_dists(res_dir):
    ps = glob(os.path.join(res_dir, "loc*noa*.csv"))
    df = pd.read_csv(ps[0])[["t", "r"]]
    for p in ps[1:]:
        df = pd.concat((df, pd.read_csv(p)[["t", "r"]]))
    df.groupby(df.t).mean().to_csv(os.path.join(res_dir, "dist_mean.csv"))

def fit_results(res_dir):
    ps = glob(os.path.join(res_dir, "mean_noa*.csv"))
    for p in ps:
        arr = pd.read_csv(p, header=None).to_numpy()
        x = np.arange(LENS_PIXELS_DIM)
        y = np.arange(LENS_PIXELS_DIM)
        x, y = np.meshgrid(x, y)

        # noinspection PyTupleAssignmentBalance
        popt, pcov = opt.curve_fit(gauss2d_simple, (x, y), arr.ravel(),
                                   p0=(np.max(arr), LENS_PIXELS_DIM/2, LENS_PIXELS_DIM/2, 10))

        data_fitted = gauss2d_simple((x, y), *popt)

        # fig, ax = plt.subplots(1, 1)
        # ax.imshow(arr, origin='lower',
        #           extent=(x.min(), x.max(), y.min(), y.max()), vmin=0, vmax=IMAGE_COUNT_LIMIT)
        # ax.contour(x, y, data_fitted.reshape(LENS_PIXELS_DIM, LENS_PIXELS_DIM), 2, colors='w')
        # plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(arr.ravel()), 0.5, 0.5, arr.ravel(), alpha=0.3, shade=True)
        ax.contour(x, y, data_fitted.reshape(LENS_PIXELS_DIM, LENS_PIXELS_DIM), 20, linewidths=3, cmap="Reds")
        ax.text(15, 25, np.max(arr), rf"$\sigma = {popt[3]:.3f}$"\
                    "\n"\
                    rf"$A = {popt[0]:.3f}$")
        plt.show()


def analyze_dist_mean(res_dir):
    df = pd.read_csv(os.path.join(res_dir, "dist_mean.csv"))
    t = df.t.to_numpy() * 1e6  # mu_s
    r = df.r.to_numpy() * 1e6  # mu_m

    # yanai_data
    exp_t = np.array([10,  20,  30,  40,  50, 60,  70,  80])  # mu_s
    exp_r = np.array([3.2, 4,   6,   7,   8,  9.5, 10,  12])  # mu_m
    exp_e = np.array([1/4, 1/2, 2/3, 5/6, 1,  1.2, 1.3, 1.5])  # mu_m

    # li_data
    li_t = np.array([5,   10,  15, 20, 25, 30,   35, 40, 45,  50])  # mu_s
    li_r = np.array([3.5, 10,  9,  10, 24, 24.5, 40, 33, 38.5, 35.5]) * 6 / 40 # mu_m

    fig = plt.figure(dpi=300)
    plt.grid()
    plt.plot(t, r, "k.", label="sim data")
    plt.plot(exp_t, exp_r, "r.-", label="yanai data (visual)")
    plt.plot(li_t, li_r, "g.-", label=r"$^6 Li$ data $\times\frac{6}{40}$ (visual)")
    plt.errorbar(exp_t, exp_r, exp_e, fmt="r+", capsize=10)
    plt.xlabel(r"t [$\mu s$]")
    plt.ylabel(r"r [$\mu m$]")
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(res_dir, "dist_mean.png"))


def main():
    # TODO: why is it suddenly brighter?
    # debug_trial()
    # run_trials(60, range(1, 4), loc_data=True)
    # average_results("results_28jun23_60_only")
    # average_dists("results_28jun23_60_locs")
    # fit_results("results_28jun23_60_only")
    analyze_dist_mean("results_28jun23_60_locs")


if __name__ == "__main__":
    main()
