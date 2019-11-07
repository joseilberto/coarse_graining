from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from load.methods_2d import radius_column_to_data
from coarse_graining.methods_2d import Coarse_Graining

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def do_coarse_graining(data, radius, skip = 1, *args, **kwargs):
    data = data[data[:, 1] > - 2*radius]
    data = radius_column_to_data(data, radius)
    min_x, max_x = data[:, 0].min(), data[:, 0].max()
    min_y, max_y = data[:, 1].min(), data[:, 1].max()
    limits = [min_x, max_x, min_y, max_y]
    kwargs.update({
        "W": radius,
        "limits": limits,
    })
    times = np.unique(data[:, 2])[::skip]
    for idx, time in tqdm(enumerate(times), total = len(times) - 1, 
                                        desc = "Running Coarse Graining"):
        coarser = Coarse_Graining(**kwargs)
        cur_data = data[data[:, 2] == time]
        X, Y = cur_data[:, 0], cur_data[:, 1]
        V_X, V_Y = cur_data[:, 4], np.abs(cur_data[:, 5])
        radii = cur_data[:, 6]
        coarser.kinetic_stress(X, Y, V_X, V_Y, radii, **kwargs)
        if idx == 0:
            density = coarser.densities_grid
            gradient = coarser.dgradient_grid
            kinetic_stress = coarser.kinetic_trace
            xx = coarser.xx
            yy = coarser.yy
            continue
        density += (coarser.densities_grid - density) / (idx + 1)
        gradient += (coarser.dgradient_grid - gradient) / (idx + 1)
        kinetic_stress += (coarser.kinetic_trace - kinetic_stress) / (idx + 1)
    gradient = np.sqrt(gradient[:, :, 0]**2 + gradient[:, :, 1]**2)
    return xx, yy, density, gradient, kinetic_stress
    

if __name__ == "__main__":
    c_params = {
                "density": 7850,
                "epsilon": 4,
                "n_points": 16,
                "sys_type": "monodisperse",
                }
    base_folder = "./mu=913cSt_diameter=2mm/degree={}/".format
    file_pattern = "video00*_spots_d=0.2cm_fps=10.npy"
    angles = np.arange(90, 50, -10, dtype = np.int32)
    radius = 1*10**(-3)
    total_files = 1
    for idx, angle in enumerate(angles):
        file_listing = sorted(glob(base_folder(angle) + file_pattern))
        for data_file in file_listing[:total_files]:
            data = radius_column_to_data(np.load(data_file), radius)
            data[:, [0, 1, 4, 5]] = data[:, [0, 1, 4, 5]]*10**(-2)
            xx, yy, density, gradient, kinetic = do_coarse_graining(data, 
                                                radius, skip = 2, **c_params)
            np.save(data_file.replace(".npy", "_xx.npy"), xx)
            np.save(data_file.replace(".npy", "_yy.npy"), yy)
            np.save(data_file.replace(".npy", "_density.npy"), density)
            np.save(data_file.replace(".npy", "_gradient.npy"), gradient)
            np.save(data_file.replace(".npy", "_kinetic.npy"), kinetic)


