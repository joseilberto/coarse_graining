from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from load.methods_2d import radius_column_to_data, velocities_column_to_data
from coarse_graining.methods_2d import Coarse_Graining

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def do_coarse_graining(data, radius, times = [], skip = 0, n_times = 10, 
                        *args, **kwargs):
    # data = data[data[:, 1] > - 2*radius]
    data = radius_column_to_data(data, radius)
    min_x, max_x = data[:, 0].min(), data[:, 0].max()
    min_y, max_y = data[:, 1].min(), data[:, 1].max()
    limits = [min_x, max_x, min_y, max_y]
    kwargs.update({
        "W": radius,
        "limits": limits,
    })
    if not np.any(times):
        times = np.unique(data[:, 2])
        times = times[::skip] if not n_times else times[:n_times]
    else:
        cur_times = data[:, 2]    
        times = times[np.isin(times, cur_times)]
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
            packing = coarser.packing_fraction
            gradient = coarser.dgradient_grid
            momenta = coarser.momenta_grid
            kinetic_stress = coarser.kinetic_trace
            xx = coarser.xx
            yy = coarser.yy
            continue
        density += coarser.densities_grid
        packing += coarser.packing_fraction
        gradient += coarser.dgradient_grid
        momenta += coarser.momenta_grid 
        kinetic_stress += coarser.kinetic_trace
    vx = np.nan_to_num(momenta[:, :, 0] / density)
    vy = np.nan_to_num(momenta[:, :, 1] / density)
    velocities = np.stack((vx, vy), axis = 2)
    density /= len(times)
    packing /= len(times)
    gradient /= len(times)
    momenta /= len(times)
    kinetic_stress /= len(times)
    gradient = np.sqrt(gradient[:, :, 0]**2 + gradient[:, :, 1]**2)
    return xx, yy, density, packing, gradient, momenta, velocities, kinetic_stress
    

def merge_files(files, redo_velocities = False):
    whole_data = np.empty(shape = (0, 7))
    limit_times = [0]
    for data_file in files:
        data = np.load(data_file)
        data = data[:, :4] if redo_velocities else data
        if data.shape[1] < 5:
            data = velocities_column_to_data(data, **{"aliasing": 1})
            np.save(data_file, data)
        if len(whole_data) > 1:
            data[:, 2] += whole_data[:, 2].max()
            limit_times.append(whole_data[:, 2].max())
        data = radius_column_to_data(data, radius)                        
        data[:, [0, 1, 4, 5]] = data[:, [0, 1, 4, 5]]*10**(-2)
        whole_data = np.append(whole_data, data, axis = 0)
    return whole_data, np.array(limit_times)


def merge_cross_times(files, limit_times):
    times = np.empty(shape = (0,))
    for idx, data_file in enumerate(files):
        cur_times = pd.read_csv(data_file).time.sort_values().values
        cur_times = np.unique(cur_times) + limit_times[idx]
        times = np.append(times, cur_times, axis = 0)
    return times


if __name__ == "__main__":
    c_params = {
                "density": 7850,
                "epsilon": 4,
                "n_points": 16,
                "sys_type": "monodisperse",
                "function": "heavyside",
                }
    redo_velocities = False
    viscosity = 0
    base_folder = "./mu={}cSt_diameter=2mm/degree={}/".format(viscosity, "{}").format
    file_pattern = "video00*_spots_*.npy"
    angles = np.arange(90, 80, -10, dtype = np.int32)        
    radius = 1*10**(-3)
    total_files = 1
    for idx, angle in enumerate(angles):
        file_listing = sorted(glob(base_folder(angle) + file_pattern))
        data, limit_times = merge_files(file_listing, redo_velocities)
        times_files = [file.replace(".npy", "_velocities.csv") 
                        for file in file_listing]
        times = merge_cross_times(times_files, limit_times)
        xx, yy, density, packing, gradient, momenta, velocities, kinetic = do_coarse_graining(
                        data, radius, skip = 1, n_times = None, 
                        **c_params)
        np.save(base_folder(angle) + "merged_field_xx.npy", xx)
        np.save(base_folder(angle) + "merged_field_yy.npy", yy)
        np.save(base_folder(angle) + "merged_density.npy", density)
        np.save(base_folder(angle) + "merged_packing.npy", packing)
        np.save(base_folder(angle) + "merged_gradient.npy", gradient)
        np.save(base_folder(angle) + "merged_px.npy", momenta[:, :, 0])
        np.save(base_folder(angle) + "merged_py.npy", momenta[:, :, 1])
        np.save(base_folder(angle) + "merged_p.npy", np.sqrt(
                        momenta[:, :, 0]**2 + momenta[:, :, 1]**2))
        np.save(base_folder(angle) + "merged_vx.npy", velocities[:, :, 0])
        np.save(base_folder(angle) + "merged_vy.npy", velocities[:, :, 1])
        np.save(base_folder(angle) + "merged_v.npy", np.sqrt(
                        velocities[:, :, 0]**2 + velocities[:, :, 1]**2))                           
        np.save(base_folder(angle) + "merged_kinetic.npy", kinetic)


