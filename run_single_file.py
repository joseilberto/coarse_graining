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


def calculate_velocities(file_name, aliasing = 1, n_procs = 8):
    data = np.load(file_name)[:, :4]
    data = velocities_column_to_data(data, **{
                                            "aliasing": aliasing, 
                                            "n_procs": n_procs})
    np.save(file_name, data)
    return data


def do_coarse_graining(base_folder, file_name, radius, skip = 0, n_times = 10, 
                        *args, **kwargs):
    data = np.load(base_folder + file_name)    
    data = radius_column_to_data(data, radius)
    min_x, max_x = data[:, 0].min(), data[:, 0].max()
    min_y, max_y = data[:, 1].min(), data[:, 1].max()
    limits = [min_x, max_x, min_y, max_y]
    kwargs.update({
        "W": radius,
        "limits": limits,
    })
    times = np.unique(data[:, 2])
    times = times[::skip] if not n_times else times[:n_times]
    base_file_name = file_name.split('_spots')[0]
    os.makedirs(base_folder + 'cg_data/' + base_file_name + '/', exist_ok = True)
    base_pattern = base_folder + 'cg_data/' + base_file_name + '{}/{}.npy'
    grid_pattern = base_folder + 'cg_data/' + base_file_name + '{}.npy'
    output_pattern = {
                "density": base_pattern.format('density', "{}").format,
                "packing": base_pattern.format('packing', "{}").format,
                "px": base_pattern.format('px', "{}").format,
                "py": base_pattern.format('py', "{}").format,                
            }
    for idx, time in tqdm(enumerate(times), total = len(times) - 1, 
                                        desc = "Running Coarse Graining"):
        coarser = Coarse_Graining(**kwargs)
        cur_data = data[data[:, 2] == time]        
        X, Y = cur_data[:, 0], cur_data[:, 1]
        V_X, V_Y = cur_data[:, 4], np.abs(cur_data[:, 5])
        radii = cur_data[:, 6]
        coarser.momenta(X, Y, V_X, V_Y, radii, **kwargs)
        if idx == 0:
            np.save(grid_pattern.format('xx'), coarser.xx)
            np.save(grid_pattern.format('yy'), coarser.yy)
        np.save(output_pattern["density"](idx), coarser.densities_grid)
        np.save(output_pattern["packing"](idx), coarser.packing_fraction)
        np.save(output_pattern["px"](idx), coarser.momenta_grid[:, :, 0])
        np.save(output_pattern["py"](idx), coarser.momenta_grid[:, :, 1])

    
    
if __name__ == "__main__":
    c_params = {
                "density": 7850,
                "epsilon": 4,
                "n_points": 16,
                "sys_type": "monodisperse",
                "function": "heavyside",
                }
    redo_velocities, aliasing, n_procs = True, 4, 64
    viscosity = 12000
    fps = 2
    base_folder = "./data/mu={}cSt/".format(viscosity)
    file_name = "video0001_spots_d=0.2cm_fps={}.npy".format(fps)
    radius = 1*10**(-3)    
    if redo_velocities:
        calculate_velocities(base_folder + file_name, aliasing, n_procs)    
    do_coarse_graining(base_folder, file_name, radius, skip = 1, 
                        n_times = None, **c_params)
