from multiprocessing import cpu_count

import argparse
import logging
import os

from methods_for_run import *
from load.methods_2d import *
from coarse_graining.methods_2d import Coarse_Graining


def set_args():
    """
    Set all arguments and return them parsed.
    """
    args_dic = {
        '-p': ['--path', str, 'Inputs a path in which it will try to find all '
               'relevant data.'],
        '-cp': ['--copy_path', str, 'A copy path directing to a folder where '
                'all raw data will be copied.'],
        '-fmt': ['--format', str, 'Define the extension of data files to be '
        'searched, if not given the standard is .npy'],
        '-indir': ['--inner_dir', str, 'Directory where the data files are. '
                   'If not specified, uses "results/"'],
    }
    parser = argparse.ArgumentParser(description = 'Performs coarse-graining '
                                    'from the relevant data.' )
    for key, value in args_dic.items():
        parser.add_argument(key, value[0], type = value[1], help = value[2])
    return parser.parse_args()


@args_logger_wrapper(set_args)
def process_args(args):
    """
    Process the arguments got from argparse.
    """
    file_format = args.format if args.format else ".npy"
    if not getattr(args, "copy_path", None):
        raise Exception("No copy path specified with -cp")
    copy_path = args.copy_path
    os.makedirs(copy_path, exist_ok = True)
    inner_dir = args.inner_dir if args.inner_dir else "results/"
    files = get_files(args.path, inner_dir, file_format)
    files_dict = get_dict_from_files(files)
    kwargs = {
                "format": file_format,
                "inner_dir": inner_dir,
                "path": copy_path,
            }
    return files, files_dict, kwargs


@process_for_file
def file_processing(file, ratio, diameter, viscosity, angle, *args,
                          **kwargs):
    sys_type = kwargs.get("sys_type", "monodisperse")
    if "monodisperse" in sys_type:
        radius = diameter / 2
    data = load_data(file)
    data = radius_column_to_data(data, radius)
    kwargs.update({
        "sys_type": "monodisperse",
        "W": radius*10**(-3),
    })
    times = np.unique(data[:, 2])
    radius_in_meter = radius*10**(-3)
    for time in times:
        coarser = Coarse_Graining(**kwargs)
        cur_data = data[data[:, 2] == time]
        X, Y = cur_data[:, 0]*10**(-2), cur_data[:, 1]*10**(-2)
        V_X, V_Y = cur_data[:, 4]*10**(-2), cur_data[:, 5]*10**(-2)
        radii = cur_data[:, 6]*10**(-3)
        coarse_graining_data = coarser.kinetic_stress(X, Y, V_X, V_Y,
                                                      radii, **kwargs)


def run_coarse_graining(stationary_path, parameters, *args, **kwargs):
    files, files_dict, kwargs = process_args()
    kwargs.update({
            "aliasing": 2,
            "x_col": 0,
            "y_col": 1,
            "time_col": 2,
            "idx_col": 3,
            "n_procs": cpu_count(),
         })
    stationary_files = get_stationary_files(files_dict, stationary_path,
                                            redo = False, *args, **kwargs)
    file_processing(stationary_files, *args, **parameters)


if __name__ == "__main__":
    stationary_path = "stationary_velocity_region=0"
    parameters = {
        "density": 7850,
        "epsilon": 4,
        "n_points": 32,
    }
    run_coarse_graining(stationary_path, parameters)
