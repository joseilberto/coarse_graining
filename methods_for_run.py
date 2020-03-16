import glob
import numpy as np
import os
import pandas as pd
import re

from load.methods_2d import velocities_column_to_data


def args_logger_wrapper(*args):
    def outer_wrapper(function):
        def inner_wrapper():
            for idx, arg in enumerate(args):
                if idx == 0:
                    args_parse = arg()
                else:
                    arg()
            return function(args_parse)
        return inner_wrapper
    return outer_wrapper


def get_files(path, inner_dir, file_format):
    """
    Search all files with the given extension.
    """
    files = []
    viscosities = [path + folder + '/' for folder in next(os.walk(path))[1]
                  if 'mu=' in folder]
    for viscosity in viscosities:
        angles = [viscosity + folder + '/'
                  for folder in next(os.walk(viscosity))[1]
                        if 'degree=' in folder]
        for angle in angles:
            rel_files = [file1
                for file1 in glob.glob(angle + inner_dir +
                                        "*{}".format(file_format))
                if "crossed" not in file1]
            files.extend(rel_files)
    return files


def files_from_pattern(files, pattern):
    return sorted(set([int(re.search(pattern, file1).group(1))
                    for file1 in files if re.search(pattern, file1)]))


def find_stationarity(file, stationary_reference, *args, **kwargs):
    fps = re.search(r"fps=(-?\d+)_", stationary_reference)
    if not fps:
        fps = re.search(r"fps=(.*?)_", stationary_reference)
        if not fps:
            return
    fps = fps.group(1)
    fps = int(fps) if float(fps) > 1 else float(fps)
    data = np.load(file)
    data[:, 2] = data[:, 2] / fps
    reference = pd.read_csv(stationary_reference)
    reference.sort_values(by = ["time"], inplace = True)
    t_min, t_max = reference["time"].min(), reference["time"].max()
    rel_idxs = ((data[:, 2] >= t_min) & (data[:, 2] <= t_max))
    rel_data = pd.DataFrame(data[rel_idxs])
    rel_data.columns = ["x", "y", "time", "p_idx"]
    rel_data.sort_values(by = ["time", "x", "y"], inplace = True)
    rel_data = velocities_column_to_data(rel_data.values, kwargs["aliasing"])
    return rel_data


def get_dict_from_files(files, *args, **kwargs):
    vis_pat = kwargs.get("vis_pattern", r'mu={}cSt'.format)
    deg_pat = kwargs.get("deg_pattern", r'degree={}/'.format)
    ratio_pat = kwargs.get("ratio_pattern", r'ratio={}/'.format)
    diameter_pat = kwargs.get("diameter_pattern", r'diameter={}mm'.format)
    ratios = files_from_pattern(files, ratio_pat(r'(.*?)'))
    files_dict = {}
    for ratio in ratios:
        ratio_files = [file1 for file1 in files if ratio_pat(ratio) in file1]
        diameters = files_from_pattern(ratio_files, diameter_pat(r'(.*?)'))[::-1]
        diameter_dict = {}
        for diameter in diameters:
            diameter_files = [file1 for file1 in ratio_files
                                            if diameter_pat(diameter) in file1]
            viscosities = files_from_pattern(diameter_files, vis_pat(r'(.*?)'))
            viscosity_dict = {}
            for viscosity in viscosities:
                vis_files = [file1 for file1 in diameter_files
                                                if vis_pat(viscosity) in file1]
                angles = files_from_pattern(vis_files, deg_pat(r'(.*?)'))[::-1]
                angle_dict = {}
                for angle in angles:
                    ang_files = [file1 for file1 in vis_files
                                                    if deg_pat(angle) in file1]
                    angle_dict[angle] = ang_files
                viscosity_dict[viscosity] = angle_dict
            diameter_dict[float(diameter)] = viscosity_dict
        files_dict[ratio] = diameter_dict
    return files_dict


def get_stationary_files(files, statio_path, redo = False, *args, **kwargs):
    path_base = kwargs["path"]
    file_format = kwargs["format"]
    stationary_pattern = kwargs.get("stationary_pattern",
                                    "_particles_crossed.csv")
    pattern_str = "{0}ratio={2}/diameter={2}mm/mu={2}cSt/degree={2}/{1}/"
    folders_pattern = pattern_str.format(path_base, statio_path, "{}").format
    files_dict = {}
    for ratio, diameter_dict in files.items():
        diameters = {}
        for diameter, vis_dict in diameter_dict.items():
            viscosities = {}
            for viscosity, angle_dict in vis_dict.items():
                angles = {}
                for angle, data_dict in angle_dict.items():
                    stationaries = []
                    for file in data_dict:
                        stationary_folder = folders_pattern(ratio, int(diameter),
                                                        viscosity, angle)
                        file_name = os.path.basename(file)
                        stationary_file = stationary_folder + file_name
                        stationaries.append(stationary_file)
                        file_pattern = file_name.replace(file_format, "")
                        stationary_reference = (stationary_folder +
                                            file_pattern + stationary_pattern)
                        if not os.path.isfile(stationary_reference):
                            continue
                        if os.path.isfile(stationary_file) and not redo:
                            continue
                        stationary_data = find_stationarity(file,
                                                        stationary_reference,
                                                        *args, **kwargs)
                        np.save(stationary_file, stationary_data)
                    if stationaries:
                        angles[angle] = stationaries
                viscosities[viscosity] = angles
            diameters[float(diameter)] = viscosities
        files_dict[ratio] = diameters
    return files_dict


def process_for_file(method):
    def wrapper(files, *args, **kwargs):
        for ratio, diameter_dict in files.items():
            for diameter, vis_dict in diameter_dict.items():
                viscosities = {}
                for viscosity, angle_dict in vis_dict.items():
                    for angle, data_dict in angle_dict.items():
                        for file in data_dict:
                            method(file, int(ratio), int(diameter), viscosity,
                                   angle, *args, **kwargs)
    return wrapper
