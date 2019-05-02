import glob
import os
import re


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


def make_stationary_files(files, output_path, *args, **kwargs):
    pass
