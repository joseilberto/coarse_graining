import argparse
import logging
import os

from methods_for_run import args_logger_wrapper, get_dict_from_files, get_files


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


def start_logger():
    logging.basicConfig(filename = os.path.basename(__file__).replace('.py',
    '.log'), filemode = 'w',
    format = '%(asctime)s --%(name)s-- %(levelname)s: %(message)s',
    datefmt = '%d/%m/%Y %H:%M:%S', level = logging.DEBUG)


@args_logger_wrapper(set_args, start_logger)
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
    files_stationary = make_stationary_files(files_dict, copy_path)
    kwargs = {
                "format": file_format,
                "inner_dir": inner_dir,
                "logger": logging.getLogger('Coarse Graining logging'),
                "path": copy_path,
            }
    return files, files_dict, kwargs


def run_coarse_graining():
    files, files_dict, kwargs = process_args()


if __name__ == "__main__":
    run_coarse_graining()
