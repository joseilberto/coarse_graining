import numpy as np
import pandas as pd


def load_data(file):
    if file.endswith(".npy"):
        data = np.load(file)
    elif file.endswith(".csv") or file.endswith(".txt"):
        data = pd.read_csv(file).values
    return data


def radius_column_to_data(file, radius):
    return np.c_[data, np.repeat(radius, len(data))]


def velocities_column_to_data(data, x_col = 0, y_col = 1, time_col = 2,
                              idx_col = 3):
    times = np.unique(data[:, time_col])
    full_data = []
    for idx, time in enumerate(times[1:]):
        prevs = data[data[:, time_col] == times[idx]]
        curs = data[data[:, time_col] == time]
        cur_data = np.zeros(curs.shape)
        intersects = np.isin(curs[:, idx_col], prevs[:, idx_col])
        intersect = curs[intersects]
        #TODO
