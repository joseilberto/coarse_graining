from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd


def make_vel_batches(data, *args, **kwargs):
    x_col = kwargs.get("x_col", 0)
    y_col = kwargs.get("x_col", 1)
    time_col = kwargs.get("time_col", 2)
    idx_col = kwargs.get("idx_col", 3)
    n_procs = kwargs.get("n_procs", cpu_count())
    aliasing = kwargs.get("aliasing", 2)
    times = np.unique(data[:, time_col])
    n_times = len(times)
    batch_size = round(n_times / n_procs)
    times_batches = times[::batch_size]
    batches = []
    time_steps = []
    for idx, time in enumerate(times_batches[1:]):
        init_time = times_batches[idx]
        final_time = time
        times_in_batch = times[(times >= init_time) &
                               (times < final_time)]
        step_data = data[np.isin(data[:, time_col], times_in_batch)]
        step_data = step_data[:, [x_col, y_col, time_col, idx_col]]
        step_data = pd.DataFrame(step_data)
        step_data.columns = ["x", "y", "time", "idx"]
        step_data.sort_values(by = ["time", "idx"], inplace = True)
        step_data = step_data.values
        batches.append([step_data, times_in_batch, aliasing])
        time_steps.append(final_time)
    return batches, time_steps


def get_velocities_and_where(batch, times_in_batch, curs, prevs, idx,
                             aliasing, size_times):
    intersects = np.isin(curs[:, 3], prevs[:, 3])
    intersect = curs[intersects]
    for p_idx in intersect[:, 3]:
        min_idx = (idx - (aliasing - 1)
                   if idx - (aliasing - 1) > 0 else 0)
        max_idx = (idx + 1 + aliasing
                   if idx + 1 + aliasing < size_times else size_times)
        avg_idxs = ((batch[:, 2] >= times_in_batch[min_idx]) &
                    (batch[:, 2] <= times_in_batch[max_idx]) &
                    (batch[:, 3] == p_idx))
        avg_data = batch[avg_idxs]
        x, y, times = avg_data[:, 0], avg_data[:, 1], avg_data[:, 2]
        v_x = np.mean(np.diff(x)) / np.sum(np.diff(times))
        v_y = np.mean(np.diff(y)) / np.sum(np.diff(times))
        idx_cur_data = np.where(curs[:, 3] == p_idx)[0]
        yield v_x, v_y, idx_cur_data


def calculate_vel_batch(*args):
    batch, times_in_batch, aliasing = args[0]
    full_data = []
    size_times = len(times_in_batch) - 1
    for idx, time in enumerate(times_in_batch[1:]):
        prevs = batch[batch[:, 2] == times_in_batch[idx]]
        curs = batch[batch[:, 2] == time]
        cur_data = np.zeros((curs.shape[0], curs.shape[1] + 2))
        cur_data[:, [0, 1, 2, 3]] = curs.copy()
        velocities_generator = get_velocities_and_where(batch, times_in_batch,
                                                        curs, prevs, idx,
                                                        aliasing, size_times)
        for v_x, v_y, idx_cur_data in velocities_generator:
            cur_data[idx_cur_data, [4, 5]] = v_x, v_y
        full_data.append(cur_data)
    if not full_data:
        return
    return np.vstack(full_data)


def calculate_vel_intersections(data, time_steps, *args, **kwargs):
    pass

def load_data(file):
    if file.endswith(".npy"):
        data = np.load(file)
    elif file.endswith(".csv") or file.endswith(".txt"):
        data = pd.read_csv(file).values
    return data


def radius_column_to_data(file, radius):
    return np.c_[data, np.repeat(radius, len(data))]


def velocities_column_to_data(data, *args, **kwargs):
    n_procs = kwargs.get("n_procs", cpu_count())
    batches, time_steps = make_vel_batches(data, **kwargs)
    pool = Pool(n_procs)
    data = np.vstack(pool.map(calculate_vel_batch, batches))
    pool.close()
    data = calculate_vel_intersections(data, time_steps, **kwargs)
    vel_data = vel_data.values
    #TODO Do intersections between batches
    return data
