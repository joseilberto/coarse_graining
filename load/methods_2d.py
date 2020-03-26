from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd


def make_vel_batches(data, *args, **kwargs):
    x_col = kwargs.get("x_col", 0)
    y_col = kwargs.get("x_col", 1)
    time_col = kwargs.get("time_col", 2)
    idx_col = kwargs.get("idx_col", 3)
    n_procs = kwargs.get("n_procs", cpu_count() * 8)
    aliasing = kwargs.get("aliasing", 1)
    times = np.unique(data[:, time_col])
    n_times = len(times)
    batch_size = int(n_times / n_procs)
    times_batches = times[::batch_size]
    times_batches[-1] = (times_batches[-1] if times_batches[-1] == times[-1] 
                                            else times[-1])
    batches = []
    time_steps = []
    for idx, time in enumerate(times_batches[1:]):
        init_time = times_batches[idx]
        final_time = time
        times_in_batch = times[(times >= init_time) &
                               (times < final_time)]
        step_data = data[np.isin(data[:, time_col], times_in_batch)]
        step_data = step_data[:, [x_col, y_col, time_col, idx_col]]
        step_data = sort_by_time(step_data)
        batches.append([step_data, times_in_batch, aliasing])
        time_steps.append(final_time)
    return batches, time_steps


def get_velocities_and_where(batch, times_in_batch, curs, prevs, idx,
                             aliasing, size_times):
    intersects = np.isin(curs[:, 3], prevs[:, 3])
    intersect = curs[intersects]
    for p_idx in intersect[:, 3]:
        min_idx = max(idx - (aliasing - 1), 0)
        max_idx = min(idx + aliasing, size_times)        
        avg_idxs = ((batch[:, 2] >= times_in_batch[min_idx]) &
                    (batch[:, 2] <= times_in_batch[max_idx]) &
                    (batch[:, 3] == p_idx))        
        avg_data = batch[avg_idxs]
        if avg_data.shape[0] > 1:        
            x, y, times = avg_data[:, 0], avg_data[:, 1], avg_data[:, 2]
            v_x = np.mean(np.diff(x)) / np.mean(np.diff(times))
            v_y = np.mean(np.diff(y)) / np.mean(np.diff(times))
        else:
            v_x, v_y = 0, 0
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
        end_pat = "\r" if idx < len(times_in_batch) - 1 else "\n"
        print("Processing {}/{}".format(idx + 1, len(times_in_batch)), 
                        end = end_pat)
    if not full_data:
        return
    return np.vstack(full_data)


def calculate_vel_intersections(data, vel_data, time_steps, *args, **kwargs):
    aliasing = kwargs.get("aliasing", 2)
    times = np.unique(data[:, 2])
    size_times = len(times) - 1
    not_tracked_data = [vel_data]
    for time in time_steps:
        idx_time_in_times = np.where(times == time)[0]
        previous = data[data[:, 2] == times[idx_time_in_times - 1]]
        curs = data[data[:, 2] == time]
        cur_data = np.zeros((curs.shape[0], curs.shape[1] + 2))
        cur_data[:, [0, 1, 2, 3]] = curs.copy()
        velocities_generator = get_velocities_and_where(data, times,
                                                        curs, previous,
                                                        idx_time_in_times,
                                                        aliasing, size_times)
        for v_x, v_y, idx_cur_data in velocities_generator:
            cur_data[idx_cur_data, [4, 5]] = v_x, v_y
        not_tracked_data.append(cur_data)
    if not not_tracked_data:
        return
    vel_data = np.vstack(not_tracked_data)
    return sort_by_time(vel_data)


def load_data(file):
    if file.endswith(".npy"):
        data = np.load(file)
    elif file.endswith(".csv") or file.endswith(".txt"):
        data = pd.read_csv(file).values
    return data


def radius_column_to_data(data, radius):
    return np.c_[data, np.repeat(radius, len(data))]


def velocities_column_to_data(data, *args, **kwargs):
    n_procs = kwargs.get("n_procs", cpu_count())
    batches, time_steps = make_vel_batches(data, **kwargs)
    pool = Pool(n_procs)
    vel_data = np.vstack(pool.map(calculate_vel_batch, batches))
    pool.close()
    data = calculate_vel_intersections(data, vel_data, time_steps, **kwargs)
    return data


def sort_by_time(data):
    data = pd.DataFrame(data)
    data.sort_values(by = [2, 3], inplace = True)
    return data.values


def get_velocity_fields(folder_base, total_frames):
   for i in range(total_frames): 
       density = np.load(folder_base + 'density/{}.npy'.format(i)) 
       px = np.load(folder_base + 'px/{}.npy'.format(i)) 
       py = np.load(folder_base + 'py/{}.npy'.format(i)) 
       vx = np.nan_to_num(px / density) 
       vy = np.nan_to_num(py / density) 
       if i == 0: 
           sed_velocity = np.stack((vx, vy), axis = 2) 
           continue 
       sed_velocity += np.stack((vx, vy), axis = 2) 
   return sed_velocity / total_frames    