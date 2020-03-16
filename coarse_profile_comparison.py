from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_idx(yy, y_level = 0):
    uniques = yy[:, 0]
    for idx, val in enumerate(uniques[:-1]):
        if val <= y_level and uniques[idx + 1] > y_level:
            return idx + 1


def fit_quadratic(x, v_c, R, a):
    return v_c * np.power(1 - (x/R)**2, a)


def find_crossing_data(data, y_level = 0):
    idxs = np.unique(data[:, 3])
    vels_data = [] 
    for idx in idxs: 
        rel_data = data[data[:, 3] == idx] 
        for idx2, pos in enumerate(rel_data[:-1]): 
            nxt_x, nxt_y = rel_data[idx2 + 1, [0, 1]] 
            if (pos[1] >= y_level and nxt_y < y_level): 
                v_y = rel_data[idx2 + 1, -1] 
                vels_data.append((rel_data[idx2 + 1, 0], v_y)) 
    return np.stack(vels_data)


def find_all_crossing_data(data, radius, R, y_level = 0):
    rel_idxs = (((data[:, 0] >= -R) & (data[:, 0] <= R) & 
                (data[:, 1] > y_level) & (data[:, 1] - radius < y_level)) |
                ((data[:, 0] >= -R) & (data[:, 0] <= R) & 
                (data[:, 1] < y_level) & (data[:, 1] + radius > y_level)))    
    return data[rel_idxs][:, [0, -1]]


def find_average_boxes(xs, v, field_xs):
    average_v = []
    for idx, cur_x in enumerate(field_xs[:-1]):
        if cur_x < -5 or cur_x > 5:
            continue
        nxt_x = field_xs[idx + 1]
        idxs = np.where((xs >= cur_x) & (xs < nxt_x))
        if not idxs[0].any():
            continue
        average_v.append((cur_x, v[idxs].mean()))
    return np.array(average_v)


def do_regression_plot(x, v, field_x, field_vy, idx): 
    fit_func = lambda x, v_c, a: fit_quadratic(x, v_c, 5.0, a)
    xs = np.linspace(-5, 5, 1000)
    # params, covs = curve_fit(fit_func, x, v, p0 = (1, 0.5), bounds = ((-200, 0), (200, 1))) 
    # xi = abs((-field_vy[idx]*10**2).max() - params[0]) / params[0]    
    # reg = fit_func(xs, *params) 
    plt.scatter(x, v, c = "b", alpha = 0.2) 
    average_points = find_average_boxes(x, v, field_x[idx])
    field_idx = int(field_vy[idx].shape[0] // 2)
    avg_idx = int(average_points.shape[0] // 2)
    xi = abs((-field_vy[idx]*10**2)[field_idx] - average_points[avg_idx, 1]) / average_points[avg_idx, 1]
    # plt.plot(xs, reg, "--r", label = r"$v = {:.3f}(1 - (x/5.0)^2)^{{{:.2f}}}$".format(*params)) 
    plt.scatter(average_points[:, 0], average_points[:, 1], c = "r", label = r"Mean $v_y$") 
    plt.plot(field_x[idx], -field_vy[idx]*10**2, "g", label = "Coarse-Graining")
    plt.xlabel(r"$x/d$")
    plt.ylabel(r"$|v_z| \, (cm/s)$")
    plt.xlim((-10, 10))
    # plt.ylim((0, 0.05))
    plt.title(r"$\mu = 12000 \, cSt$      $\xi = {:.2f}$".format(xi))
    plt.legend() 
    plt.show()


if __name__ == "__main__":
    viscosity = 0
    fps = 1000
    base_folder = "./mu={}cSt_diameter=2mm/degree=90/".format(viscosity)
    data_dict = {
        "data_pattern": "video0001_spots_d=0.2cm_fps={}.npy".format(fps),
        "xx": "merged_field_xx.npy",
        "yy": "merged_field_yy.npy",
        "vy": "merged_vy.npy",
    }
     
    data = np.load(base_folder + data_dict["data_pattern"])
    vels_data = find_all_crossing_data(data, 0.1, 1, 0)
    # vels_data = find_crossing_data(data, 0)        
    xx = np.load(base_folder + data_dict["xx"])
    yy = np.load(base_folder + data_dict["yy"])
    cr_idx = find_idx(yy)
    vy = np.load(base_folder + data_dict["vy"])
    do_regression_plot(vels_data[:, 0]/0.2, -vels_data[:, 1],
                            xx/0.002, -vy, cr_idx)

    

