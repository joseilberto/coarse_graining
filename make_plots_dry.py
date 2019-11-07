from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import os


def find_limits(angles, field, base_folder, file_pattern, replacer, total_files = 1):
    vmin, vmax = float("inf"), -float("inf")
    for idx, angle in enumerate(angles):
        file_listing = sorted(glob(base_folder(angle) + file_pattern))        
        for data_file in file_listing[:total_files]:
            field_data = np.load(replacer(data_file, field))
            vmin = field_data.min() if field_data.min() < vmin else vmin
            vmax = field_data.max() if field_data.max() > vmax else vmax
    return vmin, vmax


if __name__ == "__main__":
    c_params = {
                "density": 7850,
                "epsilon": 4,
                "n_points": 16,
                "sys_type": "monodisperse",
                }
    base_folder = "./mu=913cSt_diameter=2mm/degree={}/".format
    file_pattern = "video00*_spots_d=0.2cm_fps=10.npy"
    colorbar_ylabel = {
        "kinetic": r"$Tr(\sigma_{ab}) (N / m)$",
        "density": r"$\rho (kg/m^2)$",
        "gradient":  r"$\nabla \rho (kg / m^3)$",
    }
    replacer = lambda x, y: x.replace(".npy", "_{}.npy".format(y))
    angles = np.arange(80, 50, -10, dtype = np.int32)
    # angles = np.array([90, 80, 60])
    radius = 1*10**(-3)
    diameter = 2*radius
    total_files = 1
    field = "kinetic"
    vmin, vmax = find_limits(angles, field, base_folder, file_pattern, 
                                        replacer, total_files)
    fig, ax = plt.subplots(ncols = len(angles), sharey = True)    
    for idx, angle in enumerate(angles):
        file_listing = sorted(glob(base_folder(angle) + file_pattern)) 
        for data_file in file_listing[:total_files]:            
            xx = np.load(replacer(data_file, "xx"))
            yy = np.load(replacer(data_file, "yy"))
            field_data = np.load(replacer(data_file, field))

            axis = ax[idx].pcolor(xx / diameter, yy, field_data, cmap = "coolwarm", 
                                vmin = vmin, vmax = vmax)            

            ax[idx].set_title(r"$\theta = {} \, \degree$".format(angle))
            ax[idx].set_xlabel(r"$x / d$")
            if field == "kinetic":
                ax[idx].set_xlim((-10, 10))
            else:
                ax[idx].set_xlim((-20, 20))
            if idx == 0:
                ax[idx].set_ylabel(r"$y \, (m)$")
                ax[idx].set_ylim((-0.005, 0.015))
    color_bar = fig.colorbar(axis, ax = ax.ravel().tolist())
    color_bar.ax.set_ylabel(colorbar_ylabel[field])
    plt.show()


