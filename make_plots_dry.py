from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import os


def find_limits(angles, field, base_folder, file_pattern, replacer, total_files = 1):
    vmin, vmax = float("inf"), -float("inf")
    for idx, angle in enumerate(angles):
        file_listing = sorted(glob(base_folder(angle) + file_pattern(field)))  
        for data_file in file_listing[:total_files]:
            field_data = np.load(data_file)
            if field in ["v", "vx", "vy"]:
                field_data *= 10**2
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
    viscosity = 0
    folder = "./mu={}cSt_diameter=2mm/".format(viscosity)
    base_folder = "{}degree={}/".format(folder, "{}").format
    file_pattern = "merged_{}.npy".format
    colorbar_ylabel = {
        # "density": r"$\rho (kg / m^2)$",
        # "packing": r"$\phi$",
        # "v": r"$|\vec{v}| (cm/s)$",
        # "vy": r"$v_y (cm/s)$",
        "vx": r"$v_x (cm/s)$",
        # "p": r"$|\vec{P}| (\frac{kg \, m}{s})$",
        # "px": r"$P_x (\frac{kg \, m}{s})$",
        # "py": r"$P_y (\frac{kg \, m}{s})$",
        # "gradient":  r"$\nabla \rho (kg / m^3)$",
        # "kinetic": r"$Tr(\sigma_{ab}) (N / m)$",
    }
    replacer = lambda x, y: x.replace(".npy", "_{}.npy".format(y))
    # angles = np.arange(80, 50, -10, dtype = np.int32)
    angles = np.array([90])
    radius = 1*10**(-3)
    diameter = 2*radius
    total_files = 1
    fields = list(colorbar_ylabel.keys())    
    for field in fields:
        vmin, vmax = find_limits(angles, field, base_folder, file_pattern, 
                                        replacer, total_files)
        fig, ax = plt.subplots(ncols = len(angles), sharey = True)
            
        ax = ax if len(angles) > 1 else [ax]
        for idx, angle in enumerate(angles):
            file_listing = sorted(glob(base_folder(angle) + file_pattern(field)))
            for data_file in file_listing:
                xx = np.load(base_folder(angle) + file_pattern("field_xx"))
                yy = np.load(base_folder(angle) + file_pattern("field_yy"))
                field_data = np.load(data_file)      
                if field in ["v", "vx", "vy"]:
                    field_data *= 10**(2)
                    idxs = np.where((yy >= 0) & (yy / diameter <= 20))
                    xx = xx[idxs].reshape(-1, xx.shape[1])
                    yy = yy[idxs].reshape(-1, yy.shape[1])
                    field_data = field_data[idxs].reshape(-1, field_data.shape[1])
                    idxs_gap = np.where((xx >= -5) & (xx <= 5))                    
                    field_data = field_data[idxs_gap].reshape(-1, field_data.shape[1])
                    vmin = field_data.min()
                    vmax = field_data.max()
                axis = ax[idx].pcolor(xx / diameter, yy / diameter, field_data, 
                                    cmap = "coolwarm", vmin = vmin, vmax = vmax)
                # ax[idx].set_title(r"$\mu = {} \, cSt$   $\theta = {} \, \degree$".format(viscosity, angle))
                ax[idx].set_xlabel(r"$x / d$")
                ax[idx].set_xlim((-6, 6))
                if idx == 0:
                    ax[idx].set_ylabel(r"$y / d$")
                    ax[idx].set_ylim((0, 16))    
        if len(angles) > 1:
            color_bar = fig.colorbar(axis, ax = ax.ravel().tolist())
        else:
            color_bar = fig.colorbar(axis, ax = ax)
        color_bar.ax.set_ylabel(colorbar_ylabel[field])
        fig.savefig(folder + "{}.pdf".format(field), format = "pdf", bbox_inches = "tight")            
        plt.close()
    


