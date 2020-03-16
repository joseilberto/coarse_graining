import numpy as np

from load.methods_2d import radius_column_to_data, velocities_column_to_data


if __name__ == "__main__":
    fps = 30
    # path = "/home/ilberto/Desktop/fluidos/datos/silicona/tolva_0_grados/ratio=10/diameter=2mm/mu=2000cSt/single_particle/results/"
    path = "/home/ilberto/Desktop/fluidos/datos/silicona/tolva_0_grados/ratio=10/diameter=2mm/mu=1000cSt/degree=90/results/"
    # path = "/home/ilberto/Desktop/fluidos/coarse_graining/mu=0cSt_diameter=2mm/degree=90/results/"
    pattern = "video{}_spots_d=0.2cm_fps={}.npy".format("{}", fps).format
    files = [path + pattern(str(i).zfill(4)) for i in range(1, 2)]
    for file in files:
        data = np.load(file)
        data[:, 2] /= fps
        data = velocities_column_to_data(data, **{"aliasing": 3, "n_procs": 8})
        np.save(file, data)
    
