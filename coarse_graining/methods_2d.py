import numpy as np
import tensorflow as tf

from .common import Coarse_Base

class Coarse_Graining(Coarse_Base):
    def calculate_densities(self, idxs, *args, **kwargs):
        masses = self.find_sphere_masses()
        centers = tf.stack([idxs[:, 2, 0], idxs[:, 2, 1]], axis = 1)        
        x_grids = tf.ragged.range(starts = idxs[:, 0, 0], limits = idxs[:, 0, 1])
        y_grids = tf.ragged.range(starts = idxs[:, 1, 0], limits = idxs[:, 1, 1])        
        densities_x = self.fill_density_grids(x_grids, centers[:, 0], masses)
        densities_y = self.fill_density_grids(y_grids, centers[:, 1], masses)

    
    def calculate_distances(self, positions, grid_centers, *args, **kwargs):
        X = tf.reshape(positions[:, 0], [-1, 1, 1])
        Y = tf.reshape(positions[:, 1], [-1, 1, 1])
        return tf.sqrt((grid_centers[0] - X)**2 + (grid_centers[1] - Y)**2)

    
    def extend_limits(self, limits, *args, **kwargs):
        min_x = limits[0]*1.25 if limits[0] < 0 else limits[0]*0.75
        max_x = limits[1]*1.25 if limits[1] >= 0 else limits[1]*0.75
        min_y = limits[2]*1.25 if limits[2] < 0 else limits[2]*0.75
        max_y = limits[3]*1.25 if limits[3] >= 0 else limits[3]*0.75
        max_x += self.cell_size
        max_y += self.cell_size
        return [min_x, max_x, min_y, max_y]


    def find_indexes(self, positions, grid_centers, *args, **kwargs):
        self.distances = self.calculate_distances(positions, grid_centers)
        min_distances = tf.reduce_min(self.distances, axis = [1, 2], 
                                        keepdims = True)
        centers = tf.where(tf.equal(self.distances, min_distances))[:, 1:3]
        minima = centers[:, 0:2] - self.n_points
        maxima = centers[:, 0:2] + self.n_points
        idxs = tf.stack([minima, maxima], axis = 2)
        xs_idxs = self.constrain_idxs(idxs[:, 0, :], 
                                            grid_centers[0].shape[0] - 1)
        ys_idxs = self.constrain_idxs(idxs[:, 1, :], 
                                            grid_centers[0].shape[1] - 1)
        minima = tf.stack([xs_idxs[:, 0], ys_idxs[:, 0]], axis = 1)
        maxima = tf.stack([xs_idxs[:, 1], ys_idxs[:, 1]], axis = 1)
        idxs = tf.stack([minima, maxima, centers], axis = 1)
        return tf.transpose(idxs, perm = [0, 2, 1])
        

    def find_sphere_masses(self, *args, **kwargs):
        return self.radii*(4/3)*np.pi*self.density


    def density_grid_updater(self, *args, **kwargs):
        self.idxs = self.find_indexes(self.pos, [self.xx, self.yy])        
        # self.densities_updates = self.calculate_densities(self.idxs)


    def start_tf_variables(self, *args, **kwargs):
        self.pos = tf.placeholder(tf.float32, shape = (None, 2))
        self.vels = tf.placeholder(tf.float32, shape = (None, 2))
        self.radii = tf.placeholder(tf.float32, shape = (None, ))


    def start_grids(self, X, Y, *args, **kwargs):
        limits = getattr(self, "limits", None)        
        if not limits:
            limits = [X.min(), X.max(), Y.min(), Y.max()]
        limits = self.extend_limits(limits)        
        self.xs = np.arange(*limits[:2], self.cell_size)
        self.ys = np.arange(*limits[2:], self.cell_size)
        self.xx, self.yy = np.meshgrid(self.ys, self.xs)
        self.densities = np.zeros(self.xx.shape)
        self.momentum = np.zeros(self.xx.shape + (2,))
        self.kinetic = np.zeros(self.xx.shape + (4,))
        self.kinect_trace = np.zeros(self.xx.shape)


    def kinetic_stress(self, X, Y, V_X, V_Y, radii, *args, **kwargs):
        self.start_tf_variables()
        self.start_grids(X, Y, *args, **kwargs)
        self.density_grid_updater()
        session = tf.Session()
        init = tf.global_variables_initializer()
        session.run(init)
        test_var = session.run(self.idxs,
                        feed_dict = {
                            self.pos: np.column_stack((X, Y)),
                            self.radii: radii,
                        })