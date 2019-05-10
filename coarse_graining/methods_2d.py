import numpy as np
import tensorflow as tf

from .common import Coarse_Base

class Coarse_Graining(Coarse_Base):
    def __init__(self, function = "gaussian", *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs


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


    def find_indexes(self, positions, grid_centers, *args, **kwargs):
        self.distances = self.calculate_distances(positions, grid_centers)
        min_distances = tf.reduce_min(self.distances, axis = [1, 2], 
                                        keepdims = True)
        centers = tf.where(tf.equal(self.distances, min_distances))[:, 1:3]
        minima = centers[:, 0:2] + self.n_points
        maxima = centers[:, 0:2] + self.
        


    def find_sphere_masses(self, *args, **kwargs):
        return self.radii*(4/3)*np.pi*self.density


    def density_grid_updater(self, *args, **kwargs):
        self.find_indexes(self.pos, [self.xx, self.yy])        
        # densities_updates = self.calculate_densities(self.idxs)


    def start_tf_variables(self, *args, **kwargs):
        self.pos = tf.placeholder(tf.float32, shape = (None, 2))
        self.vels = tf.placeholder(tf.float32, shape = (None, 2))
        self.radii = tf.placeholder(tf.float32, shape = (None, ))


    def start_grids(self, X, Y, *args, **kwargs):
        min_x = X.min()*1.25 if X.min() < 0 else X.min()*0.75
        max_x = X.max()*1.25 if X.max() >= 0 else X.max()*0.75
        min_y = Y.min()*1.25 if Y.min() < 0 else Y.min()*0.75
        max_y = Y.max()*1.25 if Y.max() >= 0 else Y.max()*0.75
        max_x += self.cell_size
        max_y += self.cell_size
        self.xs = np.arange(min_x, max_x, self.cell_size)
        self.ys = np.arange(min_y, max_y, self.cell_size)
        self.xx, self.yy = np.meshgrid(self.ys, self.xs)
        self.densities = tf.zeros(self.xx.shape)
        self.momentum = tf.zeros(self.xx.shape + (2,))
        self.kinetic = tf.zeros(self.xx.shape + (4,))
        self.kinect_trace = tf.zeros(self.xx.shape)


    def kinetic_stress(self, X, Y, V_X, V_Y, radii, *args, **kwargs):
        self.start_tf_variables()
        self.start_grids(X, Y, *args, **kwargs)
        self.density_grid_updater()
        session = tf.Session()
        init = tf.global_variables_initializer()
        session.run(init)
        test_var = session.run(self.idxs_min,
                        feed_dict = {
                            self.pos: np.column_stack((X, Y)),
                            self.radii: radii,
                        })
        import ipdb; ipdb.set_trace()