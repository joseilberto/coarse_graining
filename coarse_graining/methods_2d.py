import numpy as np
import tensorflow as tf

from .common import Coarse_Base, idx_nearest

class Coarse_Graining(Coarse_Base):
    def __init__(self, function = "gaussian", *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs


    def calculate_densities(self, idxs, *args, **kwargs):
        masses = self.find_sphere_masses()
        x_grids = tf.ragged.range(starts = idxs[:, 0, 0], limits = idxs[:, 0, 1])
        y_grids = tf.ragged.range(starts = idxs[:, 1, 0], limits = idxs[:, 1, 1])


    def find_indexes(self, array, values, grid, *args, **kwargs):
        idxs_center = idx_nearest(array, values)
        minima = tf.subtract(idxs_center, self.n_points)
        maxima = tf.add(idxs_center, self.n_points)
        shape_tile = tf.constant([len(grid) - 1], dtype = tf.int32)
        zeros_reference = tf.zeros(shape = tf.shape(minima), dtype = tf.int32)
        maxima_reference = tf.tile(shape_tile, tf.shape(maxima))
        minima = tf.where(minima <= 0, zeros_reference, minima)
        maxima = tf.where(maxima > len(grid) - 1, maxima_reference, maxima)
        return tf.stack([minima, maxima, idxs_center], axis = 1)


    def find_sphere_masses(self, *args, **kwargs):
        return tf.multiply(self.radii, (4/3)*np.pi*self.density)


    def density_grid_updater(self, *args, **kwargs):
        idxs_x = self.find_indexes(self.xs, self.pos[:, 0], self.xx)
        idxs_y = self.find_indexes(self.ys, self.pos[:, 1], self.yy)
        self.idxs = tf.stack([idxs_x, idxs_y], axis = 2)
        densities_updates = self.calculate_densities(self.idxs)


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
        with tf.Session() as session:
            tf.global_variables_initializer().run()
            test_var = session.run(self.idxs,
                                  feed_dict = {
                                      self.pos: np.column_stack((X, Y)),
                                      self.radii: radii,
                                  })
