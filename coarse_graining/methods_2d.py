import numpy as np
import tensorflow as tf

from .common import Coarse_Base, idx_nearest

class Coarse_Graining(Coarse_Base):
    def __init__(self, function = "gaussian", *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs


    def density_grid_updater(self, *args, **kwargs):
        idx_x_center = idx_nearest(self.xs, self.pos[:, 0])
        idx_y_center = idx_nearest(self.ys, self.pos[:, 1])
        #TODO Generalize it, cause it stinks
        min_x = tf.subtract(idx_x_center, self.n_points)
        max_x = tf.add(idx_x_center, self.n_points)
        shape_maxs = tf.constant([len(self.xx) - 1], dtype = tf.int64)
        mins_x = tf.zeros(shape = tf.shape(min_x), dtype = tf.int64)
        maxs_x = tf.tile(shape_maxs, max_x)
        min_x = tf.where(min_x <= 0, mins_x, min_x)
        max_x = tf.where(max_x > len(self.xx) - 1, maxs_x, max_x)

        self.densities = tf.ones(self.xx.shape)



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
            session.run(tf.global_variables_initializer())
            density = session.run(self.densities,
                                  feed_dict = {
                                      self.pos: np.column_stack((X, Y)),
                                      self.radii: radii,
                                  })
