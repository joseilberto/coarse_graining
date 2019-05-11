import numpy as np
import tensorflow as tf

from .common import Coarse_Base

class Coarse_Graining(Coarse_Base):
    def calculate_densities(self, regions, masses, *args, **kwargs):
        pass


    def calculate_distances(self, positions, grid_centers, *args, **kwargs):
        X = tf.reshape(positions[:, 0], [-1, 1, 1])
        Y = tf.reshape(positions[:, 1], [-1, 1, 1])
        return tf.sqrt((grid_centers[0] - X)**2 + (grid_centers[1] - Y)**2)

    
    def extend_limits(self, limits, *args, **kwargs):
        min_x = limits[0] - 2*self.n_points*self.cell_size
        max_x = limits[1] + 2*self.n_points*self.cell_size
        min_y = limits[2] - 2*self.n_points*self.cell_size
        max_y = limits[3] + 2*self.n_points*self.cell_size
        return [min_x, max_x, min_y, max_y]


    def find_indexes(self, positions, grid_centers, *args, **kwargs):
        """
        Find the minima and maxima for the batch of particles presented. 
        When applying tf.where, it returns a tensor with shape 
        (?, [idx_y, idx_x]), this is why we invert it and create maximas 
        and minimas stacking then together. Idxs ordering is (?, [[min_x, min_y],
        [max_x, max_y]])
        """
        self.distances = self.calculate_distances(positions, grid_centers)
        min_distances = tf.reduce_min(self.distances, axis = [1, 2], 
                                        keepdims = True)
        centers = tf.where(tf.equal(self.distances, min_distances))[:, 1:3] 
        xs = centers[:, 1]
        ys = centers[:, 0]
        self.centers = tf.stack([xs, ys], axis = 1)
        minima = self.centers[:, 0:2] - self.n_points
        maxima = self.centers[:, 0:2] + self.n_points
        idxs = tf.stack([minima, maxima], axis = 2)
        return tf.transpose(idxs, perm = [0, 2, 1])


    def find_regions(self, X, Y, *args, **kwargs):
        region_idxs, distances = self.session.run((self.idxs, self.distances), 
                            feed_dict = {self.pos: np.column_stack((X, Y))})
        regions = []
        for idx, region in enumerate(region_idxs):
            min_x, max_x = region[:, 0]
            min_y, max_y = region[:, 1]
            regions.append(distances[idx][min_y:max_y, min_x:max_x])
        return tf.Variable(np.stack(regions, axis = 0), dtype = tf.float32)
        

    def find_sphere_masses(self, *args, **kwargs):
        return self.radii*(4/3)*np.pi*self.density


    def density_grid_updater(self, X, Y, *args, **kwargs):
        self.idxs = self.find_indexes(self.pos, [self.xx, self.yy])        
        self.regions = self.find_regions(X, Y)
        masses = self.find_sphere_masses()
        density_updates = self.calculate_densities(self.regions, masses)        


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
        self.xx, self.yy = np.meshgrid(self.xs, self.ys)
        self.densities = np.zeros(self.xx.shape)
        self.momentum = np.zeros(self.xx.shape + (2,))
        self.kinetic = np.zeros(self.xx.shape + (4,))
        self.kinect_trace = np.zeros(self.xx.shape)


    def kinetic_stress(self, X, Y, V_X, V_Y, radii, *args, **kwargs):
        self.session = tf.InteractiveSession()
        self.start_tf_variables()
        self.start_grids(X, Y, *args, **kwargs)
        self.density_grid_updater(X, Y)
        init = tf.global_variables_initializer()
        self.session.run(init)
        test_var, test_var2 = self.session.run((self.idxs, self.x_grids),
                        feed_dict = {
                            self.pos: np.column_stack((X, Y)),
                            self.radii: radii,
                        })        
        import ipdb; ipdb.set_trace()
        self.session.close()