import tensorflow as tf


class Coarse_Base:
    def __init__(self, density = None, epsilon = None, function = "gaussian",
                    limits = [], n_points = None, W = None, *args, **kwargs):
        self._density = density
        self.function = function
        self._epsilon = epsilon
        self._n_points = n_points
        self._limits = limits
        self._W = W
        self.args = args
        self.kwargs = kwargs


    @property
    def density(self):
        density = getattr(self, "_density", None)
        if not density:            
            density = self.kwargs.get("density", 7850)
        return density


    @property
    def epsilon(self):
        epsilon = getattr(self, "_epsilon", None)
        if not epsilon:
            epsilon = self.kwargs.get("epsilon", 4)
        return epsilon

    
    @property
    def limits(self):
        limits = getattr(self, "_limits", None)
        if not limits:
            limits = self.kwargs.get("limits", None)
        return limits


    @property
    def W(self):
        W = getattr(self, "_W", None)
        if not W:
            W = self.kwargs.get("W", 4)
        return W


    @property
    def n_points(self):
        n_points = getattr(self, "_n_points", None)
        if not n_points:            
            n_points = self.kwargs.get("n_points", self.epsilon * 4)
            if n_points < self.epsilon * 2:
                n_points = self.epsilon * 2
            while not n_points % self.epsilon == 0:
                n_points += 1
        return n_points


    @property
    def cell_size(self):
        return self.epsilon * self.W / self.n_points

    
    def fill_density_grids(self, grids, centers, masses, *args, **kwargs):
        pass


    def constrain_idxs(self, idxs, grid_length, *args, **kwargs):
        shape_tile = tf.constant([grid_length], dtype = tf.int64)
        zeros_reference = tf.zeros(shape = tf.shape(idxs[:, 0]), 
                                    dtype = tf.int64)        
        maxima_reference = tf.tile(shape_tile, tf.shape(idxs[:, 1]))
        minimum = tf.where(idxs[:, 0] <= 0, zeros_reference, idxs[:, 0])
        maximum = tf.where(idxs[:, 1] > grid_length, maxima_reference, 
                            idxs[:, 1])
        length = maximum - minimum
        return tf.stack([minimum, length], axis = 1)