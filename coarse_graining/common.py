import tensorflow as tf


class Coarse_Base:
    @property
    def density(self):
        return self.kwargs.get("density", 7850)


    @property
    def epsilon(self):
        return self.kwargs.get("epsilon", 4)


    @property
    def W(self):
        return self.kwargs.get("W", None)


    @property
    def n_points(self):
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
        return tf.stack([minimum, maximum], axis = 1)