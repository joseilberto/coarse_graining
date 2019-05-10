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


