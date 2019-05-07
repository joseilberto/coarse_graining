class Coarse_Base:
    @property
    def density(self):
        return self.kwargs.get("density", 7850)


    @property
    def epsilon(self):
        return self.kwargs.get("epsilon", 4)


    @property
    def R(self):
        return self.kwargs.get("R", None)


    @property
    def cell_size(self):
        n_points = self.kwargs.get("n_points", self.epsilon * 4)
        if n_points < self.epsilon * 2:
            n_points = self.epsilon * 2
        while not n_points % self.epsilon == 0:
            n_points += 1
        return self.epsilon * self.R / n_points
