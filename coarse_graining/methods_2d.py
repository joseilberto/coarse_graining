import numpy as np
import tensorflow as tf

from .common import Coarse_Base


class Coarse_Graining(Coarse_Base):
    def __init__(self, density = None, epsilon = None, function = "gaussian",
                    limits = [], n_points = None, W = None, *args, **kwargs):
        tf.reset_default_graph()
        self._calc_class = CG_Calculator(density = density, epsilon = epsilon, 
                    function = function, limits = limits, n_points = n_points, 
                    W = W, *args, **kwargs)
        super().__init__(density = density, epsilon = epsilon, 
                    function = function, limits = limits, n_points = n_points, 
                    W = W, *args, **kwargs)


    def densities(self, X, Y, radii, *args, **kwargs):
        self._calc_class.fill_density_grid(X, Y, radii, *args, **kwargs)
        extras = ["densities_grid", "densities_grid_raveled"]
        self._transfer_variables(from_class = self._calc_class, to_class = self, 
                                    extras = extras)
        return np.column_stack((self.positions, self.densities_grid_raveled))
        

    def kinetic_stress(self, X, Y, V_X, V_Y, radii, *args, **kwargs):
        self._calc_class.fill_kinetic_stress_grid(X, Y, V_X, V_Y, radii, *args, 
                                                    **kwargs)
        extras = ["densities_grid", "densities_grid_raveled", "kinetic_grid", 
                "kinetic_trace", "kinetic_grid_raveled", "velocities_grid", 
                "velocities_grid_raveled", "momenta_grid", 
                "momenta_grid_raveled"]
        self._transfer_variables(from_class = self._calc_class, to_class = self, 
                                    extras = extras)
        return np.column_stack((self.positions, self.kinetic_grid_raveled))

    
    def momenta(self, X, Y, V_X, V_Y, radii, *args, **kwargs):
        self._calc_class.fill_momenta_grid(X, Y, V_X, V_Y, radii, *args, 
                                                    **kwargs)
        extras = ["densities_grid", "densities_grid_raveled", "velocities_grid", 
                    "velocities_grid_raveled", "momenta_grid", 
                    "momenta_grid_raveled"]
        self._transfer_variables(from_class = self._calc_class, to_class = self, 
                                    extras = extras)
        return np.column_stack((self.positions, self.momenta_grid_raveled))
        

class CG_Calculator(Coarse_Base):
    @staticmethod
    def find_centers(distances, minima):
        minima = np.ravel(minima)        
        centers = np.empty((len(minima), 2)).astype(np.int32)        
        for idx, dist_matrix in enumerate(distances):
            min_idxs = np.where(dist_matrix == minima[idx])
            assert len(min_idxs[0]) > 0
            centers[idx] = np.array([min_idxs[0][0], min_idxs[1][0]])
        return centers
        

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

    
    def densities_at_borders(self, updates, *args, **kwargs):
        borders = tf.stack([updates[:, :, 0], updates[:, :, -1], 
        updates[:, 0, :], updates[:, -1, :]], axis = 2)
        return tf.reduce_mean(borders, axis = [1, 2])


    def fill_density_grid(self, X, Y, radii, *args, **kwargs):
        if not hasattr(self, "xx"):
            self.start_grids_and_variables(X, Y, *args, **kwargs)
        if not hasattr(self, "session"):
            self.session = tf.InteractiveSession()
        if not hasattr(self, "idxs"):
            self.idxs = self.find_indexes(self.pos, [self.xx, self.yy])
        if not hasattr(self, "regions"):           
            self.regions = self.find_regions(X, Y)
        masses = self.find_sphere_masses(self.density, self.radii)
        density_updates = self.updater_densities(self.regions, masses)
        densities_borders = self.densities_at_borders(density_updates)
        init = tf.global_variables_initializer()
        self.session.run(init)        
        idxs, density_updates, self.densities_borders = self.session.run(
                        (self.idxs, density_updates, densities_borders),
                        feed_dict = {
                            self.pos: np.column_stack((X, Y)),
                            self.radii: radii,
                        })   
        self.session.close()
        del self.session
        self.densities_grid = self.update_grid(self.densities_grid, 
                                                        density_updates, idxs)
        self.densities_grid_raveled = self.ravel_grid(self.densities_grid)         

    def fill_kinetic_stress_grid(self, X, Y, V_X, V_Y, radii, *args, **kwargs):       
        if not hasattr(self, "velocity_updates"):
            self.fill_momenta_grid(X, Y, V_X, V_Y, radii)
        if not hasattr(self, "session"):            
            self.session = tf.InteractiveSession()
        masses = self.find_sphere_masses(self.density, self.radii)
        kinetic_updates = self.updater_kinetic_stresses(self.fn_term, masses,
                                                    self.vels, 
                                                    self.velocities_grid)
        init = tf.global_variables_initializer()
        self.session.run(init)
        idxs, kinetic_updates = self.session.run((self.idxs, kinetic_updates),
                            feed_dict = {
                                self.pos: np.column_stack((X, Y)),
                                self.vels: np.column_stack((V_X, V_Y)),
                                self.radii: radii,
                            })
        self.session.close()
        del self.session        
        self.kinetic_grid = self.update_grid(self.kinetic_grid, kinetic_updates)
        self.kinetic_trace = (self.kinetic_grid[:, :, 0, 0] + 
                                self.kinetic_grid[:, :, 1, 1])
        self.kinetic_grid_raveled = self.ravel_grid(self.kinetic_trace)   


    def fill_momenta_grid(self, X, Y, V_X, V_Y, radii, *args, **kwargs): 
        if not hasattr(self, "densities_grid_raveled"):
            self.fill_density_grid(X, Y, radii)
        if not hasattr(self, "session"):
            self.session = tf.InteractiveSession()        
        masses = self.find_sphere_masses(self.density, self.radii)        
        momenta_updates = self.updater_momenta(self.regions, masses, 
                                                    self.vels)
        init = tf.global_variables_initializer()
        self.session.run(init)
        idxs, momenta_updates = self.session.run(
                        (self.idxs, momenta_updates),
                        feed_dict = {
                            self.pos: np.column_stack((X, Y)),
                            self.vels: np.column_stack((V_X, V_Y)),
                            self.radii: radii,
                        })
        self.session.close()
        del self.session
        self.momenta_grid = self.update_grid(self.momenta_grid, momenta_updates,
                                                idxs)        
        self.momenta_grid_raveled = self.ravel_grid(self.momenta_grid)
        self.velocities_grid[:, :, 0] = np.nan_to_num(
                            self.momenta_grid[:, :, 0] / self.densities_grid)
        self.velocities_grid[:, :, 1] = np.nan_to_num(
                            self.momenta_grid[:, :, 1] / self.densities_grid)
        self.velocities_grid_raveled = self.ravel_grid(self.velocities_grid)                


    def find_indexes(self, positions, grid_centers, *args, **kwargs):
        """
        Find the minima and maxima for the batch of particles presented. 
        When applying tf.where, it returns a tensor with shape 
        (?, [idx_y, idx_x]), this is why we invert it and create maximas 
        and minimas stacking then together. Idxs ordering is (?, [[min_x, min_y],
        [max_x, max_y]])
        """
        self.distances = self.calculate_distances(positions, grid_centers)
        zeros = tf.zeros(shape = tf.shape(self.distances))
        fn_term = tf.exp(-self.distances**2 / (2 * self.W**2))        
        fn_term = tf.where(self.distances > self.epsilon*self.W, zeros, fn_term)
        volume_fraction = tf.reshape(tf.reduce_sum(fn_term, axis = [1, 2]),
                                        [-1, 1, 1])        
        self.fn_term = fn_term / (volume_fraction * self.cell_size**2)
        min_distances = tf.reduce_min(self.distances, axis = [1, 2], 
                                        keepdims = True)         
        centers = tf.py_func(self.find_centers, [self.distances, min_distances], 
                                            [tf.int32])        
        centers = tf.convert_to_tensor(centers[0], dtype = tf.int32) 
        xs = centers[:, 1]
        ys = centers[:, 0]
        self.centers = tf.stack([xs, ys], axis = 1)
        minima = self.centers[:, 0:2] - self.n_points
        maxima = self.centers[:, 0:2] + self.n_points
        idxs = tf.stack([minima, maxima], axis = 2)
        return tf.transpose(idxs, perm = [0, 2, 1])


    def find_regions(self, X, Y, *args, **kwargs):
        region_idxs, fn_term = self.session.run((self.idxs, self.fn_term), 
                            feed_dict = {self.pos: np.column_stack((X, Y))})
        fn_term_regions = []
        for idx, region in enumerate(region_idxs):
            min_x, max_x = region[:, 0]
            min_y, max_y = region[:, 1]
            fn_term_regions.append(fn_term[idx][min_y:max_y, min_x:max_x])
        return tf.Variable(np.stack(fn_term_regions, axis = 0), 
                                                        dtype = tf.float32)
        

    def find_sphere_masses(self, density, radii, *args, **kwargs):
        return density*np.pi*radii**2

    
    def ravel_meshes(self, xs, ys, *args, **kwargs):
        combinations = np.stack((xs, ys), axis = 2)
        y_dim, x_dim, dims = combinations.shape
        return combinations.reshape((y_dim*x_dim, dims))


    def ravel_grid(self, grid, *args, **kwargs):
        try:
            y_dim, x_dim, dims = grid.shape
            return grid.reshape((y_dim*x_dim, dims))
        except ValueError:
            y_dim, x_dim = grid.shape
            return grid.reshape(y_dim*x_dim)

 
    def start_grids_and_variables(self, X, Y, *args, **kwargs):
        limits = getattr(self, "limits", None)        
        if not limits:
            limits = [X.min(), X.max(), Y.min(), Y.max()]
        limits = self.extend_limits(limits)        
        xs = np.arange(*limits[:2], self.cell_size)
        ys = np.arange(*limits[2:], self.cell_size)
        self.xx, self.yy = np.meshgrid(xs, ys)
        self.positions = self.ravel_meshes(self.xx, self.yy)        
        self.densities_grid = np.zeros(self.xx.shape)
        self.velocities_grid = np.zeros(self.xx.shape + (2,))
        self.momenta_grid = np.zeros(self.xx.shape + (2,))
        self.kinetic_grid = np.zeros(self.xx.shape + (2, 2))
        self.pos = tf.placeholder(tf.float32, shape = (None, 2))
        self.vels = tf.placeholder(tf.float32, shape = (None, 2))
        self.radii = tf.placeholder(tf.float32, shape = (None, ))


    def update_grid(self, grid, updates, idxs = [], *args, **kwargs):
        if np.any(idxs):
            for idx, region in enumerate(idxs):
                min_x, max_x = region[:, 0]
                min_y, max_y = region[:, 1]
                grid[min_y:max_y, min_x:max_x] += updates[idx]
        else:
            for update in updates:
                grid += update            
        return grid


    def updater_densities(self, fn_terms, masses, *args, **kwargs):
        masses = tf.reshape(masses, [-1, 1, 1])                                
        return masses * fn_terms

    
    def updater_kinetic_stresses(self, fn_term, masses, velocities, 
                                    velocities_grid, *args, **kwargs):
        masses = tf.reshape(masses, [-1, 1, 1])
        vx = tf.reshape(velocities[:, 0], [-1, 1, 1])
        vy = tf.reshape(velocities[:, 1], [-1, 1, 1])
        prime_vx = vx - velocities_grid[:, :, 0]
        prime_vy = vy - velocities_grid[:, :, 1]
        vxx = masses * prime_vx * prime_vx * fn_term
        vxy = masses * prime_vx * prime_vy * fn_term
        vyx = masses * prime_vy * prime_vx * fn_term
        vyy = masses * prime_vy * prime_vy * fn_term
        vxs = tf.stack((vxx, vxy), axis = 3)
        vys = tf.stack((vyx, vyy), axis = 3)
        return tf.stack((vxs, vys), axis = 4)


    def updater_momenta(self, fn_terms, masses, velocities, *args, **kwargs):                
        masses = tf.reshape(masses, [-1, 1, 1])
        velocity_x = tf.reshape(velocities[:, 0], [-1, 1, 1])*masses*fn_terms
        velocity_y = tf.reshape(velocities[:, 1], [-1, 1, 1])*masses*fn_terms
        return tf.stack([velocity_x, velocity_y], axis = 3)
