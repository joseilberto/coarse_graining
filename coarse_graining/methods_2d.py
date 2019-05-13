import numpy as np
import tensorflow as tf

from .common import Coarse_Base

class Coarse_Graining(Coarse_Base):
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


    def fill_density_grid(self, X, Y, radii, *args, **kwargs):
        if not hasattr(self, "session"):
            self.session = tf.InteractiveSession()
        if not hasattr(self, "idxs"):
            self.idxs = self.find_indexes(self.pos, [self.xx, self.yy])
        if not hasattr(self, "regions"):           
            self.regions = self.find_regions(X, Y)
        masses = self.find_sphere_masses()
        density_updates = self.updater_densities(self.regions, masses)
        init = tf.global_variables_initializer()
        self.session.run(init)        
        idxs, density_updates = self.session.run((self.idxs, density_updates),
                        feed_dict = {
                            self.pos: np.column_stack((X, Y)),
                            self.radii: radii,
                        })   
        self.session.close()
        del self.session
        self.densities = self.update_grid(self.densities, idxs, density_updates)
        self.densities_raveled = self.ravel_grid(self.densities)
        self.densities_grid_plot = self.plot_grid(self.xx, self.yy, 
                                    self.densities, plot_type = "density")                        


    def fill_momenta_grid(self, X, Y, V_X, V_Y, radii, *args, **kwargs):
        if not hasattr(self, "densities_raveled"):
            self.fill_density_grid(X, Y, radii)
        if not hasattr(self, "session"):
            self.session = tf.InteractiveSession()
        if not hasattr(self, "idxs"):
            self.idxs = self.find_indexes(self.pos, [self.xx, self.yy])
        if not hasattr(self, "regions"):           
            self.regions = self.find_regions(X, Y)
        masses = self.find_sphere_masses()
        velocity_updates = self.updater_velocities(self.regions)
        init = tf.global_variables_initializer()
        self.session.run(init)
        idxs, velocity_updates = self.session.run((self.idxs, velocity_updates),
                        feed_dict = {
                            self.pos: np.column_stack((X, Y)),
                            self.vels: np.column_stack((V_X, V_Y)),
                            self.radii: radii,
                        })
        self.session.close()
        del self.session
        self.velocities = self.update_grid(self.velocities, idxs, 
                                                    velocity_updates)
        total_velocity = np.sqrt(self.velocities[:, :, 0]**2 
                                    + self.velocities[:, :, 1]**2)
        self.velocities_raveled = self.ravel_grid(self.velocities)
        self.velocity_grid_plot = self.plot_grid(self.xx, self.yy, 
                                    total_velocity, plot_type = "velocity")        
        self.momenta[:, :, 0] = self.velocities[:, :, 0] * self.densities
        self.momenta[:, :, 1] = self.velocities[:, :, 1] * self.densities
        self.momenta_raveled = self.ravel_grid(self.momenta)                        


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
        return 4*np.pi*self.radii
        

    def kinetic_stress(self, X, Y, V_X, V_Y, radii, *args, **kwargs):
        self.make_updates(X, Y, V_X, V_Y, radii, *args, **kwargs)


    def make_updates(self, X, Y, V_X, V_Y, radii, *args, **kwargs):
        self.start_grids_and_variables(X, Y, *args, **kwargs)        
        self.fill_density_grid(X, Y, radii)
        self.fill_momenta_grid(X, Y, V_X, V_Y, radii)              

    
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
            grid.reshape(y_dim*x_dim)

 
    def start_grids_and_variables(self, X, Y, *args, **kwargs):
        limits = getattr(self, "limits", None)        
        if not limits:
            limits = [X.min(), X.max(), Y.min(), Y.max()]
        limits = self.extend_limits(limits)        
        self.xs = np.arange(*limits[:2], self.cell_size)
        self.ys = np.arange(*limits[2:], self.cell_size)
        self.xx, self.yy = np.meshgrid(self.xs, self.ys)
        self.positions = self.ravel_meshes(self.xx, self.yy)        
        self.densities = np.zeros(self.xx.shape)
        self.velocities = np.zeros(self.xx.shape + (2,))
        self.momenta = np.zeros(self.xx.shape + (2,))
        self.kinetic = np.zeros(self.xx.shape + (4,))
        self.kinect_trace = np.zeros(self.xx.shape)
        self.pos = tf.placeholder(tf.float32, shape = (None, 2))
        self.vels = tf.placeholder(tf.float32, shape = (None, 2))
        self.radii = tf.placeholder(tf.float32, shape = (None, ))


    def update_grid(self, grid, idxs, updates, *args, **kwargs):
        for idx, region in enumerate(idxs):
            min_x, max_x = region[:, 0]
            min_y, max_y = region[:, 1]
            grid[min_y:max_y, min_x:max_x] += updates[idx]            
        return grid


    def updater_densities(self, regions, masses, *args, **kwargs):
        if not hasattr(self, "volume_fraction"):
            self.volume_fraction = tf.exp(-regions**2 / (2 * self.W**2))
        total_volume = tf.reduce_sum(self.volume_fraction, axis = [1, 2])
        mass_density = masses*(1/self.cell_size**2)
        scaled_mass = tf.reshape(mass_density / total_volume, [-1, 1, 1])        
        return scaled_mass * self.volume_fraction
    

    def updater_velocities(self, regions, *args, **kwargs):
        if not hasattr(self, "volume_fraction"):
            self.volume_fraction = tf.exp(-regions**2 / (2 * self.W**2))
        total_volume = tf.reduce_sum(self.volume_fraction, axis = [1, 2])
        velocity_x = tf.reshape(self.vels[:, 0]/total_volume, 
                                        [-1, 1, 1])*self.volume_fraction
        velocity_y = tf.reshape(self.vels[:, 1]/total_volume, 
                                        [-1, 1, 1])*self.volume_fraction
        return tf.stack([velocity_x, velocity_y], axis = 3)