import matplotlib
matplotlib.interactive(False)
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
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
    

    def _fill_plot(self, X, Y, data, plot_type = None, ax = None, 
                    label_axis = None, *args, **kwargs):
        if "density" == plot_type:
            colorbar_ylabel = r"$\rho (kg/m^2)$"
        elif "density_gradient" == plot_type:
            colorbar_ylabel = r"$\nabla \rho (kg / m^3)$"
        elif "velocity" in plot_type:
            colorbar_ylabel = r"$v (m/s)$"
        elif "momentum" in plot_type:
            colorbar_ylabel = r"$p (N \, \cdotp s)$"
        elif "kinetic" in plot_type:
            colorbar_ylabel = r"$Tr(\sigma_{ab}) (N \, \cdotp m)$"
        scatter_plot = ax.scatter(X, Y, c = data, cmap = "coolwarm", s = 0.2)
        if label_axis:
            ax.set_xlabel(r"$x/D$")
            ax.set_ylabel(r"$y/D$")
        color_bar = plt.colorbar(scatter_plot, ax = ax)
        color_bar.ax.set_ylabel(colorbar_ylabel)                
    

    def plot_raveled(self, X, Y, data, plot_type = None, label_axis = None, 
                        *args, **kwargs):
        fig, ax = plt.subplots()
        self._fill_plot(X, Y, data, plot_type, ax = ax, label_axis = label_axis, 
                        *args, **kwargs)
        plt.show()

    
    def plot_grid(self, xs, ys, grid, plot_type = None, *args, **kwargs):
        if "density" == plot_type:
            colorbar_ylabel = r"$\rho (kg/m^2)$"
        elif "density_gradient" == plot_type:
            colorbar_ylabel = r"$\nabla \rho (kg / m^3)$"
        elif "velocity" in plot_type:
            colorbar_ylabel = r"$|\mathbf{v}| (m/s)$"
        elif "momentum" in plot_type:
            colorbar_ylabel = r"$p (N \, \cdotp s)$"
        elif "kinetic" in plot_type:
            colorbar_ylabel = r"$Tr(\sigma_{ab}) (N / m)$"
        fig, ax = plt.subplots()
        pcolor = ax.pcolor(xs, ys, grid, cmap = "coolwarm")
        ax.set_xlabel(r"$x (m)$")
        ax.set_ylabel(r"$y (m)$")
        color_bar = plt.colorbar(pcolor)
        color_bar.ax.set_ylabel(colorbar_ylabel)
        plt.show()


    def _transfer_variables(self, from_class, to_class, extras, *args, **kwargs):
        variables = ["xx", "yy", "positions"]
        variables.extend(extras)
        for variable in variables:
            if hasattr(from_class, variable):                
                setattr(to_class, variable, getattr(from_class, variable))


    #TODO Plotly can be used to make the plots and animations