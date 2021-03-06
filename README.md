# Particle coarse-graining for 2D and 3D systems #

A simple implementation of coarse-graining for monodisperse or polydisperse spherical particles in both 2D and 3D systems.
The relevant coarse-grained quantities are the required to calculate the kinetic stress tensor defined by
[Goldhirsch](https://link.springer.com/article/10.1007/s10035-010-0181-z). All the requirements are in the file requirements.sh.

## Usage ##

Add some example later.

## TODO List  ##

Coarse-graining parameters:
- [X] System type: 2D;
- [ ] System type: 3D;
- [X] Particle density (rho);
- [X] Particle radius if not contained in file (R);
- [X] Grid size of particle coarse-graining ("n_points");
- [X] Threshold where function is truncated (threshold);
- [X] Coarse-graining radius (w) with two options (max radius or min radius if polydisperse).

Reading data:
- [X] Read data from different formats (npy, csv and txt supported only);
- [X] Allow data to be read by specifying columns where the data is present.
- [X] Calculating velocities (if not given) with aliasing for indexed data.

Lattice:
- [ ] Handle lattices that does not fit into memory (CPU and GPU cases);
- [X] Methods for 2D and 3D systems designed separately;
- [X] Momentum fields;
- [X] Density fields;
- [X] Trace of kinetic stress tensor fields;
- [ ] Data should be stored in .npy or .csv files.

Visualization:
- [X] For all fields:
- [ ] Images should be retrieved;
- [ ] Videos should be recorded.
