import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng()
class LeonardJones:
    def __init__(self, size, n_particles, sigma, v_th):
        self.size = size
        self.n_particles = n_particles
        self.positions = rng.uniform(low=0, high=size, size=(n_particles, 3))
        self.velocities = rng.normal(loc=0, scale=v_th, size=(n_particles, 3))
        self.sigma = sigma
        self.particle_tracker = (self.positions * self.boxes_by_axis) // (self.size)
    