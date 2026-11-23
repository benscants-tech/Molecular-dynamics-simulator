import numpy as np
from scipy.stats import maxwell
from numba import njit
rng = np.random.default_rng


class ideal_sim:
    def __init__(self, size, n_particles, particle_radius, v_th, boxes_by_axis):
        self.size = size
        self.n_particles = n_particles
        self.positions = rng.uniform(low=0, high=size, size=(n_particles, 3))
        self.velocities = rng.normal(loc=0, scale=v_th, size=(n_particles, 3))
        self.radius = particle_radius
        self.boxes = np.moveaxis(np.indices((boxes_by_axis, boxes_by_axis, boxes_by_axis)), 0, -1)
        self.particle_tracker = self.positions // ((boxes_by_axis) * self.size)
    def check_collisions(self):
        collisions = []
        b = np.size(self.boxes_by_axis, axis=0)
        vecs = np.moveaxis(self.boxes, 0, -1).reshape(-1, 3)
        for v in vecs:
            particle_indexes = np.where(np.linalg.norm(self.particle_tracker - v) <= 1)[0]
            for i in particle_indexes:
                for j in particle_indexes:
                    if np.linalg.norm(self.positions[i] - self.positions[j]) < self.radius:
                        collisions.append([i, j])
        return collisions
