import numpy as np
from numba import njit
rng = np.random.default_rng()
import matplotlib.pyplot as plt
import time


class IdealSim:
    def __init__(self, size, n_particles, particle_radius, v_th):
        self.size = size
        self.n_particles = n_particles
        self.positions = rng.uniform(low=0, high=size, size=(n_particles, 3))
        self.velocities = rng.normal(loc=0, scale=v_th, size=(n_particles, 3))
        self.radius = particle_radius
        self.boxes_by_axis = max(1, int(self.size / (2 * particle_radius))
        self.boxes = np.moveaxis(np.indices((self.boxes_by_axis, self.boxes_by_axis, self.boxes_by_axis)), 0, -1)
        self.particle_tracker = (self.positions * self.boxes_by_axis) // (self.size)
        self.timestep = 0.1 * (collision_radius / v_th)
    def check_collisions(self):
        collisions = set()
        vecs = self.boxes.reshape(-1, 3)
        for v in vecs:
            particle_indexes = np.where(np.max(np.abs(self.particle_tracker - v),axis=1) <= 1)[0]
            l = len(particle_indexes)
            # next we should use combinations to make this for loop run faster
            for i in range(l):
                for j in range(l)[i + 1:]:
                    r1, r2, u1, u2 = self.positions[particle_indexes[i]], self.positions[particle_indexes[j]], self.velocities[particle_indexes[i]], self.velocities[particle_indexes[j]]
                    if np.linalg.norm(r1 - r2) < 2 * self.radius and np.dot(u1 - u2, r1 - r2) < 0:
                        collisions.add((particle_indexes[i], particle_indexes[j]))
        x_crosses = np.where((0 > self.positions[:, 0]) | (self.positions[:, 0] > self.size))[0]
        y_crosses = np.where((0 > self.positions[:, 1]) | (self.positions[:, 1] > self.size))[0]
        z_crosses = np.where((0 > self.positions[:, 2]) | (self.positions[:, 2] > self.size))[0]
        return collisions, x_crosses, y_crosses, z_crosses
    def collision_velocity_change(self, p1, p2,):
        r1, r2, u1, u2 = self.positions[p1, :], self.positions[p2, :], self.velocities[p1, :], self.velocities[p2, :]
        r = r2 - r1
        alpha = np.dot(r, u2 - u1) / np.dot(r, r)
        v1 = u1 + alpha * r
        v2 = u2 - alpha * r
        self.velocities[p1] = v1
        self.velocities[p2] = v2
    def apply_collisions(self):
        collisions, x_crosses, y_crosses, z_crosses = self.check_collisions()
        for c in collisions:
            i, j = c
            self.collision_velocity_change(i, j)
        for x in x_crosses:
            self.velocities[x, 0] *= -1
            if self.positions[x, 0] < 0:
                self.positions[x, 0] *= -1
            else:
                self.positions[x, 0] = 2 * self.size - self.positions[x, 0]
        for y in y_crosses:
            self.velocities[y, 1] *= -1
            if self.positions[y, 1] < 0:
                self.positions[y, 1] *= -1
            else:
                self.positions[y, 1] = 2 * self.size - self.positions[y, 1]
        for z in z_crosses:
            self.velocities[z, 2] *= -1
            if self.positions[z, 2] < 0:
                self.positions[z, 2] *= -1
            else:
                self.positions[z, 2] = 2 * self.size - self.positions[z, 2]
    def update_positions(self):
        self.positions += self.timestep * self.velocities
        self.particle_tracker = (self.positions * self.boxes_by_axis) // self.size


t1 = time.time()
sim = IdealSim(1, 100, 0.1, 10, 10)
t_change = 0.00001

print(sim.velocities[:10])
positions = []
for i in range(100):
    sim.update_positions()
    sim.apply_collisions()
    positions.append(sim.positions[0])

print(sim.velocities[:10])