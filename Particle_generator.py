import numpy as np
from numba import njit
rng = np.random.default_rng()


class IdealSim:
    def __init__(self, size, n_particles, particle_radius, v_th, boxes_by_axis):
        self.size = size
        self.n_particles = n_particles
        self.positions = rng.uniform(low=0, high=size, size=(n_particles, 3))
        self.velocities = rng.normal(loc=0, scale=v_th, size=(n_particles, 3))
        self.radius = particle_radius
        self.boxes = np.moveaxis(np.indices((boxes_by_axis, boxes_by_axis, boxes_by_axis)), 0, -1)
        self.particle_tracker = (self.positions * boxes_by_axis) // (self.size)
        adj = set()
        for i in [-1 0, 1]:
            for j in [-1 0, 1]:
                for k in [-1, 0, 1]:
                    adj.add(np.array([i, j, k]))
        self.adj = adj
    def check_collisions(self):
        collisions = []
        vecs = self.boxes.reshape(-1, 3)
        for v in vecs:
            particle_indexes = np.where(np.max(np.abs(self.particle_tracker - v),axis=1) <= 1)[0]
            l = len(particle_indexes)
            for i in range(l):
                for j in range(l)[i + 1:]:
                    if np.linalg.norm(self.positions[particle_indexes[i]] - self.positions[particle_indexes[j]]) < 2 * self.radius and i != j:
                        collisions.append([i, j])
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
        for y in y_crosses:
            self.velocities[y, 1] *= -1
        for z in z_crosses:
            self.velocities[z, 2] *= -1
    def update_positions(self, t_change):
        self.positions += t_change * self.velocities

sim = IdealSim(1, 1000, 0.01, 100, 10)
t_change = 0.00001

for i in range(10000):
    sim.update_positions(t_change)
    sim.apply_collisions()




    

