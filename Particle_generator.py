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
    def check_collisions(self):
        collisions = []
        vecs = self.boxes.reshape(-1, 3)
        for v in vecs:
            particle_indexes = np.where(np.linalg.norm(self.particle_tracker - v, axis=1) <= 2)[0]
            l = len(particle_indexes)
            for i in range(l):
                for j in range(l)[i + 1:]:
                    if np.linalg.norm(self.positions[i] - self.positions[j]) < 2 * self.radius and i != j:
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


sim = IdealSim(10, 10, 1, 10, 4)
print(sim.boxes)
print(sim.check_collisions())
print(sim.velocities)
v1, v2 = sim.velocities[0,:], sim.velocities[1, :]
v1 = v1.copy()
v2 = v2.copy()
sim.collision_velocity_change(0, 1)
print(sim.velocities)
x1, x2 = sim.velocities[0, :], sim.velocities[1, :]
print(x1)
print(x1.dtype)
print(np.linalg.norm(x1) **2 + np.linalg.norm(x2) ** 2 - np.linalg.norm(v1) **2 - np.linalg.norm(v2) **2 )
