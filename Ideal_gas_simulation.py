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
        self.boxes_by_axis = max(1, int(self.size / (2 * particle_radius)))
        self.boxes = np.moveaxis(np.indices((self.boxes_by_axis, self.boxes_by_axis, self.boxes_by_axis)), 0, -1)
        self.particle_tracker = (self.positions * self.boxes_by_axis) // (self.size)
        self.timestep = 0.2 * (particle_radius / v_th)
        self.boxes_total = self.boxes_by_axis ** 3
        adjacency_list = []
        b = self.boxes_by_axis
        for c in range(self.boxes_by_axis ** 3):
            adjacents = []
            bz = c // (b ** 2)
            by = (c - bz * b ** 2) // (b)
            bx = (c - by * b - bz * b ** 2)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx + 2 * dy + 4 * dz > 0:
                            if max(bx + dx, by + dy, bz + dz) < b and min(bx + dx, by + dy, bz +dz) > -1:
                                adjacents.append(c + dx + b * dy + b **2 * dz)
            adjacency_list.append(adjacents)
        self.adjacency = adjacency_list  
             
    def check_collisions(self):
        n_boxes = self.boxes_by_axis ** 3
        b = self.boxes_by_axis
        cells = [[] for _ in range(n_boxes)]
        collisions = set()
        for i in range(self.n_particles):
            cell_id = self.particle_tracker[i, 0] + b * self.particle_tracker[i, 1] + self.particle_tracker[i, 2] * b ** 2
            cells[int(cell_id)].append(i)
        c = 0
        while c < n_boxes:
            particles = cells[c]
            if not particles:
                c += 1
                break
            list = range(len(particles))
            for i in list:
                p1 = particles[i]
                r1 = self.positions[p1]
                u1 = self.velocities[p1]
                for adj in self.adjacency:
                    for p2 in cells[adj]:
                        r2, u2 = self.positions[p2], self.velocities[p2]
                        if np.linalg.norm(r1 - r2) < 2 * self.radius and np.dot(u1 - u2, r1 - r2) < 0:
                            collisions.add((p1, p2))
                for j in list[i + 1:]:
                    r2, u2 = self.positions[particles[j]], self.velocities[particles[j]]
                    if np.linalg.norm(r1 - r2) < 2 * self.radius and np.dot(u1 - u2, r1 - r2) < 0:
                                                collisions.add((p1, particles[j]))
            c += 1
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
        self.particle_tracker = np.clip((self.positions * self.boxes_by_axis) // self.size, 0, self.boxes_by_axis - 1)


t1 = time.time()
sim = IdealSim(1, 40, 0.02, 10)

positions = []
for i in range(500):
    sim.update_positions()
    sim.apply_collisions()
    positions.append(sim.positions[0].copy())
t2 = time.time()
print(t2 - t1)
p = np.stack(positions, axis=0)
plt.plot(p[:, 0])
plt.show()