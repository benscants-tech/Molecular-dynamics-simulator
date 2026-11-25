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
        self.timestep = 0.1 * (particle_radius / v_th)
        self.boxes_total = self.boxes_by_axis ** 3
        self.colls = []
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
        self.wall_hits = []
    def check_collisions(self):
        n_boxes = self.boxes_by_axis ** 3
        b = self.boxes_by_axis
        cells = {}
        collisions = set()
        for i in range(self.n_particles):
            cell_id = self.particle_tracker[i, 0] + b * self.particle_tracker[i, 1] + self.particle_tracker[i, 2] * b ** 2
            if int(cell_id) in cells.keys():
                cells[int(cell_id)].append(i)
            else:
                cells[int(cell_id)] = [i]
        c = 0
        while c < n_boxes:
            if c in cells.keys():
                particles = cells[c]
                list = range(len(particles))
                for i in list:
                    p1 = particles[i]
                    r1 = self.positions[p1]
                    u1 = self.velocities[p1]
                    for adj in self.adjacency[c]:
                        if adj in cells.keys():
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
        self.colls.append(len(collisions))
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
        wall_momentum = 0
        for c in collisions:
            i, j = c
            self.collision_velocity_change(i, j)
        for x in x_crosses: 
            if self.positions[x, 0] < 0  :
                self.positions[x, 0] *= -1
                if self.velocities[x, 0] < 0:
                    self.velocities[x, 0] *= -1
                    wall_momentum += 2 * np.abs(self.velocities[x, 0])
            else:
                self.positions[x, 0] = 2 * self.size - self.positions[x, 0]
                if self.velocities[x, 0] > 0:
                    self.velocities[x, 0] *= -1
                    wall_momentum += 2 * np.abs(self.velocities[x, 0])
        for y in y_crosses:
            if self.positions[y, 1] < 0:
                self.positions[y, 1] *= -1
                if self.velocities[y, 1] < 0:
                    self.velocities[y, 1] *= -1
                    wall_momentum += 2 * np.abs(self.velocities[y, 1])
            else:
                self.positions[y, 1] = 2 * self.size - self.positions[y, 1]
                if self.velocities[y, 1] > 0:
                    self.velocities[y, 1] *= -1
                    wall_momentum += 2 * np.abs(self.velocities[y, 1])
        for z in z_crosses:
            if self.positions[z, 2] < 0:
                self.positions[z, 2] *= -1
                if self.velocities[z, 2] < 0:
                    self.velocities[z, 2] *= -1
                    wall_momentum += 2 * np.abs(self.velocities[z, 2])
            else:
                self.positions[z, 2] = 2 * self.size - self.positions[z, 2]
                if self.velocities[z, 2] > 0:
                    self.velocities[z, 2] *= -1
                    wall_momentum += 2 * np.abs(self.velocities[z, 2])
        self.wall_hits.append(wall_momentum)
    def update_positions(self):
        self.positions += self.timestep * self.velocities
        self.particle_tracker = np.clip((self.positions * self.boxes_by_axis) // self.size, 0, self.boxes_by_axis - 1)
    def run(self, n_steps):
        for i in range(n_steps):
            self.update_positions()
            self.apply_collisions()
    def measure_pressure(self):
            self.run(100)
            momentum_per_step = np.mean(np.array(self.wall_hits))
            pressure = momentum_per_step / (self.timestep * 6 * self.size ** 2)
            print(pressure)
    def measure_collision_frequency(self):
        self.run(100)
        print(np.mean(np.array(self.colls)) / self.timestep)
    def clear_data(self):
        self.colls = []
        self.wall_hits = []


positions = []
sim = IdealSim(10, 10000, 0.15, 10)

sim.measure_pressure()
sim.clear_data()
sim.measure_collision_frequency()


positions = []
t1 = time.time()
for i in range(1000):
    sim.update_positions()
    sim.apply_collisions()
    positions.append(sim.positions[0, 0])

t2 = time.time()
plt.plot(np.arange(1000) * timestep, np.array(positions))
plt.show()

print(t2 - t1)


