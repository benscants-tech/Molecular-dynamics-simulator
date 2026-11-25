import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng()
class LennardJones:
    def __init__(self, size, n_particles, sigma, epsilon, v_th, tstep):
        self.size = size
        self.radius = 3 * sigma
        self.boxes_by_axis = max(1, int(self.size / (2 * self.radius)))
        self.n_particles = n_particles
        self.positions = rng.uniform(low=0, high=size, size=(n_particles, 3))
        self.velocities = rng.normal(loc=0, scale=v_th, size=(n_particles, 3))
        self.sigma = sigma
        self.epsilon = epsilon
        self.timestep = tstep
        self.particle_tracker = (self.positions * self.boxes_by_axis) // (self.size)
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
    def check_interactions(self):
        n_boxes = self.boxes_by_axis ** 3
        b = self.boxes_by_axis
        cells = [[] for _ in range(n_boxes)]
        interactions = set()
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
                for adj in self.adjacency[c]:
                    for p2 in cells[adj]:
                        r2 = self.positions[p2]
                        if np.linalg.norm(r1 - r2) < 2 * self.radius:
                            interactions.add((p1, p2))
                for j in list[i + 1:]:
                    r2 = self.positions[particles[j]]
                    if np.linalg.norm(r1 - r2) < 2 * self.radius:
                                                interactions.add((p1, particles[j]))
            c += 1
    def compute_acceleration(self, p1, p2):
        r1, r2 = self.positions[p1], self.positions[p2]
        r = np.linalg.norm(r2 - r1)
        ratio = self.sigma / r
        factor = (self.epsilon / (self.sigma * r)) * (6 * ratio** 7 - 12 * ratio ** 13)
        return (r2 - r1) * factor
    def apply_forward_step(self):
        interactions = self.check_interactions()
        a_n = np.zeros(self.n_particles, 3)
        for p1, p2 in interactions:
            acc = self.compute_acceleration(p1, p2)
            a_n[p1, :] += acc
            a_n[p2, :] -= acc
        self.velocities +=  0.5 * self.timestep * a_n
        self.positions += self.timestp * self.velocities
        a_n_add_one = np.zeros(self.n_particles, 3)
        for p1_p2 in interactions:
            acc = self.compute_acceleration(p1, p2)
            a_n[p1, :] += acc
            a_n[p2, :] -= acc
        self.velocities += 0.5 * self.timestep * a_n
        self.positions = self.positions // self.size
    
        
             
            


# structure: look at the interactions
# sum over each one for each particle
# implement a verlet timestep
# check if anything has crossed the axes, in which case put them on the opposite side.