import numpy as np
import matplotlib.pyplot as plt


rng = np.random.default_rng()
class LennardJones:
    def __init__(self, size: float, particles_per_axis: int, sigma: float, epsilon: float, v_th: float, tstep: float):
        self.size = size
        self.cutoff = 3 * sigma
        self.boxes_by_axis = max(1, int(self.size / (self.cutoff)))
        self.n_particles = particles_per_axis ** 3
        grid =  np.moveaxis(np.indices((particles_per_axis, particles_per_axis, particles_per_axis)), 0, -1).reshape(-1, 3)
        self.positions = (grid + 0.5) * (self.size / particles_per_axis)
        self.velocities = rng.normal(loc=0, scale=v_th, size=(self.n_particles, 3))
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
                            if b > 2:
                                nx, ny, nz = (bx + dx) % b - bx, (by + dy) % b - by, (bz + dz) % b - bz
                                adjacents.append(c + nx + (b * ny)  + b **2 * nz)
                            else:
                                if min(bx + dx, by + dy, bz + dz) > -1 and max(bx + dx, by + dy, bz + dz) < b:
                                    adjacents.append(c + dx + (b * dy) + (b ** 2 * dz))
            adjacency_list.append(adjacents)
        self.adjacency = adjacency_list
    def check_interactions(self) -> list: 
        n_boxes = self.boxes_by_axis ** 3
        b = self.boxes_by_axis
        cells = {}
        interactions = []
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
                l = range(len(particles))
                for i in l:
                    p1 = particles[i]
                    r1 = self.positions[p1]
                    if self.boxes_by_axis > 1:
                        for adj in self.adjacency[c]:
                            if adj in cells.keys():
                                for p2 in cells[adj]:
                                    r2 = self.positions[p2]
                                    d = r2 - r1
                                    d -= self.size * np.round(d / self.size)
                                    if np.linalg.norm(d) < self.cutoff:
                                        interactions.append([(p1, p2), d])
                    for j in l[i + 1:]:
                        r2 = self.positions[particles[j]]
                        if np.linalg.norm(d) < self.cutoff:
                            interactions.append([(p1, particles[j]), d])
            c += 1
        return interactions
    def compute_acceleration(self, d: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(d)
        ratio = self.sigma / r
        factor = 4 * (self.epsilon / (self.sigma * r)) * (6 * ratio** 7 - 12 * ratio ** 13)
        return (d) * factor
    def apply_forward_step(self):
        interactions = self.check_interactions()
        a_n = np.zeros((self.n_particles, 3))
        for (p1, p2), d in interactions:
            acc = self.compute_acceleration(d)
            a_n[p1, :] += acc
            a_n[p2, :] -= acc
        self.velocities +=  0.5 * self.timestep * a_n
        self.positions += self.timestep * self.velocities
        a_n_add_one = np.zeros((self.n_particles, 3))
        for ((p1, p2), d) in interactions:
            acc = self.compute_acceleration(d + self.timestep * (self.velocities[p2] - self.velocities[p1]))
            a_n_add_one[p1, :] += acc
            a_n_add_one[p2, :] -= acc
        self.velocities += 0.5 * self.timestep * a_n_add_one
        self.positions = self.positions % self.size
        self.particle_tracker = (self.positions * self.boxes_by_axis) // self.size
    def run(self, n_steps: int):
        for i in range(n_steps):
            self.apply_forward_step()

sim = LennardJones(10, 10, 0.1, 0.1, 1, 0.001)

sim.run(10)