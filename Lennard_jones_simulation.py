import numpy as np
import matplotlib.pyplot as plt
import time

rng = np.random.default_rng()
class LennardJones:
    def __init__(self, size: float, particles_per_axis: int, sigma: float, epsilon: float, v_th: float, tstep: float):
        self.size = size
        self.cutoff = 3 * sigma
        self.cutoff_sq = self.cutoff ** 2
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
        delta = 0.2 * self.cutoff_sq
        n_boxes = self.boxes_by_axis ** 3
        b = self.boxes_by_axis
        cells = {}
        interactions = []
        for i in range(self.n_particles):
            cell_id = self.particle_tracker[i, 0] + b * self.particle_tracker[i, 1] + self.particle_tracker[i, 2] * b ** 2
            if int(cell_id) in cells:
                cells[int(cell_id)].append(i)
            else:
                cells[int(cell_id)] = [i]
        c = 0
        for c in cells.keys():
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
                                if np.dot(d, d) < self.cutoff_sq + delta:
                                    interactions.append([(p1, p2), d])
                for j in l[i + 1:]:
                    r2 = self.positions[particles[j]]
                    d = r2 - r1
                    if np.dot(d, d) < self.cutoff_sq + delta:
                        interactions.append([(p1, particles[j]), d])
        return interactions
    def compute_acceleration(self, d: np.ndarray, sigma_sq: float) -> np.ndarray:
        r_sq = np.dot(d, d)
        if r_sq > self.cutoff_sq:
            return 0
        ratio = sigma_sq / r_sq
        factor = 4 * (self.epsilon / sigma_sq) * (6 * ratio** 4 - 12 * ratio ** 7)
        return (d) * factor
    def apply_forward_step(self):
        sigma_sq = self.sigma ** 2
        interactions = self.check_interactions()
        a_n = np.zeros((self.n_particles, 3))
        for (p1, p2), d in interactions:
            acc = self.compute_acceleration(d, sigma_sq)
            a_n[p1, :] += acc
            a_n[p2, :] -= acc
        self.velocities +=  0.5 * self.timestep * a_n
        self.positions += self.timestep * self.velocities
        a_n_add_one = np.zeros((self.n_particles, 3))
        for ((p1, p2), d) in interactions:
            acc = self.compute_acceleration(d + self.timestep * (self.velocities[p2] - self.velocities[p1]), sigma_sq)
            a_n_add_one[p1, :] += acc
            a_n_add_one[p2, :] -= acc
        self.velocities += 0.5 * self.timestep * a_n_add_one
        self.positions = self.positions % self.size
        self.particle_tracker = np.clip((self.positions * self.boxes_by_axis) // self.size, 0, self.boxes_by_axis - 1)
    def run(self, n_steps: int):
        for i in range(n_steps):
            self.apply_forward_step()
    def compute_potential_energy(self):
        m = self.positions[:, None, :] - self.positions[None, :, :]
        m -= self.size * np.round(m / self.size)
        sigma_sq = self.sigma ** 2
        r_sq = np.sum(m** 2, axis=-1)
        a = np.arange(self.n_particles)
        r_sq[a, a] = 1
        squared_ratio = sigma_sq / r_sq
        potentials = 4 * self.epsilon * (squared_ratio ** 6 - squared_ratio ** 3)
        return np.sum(potentials[np.triu_indices(potentials.shape[0], k=1)])
    def compute_kinetic_energy(self):
        return 0.5 * np.sum(self.velocities ** 2)
    def mean_squared_displacement(self):
        original_positions = self.positions.copy()
        msd = []
        for i in range(10):
            self.run(10)
            disp = self.positions - original_positions
            msd.append(np.mean(disp ** 2))
        return msd

sim = LennardJones(5, 5, 1, 1, 1, 0.005)
print(sim.compute_potential_energy())

r = range(5, 10)
times = []
for i in range(5):
    sim = LennardJones(r[i], r[i], 1, 1, 1, 0.005)
    t1 = time.time()
    sim.run(50)
    t2 = time.time()
    print(t2-t1)
    times.append(t2 - t1)

fig1, ax1 = plt.subplots()

ax1.plot(np.array(r) ** 3, times)
ax1.set_title("Runtime against number of particles")

sim = LennardJones(10, 10, 1, 1, 1, 0.005)
energies = []
for i in range(100):
    energies.append(sim.compute_potential_energy() + sim.compute_kinetic_energy())
    sim.take_step_forward

E_0 = energies[0]
energy_change = (np.array(energies) - E_0) / E_0

fig2, ax2 = plt.subplots()
ax2.plot(energy_change)

ax2.set_title("Energy change against time")


plt.show()