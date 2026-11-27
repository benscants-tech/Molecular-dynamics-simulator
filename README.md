# Molecular Dynamics simulator

**This project is a work in progress**

What it does currently:
- Simulates a hard-sphere gas with entirely elastic collisions using spatial partitioning which reduces time complexity of each position update from $O(n^2)$ to $O(n)$ under appropriate density conditions
- Displays a plot of the x-coordinate of one of the particles over time

What it will do in future:
- Simulate a Lennard-Jones fluid using Verlet integration and spatial partitioning
- Include measurements of: temperature, pressure and diffusion
- Use compiled code to allow for higher-volume simulation

## Requirements

- Python 3.9 or later
- NumPy
- Matplotlib
- Numba

Install the required dependencies using:

```bash
pip install numpy matplotlib numba
```

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/benscants-tech/Molecular-dynamics-simulator.git
   cd Molecular-dynamics-simulator
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the simulation:

   ```bash
   python simulation.py
   ```

The simulation initializes particles with randomly distributed positions and normally distributed velocity components, then evolves the system through elastic particle collisions and reflections at the container walls.

By default, the example simulates 10,000 particles over 1,000 timesteps and displays a plot of the x-coordinate of one particle over time.

### Configuration

Simulation parameters can be modified in the following line of `simulation.py`:

```python
sim = IdealSim(10, 10000, 0.15, 10)
```

| Parameter | Description |
|---|---|
| `size` | Side length of the cubic simulation box |
| `n_particles` | Number of particles |
| `particle_radius` | Radius of each particle |
| `v_th` | Standard deviation of the initial velocity distribution along each axis |

The timestep is automatically calculated from the particle radius and thermal velocity scale.
