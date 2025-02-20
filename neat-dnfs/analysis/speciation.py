import numpy as np
import matplotlib.pyplot as plt

# Example: Replace this with your actual data
# Assume each species is a list where index = generation, value = population size
species_data = [
    [150, 140, 130, 120, 100,  80,  50,  30,  10,   0,   0,  0,  0],  # Species 1 (goes extinct)
    [0,   0,  30,  50,  80, 100, 120, 130, 140, 150, 130, 110,  90],  # Species 2 (newer)
    [0,   0,   0,   0,   0,  20,  50,  70,  90, 110, 120, 140, 150]   # Species 3 (even newer)
]

num_generations = len(species_data[0])
num_species = len(species_data)

# Convert to NumPy array for easier manipulation
species_populations = np.array(species_data, dtype=float)

# Normalize for better visualization
species_populations /= species_populations.max()

# Plot
fig, ax = plt.subplots(figsize=(8, 10))

for s in range(num_species):
    y = np.arange(num_generations)  # Generations
    width = species_populations[s]  # Population (scaled)
    
    ax.fill_betweenx(y, s - width / 2, s + width / 2, color='gray', edgecolor="white", linewidth=0.5)

# Labels and formatting
ax.set_xlabel("Species")
ax.set_ylabel("Generations")
ax.set_title("NEAT Speciation Visualization")
ax.invert_yaxis()  # Make generations go from top to bottom

plt.show()

