import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

periods = 100
simulations = 100

# Generate random numbers
random = np.random.randn(periods, simulations)

# Generate random walk
random_walk = np.cumsum(random, axis=0)

# Set figure size
fig, ax = plt.subplots(figsize=(12, 8))

# Initialize the plot with empty lines
lines = [ax.plot([], [], lw=1)[0] for _ in range(simulations)]

# Set the limits of the plot
ax.set_xlim(0, periods)
ax.set_ylim(np.min(random_walk), np.max(random_walk))
ax.set_title(f'Monte Carlo Simulation ({simulations} times)')
ax.set_xlabel('Periods')
ax.set_ylabel('Value')

# Function to update the plot
def update(num, random_walk, lines):
    for i, (line, data) in enumerate(zip(lines, random_walk.T)):
        line.set_data(range(num), data[:num])
        if i == 0:  # Increase the line width of the first line
            line.set_linewidth(3)
            line.set_color('red')
        else:
            line.set_linewidth(1)
            line.set_alpha(0.3)  # Decreased alpha for other lines
            
    return lines

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=periods, fargs=[random_walk, lines], interval=50, blit=True)

# Show the animation
plt.show()
