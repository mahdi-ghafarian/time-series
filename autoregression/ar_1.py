import numpy as np
import matplotlib.pyplot as plt 

# Parameters 
n = 100  # Number of observations 
phi = 0.5  # AR(1) parameter 
sigma = 1  # Standard deviation of the noise
mean = 0  # Theoretical mean
# Initialize the time series 
ar1_process = np.zeros(n) 

# Generate AR(1) process 
for t in range(1, n): 
    ar1_process[t] = phi * ar1_process[t-1] + np.random.normal(mean, sigma) 

# Calculate average run size 
def calculate_average_run_size(series): 
    runs = [1] 
    for i in range(2, len(series)): 
        if (series[i] > series[i-1] and series[i-1] > series[i-2]) or (series[i] < series[i-1] and series[i-1] < series[i-2]): 
            runs[-1] += 1 
        else: 
            runs.append(1) 
    average_run_size = np.mean(runs) 
    return runs, average_run_size 

runs, average_run_size = calculate_average_run_size(ar1_process) 

print('Runs:', runs) 
print(f"Average Run Size: {average_run_size:.2f}") 

# Calculate actual mean of the generated series
actual_mean = np.mean(ar1_process)

# Plot the AR(1) process with points and horizontal lines
plt.figure(figsize=(10, 5)) 
plt.plot(ar1_process, label='AR(1) Process', linestyle='-', marker='o', markersize=7) 
plt.axhline(y=mean, color='red', linestyle='--', linewidth=2, label='Theoretical Mean (μ = 0)') 
plt.axhline(y=actual_mean, color='blue', linestyle='-.', linewidth=2, label=f'Actual Mean ≈ {actual_mean:.2f}') 
plt.xlabel('Time') 
plt.ylabel('Value') 
plt.title(f'AR(1) Process: phi = {phi}, n = {n}') 
plt.legend() 
plt.grid(True) 
plt.show()
