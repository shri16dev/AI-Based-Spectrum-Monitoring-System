import numpy as np
import os

os.makedirs("../data", exist_ok=True)

def generate_signal(freq=50, noise_level=0.2):
    t = np.linspace(0, 1, 1024)
    signal = np.sin(2*np.pi*freq*t)
    noise = np.random.normal(0, noise_level, 1024)
    return signal + noise

# Generate datasets
for i in range(50):
    np.save(f"../data/signal_{i}.npy", generate_signal(50, 0.1))   # clean signal
    np.save(f"../data/noise_{i}.npy", np.random.normal(0, 1, 1024)) # noise
    np.save(f"../data/multi_{i}.npy", generate_signal(50,0.2) + generate_signal(120,0.2)) # multi