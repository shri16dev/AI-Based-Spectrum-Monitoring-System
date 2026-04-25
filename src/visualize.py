import numpy as np
import matplotlib.pyplot as plt

data = np.load("../data/signal_0.npy")

plt.plot(data)
plt.title("Sample Signal")
plt.show()