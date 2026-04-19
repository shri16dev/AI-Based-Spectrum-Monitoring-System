import numpy as np

def extract_features(x):
    fft = np.fft.fft(x)
    fft_mag = np.abs(fft)

    return [
        np.mean(x),
        np.std(x),
        np.max(x),
        np.sum(x**2),          # energy
        np.max(fft_mag)        # frequency peak
    ]