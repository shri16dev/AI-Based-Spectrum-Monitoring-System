"""
simulate.py — HARDER RF signal simulator forcing real class confusion.

Why 0.99 happened:
  - fft_peak_count is trivially 1 for signal, 2 for multi, 0 for noise
  - kurtosis cleanly separates sine (low) from Gaussian noise (=0)
  - SNR was still too high — features were obvious

Fixes:
  1. LOW SNR range (2-12 dB) — signal barely above noise floor
  2. Noise gets occasional narrowband spike — looks like signal in FFT
  3. Multi uses close frequencies (~33%) — peaks merge, count looks like 1
  4. Signal gets occasional harmonic (~25%) — confuses peak_count
  5. Frequency ranges overlap across all three classes
"""

import numpy as np
import os
import glob

SAMPLE_RATE = 1024
N_SAMPLES   = 1024
N_EACH      = 150    # 150 per class -> 450 total
DATA_DIR    = "../data"
RNG         = np.random.default_rng(0)

os.makedirs(DATA_DIR, exist_ok=True)


def _time():
    return np.linspace(0, 1, N_SAMPLES, endpoint=False)


def _awgn(signal, snr_db):
    signal_power = np.mean(signal ** 2) + 1e-12
    noise_power  = signal_power / (10 ** (snr_db / 10))
    return signal + RNG.normal(0, np.sqrt(noise_power), len(signal))


def _pink_noise(scale=1.0):
    white = RNG.standard_normal(N_SAMPLES)
    fft   = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(N_SAMPLES)
    freqs[0] = 1e-6
    pink  = np.fft.irfft(fft / np.sqrt(freqs), n=N_SAMPLES)
    pink  = pink / (np.std(pink) + 1e-12)
    return pink * scale * RNG.uniform(0.5, 1.5)


def generate_signal(i):
    t   = _time()
    # Every 3rd example is low-SNR borderline
    snr = RNG.uniform(2, 5) if i % 3 == 0 else RNG.uniform(5, 12)
    f   = RNG.integers(40, 130)    # wide range, overlaps with multi f1
    amp = RNG.uniform(0.3, 1.8)
    pure = amp * np.sin(2 * np.pi * f * t)
    # ~25% chance of a weak harmonic -> confuses fft_peak_count
    if RNG.random() < 0.25:
        pure += (amp * 0.15) * np.sin(2 * np.pi * f * 2 * t)
    return _awgn(pure, snr)


def generate_noise(i):
    base = _pink_noise()
    # ~33% chance of narrowband spike -> looks like weak signal
    if i % 3 == 0:
        t = _time()
        f_narrow = RNG.integers(40, 200)
        amp_narrow = RNG.uniform(0.05, 0.25)
        base = base + amp_narrow * np.sin(2 * np.pi * f_narrow * t)
    return base


def generate_multi(i):
    t   = _time()
    snr = RNG.uniform(3, 6) if i % 3 == 0 else RNG.uniform(6, 14)
    if i % 3 == 0:
        # Close frequencies -> peaks merge -> indistinguishable from signal
        f1 = RNG.integers(50, 100)
        f2 = f1 + RNG.integers(5, 20)
    else:
        f1 = RNG.integers(40, 90)
        f2 = RNG.integers(100, 200)
    a1 = RNG.uniform(0.2, 1.6)
    a2 = RNG.uniform(0.2, 1.6)
    pure = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)
    return _awgn(pure, snr)


if __name__ == "__main__":
    # Remove old files
    for f in (glob.glob(f"{DATA_DIR}/signal_*.npy") +
              glob.glob(f"{DATA_DIR}/noise_*.npy")  +
              glob.glob(f"{DATA_DIR}/multi_*.npy")):
        os.remove(f)

    for i in range(N_EACH):
        np.save(f"{DATA_DIR}/signal_{i}.npy", generate_signal(i))
        np.save(f"{DATA_DIR}/noise_{i}.npy",  generate_noise(i))
        np.save(f"{DATA_DIR}/multi_{i}.npy",  generate_multi(i))

    print(f"Generated {N_EACH * 3} files in '{DATA_DIR}/'")
    print(f"  {N_EACH} signal  - single tone, SNR 2-12 dB, ~25% with harmonic")
    print(f"  {N_EACH} noise   - pink noise, ~33% with narrowband spike")
    print(f"  {N_EACH} multi   - dual tone, ~33% with close freqs (peaks merge)")
    print(f"\nExpected CV accuracy after this: 0.78 - 0.88")