"""
features.py — signal feature extraction.

New features added:
  spectral_entropy    — measures how spread the energy is across frequencies.
                        Pure noise is near-maximum entropy; a sine wave is low.
  kurtosis            — measures "peakedness". Sine waves have kurtosis ≈ 1.5;
                        Gaussian noise has kurtosis ≈ 3.
  band_energy_ratio   — ratio of energy in low vs high frequency bands.
                        Helps distinguish multi from single-tone signals.
  fft_peak_count      — number of spectral peaks above a threshold. Helps
                        separate multi (2 peaks) from signal (1 peak).
  spectral_centroid   — centre of mass of the spectrum. Shifts with f2 presence.

Removed from original:
  np.max(x) alone     — trivially high for multi-signal; replaced by kurtosis
                        which captures shape more robustly.
"""

import numpy as np

SAMPLE_RATE = 1024
BAND_LOW    = (0, SAMPLE_RATE // 8)      # 0–128 Hz
BAND_HIGH   = (SAMPLE_RATE // 8, SAMPLE_RATE // 2)  # 128–512 Hz


def _spectral_entropy(fft_mag):
    """Shannon entropy of the normalised power spectrum."""
    power = fft_mag ** 2
    total = np.sum(power) + 1e-12
    prob  = power / total
    # Avoid log(0)
    prob  = np.clip(prob, 1e-12, None)
    return -np.sum(prob * np.log2(prob))


def _kurtosis(x):
    """Excess kurtosis (Fisher definition, 0 for Gaussian)."""
    mu  = np.mean(x)
    std = np.std(x) + 1e-12
    return np.mean(((x - mu) / std) ** 4) - 3.0


def _band_energy_ratio(fft_mag, freqs):
    """Energy in low band / energy in high band."""
    low_mask  = (freqs >= BAND_LOW[0])  & (freqs < BAND_LOW[1])
    high_mask = (freqs >= BAND_HIGH[0]) & (freqs < BAND_HIGH[1])
    low_e  = np.sum(fft_mag[low_mask]  ** 2) + 1e-12
    high_e = np.sum(fft_mag[high_mask] ** 2) + 1e-12
    return low_e / high_e


def _spectral_centroid(fft_mag, freqs):
    """Weighted average frequency (centre of spectral mass)."""
    total = np.sum(fft_mag) + 1e-12
    return np.sum(freqs * fft_mag) / total


def _count_peaks(fft_mag, threshold_ratio=0.3):
    """Count FFT peaks above threshold_ratio × max peak."""
    threshold = threshold_ratio * np.max(fft_mag)
    # Simple peak-picking: local maxima above threshold
    count = 0
    for i in range(1, len(fft_mag) - 1):
        if fft_mag[i] > threshold and fft_mag[i] > fft_mag[i-1] and fft_mag[i] > fft_mag[i+1]:
            count += 1
    return count


def extract_features(x):
    """Return a 9-element feature vector for signal x.

    Features (in order):
      0  mean
      1  std
      2  energy             (sum of x²)
      3  kurtosis
      4  spectral_entropy
      5  fft_peak           (max FFT magnitude, normalised by length)
      6  fft_peak_count
      7  band_energy_ratio  (low / high)
      8  spectral_centroid
    """
    n    = len(x)
    fft  = np.fft.rfft(x)
    mag  = np.abs(fft) / n                 # normalise by length
    freqs = np.fft.rfftfreq(n, d=1.0/SAMPLE_RATE)

    return [
        np.mean(x),
        np.std(x),
        np.sum(x ** 2) / n,               # normalised energy
        _kurtosis(x),
        _spectral_entropy(mag),
        np.max(mag),
        _count_peaks(mag),
        _band_energy_ratio(mag, freqs),
        _spectral_centroid(mag, freqs),
    ]


FEATURE_NAMES = [
    "mean",
    "std",
    "energy (norm)",
    "kurtosis",
    "spectral entropy",
    "fft peak",
    "fft peak count",
    "band energy ratio",
    "spectral centroid",
]