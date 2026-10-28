"""Bounded, background-aware fluorescence display adjustment (pixels stay raw)."""

import numpy as np


def automatic_levels(volume, sample_size=500_000):
    flat = np.asarray(volume).reshape(-1)
    if not flat.size:
        return 0., 1., 1.
    # Even coverage avoids aliasing a repeated row/plane pattern with a stride.
    sample = flat[np.linspace(0, len(flat) - 1, min(len(flat), sample_size), dtype=np.int64)]
    sample = sample[np.isfinite(sample)].astype(float)
    if not len(sample):
        return 0., 1., 1.
    low, p10, median = np.percentile(sample, [1, 10, 50])
    threshold = median + 3 * (median - p10)
    foreground = sample[sample > threshold]
    if not len(foreground):
        foreground = sample[sample > median]
    if not len(foreground):
        value = float(sample[0])
        return value, value + max(1., abs(value) * .01), 1.
    # Tail percentiles are measured among foreground, so sparse puncta survive
    # even when >99.8% of the scan is empty. Isolated hot pixels do not set white.
    high = float(np.percentile(foreground, 98.5))
    if high <= low:
        high = float(foreground.max())
    high = max(high, low + max(1e-6, abs(low) * 1e-6))
    quarter = np.clip((np.percentile(foreground, 25) - low) / (high - low), .01, .99)
    gamma = float(np.clip(np.log(quarter) / np.log(.35), 1., 1.6))
    return float(low), high, gamma


def auto_adjust(channel, volume):
    channel.low, channel.high, channel.gamma = automatic_levels(volume)
    channel.brightness, channel.contrast = 0., 1.
