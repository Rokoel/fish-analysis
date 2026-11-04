"""Apply saved signal cutoffs by stain/channel identity rather than channel order."""
import re

import numpy as np

from ..signals.evaluation import suppress


def channel_key(channel):
    wavelength = channel.get('excitation_nm')
    if wavelength is not None and np.isfinite(wavelength) and wavelength > 0:
        return f'excitation:{float(wavelength):g}nm'
    name = str(channel.get('name', '')).strip().casefold()
    if not name or re.fullmatch(r'(?:ch(?:annel)?|fluor)[\s\d_:-]*(?:t\d+)?', name):
        return None
    return 'name:' + name


def key_for(case, point):
    channels = case.get('channels', [])
    channel = point['channel']
    return channel_key(channels[channel]) if 0 <= channel < len(channels) else None


def accepted_points(case, manifest, separation):
    thresholds = manifest.get('channel_thresholds', {})
    rows = [p for p in case['candidates'] if p['score'] >= thresholds.get(key_for(case, p), manifest['threshold'])]
    return suppress(rows, [p['score'] for p in rows], 0., separation)


