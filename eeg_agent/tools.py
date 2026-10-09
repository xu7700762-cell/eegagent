# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
from scipy.integrate import trapezoid
FS = 256
BANDS = {"delta": (.5,4.), "theta": (4.,8.), "alpha": (8.,13.), "beta": (13.,30.), "gamma_low": (30.,45.)}

def _spectral(x):
    frequencies, psd = signal.welch(x, fs=FS, window='hann', nperseg=512,
                                    noverlap=256, detrend='constant', axis=-1,
                                    scaling='density')
    if not np.isfinite(psd).all():
        raise ValueError('nonfinite spectral computation')
    power = {}
    for band, (low, high) in BANDS.items():
        # Include exact edges. Adjacent bands share boundary points, not area.
        take = (frequencies >= low) & (frequencies <= high)
        power[band] = trapezoid(psd[:, take], frequencies[take], axis=-1)
    return frequencies, psd, power
