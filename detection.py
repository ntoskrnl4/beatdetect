import time
from random import random
from typing import Any, Tuple

import numpy as np
import soundcard as sc
from matplotlib import pyplot as plot
from scipy import fft, signal


source = sc.get_microphone("Headset Sink", include_loopback=True)


beat_n = 0
sr = 32000
bufsize = 256  # should be a power of 2
histsize = 2**17  # should be a power of 2, long enough for a few seconds @ SR
last_beat = time.time()-1
bpm = None
bpm_allow_max_hz = 3.1

envelope_smoothing = 1-(60/sr)  # lower values are smoother, higher values are more responsive (complementary filter)
# you don't want to pick a value too low though: more lag detecting beat hits; and lower SNR in BPM detection

lows = (100, 350)  # frequency filters that can be selected for BPM detection
mids = (320, 750)
hpf_freq = 6000

history = np.zeros((histsize,), dtype=np.float32)
# Find which signal to sync to based on autocorrelation with highest peaks (raw, envelope, lows, mids, high)


def to_bpm(x) -> float:
    return (sr/x)*60


def analyze_peaks(wf, distance=None, zheight=2):
    peaks = signal.find_peaks(wf, distance=distance, height=wf.mean() + wf.std() * zheight)[0]

    dist = 0
    if len(peaks) > 4:
        for i in range(len(peaks) - 1):
            dist += peaks[i + 1] - peaks[i]
        dist = dist / (len(peaks) - 1)
    heights = wf[peaks]
    zindexs = (heights - wf.mean()) / wf.std()

    return dist, peaks, zindexs


def detect_bpm_raw() -> tuple[float | None, float]:
    autocorr = signal.correlate(history, history, mode="full")
    dist, peaks, zindexs = analyze_peaks(autocorr, distance=sr / bpm_allow_max_hz, zheight=2)
    
    # print(f"detect_bpm_raw: found {len(peaks)} peaks, dist={int(dist)}(={to_bpm(dist):.1f}), z={zindexs.mean():.2f}")
    if dist == 0:
        return None, -1
    else:
        return dist, zindexs.mean()


def detect_bpm_envelope() -> tuple[float | None, float]:
    hist_fft = fft.fft(history)
    hist_fft[len(hist_fft)//2:] = 0
    envelope = abs(2*fft.ifft(hist_fft))
    envelope[0] = envelope.mean()
    
    for i in range(1, len(envelope)):
        envelope[i] = (1-envelope_smoothing)*envelope[i] + envelope_smoothing*envelope[i-1]
    
    envelope -= envelope.mean()
    env_ac = signal.correlate(envelope, envelope, mode="full")
    dist, peaks, zindexs = analyze_peaks(env_ac, distance=sr / bpm_allow_max_hz)
    
    # print(f"detect_bpm_env: found {len(peaks)} peaks, dist={int(dist)}(={to_bpm(dist):.1f}), z={zindexs.mean():.2f}")
    if dist == 0:
        return None, -1
    else:
        return dist, zindexs.mean()


def detect_bpm_lows_env() -> tuple[float | None, float]:
    hist_fft = fft.fft(history)
    hist_fft[len(hist_fft)//2:] = 0
    hist_fft[int(len(hist_fft)*(lows[0]/sr))+1 : int(len(hist_fft)*(lows[1]/sr))+1] = 0
    
    envelope = abs(2*fft.ifft(hist_fft))
    envelope[0] = envelope.mean()

    for i in range(1, len(envelope)):
        envelope[i] = (1-envelope_smoothing)*envelope[i] + envelope_smoothing*envelope[i-1]
    
    envelope -= envelope.mean()
    env_ac = signal.correlate(envelope, envelope, mode="full")
    dist, peaks, zindexs = analyze_peaks(env_ac, distance=sr / bpm_allow_max_hz, zheight=1)
    
    # print(f"detect_bpm_low: found {len(peaks)} peaks, dist={int(dist)}(={to_bpm(dist):.1f}), z={zindexs.mean():.2f}")
    if dist == 0:
        return None, -1
    else:
        return dist, zindexs.mean()


def detect_bpm_mids_env() -> tuple[float | None, float]:
    hist_fft = fft.fft(history)
    hist_fft[len(hist_fft)//2:] = 0
    hist_fft[int(len(hist_fft)*(mids[0]/sr))+1 : int(len(hist_fft)*(mids[1]/sr))+1] = 0
    
    envelope = abs(2*fft.ifft(hist_fft))
    envelope[0] = envelope.mean()

    for i in range(1, len(envelope)):
        envelope[i] = (1-envelope_smoothing)*envelope[i] + envelope_smoothing*envelope[i-1]
    
    envelope -= envelope.mean()
    env_ac = signal.correlate(envelope, envelope, mode="full")
    dist, peaks, zindexs = analyze_peaks(env_ac, distance=sr / bpm_allow_max_hz, zheight=1)
    
    # print(f"detect_bpm_mid: found {len(peaks)} peaks, dist={int(dist)}(={to_bpm(dist):.1f}), z={zindexs.mean():.2f}")
    if dist == 0:
        return None, -1
    else:
        return dist, zindexs.mean()


def detect_bpm_high_env() -> tuple[float | None, float]:
    hist_fft = fft.fft(history)
    hist_fft[len(hist_fft)//2:] = 0
    hist_fft[:int(len(hist_fft)*(hpf_freq/sr))] = 0  # cut out all the frequencies below hpf (brick wall filter)
    
    envelope = abs(2*fft.ifft(hist_fft))
    envelope[0] = envelope.mean()
    
    for i in range(1, len(envelope)):
        envelope[i] = (1-envelope_smoothing)*envelope[i] + envelope_smoothing*envelope[i-1]
    
    envelope -= envelope.mean()
    env_ac = signal.correlate(envelope, envelope, mode="full")
    dist, peaks, zindexs = analyze_peaks(env_ac, distance=sr / bpm_allow_max_hz, zheight=1)
    
    # print(f"detect_bpm_hi : found {len(peaks)} peaks, dist={int(dist)}(={to_bpm(dist):.1f}), z={zindexs.mean():.2f}")
    if dist == 0:
        return None, -1
    else:
        return dist, zindexs.mean()


with source.recorder(samplerate=sr, blocksize=histsize) as mic:
    while True:
        live = mic.record(numframes=bufsize).mean(axis=1)

        history[:-bufsize] = history[bufsize:]
        history[-bufsize:] = live

        if history[0] != 0.000 and (time.time()-last_beat) > 2:
            last_beat = time.time()
            raw = detect_bpm_raw()
            env = detect_bpm_envelope()
            low = detect_bpm_lows_env()
            mid = detect_bpm_mids_env()
            his = detect_bpm_high_env()
            print(f"raw:{f'{to_bpm(raw[0]):05.1f}({raw[1]:.2f}) ' if raw[0] else '----------- '}\t"
                  f"env:{f'{to_bpm(env[0]):05.1f}({env[1]:.2f}) ' if env[0] else '----------- '}\t"
                  f"low:{f'{to_bpm(low[0]):05.1f}({low[1]:.2f}) ' if low[0] else '----------- '}\t"
                  f"mid:{f'{to_bpm(mid[0]):05.1f}({mid[1]:.2f}) ' if mid[0] else '----------- '}\t"
                  f"his:{f'{to_bpm(his[0]):05.1f}({his[1]:.2f}) ' if his[0] else '----------- '}\t")

        # how to time out beats:
        # continuously update audio history buffer
        # detect BPM from that
        # periodically take last BPM samples and find location of beats (peak finding on envelopes)
        # extrapolate out to next hit
        # at that time, confirm if hit found (<last> corr <history>, if peaks match dist, then yes)
        # if yes, sticky to that BPM and predict next one
        # if no, check for updated BPM


