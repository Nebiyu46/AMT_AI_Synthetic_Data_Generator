"""
Shared DSP augmentation helpers used by build_pool.py and the composition generator.

All functions operate on mono 1-D float32 numpy arrays unless noted otherwise.
"""

from __future__ import annotations

import numpy as np
import scipy.signal
import librosa
from pedalboard import (
    Pedalboard,
    PitchShift,
    LowShelfFilter,
    HighShelfFilter,
    PeakFilter,
)


# Vibrato

def add_vibrato(
    y: np.ndarray,
    sr: int,
    rate_hz: float | None = None,
    depth_cents: float | None = None,
    attack_skip_ms: float = 80.0,
    ramp_ms: float = 50.0,
) -> np.ndarray:
    """
    Add pitch vibrato via fractional-delay LFO modulation.

    Approximation: for sinusoidal delay variation A * sin(2*pi*f*t), the resulting
    pitch deviation is small-angle ~= 1200 / ln(2) * (2*pi*f*A / sr) cents peak.
    Solving for A in samples:  A = depth_cents * sr * ln(2) / (1200 * 2*pi*f)
    """
    n = len(y)
    if n == 0:
        return y
    if rate_hz is None:
        rate_hz = float(np.random.uniform(4.5, 6.5))
    if depth_cents is None:
        depth_cents = float(np.random.uniform(8.0, 30.0))

    A_samples = depth_cents * sr * np.log(2) / (1200.0 * 2 * np.pi * rate_hz)

    t = np.arange(n) / sr
    lfo = A_samples * np.sin(2 * np.pi * rate_hz * t)

    skip_n = min(int(attack_skip_ms * sr / 1000.0), max(0, n // 4))
    ramp_n = min(int(ramp_ms * sr / 1000.0), max(0, n - skip_n))

    envelope = np.ones(n, dtype=np.float64)
    if skip_n > 0:
        envelope[:skip_n] = 0.0
    if ramp_n > 0 and skip_n + ramp_n <= n:
        envelope[skip_n:skip_n + ramp_n] = np.linspace(0.0, 1.0, ramp_n)
    lfo *= envelope

    indices = np.arange(n) - lfo
    indices = np.clip(indices, 0.0, n - 1.0)
    return np.interp(indices, np.arange(n), y).astype(np.float32)


# Pink noise

def _generate_pink_noise(n: int) -> np.ndarray:
    """Pink noise via 1/sqrt(f) FFT shaping of white noise. Output unit peak."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    white = np.random.randn(n).astype(np.float64)
    freqs = np.fft.rfftfreq(n)
    if len(freqs) > 1:
        shape = np.zeros_like(freqs)
        shape[1:] = 1.0 / np.sqrt(freqs[1:])
    else:
        shape = np.zeros_like(freqs)
    spectrum = np.fft.rfft(white) * shape
    pink = np.fft.irfft(spectrum, n=n)
    peak = np.max(np.abs(pink))
    if peak > 0:
        pink /= peak
    return pink.astype(np.float32)


def add_pink_noise(
    y: np.ndarray,
    sr: int,
    snr_db_range: tuple[float, float] = (20.0, 40.0),
) -> np.ndarray:
    """Mix pink noise into y at a random SNR sampled from snr_db_range."""
    snr_db = float(np.random.uniform(*snr_db_range))
    n = len(y)
    if n == 0:
        return y
    pink = _generate_pink_noise(n)

    signal_power = float(np.mean(y.astype(np.float64) ** 2)) + 1e-12
    noise_power = float(np.mean(pink.astype(np.float64) ** 2)) + 1e-12
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    pink = pink * np.sqrt(target_noise_power / noise_power)
    return (y + pink).astype(np.float32)


# Releases

def apply_random_release(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply one of several release shapes to the tail of y, selected uniformly:
      - 'sharp' : 20-50 ms cosine fade-out (bow lifts quickly).
      - 'soft'  : 150-300 ms exponential decay (slow lift).
      - 'natural': 80-200 ms cos^1.5 fall (mimics natural string decay).
    Length of y is preserved.
    """
    n = len(y)
    if n < 16:
        return y

    choice = np.random.choice(['sharp', 'soft', 'natural'])
    out = y.astype(np.float32, copy=True)

    if choice == 'sharp':
        ms = float(np.random.uniform(20.0, 50.0))
        rel_n = min(int(ms * sr / 1000.0), n)
        if rel_n > 1:
            t = np.linspace(0.0, np.pi / 2.0, rel_n, dtype=np.float32)
            out[-rel_n:] *= np.cos(t)
    elif choice == 'soft':
        ms = float(np.random.uniform(150.0, 300.0))
        rel_n = min(int(ms * sr / 1000.0), n)
        if rel_n > 1:
            tau = max(rel_n / 4.0, 1.0)
            envelope = np.exp(-np.arange(rel_n, dtype=np.float32) / tau)
            out[-rel_n:] *= envelope
    else:
        ms = float(np.random.uniform(80.0, 200.0))
        rel_n = min(int(ms * sr / 1000.0), n)
        if rel_n > 1:
            t = np.linspace(0.0, np.pi / 2.0, rel_n, dtype=np.float32)
            cosine = np.maximum(np.cos(t), 0.0)
            out[-rel_n:] *= cosine ** 1.5

    return out


# EQ

def random_eq(y: np.ndarray, sr: int) -> np.ndarray:
    """Random low-shelf (200 Hz +/-3 dB), high-shelf (6 kHz +/-3 dB), peaking (500-4000 Hz +/-2 dB)."""
    low_gain = float(np.random.uniform(-3.0, 3.0))
    high_gain = float(np.random.uniform(-3.0, 3.0))
    peak_freq = float(np.random.uniform(500.0, 4000.0))
    peak_gain = float(np.random.uniform(-2.0, 2.0))

    board = Pedalboard([
        LowShelfFilter(cutoff_frequency_hz=200.0, gain_db=low_gain),
        HighShelfFilter(cutoff_frequency_hz=6000.0, gain_db=high_gain),
        PeakFilter(cutoff_frequency_hz=peak_freq, gain_db=peak_gain),
    ])
    return board(y.astype(np.float32), sr)


# Pitch shift

def pitch_shift_relabel(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Pitch-shift in semitones using pedalboard. Caller is responsible for relabeling MIDI."""
    if semitones == 0:
        return y.astype(np.float32, copy=True)
    board = Pedalboard([PitchShift(semitones=float(semitones))])
    return board(y.astype(np.float32), sr)


# Synthetic IRs

def make_synthetic_ir(sr: int, rt60_s: float | None = None) -> np.ndarray:
    """
    Synthetic short-room impulse response: direct + early reflections + exponential late tail.
    RT60 is sampled from [50 ms, 300 ms] if not provided.
    """
    if rt60_s is None:
        rt60_s = float(np.random.uniform(0.05, 0.30))
    n = max(int(rt60_s * sr), 32)

    ir = np.zeros(n, dtype=np.float32)
    ir[0] = 1.0

    early_n = min(int(0.030 * sr), n)
    if early_n > 4:
        num_reflections = int(np.random.randint(3, 8))
        for _ in range(num_reflections):
            pos = int(np.random.randint(int(0.001 * sr), max(int(0.001 * sr) + 1, early_n)))
            if pos < n:
                amp = float(np.random.uniform(0.3, 0.7)) * float(np.sign(np.random.randn()))
                ir[pos] += amp

    tau = max(n / 6.91, 1.0)
    noise = np.random.randn(n).astype(np.float32)
    envelope = np.exp(-np.arange(n, dtype=np.float32) / tau)
    ir = ir + 0.3 * noise * envelope

    peak = float(np.max(np.abs(ir)))
    if peak > 0:
        ir = ir / peak
    return ir.astype(np.float32)


def convolve_reverb(
    y: np.ndarray,
    sr: int,
    ir: np.ndarray,
    wet: float = 0.18,
) -> np.ndarray:
    """Mix dry y with convolution reverb at wet ratio. Wet is hard-capped at 0.20."""
    wet = float(min(max(wet, 0.0), 0.20))
    if len(y) == 0 or len(ir) == 0 or wet == 0.0:
        return y.astype(np.float32, copy=True)

    wet_signal = scipy.signal.fftconvolve(y, ir, mode='full')[:len(y)]
    wet_peak = float(np.max(np.abs(wet_signal))) + 1e-12
    dry_peak = float(np.max(np.abs(y))) + 1e-12
    wet_signal = wet_signal * (dry_peak / wet_peak)
    return ((1.0 - wet) * y + wet * wet_signal).astype(np.float32)


# Normalization

def peak_normalize(y: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    """Scale y so its peak sits at target_dbfs (default -3 dBFS)."""
    if len(y) == 0:
        return y
    peak = float(np.max(np.abs(y)))
    if peak <= 0:
        return y.astype(np.float32, copy=True)
    target_amp = 10.0 ** (target_dbfs / 20.0)
    return (y * (target_amp / peak)).astype(np.float32)


# Crossfade

def equal_power_crossfade_pair(
    prev_tail: np.ndarray,
    next_head: np.ndarray,
) -> np.ndarray:
    """Equal-power cosine crossfade of two equal-length buffers. Returns the blended segment."""
    n = min(len(prev_tail), len(next_head))
    if n < 2:
        return next_head[:n].astype(np.float32, copy=True)
    t = np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    return (prev_tail[-n:] * fade_out + next_head[:n] * fade_in).astype(np.float32)


# Sustain stretching

def stretch_to_durations(
    y: np.ndarray,
    sr: int,
    durations,
    target_attack_s: float = 0.08,
    anti_click_ms: float = 5.0,
):
    """
    Take a single recorded note, isolate the attack, time-stretch the steady-state sustain
    to each target duration in `durations`, stitch back together with phase-aligned crossfade,
    and apply only a tiny anti-click cosine fade at the very end (the 100 ms universal release
    has been intentionally removed -- per-note release randomization happens in the generator).

    Returns: list of (target_duration_seconds, audio_float32) tuples, one per duration that fit.
    """
    if len(y) < int(0.2 * sr):
        return []

    target_attack = int(target_attack_s * sr)
    zero_crossings = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
    if len(zero_crossings) == 0:
        attack_samples = min(target_attack, max(1, len(y) // 8))
    else:
        attack_samples = int(zero_crossings[np.argmin(np.abs(zero_crossings - target_attack))])
    attack_samples = max(1, attack_samples)
    attack_part = y[:attack_samples]

    tail = y[attack_samples:]
    if len(tail) < 256:
        return []

    rms = librosa.feature.rms(y=tail)[0]
    if len(rms) == 0:
        sustain_part = tail
    else:
        threshold = float(np.max(rms)) * 0.3
        valid_frames = np.where(rms >= threshold)[0]
        if len(valid_frames) > 0:
            last_loud_sample = int(librosa.frames_to_samples(valid_frames[-1]))
            sustain_part = tail[:last_loud_sample] if last_loud_sample > 0 else tail
        else:
            sustain_part = tail

    if len(sustain_part) < 256:
        sustain_part = tail

    buffer_samples = 2048
    if attack_samples > buffer_samples and (attack_samples + len(sustain_part)) <= len(y):
        buffered_sustain = y[attack_samples - buffer_samples:attack_samples + len(sustain_part)]
    else:
        buffered_sustain = sustain_part
        buffer_samples = 0

    results = []
    anti_click_n = int(anti_click_ms * sr / 1000.0)

    for target_duration in durations:
        target_duration = float(target_duration)
        target_sustain_duration = target_duration - (attack_samples / sr)
        if target_sustain_duration <= 0:
            continue

        original_sustain_duration = len(sustain_part) / sr
        if original_sustain_duration <= 0:
            continue
        rate = original_sustain_duration / target_sustain_duration

        try:
            stretched_buffered = librosa.effects.time_stretch(
                buffered_sustain.astype(np.float32),
                rate=rate,
            )
        except Exception:
            continue

        if buffer_samples > 0 and rate > 0:
            stretched_buffer_len = int(buffer_samples / rate)
            stretched_sustain = stretched_buffered[stretched_buffer_len:]
        else:
            stretched_sustain = stretched_buffered

        stitch_samples = int(0.04 * sr)
        max_shift = int(0.015 * sr)

        if (
            len(attack_part) > stitch_samples
            and len(stretched_sustain) > (stitch_samples + max_shift)
        ):
            base_sig = attack_part[-stitch_samples:]
            best_corr = -np.inf
            best_shift = 0
            for shift in range(0, max_shift):
                test_sig = stretched_sustain[shift:stitch_samples + shift]
                corr = float(np.sum(base_sig * test_sig))
                if corr > best_corr:
                    best_corr = corr
                    best_shift = shift
            aligned_sustain = stretched_sustain[best_shift:]

            t = np.linspace(0.0, np.pi / 2.0, stitch_samples, dtype=np.float32)
            fade_out = np.cos(t)
            fade_in = np.sin(t)
            attack_overlap = attack_part[-stitch_samples:] * fade_out
            sustain_overlap = aligned_sustain[:stitch_samples] * fade_in
            stitched_seam = attack_overlap + sustain_overlap

            out_audio = np.concatenate([
                attack_part[:-stitch_samples],
                stitched_seam,
                aligned_sustain[stitch_samples:],
            ])
        else:
            out_audio = np.concatenate([attack_part, stretched_sustain])

        target_samples = int(target_duration * sr)
        if len(out_audio) < target_samples:
            out_audio = np.pad(out_audio, (0, target_samples - len(out_audio)))
        else:
            out_audio = out_audio[:target_samples]

        if len(out_audio) > anti_click_n and anti_click_n > 1:
            t = np.linspace(0.0, np.pi / 2.0, anti_click_n, dtype=np.float32)
            out_audio = out_audio.astype(np.float32, copy=True)
            out_audio[-anti_click_n:] *= np.cos(t)

        results.append((target_duration, out_audio.astype(np.float32)))

    return results
