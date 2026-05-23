"""
Composition generator (v2).

Reads from the augmented source pool produced by build_pool.py and emits paired
audio/MIDI files. Per-note continuous augmentation (micro-detune, time-stretch,
gain, vibrato, randomized release) and inter-note structure (legato overlap /
normal rest / phrase rest with declick crossfades) happen here. Master-level
random EQ and optional convolutional reverb (synthetic short IRs, wet <= 0.18)
are applied once per output file -- never as a discrete pool variant.

Default output: Generated_midi/Dataset_v2/{audio,midi}/melody_NNNN.{wav,mid}
Default volume: 2000 files, 20-50 notes each.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augment.dsp import (
    add_vibrato,
    apply_random_release,
    convolve_reverb,
    make_synthetic_ir,
    peak_normalize,
    pitch_shift_relabel,
    random_eq,
)


def discover_pool(pool_dir: str) -> dict[int, list[str]]:
    """Return {midi_pitch: [path, ...]} from pool/<midi_pitch>/<variant>/*.wav."""
    note_paths: dict[int, list[str]] = {}
    if not os.path.isdir(pool_dir):
        return note_paths
    for pitch_str in sorted(os.listdir(pool_dir)):
        pitch_dir = os.path.join(pool_dir, pitch_str)
        if not os.path.isdir(pitch_dir):
            continue
        try:
            midi_pitch = int(pitch_str)
        except ValueError:
            continue
        files: list[str] = []
        for variant in os.listdir(pitch_dir):
            variant_dir = os.path.join(pitch_dir, variant)
            if os.path.isdir(variant_dir):
                files.extend(glob.glob(os.path.join(variant_dir, '*.wav')))
        if files:
            note_paths[midi_pitch] = files
    return note_paths


def _augment_note(y: np.ndarray, sr: int, detune_semitones: float = 0.0):
    """
    Per-note continuous augments. Returns (audio, effective_duration_s, gain).
    Audio length may change due to time-stretch.
    """
    if len(y) > int(0.7 * sr):
        stretch = random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y.astype(np.float32), rate=1.0 / stretch)

    if abs(detune_semitones) > 1e-6:
        y = pitch_shift_relabel(y.astype(np.float32), sr, detune_semitones)

    gain = random.uniform(0.7, 1.3)
    y = y.astype(np.float32) * gain

    if random.random() < 0.7:
        y = add_vibrato(y, sr)

    y = apply_random_release(y, sr)

    return y.astype(np.float32), len(y) / sr, gain


def _build_one_file(
    note_paths: dict[int, list[str]],
    notes_per_file: tuple[int, int],
    sr_target: int | None,
    reverb_prob: float = 0.5,
    global_detune_range: float = 0.03,
    pitch_detune_range: float = 0.20,
    event_detune_range: float = 0.02,
):
    """Build one audio array + PrettyMIDI for a single training file."""
    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    chunks: list[np.ndarray] = []
    current_time = 0.0
    prev_note_chunk_idx: int | None = None
    prev_pitch: int | None = None
    sr = sr_target

    num_notes = random.randint(notes_per_file[0], notes_per_file[1])
    pitches = list(note_paths.keys())
    global_detune = random.uniform(-global_detune_range, global_detune_range)
    pitch_detunes: dict[int, float] = {}

    for _ in range(num_notes):
        pitch = random.choice(pitches)
        audio_file = random.choice(note_paths[pitch])

        y_raw, sr_file = sf.read(audio_file, dtype='float32')
        if y_raw.ndim > 1:
            y_raw = y_raw.mean(axis=1)
        if sr is None:
            sr = sr_file
        elif sr_file != sr:
            y_raw = librosa.resample(y_raw, orig_sr=sr_file, target_sr=sr)

        if pitch not in pitch_detunes:
            pitch_detunes[pitch] = random.uniform(-pitch_detune_range, pitch_detune_range)
        event_jitter = random.uniform(-event_detune_range, event_detune_range)
        detune = global_detune + pitch_detunes[pitch] + event_jitter

        y, duration, _gain = _augment_note(y_raw, sr, detune_semitones=detune)

        # Decide spacing relative to previous note.
        r = random.random()
        if r < 0.35 and prev_note_chunk_idx is not None and prev_pitch != pitch:
            spacing = 'legato'
        elif r < 0.85:
            spacing = 'normal'
        else:
            spacing = 'phrase'

        if spacing == 'legato':
            overlap_n = int(random.uniform(0.030, 0.080) * sr)
            prev_chunk = chunks[prev_note_chunk_idx]
            overlap_n = min(overlap_n, len(prev_chunk), len(y))
            if overlap_n > 1:
                t = np.linspace(0.0, np.pi / 2.0, overlap_n, dtype=np.float32)
                fade_out = np.cos(t)
                fade_in = np.sin(t)
                blended = (prev_chunk[-overlap_n:] * fade_out + y[:overlap_n] * fade_in).astype(np.float32)
                # Replace the previous chunk with its truncated head, then insert the
                # blended seam, then the rest of y.
                chunks[prev_note_chunk_idx] = prev_chunk[:-overlap_n]
                chunks.append(blended)
                chunks.append(y[overlap_n:])
                prev_note_chunk_idx = len(chunks) - 1
                note_start = current_time - overlap_n / sr
                note_end = note_start + duration
                current_time = note_end
            else:
                chunks.append(y)
                prev_note_chunk_idx = len(chunks) - 1
                note_start = current_time
                note_end = note_start + duration
                current_time = note_end
        else:
            if prev_note_chunk_idx is not None:
                if spacing == 'normal':
                    rest = random.uniform(0.030, 0.200)
                else:
                    rest = random.uniform(0.4, 1.5)
                rest_audio = np.zeros(int(rest * sr), dtype=np.float32)
                chunks.append(rest_audio)
                current_time += rest
            chunks.append(y)
            prev_note_chunk_idx = len(chunks) - 1
            note_start = current_time
            note_end = note_start + duration
            current_time = note_end

        instrument.notes.append(pretty_midi.Note(
            velocity=100,
            pitch=int(pitch),
            start=float(note_start),
            end=float(note_end),
        ))
        prev_pitch = pitch

    midi_data.instruments.append(instrument)

    chunks = [c for c in chunks if len(c) > 0]
    if not chunks:
        return None, None, sr
    final_audio = np.concatenate(chunks).astype(np.float32)

    # Master-level effects: peak norm -> EQ -> optional reverb -> peak norm again.
    final_audio = peak_normalize(final_audio, target_dbfs=-3.0)
    final_audio = random_eq(final_audio, sr)
    if random.random() < reverb_prob:
        ir = make_synthetic_ir(sr)
        wet = random.uniform(0.05, 0.18)
        final_audio = convolve_reverb(final_audio, sr, ir, wet=wet)
    final_audio = peak_normalize(final_audio, target_dbfs=-3.0)

    return final_audio, midi_data, sr


def build_masinqo_dataset(
    num_files: int = 2000,
    notes_per_file: tuple[int, int] = (20, 50),
    pool_dir: str | None = None,
    out_dir: str | None = None,
    reverb_prob: float = 0.5,
    global_detune_range: float = 0.03,
    pitch_detune_range: float = 0.20,
    event_detune_range: float = 0.02,
    seed: int | None = None,
) -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if pool_dir is None:
        pool_dir = os.path.join(base_dir, '..', 'pool')
    if out_dir is None:
        out_dir = os.path.join(base_dir, 'Dataset_v2')

    out_audio = os.path.join(out_dir, 'audio')
    out_midi = os.path.join(out_dir, 'midi')
    os.makedirs(out_audio, exist_ok=True)
    os.makedirs(out_midi, exist_ok=True)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    note_paths = discover_pool(pool_dir)
    if not note_paths:
        print(f"Error: pool not found or empty at {pool_dir}. "
              f"Run build_pool.py first.")
        return

    total_files = sum(len(v) for v in note_paths.values())
    print(f"Pool: {len(note_paths)} pitches, {total_files} samples total.")
    print("Micro-detune: "
          f"global +/-{global_detune_range:.2f}, "
          f"per-pitch +/-{pitch_detune_range:.2f}, "
          f"per-event +/-{event_detune_range:.2f} semitones.")
    print(f"Generating {num_files} audio/MIDI pairs into {out_dir}/.\n")

    sr_target: int | None = None
    started = time.time()

    for i in range(num_files):
        final_audio, midi_data, sr_target = _build_one_file(
            note_paths,
            notes_per_file,
            sr_target,
            reverb_prob=reverb_prob,
            global_detune_range=global_detune_range,
            pitch_detune_range=pitch_detune_range,
            event_detune_range=event_detune_range,
        )
        if final_audio is None:
            print(f"  [{i+1}/{num_files}] empty composition, skipping")
            continue

        idx_str = str(i + 1).zfill(4)
        sf.write(os.path.join(out_audio, f'melody_{idx_str}.wav'), final_audio, sr_target)
        midi_data.write(os.path.join(out_midi, f'melody_{idx_str}.mid'))

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - started
            eta = (elapsed / max(i + 1, 1)) * (num_files - i - 1)
            print(f"  [{i+1}/{num_files}] length={len(final_audio)/sr_target:.2f}s "
                  f"notes={len(midi_data.instruments[0].notes)} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    print(f"\nDone. Wrote up to {num_files} pairs to {out_dir}/.")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-files', type=int, default=2000)
    p.add_argument('--notes-min', type=int, default=20)
    p.add_argument('--notes-max', type=int, default=50)
    p.add_argument('--pool-dir', default=None)
    p.add_argument('--out-dir', default=None)
    p.add_argument('--reverb-prob', type=float, default=0.5)
    p.add_argument('--global-detune-range', type=float, default=0.03,
                   help='Max absolute file-wide detune in semitones.')
    p.add_argument('--pitch-detune-range', type=float, default=0.20,
                   help='Max absolute per-pitch detune in semitones, fixed for repeated pitches in one file.')
    p.add_argument('--event-detune-range', type=float, default=0.02,
                   help='Max absolute extra per-note-occurrence detune in semitones.')
    p.add_argument('--seed', type=int, default=None)
    args = p.parse_args()

    build_masinqo_dataset(
        num_files=args.num_files,
        notes_per_file=(args.notes_min, args.notes_max),
        pool_dir=args.pool_dir,
        out_dir=args.out_dir,
        reverb_prob=args.reverb_prob,
        global_detune_range=args.global_detune_range,
        pitch_detune_range=args.pitch_detune_range,
        event_detune_range=args.event_detune_range,
        seed=args.seed,
    )


if __name__ == "__main__":
    _cli()
