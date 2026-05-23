"""
Evaluate a basic-pitch checkpoint against a directory of audio + ground-truth MIDI.

Reports per-file and aggregate:
  - onset F1 (mir_eval onset-only, 50 ms tolerance)
  - note F1 (mir_eval transcription with offset_ratio=0.2)
  - frame F1 (mir_eval multipitch frame-level, 10 cents tolerance)

Usage
-----
    python eval_basic_pitch.py \
        --audio-dir validation/audio \
        --midi-dir  validation/midi \
        --out-json  validation/results_<tag>.json

Compare base vs fine-tuned by running twice with different --model-path values.

Notes
-----
- Designed for the synthetic Dataset_v2 held-out subset AND for a future real
  validation set (validation/audio + validation/midi).
- This script does NOT split or train. Splitting belongs in the training pipeline.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Optional

import librosa
import mir_eval
import numpy as np
import pretty_midi


def _midi_note_events(midi_path: str):
    """Return ndarray (intervals shape (N,2), pitches_hz shape (N,))."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    intervals = []
    pitches_hz = []
    for inst in pm.instruments:
        for note in inst.notes:
            if note.end <= note.start:
                continue
            intervals.append([float(note.start), float(note.end)])
            pitches_hz.append(float(librosa.midi_to_hz(int(note.pitch))))
    if intervals:
        return np.asarray(intervals, dtype=float), np.asarray(pitches_hz, dtype=float)
    return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float)


def _midi_frame_pitches(
    midi_path: str,
    hop_s: float = 0.0116,
    total_duration: Optional[float] = None,
):
    pm = pretty_midi.PrettyMIDI(midi_path)
    if total_duration is None:
        total_duration = pm.get_end_time()
    if total_duration <= 0:
        return np.zeros(0), [np.zeros(0) for _ in range(0)]

    n_frames = int(total_duration / hop_s) + 1
    times = np.arange(n_frames, dtype=float) * hop_s
    frames: list[list[float]] = [[] for _ in range(n_frames)]
    for inst in pm.instruments:
        for note in inst.notes:
            start_frame = int(note.start / hop_s)
            end_frame = int(note.end / hop_s)
            freq_hz = float(librosa.midi_to_hz(int(note.pitch)))
            for f in range(max(0, start_frame), min(end_frame, n_frames)):
                frames[f].append(freq_hz)
    frames_np = [np.asarray(f, dtype=float) for f in frames]
    return times, frames_np


def _events_to_frames(intervals, pitches_midi, hop_s, n_frames):
    frames: list[list[float]] = [[] for _ in range(n_frames)]
    for (s, e), p in zip(intervals, pitches_midi):
        sf_ = int(s / hop_s)
        ef_ = int(e / hop_s)
        freq_hz = float(librosa.midi_to_hz(int(p)))
        for f in range(max(0, sf_), min(ef_, n_frames)):
            frames[f].append(freq_hz)
    return [np.asarray(f, dtype=float) for f in frames]


def _predict(audio_path: str, model_path: Optional[str]):
    """Run basic-pitch and return (note_events, midi_data)."""
    from basic_pitch.inference import predict
    if model_path is None:
        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            model_or_path = ICASSP_2022_MODEL_PATH
        except Exception:
            model_or_path = None
    else:
        model_or_path = model_path

    if model_or_path is None:
        model_output, midi_data, note_events = predict(audio_path)
    else:
        model_output, midi_data, note_events = predict(audio_path, model_or_model_path=model_or_path)
    return note_events, midi_data


def evaluate(
    audio_dir: str,
    midi_dir: str,
    model_path: Optional[str] = None,
    out_json: Optional[str] = None,
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
    hop_s: float = 0.0116,
):
    audio_files = sorted(glob.glob(os.path.join(audio_dir, '*.wav')))
    if not audio_files:
        print(f"No .wav files found in {audio_dir}")
        return None

    print(f"Evaluating {len(audio_files)} files. Model: {model_path or 'base ICASSP_2022'}\n")

    per_file = []
    onset_fs, note_fs, frame_fs = [], [], []

    for k, audio_path in enumerate(audio_files, start=1):
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        gt_midi = os.path.join(midi_dir, basename + '.mid')
        if not os.path.exists(gt_midi):
            print(f"  [{k}/{len(audio_files)}] {basename}: no ground truth -- skipping")
            continue

        t0 = time.time()
        try:
            note_events, _ = _predict(audio_path, model_path)
        except Exception as e:
            print(f"  [{k}/{len(audio_files)}] {basename}: predict failed: {e}")
            continue

        gt_intervals, gt_pitches_hz = _midi_note_events(gt_midi)

        if note_events:
            pred_intervals = np.asarray([[ev[0], ev[1]] for ev in note_events], dtype=float)
            pred_pitches_hz = np.asarray([float(librosa.midi_to_hz(int(ev[2]))) for ev in note_events], dtype=float)
            pred_pitches_midi = [int(ev[2]) for ev in note_events]
        else:
            pred_intervals = np.zeros((0, 2), dtype=float)
            pred_pitches_hz = np.zeros(0, dtype=float)
            pred_pitches_midi = []

        if len(gt_intervals) and len(pred_intervals):
            on_p, on_r, on_f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                gt_intervals, gt_pitches_hz, pred_intervals, pred_pitches_hz,
                onset_tolerance=onset_tolerance, offset_ratio=None,
            )
            n_p, n_r, n_f, _ = mir_eval.transcription.precision_recall_f1_overlap(
                gt_intervals, gt_pitches_hz, pred_intervals, pred_pitches_hz,
                onset_tolerance=onset_tolerance,
                offset_ratio=offset_ratio,
                offset_min_tolerance=onset_tolerance,
            )
        else:
            on_p = on_r = on_f = 0.0
            n_p = n_r = n_f = 0.0

        # Frame-level
        end_t_gt = float(gt_intervals[:, 1].max()) if len(gt_intervals) else 0.0
        end_t_pred = float(pred_intervals[:, 1].max()) if len(pred_intervals) else 0.0
        end_t = max(end_t_gt, end_t_pred)
        if end_t > 0:
            n_frames = int(end_t / hop_s) + 1
            times = np.arange(n_frames, dtype=float) * hop_s
            gt_pitches_midi = [int(round(librosa.hz_to_midi(p))) for p in gt_pitches_hz]
            gt_frames = _events_to_frames(gt_intervals, gt_pitches_midi, hop_s, n_frames)
            pred_frames = _events_to_frames(pred_intervals, pred_pitches_midi, hop_s, n_frames)
            try:
                mp = mir_eval.multipitch.metrics(times, gt_frames, times, pred_frames)
                p_frame = float(mp[0]) if isinstance(mp, tuple) else float(mp.get('Precision', 0.0))
                r_frame = float(mp[1]) if isinstance(mp, tuple) else float(mp.get('Recall', 0.0))
                if (p_frame + r_frame) > 0:
                    f_frame = 2 * p_frame * r_frame / (p_frame + r_frame)
                else:
                    f_frame = 0.0
            except Exception as e:
                print(f"    frame eval failed: {e}")
                p_frame = r_frame = f_frame = 0.0
        else:
            p_frame = r_frame = f_frame = 0.0

        elapsed = time.time() - t0
        row = {
            'file': basename,
            'onset_p': float(on_p), 'onset_r': float(on_r), 'onset_f': float(on_f),
            'note_p': float(n_p), 'note_r': float(n_r), 'note_f': float(n_f),
            'frame_p': float(p_frame), 'frame_r': float(r_frame), 'frame_f': float(f_frame),
            'gt_notes': int(len(gt_intervals)),
            'pred_notes': int(len(pred_intervals)),
            'predict_seconds': float(elapsed),
        }
        per_file.append(row)
        onset_fs.append(on_f)
        note_fs.append(n_f)
        frame_fs.append(f_frame)
        print(f"  [{k}/{len(audio_files)}] {basename}: "
              f"onset_F1={on_f:.3f}  note_F1={n_f:.3f}  frame_F1={f_frame:.3f}  "
              f"gt={len(gt_intervals)} pred={len(pred_intervals)}  ({elapsed:.1f}s)")

    aggregate = {
        'n_files': len(per_file),
        'onset_f_mean': float(np.mean(onset_fs)) if onset_fs else 0.0,
        'note_f_mean': float(np.mean(note_fs)) if note_fs else 0.0,
        'frame_f_mean': float(np.mean(frame_fs)) if frame_fs else 0.0,
        'onset_f_median': float(np.median(onset_fs)) if onset_fs else 0.0,
        'note_f_median': float(np.median(note_fs)) if note_fs else 0.0,
        'frame_f_median': float(np.median(frame_fs)) if frame_fs else 0.0,
    }

    print("\n=== AGGREGATE ===")
    for k_, v in aggregate.items():
        if isinstance(v, float):
            print(f"  {k_:>16s}: {v:.4f}")
        else:
            print(f"  {k_:>16s}: {v}")

    output = {
        'aggregate': aggregate,
        'per_file': per_file,
        'config': {
            'audio_dir': audio_dir,
            'midi_dir': midi_dir,
            'model_path': model_path,
            'onset_tolerance': onset_tolerance,
            'offset_ratio': offset_ratio,
            'hop_s': hop_s,
        },
    }
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)) or '.', exist_ok=True)
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        print(f"\nWrote {out_json}")

    return output


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--audio-dir', required=True)
    p.add_argument('--midi-dir', required=True)
    p.add_argument('--model-path', default=None,
                   help='Path to a basic-pitch model checkpoint. Defaults to ICASSP_2022_MODEL_PATH.')
    p.add_argument('--out-json', default=None)
    p.add_argument('--onset-tolerance', type=float, default=0.05)
    p.add_argument('--offset-ratio', type=float, default=0.2)
    p.add_argument('--hop-s', type=float, default=0.0116)
    args = p.parse_args()
    evaluate(
        audio_dir=args.audio_dir,
        midi_dir=args.midi_dir,
        model_path=args.model_path,
        out_json=args.out_json,
        onset_tolerance=args.onset_tolerance,
        offset_ratio=args.offset_ratio,
        hop_s=args.hop_s,
    )


if __name__ == "__main__":
    _cli()
