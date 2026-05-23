"""
Build the augmented source pool from the 25 original masinko recordings.

For each original recording:
  - Time-stretch the source to 10 sustain durations (0.5 s ... 2.0 s).
  - Save as 'clean' variant.
  - Also save a 'noise' variant (pink noise at 20-40 dB SNR).

Output layout:
    pool/<midi_pitch>/<variant>/from_<src_note>_idx<i>_<dur>s.wav

No reverb is applied at this stage -- reverb is master-level only inside the composition
generator (see Generated_midi/Generate_midi_and_audio.py). Micro pitch variation is also
applied in the generator, not baked into the pool, so the label remains faithful and the
distribution stays continuous.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from augment.dsp import (
    add_pink_noise,
    stretch_to_durations,
)


DEFAULT_DURATIONS = tuple(np.linspace(0.5, 2.0, 10).tolist())


def parse_note_name(filename: str) -> str:
    """Extract note name from a filename like '01 Masinko By Nuhi_Legato C5.wav' -> 'C5'."""
    base = os.path.splitext(os.path.basename(filename))[0]
    return base.split(' ')[-1].strip()


def build_pool(
    original_dir: str,
    output_dir: str,
    durations=DEFAULT_DURATIONS,
    limit: int | None = None,
    overwrite: bool = False,
) -> None:
    originals = sorted(glob.glob(os.path.join(original_dir, '*.wav')))
    if limit is not None:
        originals = originals[:limit]
    if not originals:
        raise FileNotFoundError(f"No .wav files found in {original_dir}")

    if os.path.isdir(output_dir) and os.listdir(output_dir):
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already exists and is not empty. "
                "Use --overwrite to rebuild it from scratch and avoid mixing stale shifted samples."
            )
        shutil.rmtree(output_dir)

    print(f"Source originals: {len(originals)}")
    print("Pitch shifts: none (micro-detune is applied on the fly in the generator)")
    print("Variants: clean + pink noise")
    print(f"Stretch durations: {[round(d, 2) for d in durations]}")
    print(f"Output: {output_dir}\n")

    durations = list(durations)

    for src_idx, src_path in enumerate(originals, start=1):
        src_note = parse_note_name(src_path)
        try:
            src_pitch = int(librosa.note_to_midi(src_note))
        except Exception as e:
            print(f"[SKIP] {src_path}: cannot parse note '{src_note}' ({e})")
            continue

        print(f"[{src_idx}/{len(originals)}] {src_note} (MIDI {src_pitch}) :: {os.path.basename(src_path)}")

        y_orig, sr = librosa.load(src_path, sr=None, mono=True)
        y_orig = y_orig.astype(np.float32)

        stretched = stretch_to_durations(y_orig, sr, durations)
        if not stretched:
            print("  no stretches produced")
            continue

        clean_dir = os.path.join(output_dir, str(src_pitch), 'clean')
        noise_dir = os.path.join(output_dir, str(src_pitch), 'noise')
        os.makedirs(clean_dir, exist_ok=True)
        os.makedirs(noise_dir, exist_ok=True)

        for i, (dur, audio) in enumerate(stretched, start=1):
            fname = f"from_{src_note}_idx{i}_{dur:.2f}s.wav"
            sf.write(os.path.join(clean_dir, fname), audio, sr)

            noisy = add_pink_noise(audio, sr, snr_db_range=(20.0, 40.0))
            sf.write(os.path.join(noise_dir, fname), noisy, sr)

        print(f"  {src_note} (MIDI {src_pitch}): {len(stretched)} clean + {len(stretched)} noise")

    print("\nPool build complete.")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--original-dir',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'expand_sustain', 'Original_version'),
    )
    p.add_argument(
        '--output-dir',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pool'),
    )
    p.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Process only the first N originals (smoke test).',
    )
    p.add_argument(
        '--overwrite',
        action='store_true',
        help='Delete the output directory before rebuilding the pool.',
    )
    args = p.parse_args()
    build_pool(args.original_dir, args.output_dir, limit=args.limit, overwrite=args.overwrite)


if __name__ == "__main__":
    _cli()
