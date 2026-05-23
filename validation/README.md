# Real Masinko Validation Set

This folder is a scaffold for a future **real-recording** validation set. It is intentionally empty for now -- you are running the augmentation pipeline against synthetic data only.

When you are ready to collect real masinko recordings, follow this workflow:

## Layout

```
validation/
  audio/                 # drop real masinko recordings here (.wav, mono, any sample rate)
  midi/                  # corresponding ground-truth annotations (.mid)
  manifests/manifest.csv # 1 row per clip
```

Audio and MIDI files must share a basename: `validation/audio/clip_001.wav` <-> `validation/midi/clip_001.mid`.

## Recommended target

- 30 to 120 seconds total, split across 5 to 10 short clips (3 to 15 s each).
- Cover the playable pitch range (ideally A#4 to D7 to match the augmented pool).
- Mix of techniques: short attacks, long sustains with vibrato, legato slurs, phrase rests.
- Ideally recorded with two different mics or rooms to test EQ/reverb robustness.

## Annotation

Quickest path: REAPER + manual MIDI item editing while listening to the audio. Export MIDI per clip.

Programmatic alternative (for fine-grained timing): use [`pretty_midi`](https://github.com/craffel/pretty-midi) to write MIDI from a CSV of `(pitch, start_s, end_s)` events.

## manifest.csv schema

```
audio_path,midi_path,duration_s,notes_count,annotator,notes
audio/clip_001.wav,midi/clip_001.mid,12.5,18,you,"clean recording in living room"
```

`notes_count` should match the number of note events in the corresponding MIDI -- a quick sanity check.

## Running evaluation against this set

Once populated, run:

```
python eval_basic_pitch.py --audio-dir validation/audio --midi-dir validation/midi --out-json validation/results_<tag>.json
```

This reports onset / note / frame F1 using `mir_eval`, both per-file and aggregate. Use the same command against the base basic-pitch checkpoint and your fine-tuned checkpoint to measure improvement.
