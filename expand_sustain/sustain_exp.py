import librosa
import numpy as np
import soundfile as sf
import os


def main():
    directory = os.path.dirname(os.path.abspath(__file__))
    filenames = [f for f in os.listdir(directory) if f.endswith('.wav')]
    for filename in filenames:
        stretch_masinqo_sustain(filename, f"Stretched {filename.split('.')[0].split(' ')[-1]}")
def stretch_masinqo_sustain(input_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    y, sr = librosa.load(input_file, sr=None)
    
    # 1. Isolate the Attack
    target_attack = int(0.08 * sr)
    zero_crossings = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
    attack_samples = zero_crossings[np.argmin(np.abs(zero_crossings - target_attack))]
    attack_part = y[:attack_samples]
    
    # 2. Strip the Natural Decay
    tail = y[attack_samples:]
    rms = librosa.feature.rms(y=tail)[0]
    
    threshold = np.max(rms) * 0.3 
    valid_frames = np.where(rms >= threshold)[0]
    
    if len(valid_frames) > 0:
        last_loud_sample = librosa.frames_to_samples(valid_frames[-1])
        sustain_part = tail[:last_loud_sample]
    else:
        sustain_part = tail
        
    print(f"Stripped {(len(tail) - len(sustain_part))/sr:.2f}s of quiet decay before stretching.")
    
    buffer_samples = 2048
    if attack_samples > buffer_samples:
        buffered_sustain = y[attack_samples - buffer_samples : attack_samples + len(sustain_part)]
    else:
        buffered_sustain = sustain_part
        buffer_samples = 0
        
    # 3. Generate Variations
    durations = np.linspace(0.5, 2.0, 10)
    
    for i, target_duration in enumerate(durations):
        target_sustain_duration = target_duration - (attack_samples / sr)
        if target_sustain_duration <= 0:
            continue
            
        original_sustain_duration = len(sustain_part) / sr
        rate = original_sustain_duration / target_sustain_duration
        
        # Stretch
        stretched_buffered = librosa.effects.time_stretch(buffered_sustain, rate=rate)
        
        if buffer_samples > 0:
            stretched_buffer_len = int(buffer_samples / rate)
            stretched_sustain = stretched_buffered[stretched_buffer_len:]
        else:
            stretched_sustain = stretched_buffered
        
        # 4. Cross-Correlation Phase Alignment
        stitch_samples = int(0.04 * sr) # 40ms crossfade
        max_shift = int(0.015 * sr)     # 15ms sliding window for phase alignment
        
        if len(attack_part) > stitch_samples and len(stretched_sustain) > (stitch_samples + max_shift):
            base_sig = attack_part[-stitch_samples:]
            best_corr = -np.inf
            best_shift = 0
            
            # Slide the stretched array to find the exact sample where peaks align
            for shift in range(0, max_shift):
                test_sig = stretched_sustain[shift : stitch_samples + shift]
                corr = np.sum(base_sig * test_sig) # Dot product measures alignment
                if corr > best_corr:
                    best_corr = corr
                    best_shift = shift
                    
            # Lock the phase by discarding the misaligned samples
            aligned_sustain = stretched_sustain[best_shift:]
            
            # 5. Equal-Power Stitching
            t = np.linspace(0, np.pi / 2, stitch_samples)
            fade_out = np.cos(t)
            fade_in = np.sin(t)
            
            attack_overlap = attack_part[-stitch_samples:] * fade_out
            sustain_overlap = aligned_sustain[:stitch_samples] * fade_in
            
            stitched_seam = attack_overlap + sustain_overlap
            
            out_audio = np.concatenate([
                attack_part[:-stitch_samples], 
                stitched_seam, 
                aligned_sustain[stitch_samples:]
            ])
        else:
            out_audio = np.concatenate([attack_part, stretched_sustain])
        
        # Trim to target length. Real release shape is randomized at composition
        # time (see augment/dsp.apply_random_release). Here we only apply a 5 ms
        # cosine anti-click fade so the wav file does not terminate on a click.
        target_samples = int(target_duration * sr)
        if len(out_audio) < target_samples:
            out_audio = np.pad(out_audio, (0, target_samples - len(out_audio)))
        else:
            out_audio = out_audio[:target_samples] 
        
        anti_click_samples = int(0.005 * sr)
        if len(out_audio) > anti_click_samples and anti_click_samples > 1:
            t = np.linspace(0, np.pi / 2, anti_click_samples)
            out_audio[-anti_click_samples:] *= np.cos(t)
            
        filename = f"masinqo_stretched_{i+1}_{target_duration:.2f}s.wav"
        filepath = os.path.join(output_folder, filename)
        sf.write(filepath, out_audio, sr)
        print(f"Exported: {filename} | Verified Length: {len(out_audio)/sr:.2f}s")

# Execution
if __name__ == "__main__":
    main()
