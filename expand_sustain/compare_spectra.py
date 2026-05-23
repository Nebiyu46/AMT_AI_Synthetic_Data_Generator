import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def compare_spectrograms(original_file, stretched_file):
    # Load audio
    y_orig, sr = librosa.load(original_file, sr=None)
    y_stretch, _ = librosa.load(stretched_file, sr=sr)
    
    # Calculate Mel-Spectrograms (Standard for AMT models)
    # Using high-res FFT window to spot artifacts
    n_fft = 2048
    hop_length = 512
    
    S_orig = librosa.feature.melspectrogram(y=y_orig, sr=sr, n_fft=n_fft, hop_length=hop_length)
    S_stretch = librosa.feature.melspectrogram(y=y_stretch, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    S_orig_db = librosa.power_to_db(S_orig, ref=np.max)
    S_stretch_db = librosa.power_to_db(S_stretch, ref=np.max)
    
    # Plotting
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=False)
    
    img1 = librosa.display.specshow(S_orig_db, sr=sr, x_axis='time', y_axis='mel', ax=ax[0], hop_length=hop_length)
    ax[0].set_title('Original Nuhi Sample (What you started with)')
    fig.colorbar(img1, ax=ax[0], format="%+2.0f dB")
    
    img2 = librosa.display.specshow(S_stretch_db, sr=sr, x_axis='time', y_axis='mel', ax=ax[1], hop_length=hop_length)
    ax[1].set_title('2.0s Stretched Sample (What the ML model sees)')
    fig.colorbar(img2, ax=ax[1], format="%+2.0f dB")
    
    plt.tight_layout()
    plt.savefig("Spectrogram_Comparison.png")
    print("Saved comparison to Spectrogram_Comparison.png")
    plt.show()

# Run it
compare_spectrograms("Masinko_C#5.wav", "masinqo_stretched_10_2.00s.wav")