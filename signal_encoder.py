"""encoder_pretty.py

Convert an image into a color audio signal.

This module synthesizes a short tone per image row, mapping the image's
R/G/B columns to three disjoint frequency bands. It writes a WAV file and
a small JSON metadata file describing width / height / duration_per_line / sample_rate / freq range.

Usage: call image_to_audio_color(image_path, out_path, meta_out, ...)
"""

import numpy as np
from scipy.io.wavfile import write
from PIL import Image
import json
from tqdm import tqdm

def image_to_audio_color(image_path, out_path="signal_color.wav",
                         meta_out="signal_color_meta.json",
                         target_w=480, target_h=319,
                         duration_per_line=0.1, sample_rate=96000,
                         freq_min=200, freq_max=20000,  # wider spectrum for 3 channels
                         min_brightness_threshold=0.03):

    # Load image as RGB and resize to target resolution
    img = Image.open(image_path).convert("RGB").resize((target_w, target_h), Image.LANCZOS)
    w, h = img.size
    # Data shape is now (h, w, 3) for R, G, B; normalize to [0,1]
    data = np.array(img).astype(np.float32) / 255.0

    total_samples = int(sample_rate * duration_per_line * h)
    signal = np.zeros(total_samples, dtype=np.float32)

    # Split the frequency band into three regions for R, G and B
    freq_range = freq_max - freq_min
    r_freqs = np.linspace(freq_min, freq_min + freq_range / 3, w)
    g_freqs = np.linspace(freq_min + freq_range / 3, freq_min + 2 * freq_range / 3, w)
    b_freqs = np.linspace(freq_min + 2 * freq_range / 3, freq_max, w)

    freq_maps = [r_freqs, g_freqs, b_freqs]

    print(f"Starting encoding of {h} lines...")
    # Use tqdm to show progress and estimated remaining time
    for i, row_data in tqdm(enumerate(data), total=h, desc="Encoding", unit="lines",
                           bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} lines [ETA: {remaining}]"):
        start = int(i * duration_per_line * sample_rate)
        end = int((i + 1) * duration_per_line * sample_rate)
        t = np.linspace(0, duration_per_line, end - start, endpoint=False)

        line_tone = np.zeros_like(t, dtype=np.float32)

        # Iterate over each pixel column and each color channel
        for j in range(w):
            for chan in range(3):  # 0=R, 1=G, 2=B
                brightness = row_data[j, chan]
                if brightness <= min_brightness_threshold:
                    continue

                base_freqs = freq_maps[chan]
                f = base_freqs[j]

                # Simple synth voice: a slightly detuned pair of sine partials
                detune = 1.0 + np.random.uniform(-0.002, 0.002)
                freq1 = f * detune
                freq2 = f * (1.005 + np.random.uniform(-0.001, 0.001))

                # Add the osc contribution scaled by pixel brightness
                partial = (np.sin(2 * np.pi * freq1 * t) + 0.6 * np.sin(2 * np.pi * freq2 * t)) * brightness
                line_tone += partial

        # If the produced line is effectively silent, skip writing it
        if np.max(np.abs(line_tone)) < 1e-9:
            continue

        # Apply envelope (attack/sustain/release) and a small smoothing filter
        attack_samples = max(3, int(0.01 * len(t)))
        release_samples = attack_samples
        sustain_len = len(t) - attack_samples - release_samples
        if sustain_len < 1:
            env = np.hanning(len(t))
        else:
            a = 0.5 * (1 - np.cos(np.linspace(0, np.pi, attack_samples)))
            r = 0.5 * (1 - np.cos(np.linspace(np.pi, 0, release_samples)))
            s = np.ones(sustain_len)
            env = np.concatenate([a, s, r])

        line_tone *= env
        kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        line_tone = np.convolve(line_tone, kernel, mode='same')

        # Normalize per-line then write into the overall signal buffer
        line_tone /= (np.max(np.abs(line_tone)) + 1e-9)
        signal[start:end] = line_tone * 0.9

    # Normalize overall signal and write WAV
    if np.max(np.abs(signal)) > 0:
        signal = signal / np.max(np.abs(signal)) * 0.95

    write(out_path, sample_rate, signal.astype(np.float32))
    print(f"✅ Saved color signal WAV: {out_path}")

    # Save metadata to help the decoder match encoding parameters
    meta = {
        "width": w,
        "height": h,
        "duration_per_line": duration_per_line,
        "sample_rate": sample_rate,
        "freq_min": freq_min,
        "freq_max": freq_max,
        "color": True,  # flag for the decoder
        "channels": 3,
    }
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Saved color metadata JSON: {meta_out}")

if __name__ == "__main__":
   
    image_to_audio_color("Dog.png", 
                         out_path="signal_color.wav",
                         meta_out="signal_color_meta.json",
                         duration_per_line=0.18,
                         sample_rate=44100,   
                         freq_min=300,
                         freq_max=18000)