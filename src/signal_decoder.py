"""decoder_live.py

Stable color WAV -> Image decoder with a live-preview style runner.
This module contains a single convenience function:

    decode_wav_to_image_live_color(wav_path, meta_path=None, out_path='decoded_live_color.png', ...)

The function will try to load encoder metadata if provided and then compute
an STFT, accumulate frequency-band magnitudes per image row, show a live
preview during processing, and save the final decoded color image.

The implementation applies a bandpass filter, chunked AGC, 2D median denoising
and robust per-row normalization to make the decoder resilient to noisy
recordings.
"""

import time
import threading
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, resample, butter, filtfilt
from scipy.ndimage import median_filter
import sounddevice as sd
from PIL import Image


def decode_wav_to_image_live_color(wav_path, meta_path=None, out_path="decoded_live_color.png",
                                   img_width=None, img_height=None,
                                   duration_per_line=None, sample_rate_expected=None,
                                   freq_min=None, freq_max=None,
                                   playback_speed=None, preview_target_duration=30.0, max_playback_speed=8.0):
    """Decode a WAV into a color image with a live preview.

    Parameters
    - wav_path: path to WAV file
    - meta_path: optional JSON metadata file produced by the encoder (preferred)
    - out_path: output PNG path
    - img_width/img_height: override decoded image size (if not in metadata)
    - duration_per_line: override per-line duration (seconds)
    - sample_rate_expected: (optional) expected sample rate from metadata
    - freq_min/freq_max: optional frequency bounds
    - playback_speed: if None, auto-select so preview finishes quickly; >1 speeds up
    - preview_target_duration: target seconds for preview when auto-scaling
    - max_playback_speed: clamp for auto playback speed

    The function displays a live preview window and saves the final image.
    """

    # ---- Load metadata (if provided) ----
    if meta_path:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        img_width = img_width or meta.get("width")
        img_height = img_height or meta.get("height")
        duration_per_line = duration_per_line or meta.get("duration_per_line")
        sample_rate_expected = sample_rate_expected or meta.get("sample_rate")
        freq_min = freq_min or meta.get("freq_min")
        freq_max = freq_max or meta.get("freq_max")
    else:
        # Fallback defaults
        if img_width is None:
            img_width = 480
        if freq_min is None:
            freq_min = 200
        if freq_max is None:
            freq_max = 20000
        print("Warning: no metadata found, using fallback defaults.")

    # ---- Load audio ----
    rate, data = wavfile.read(wav_path)
    data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Apply a bandpass filter to remove noise outside the expected frequency range
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    # Add a small margin to the expected frequency range but stay under Nyquist
    filter_low = max(20.0, (freq_min or 200.0) * 0.9)
    filter_high = min(rate / 2.0 * 0.95, (freq_max or (rate / 2.0)) * 1.1)
    try:
        b, a = butter_bandpass(filter_low, filter_high, rate)
        data = filtfilt(b, a, data)
    except Exception:
        # If filter fails (bad filter design because of parameters), ignore filtering
        pass

    # Simple automatic gain control (AGC): normalize each 100 ms chunk to avoid large level jumps
    chunk_size = max(1, int(rate * 0.1))  # 100 ms chunks
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        m = np.abs(chunk).max()
        if m > 0:
            data[i:i + chunk_size] = chunk / m

    total_duration = len(data) / float(rate)

    # ---- Playback speed / preview duration auto-adjust ----
    if playback_speed is None:
        if total_duration > preview_target_duration:
            playback_speed = min(max_playback_speed, total_duration / preview_target_duration)
        else:
            playback_speed = 1.0
    else:
        playback_speed = float(playback_speed) if playback_speed > 0 else 1.0

    # ---- Adjust parameters ----
    if img_height is None:
        img_height = int(total_duration / (duration_per_line or 0.1))
    else:
        img_height = int(img_height)
        duration_per_line = total_duration / img_height

    if sample_rate_expected and rate != sample_rate_expected:
        print(f"Warning: WAV sample rate ({rate} Hz) does not match metadata ({sample_rate_expected} Hz).")

    print(f"Playing audio ({total_duration:.2f}s)...")
    print(f"Preview speed: {playback_speed:.2f}x -> preview length ≈ {total_duration / playback_speed:.2f}s")
    print(f"Target resolution: {img_width}x{img_height} (color)")
    print(f"Estimated duration per line ≈ {duration_per_line:.4f}s")

    # ---- Compute STFT ----
    nperseg = 16384
    noverlap = int(nperseg * 0.75)
    try:
        f, t, Zxx = stft(data, fs=rate, window='blackman', nperseg=nperseg, noverlap=noverlap)
    except Exception:
        # Fallback to smaller window if memory or parameter issues occur
        nperseg = 4096
        noverlap = int(nperseg * 0.75)
        f, t, Zxx = stft(data, fs=rate, window='blackman', nperseg=nperseg, noverlap=noverlap)

    magnitude = np.abs(Zxx)

    # Apply a small 2D median filter to reduce transient noise while preserving edges
    try:
        magnitude = median_filter(magnitude, size=(3, 3))
    except Exception:
        pass

    # Slight dynamic compression to boost weak frequencies that may carry image info
    magnitude = np.power(magnitude, 0.8)

    # ---- Frequency bands for R, G, B ----
    freq_range = float(freq_max - freq_min)
    r_freqs = np.linspace(freq_min, freq_min + freq_range / 3.0, img_width)
    g_freqs = np.linspace(freq_min + freq_range / 3.0, freq_min + 2.0 * freq_range / 3.0, img_width)
    b_freqs = np.linspace(freq_min + 2.0 * freq_range / 3.0, freq_max, img_width)

    r_bin_indices = np.array([np.argmin(np.abs(f - freq)) for freq in r_freqs])
    g_bin_indices = np.array([np.argmin(np.abs(f - freq)) for freq in g_freqs])
    b_bin_indices = np.array([np.argmin(np.abs(f - freq)) for freq in b_freqs])

    t_corrected = t - t[0]
    frame_line_indices = np.clip((t_corrected / duration_per_line).astype(int), 0, img_height - 1)

    # ---- Accumulation arrays ----
    img_accumulated = np.zeros((img_height, img_width, 3), dtype=np.float32)
    counts = np.zeros(img_height, dtype=int)

    # ---- Play audio (optionally sped-up for preview) ----
    start_time = time.time()

    def _play_thread():
        try:
            if playback_speed is None or abs(playback_speed - 1.0) < 1e-6:
                sd.play(data, rate)
            else:
                n_out = max(1, int(len(data) / float(playback_speed)))
                try:
                    resampled = resample(data, n_out)
                except Exception:
                    resampled = data
                resampled = np.asarray(resampled, dtype=np.float32)
                sd.play(resampled, rate)
        except Exception as e:
            print(f"Warning: audio playback failed: {e}")

    threading.Thread(target=_play_thread, daemon=True).start()

    # ---- Live preview setup ----
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, max(2, 8 * img_height / float(img_width))))
    im = ax.imshow(np.zeros((img_height, img_width, 3)), vmin=0, vmax=1)
    plt.axis('off')
    ax.set_title("Live decoding (color, stable)")

    last_preview_display = np.zeros((img_height, img_width, 3), dtype=np.float32)

    # ---- Live decoding ----
    update_interval_frames = max(1, int(8 * playback_speed))

    for frame_idx in range(magnitude.shape[1]):
        target_time = t[frame_idx]
        # Wait until the time for this frame (respecting playback speed)
        while (time.time() - start_time) < (target_time / playback_speed):
            time.sleep(0.001)

        line_idx = int(frame_line_indices[frame_idx])

        # Extract magnitudes for each color band (per-column)
        r_mags = magnitude[r_bin_indices, frame_idx]
        g_mags = magnitude[g_bin_indices, frame_idx]
        b_mags = magnitude[b_bin_indices, frame_idx]

        # Accumulate
        img_accumulated[line_idx, :, 0] += r_mags
        img_accumulated[line_idx, :, 1] += g_mags
        img_accumulated[line_idx, :, 2] += b_mags
        counts[line_idx] += 1

        # Update the live display occasionally to reduce overhead when sped up
        if frame_idx % update_interval_frames == 0:
            display_img = np.zeros_like(img_accumulated)
            for i in range(img_height):
                if counts[i] > 0:
                    display_img[i, :, :] = img_accumulated[i, :, :] / counts[i]

            # Stable live normalization per-row with a small smoothing window
            display_img_normalized = np.zeros_like(display_img)
            window_size = 5
            for i in range(min(line_idx + 1, img_height)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(line_idx + 1, i + window_size // 2 + 1)
                row_window = display_img[start_idx:end_idx, :, :]

                non_zero_vals = row_window[row_window > 1e-9]
                if non_zero_vals.size > 0:
                    p95 = np.percentile(non_zero_vals, 95.0)
                    p50 = np.percentile(non_zero_vals, 50.0)
                    row_max = p95 * 0.7 + p50 * 0.3
                else:
                    row_max = 1.0

                if row_max < 1e-9:
                    row_max = 1.0

                row = display_img[i, :, :]
                normalized = (row / row_max) ** 0.9
                display_img_normalized[i, :, :] = normalized

            display_img_normalized = np.clip(display_img_normalized, 0.0, 1.0)
            im.set_data(display_img_normalized)
            last_preview_display = display_img_normalized.copy()
            plt.pause(0.001)

    # ---- Final normalization and save ----
    if last_preview_display is not None and np.any(last_preview_display > 1e-9):
        final_img_normalized = last_preview_display.copy()
        final_img_normalized = np.clip(final_img_normalized, 0.0, 1.0)
    else:
        final_img_normalized = np.zeros_like(img_accumulated)
        for i in range(img_height):
            if counts[i] > 0:
                final_img_normalized[i, :, :] = img_accumulated[i, :, :] / counts[i]

        zero_rows = np.where(counts == 0)[0]
        valid_rows = np.where(counts > 0)[0]
        if valid_rows.size == 0:
            print("Warning: no rows decoded; output image will be blank.")
        elif zero_rows.size > 0:
            for r in zero_rows:
                below = valid_rows[valid_rows < r]
                above = valid_rows[valid_rows > r]
                if below.size > 0 and above.size > 0:
                    low = below[-1]
                    high = above[0]
                    w = (r - low) / float(high - low)
                    final_img_normalized[r, :, :] = (1.0 - w) * final_img_normalized[low, :, :] + w * final_img_normalized[high, :, :]
                elif below.size > 0:
                    final_img_normalized[r, :, :] = final_img_normalized[below[-1], :, :]
                elif above.size > 0:
                    final_img_normalized[r, :, :] = final_img_normalized[above[0], :, :]

            try:
                from scipy.ndimage import gaussian_filter1d
                for ch in range(3):
                    final_img_normalized[:, :, ch] = gaussian_filter1d(final_img_normalized[:, :, ch], sigma=1.0, axis=0, mode='nearest')
            except Exception:
                pass

        non_zero_vals = final_img_normalized[final_img_normalized > 1e-9]
        max_val = np.percentile(non_zero_vals, 99.0) if non_zero_vals.size > 0 else 1.0
        if max_val < 1e-9:
            max_val = 1.0

        final_img_normalized /= max_val
        final_img_normalized = np.clip(final_img_normalized, 0, 1)

    decoded_img_8bit = (final_img_normalized * 255).astype(np.uint8)
    Image.fromarray(decoded_img_8bit, 'RGB').save(out_path)
    print(f"✅ Saved decoded color image: {out_path}")

    plt.ioff()
    plt.figure()
    plt.imshow(decoded_img_8bit)
    plt.title("Final result (stable)")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    decode_wav_to_image_live_color("signal_color.wav", 
                                   meta_path="signal_color_meta.json", 
                                   out_path="decoded_live_color.png")
