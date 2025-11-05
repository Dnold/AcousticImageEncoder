"""
decoder_live.py

Stable color WAV -> Image decoder with live preview + synced mp4 recording.
"""

import time
import threading
import json
import subprocess
import os
import shutil
import tempfile

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Ensure GUI backend on Windows
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, resample, butter, filtfilt
from scipy.ndimage import median_filter
import sounddevice as sd
from PIL import Image
import cv2


def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def _merge_audio_video_ffmpeg(video_path, audio_path, out_path):
    """Merge audio (audio_path) into video (video_path) using ffmpeg with explicit mapping.
    Returns True on success, False on failure. Prints ffmpeg output on failure.
    """
    video_abs = os.path.abspath(video_path)
    audio_abs = os.path.abspath(audio_path)
    out_abs = os.path.abspath(out_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_abs,
        "-i", audio_abs,
        "-map", "0:v:0",    # first input's video
        "-map", "1:a:0",    # second input's audio
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        out_abs
    ]

    try:
        print("🔁 Running ffmpeg merge:", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ ffmpeg merge completed.")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ ffmpeg merge failed. stdout/stderr:\n", e.stdout, e.stderr)
        return False
    except FileNotFoundError:
        print("❌ ffmpeg not found on PATH.")
        return False


def decode_wav_to_image_live_color(wav_path, meta_path=None, out_path="decoded_live_color.png",
                                   img_width=None, img_height=None,
                                   duration_per_line=None, sample_rate_expected=None,
                                   freq_min=None, freq_max=None,
                                   playback_speed=None, preview_target_duration=30.0, max_playback_speed=8.0,
                                   video_out_path=None):
    """Decode a WAV into a color image with live preview + optional mp4 recording."""

    print("🚀 Starting decode_wav_to_image_live_color")
    # ---- Load metadata ----
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        img_width = img_width or meta.get("width")
        img_height = img_height or meta.get("height")
        duration_per_line = duration_per_line or meta.get("duration_per_line")
        sample_rate_expected = sample_rate_expected or meta.get("sample_rate")
        freq_min = freq_min or meta.get("freq_min")
        freq_max = freq_max or meta.get("freq_max")
    else:
        img_width = img_width or 480
        freq_min = freq_min or 200
        freq_max = freq_max or 20000
        if meta_path:
            print(f"⚠️ Meta path {meta_path} not found — using defaults.")

    # ---- Load audio ----
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"WAV file not found: {wav_path}")
    rate, data = wavfile.read(wav_path)
    print(f"Loaded WAV: {wav_path} ({rate} Hz, {data.shape})")
    data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)

    # ---- Bandpass filter (best-effort) ----
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    filter_low = max(20.0, (freq_min or 200.0) * 0.9)
    filter_high = min(rate / 2.0 * 0.95, (freq_max or (rate / 2.0)) * 1.1)
    try:
        b, a = butter_bandpass(filter_low, filter_high, rate)
        data = filtfilt(b, a, data)
    except Exception as e:
        print("⚠️ Bandpass filter failed (continuing):", e)

    # ---- Automatic gain control ----
    chunk_size = max(1, int(rate * 0.1))
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        m = np.abs(chunk).max()
        if m > 0:
            data[i:i + chunk_size] = chunk / m

    total_duration = len(data) / float(rate)

    # ---- Playback speed auto-select ----
    if playback_speed is None:
        if total_duration > preview_target_duration:
            playback_speed = min(max_playback_speed, total_duration / preview_target_duration)
        else:
            playback_speed = 1.0
    else:
        playback_speed = float(playback_speed) if playback_speed > 0 else 1.0

    # ---- Image dimensions / duration per line ----
    if img_height is None:
        img_height = int(total_duration / (duration_per_line or 0.1))
    else:
        img_height = int(img_height)
        duration_per_line = total_duration / img_height

    if sample_rate_expected and rate != sample_rate_expected:
        print(f"⚠️ WAV sample rate ({rate}) != expected ({sample_rate_expected})")

    print(f"Duration: {total_duration:.2f}s, playback_speed: {playback_speed:.2f}x")
    print(f"Target image: {img_width}x{img_height}, duration_per_line ≈ {duration_per_line:.4f}s")

    # ---- STFT ----
    nperseg = 16384
    noverlap = int(nperseg * 0.75)
    try:
        f, t, Zxx = stft(data, fs=rate, window='blackman', nperseg=nperseg, noverlap=noverlap)
    except Exception:
        nperseg = 4096
        noverlap = int(nperseg * 0.75)
        f, t, Zxx = stft(data, fs=rate, window='blackman', nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(Zxx)
    try:
        magnitude = median_filter(magnitude, size=(3, 3))
    except Exception:
        pass
    magnitude = np.power(magnitude, 0.8)

    if magnitude.size == 0 or magnitude.shape[1] == 0:
        raise RuntimeError("STFT produced no frames — check input WAV or STFT parameters")

    # ---- Frequency -> RGB mapping ----
    freq_range = float(freq_max - freq_min)
    r_freqs = np.linspace(freq_min, freq_min + freq_range / 3.0, img_width)
    g_freqs = np.linspace(freq_min + freq_range / 3.0, freq_min + 2.0 * freq_range / 3.0, img_width)
    b_freqs = np.linspace(freq_min + 2.0 * freq_range / 3.0, freq_max, img_width)

    # convert f to numpy array if not
    f = np.array(f)
    r_bin_indices = np.array([np.argmin(np.abs(f - freq)) for freq in r_freqs])
    g_bin_indices = np.array([np.argmin(np.abs(f - freq)) for freq in g_freqs])
    b_bin_indices = np.array([np.argmin(np.abs(f - freq)) for freq in b_freqs])

    t_corrected = t - t[0]
    frame_line_indices = np.clip((t_corrected / duration_per_line).astype(int), 0, img_height - 1)

    # ---- Accumulators ----
    img_accumulated = np.zeros((img_height, img_width, 3), dtype=np.float32)
    counts = np.zeros(img_height, dtype=int)

    # ---- Audio playback thread (plays locally, not embedded in mp4) ----
    start_time = time.time()

    def _play_thread():
        try:
            if abs(playback_speed - 1.0) < 1e-6:
                sd.play(data, rate)
            else:
                n_out = max(1, int(len(data) / float(playback_speed)))
                try:
                    resampled = resample(data, n_out)
                except Exception:
                    resampled = data
                sd.play(np.asarray(resampled, dtype=np.float32), rate)
        except Exception as e:
            print("⚠️ sounddevice playback failed:", e)

    threading.Thread(target=_play_thread, daemon=True).start()
    print("🔊 Playback thread launched (local playback)")

    # ---- Matplotlib live preview setup ----
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, max(2, 8 * img_height / float(img_width))))
    im = ax.imshow(np.zeros((img_height, img_width, 3)), vmin=0, vmax=1)
    plt.axis('off')
    ax.set_title("Live decoding (color, synced)")
    plt.show(block=False)

    # ---- Video writer setup (video only) ----
    video_writer = None
    fps = 30
    frame_duration = 1.0 / fps
    last_frame_time = time.time()
    if video_out_path:
        # ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(video_out_path)) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # OpenCV expects width,height as ints
        video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (int(img_width), int(img_height)))
        if not video_writer or not video_writer.isOpened():
            print("❌ VideoWriter failed to open — check codec/size")
            video_writer = None
        else:
            print("🎥 VideoWriter opened:", os.path.abspath(video_out_path))

    last_preview_display = np.zeros((img_height, img_width, 3), dtype=np.float32)
    update_interval_frames = max(1, int(8 * playback_speed))

    # ---- Live decoding loop ----
    print("▶️ Starting live decode loop")
    for frame_idx in range(magnitude.shape[1]):
        target_time = t[frame_idx]
        # Wait according to playback_speed so preview syncs to local playback
        while (time.time() - start_time) < (target_time / playback_speed):
            time.sleep(0.001)

        line_idx = int(frame_line_indices[frame_idx])

        r_mags = magnitude[r_bin_indices, frame_idx]
        g_mags = magnitude[g_bin_indices, frame_idx]
        b_mags = magnitude[b_bin_indices, frame_idx]

        img_accumulated[line_idx, :, 0] += r_mags
        img_accumulated[line_idx, :, 1] += g_mags
        img_accumulated[line_idx, :, 2] += b_mags
        counts[line_idx] += 1

        # occasionally update the preview (reduces overhead)
        if frame_idx % update_interval_frames == 0:
            display_img = np.zeros_like(img_accumulated)
            for i in range(img_height):
                if counts[i] > 0:
                    display_img[i, :, :] = img_accumulated[i, :, :] / counts[i]

            # per-row normalization with small smoothing window
            display_img_normalized = np.zeros_like(display_img)
            window_size = 5
            for i in range(min(frame_line_indices[frame_idx] + 1, img_height)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(frame_line_indices[frame_idx] + 1, i + window_size // 2 + 1)
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

            # --- Write video frames in real time to meet fps ---
            if video_writer is not None:
                now = time.time()
                # write as many frames as needed to catch up (frame_duration steps)
                while (now - last_frame_time) >= frame_duration:
                    frame_rgb = (display_img_normalized * 255).astype(np.uint8)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)
                    last_frame_time += frame_duration
                    now = time.time()

    print("⏹ Live decode loop finished")

    # ---- Finalize video writer and merge audio ----
    merged_output = None
    if video_writer is not None:
        video_writer.release()
        print("🎬 Video saved (video-only):", os.path.abspath(video_out_path))

        # Attempt to merge audio using ffmpeg if available
        if _ffmpeg_available():
            merged_output = video_out_path.replace(".mp4", "_with_audio.mp4")
            ok = _merge_audio_video_ffmpeg(video_out_path, wav_path, merged_output)
            if not ok:
                print("⚠️ ffmpeg merge failed — leaving silent video-only file.")
                merged_output = None
        else:
            print("⚠️ ffmpeg not found — cannot merge audio. Install ffmpeg and add to PATH.")

    # ---- Final image creation ----
    if last_preview_display is not None and np.any(last_preview_display > 1e-9):
        final_img_normalized = np.clip(last_preview_display.copy(), 0.0, 1.0)
    else:
        final_img_normalized = np.zeros_like(img_accumulated)
        for i in range(img_height):
            if counts[i] > 0:
                final_img_normalized[i, :, :] = img_accumulated[i, :, :] / counts[i]

        # fill empty rows linearly between valid rows
        zero_rows = np.where(counts == 0)[0]
        valid_rows = np.where(counts > 0)[0]
        if valid_rows.size > 0 and zero_rows.size > 0:
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
        final_img_normalized = np.clip(final_img_normalized, 0.0, 1.0)

    decoded_img_8bit = (final_img_normalized * 255).astype(np.uint8)
    Image.fromarray(decoded_img_8bit, 'RGB').save(out_path)
    print("✅ Final image saved:", os.path.abspath(out_path))
    if merged_output:
        print("✅ Video with audio saved:", os.path.abspath(merged_output))
    elif video_out_path:
        print("⚠️ Video-only file saved (no audio merged).")

    # ---- Show final result and wait ----
    plt.ioff()
    plt.figure()
    plt.imshow(decoded_img_8bit)
    plt.title("Final result (stable)")
    plt.axis('off')
    plt.show(block=True)
    input("Press Enter to exit...")



if __name__ == "__main__":
    decode_wav_to_image_live_color(
        "signal_color.wav",
        meta_path="signal_color_meta.json",
        out_path="decoded_live_color.png",
        video_out_path="decoded_live_color.mp4"
    )
