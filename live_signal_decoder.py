"""
live_signal_detector.py

Real-time microphone signal detector and visualizer.

Features:
- Listens to microphone input using `sounddevice` InputStream
- Computes chunk RMS and band energy (configurable freq_min/freq_max)
- Displays live waveform, band-energy bar, threshold line, and "Signal Detected" status
- Uses a running background noise estimate to compute an approximate SNR

Usage:
    python live_signal_detector.py

Configuration can be changed in the top-level constants or by editing the file.

Dependencies:
    pip install sounddevice numpy matplotlib scipy

Works on Windows / macOS / Linux where sounddevice can access the microphone.
"""

import argparse
import os
import json
import queue
import sys
import threading
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import sounddevice as sd
from scipy.fft import rfft, rfftfreq


# --- Configuration defaults (change as needed) ---
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BLOCK_DURATION = 0.05  # seconds per audio block (50 ms)
WAVEFORM_DISPLAY_SECONDS = 3.0  # how many seconds of waveform to show
FREQ_MIN = 500.0   # detection band lower bound (Hz)
FREQ_MAX = 22000.0  # detection band upper bound (Hz)
ENERGY_THRESHOLD_FACTOR = 6.0  # detection ratio above noise floor
NOISE_HISTORY_SECONDS = 10.0  # running noise estimate window


def db(x):
    """Convert linear ratio to dB (safeguarded)."""
    x = max(x, 1e-12)
    return 10.0 * np.log10(x)


class LiveSignalDetector:
    def __init__(self, samplerate=DEFAULT_SAMPLE_RATE, block_duration=DEFAULT_BLOCK_DURATION,
                 freq_min=FREQ_MIN, freq_max=FREQ_MAX,
                 display_seconds=WAVEFORM_DISPLAY_SECONDS,
                 energy_threshold_factor=ENERGY_THRESHOLD_FACTOR,
                 noise_history_seconds=NOISE_HISTORY_SECONDS,
                 img_width=480, img_height=319, duration_per_line=0.1):

        self.samplerate = int(samplerate)
        self.block_duration = float(block_duration)
        self.block_size = int(self.samplerate * self.block_duration)
        self.freq_min = float(freq_min)
        self.freq_max = float(freq_max)
        self.display_seconds = float(display_seconds)
        self.display_samples = int(self.display_seconds * self.samplerate)
        self.energy_threshold_factor = float(energy_threshold_factor)
        self.noise_history_max_blocks = max(1, int(noise_history_seconds / block_duration))

        self.q = queue.Queue()
        self.running = False

        # buffers
        self.waveform = np.zeros(self.display_samples, dtype=np.float32)
        self.band_energy_history = []  # recent band energies for noise estimate

        # build freq bins for rfft of block_size
        self.freqs = rfftfreq(self.block_size, d=1.0 / self.samplerate)
        self.band_indices = np.where((self.freqs >= self.freq_min) & (self.freqs <= self.freq_max))[0]

        # --- Image decode structures ---
        self.img_width = int(img_width)
        self.img_height = int(img_height)
        self.duration_per_line = float(duration_per_line)

        # Precompute per-column target frequencies for R,G,B like the encoder
        freq_range = self.freq_max - self.freq_min
        self.r_freqs = np.linspace(self.freq_min, self.freq_min + freq_range / 3.0, self.img_width)
        self.g_freqs = np.linspace(self.freq_min + freq_range / 3.0, self.freq_min + 2.0 * freq_range / 3.0, self.img_width)
        self.b_freqs = np.linspace(self.freq_min + 2.0 * freq_range / 3.0, self.freq_max, self.img_width)

        # Convert these to bin indices in the FFT result
        self.r_bin_indices = np.array([np.argmin(np.abs(self.freqs - f)) for f in self.r_freqs])
        self.g_bin_indices = np.array([np.argmin(np.abs(self.freqs - f)) for f in self.g_freqs])
        self.b_bin_indices = np.array([np.argmin(np.abs(self.freqs - f)) for f in self.b_freqs])

        # Accumulators for live image (sum of mags) and counts per row
        self.img_accumulated = np.zeros((self.img_height, self.img_width, 3), dtype=np.float32)
        self.img_counts = np.zeros(self.img_height, dtype=np.int32)
        self.start_time = None

    def audio_callback(self, indata, frames, time_info, status):
        # called in a background audio thread; push a copy to the queue
        if status:
            # status may contain underrun/overrun info
            print(f"Audio status: {status}", file=sys.stderr)
        self.q.put(indata.copy())

    def start_stream(self):
        self.stream = sd.InputStream(samplerate=self.samplerate, channels=1,
                                     blocksize=self.block_size, callback=self.audio_callback)
        self.stream.start()
        # mark start time for mapping frames to image lines
        self.start_time = time.time()

    def stop_stream(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

    def process_block(self, block):
        # block is shape (frames, channels) - we use mono
        if block.ndim > 1:
            block = block.mean(axis=1)
        block = block.flatten()

        # update waveform buffer (circular-like shift)
        ns = len(block)
        if ns >= self.display_samples:
            self.waveform = block[-self.display_samples:].astype(np.float32)
        else:
            self.waveform = np.roll(self.waveform, -ns)
            self.waveform[-ns:] = block

        # compute RMS
        rms = np.sqrt(np.mean(block.astype(np.float64) ** 2))

        # compute band energy via FFT
        try:
            spec = np.abs(rfft(block * np.hanning(len(block))))
            band_energy = np.sum(spec[self.band_indices] ** 2)
            # also extract per-column color band magnitudes for image decoding
            try:
                r_mags = spec[self.r_bin_indices]
                g_mags = spec[self.g_bin_indices]
                b_mags = spec[self.b_bin_indices]
            except Exception:
                r_mags = np.zeros(self.img_width, dtype=np.float32)
                g_mags = np.zeros(self.img_width, dtype=np.float32)
                b_mags = np.zeros(self.img_width, dtype=np.float32)
        except Exception:
            # fallback to time-domain band energy proxy
            band_energy = np.sum(block ** 2)
            r_mags = np.zeros(self.img_width, dtype=np.float32)
            g_mags = np.zeros(self.img_width, dtype=np.float32)
            b_mags = np.zeros(self.img_width, dtype=np.float32)

        # update history for noise estimate
        self.band_energy_history.append(band_energy)
        if len(self.band_energy_history) > self.noise_history_max_blocks:
            self.band_energy_history.pop(0)

        # noise floor estimate is median of recent energies
        noise_floor = float(np.median(self.band_energy_history)) if len(self.band_energy_history) > 0 else 1e-12

        signal_ratio = (band_energy / (noise_floor + 1e-24)) if noise_floor > 0 else np.inf
        detected = signal_ratio > self.energy_threshold_factor

        # estimated SNR in dB (rough)
        snr_db = db(band_energy / (noise_floor + 1e-24))

        return {
            "rms": float(rms),
            "band_energy": float(band_energy),
            "noise_floor": float(noise_floor),
            "signal_ratio": float(signal_ratio),
            "detected": bool(detected),
            "snr_db": float(snr_db),
            "r_mags": r_mags.tolist(),
            "g_mags": g_mags.tolist(),
            "b_mags": b_mags.tolist(),
        }

    def run_visualizer(self):
        # Prefer a dark seaborn-like style if available; fall back gracefully
        preferred_styles = ['seaborn-dark', 'seaborn', 'dark_background', 'ggplot']
        for s in preferred_styles:
            if s in plt.style.available:
                try:
                    plt.style.use(s)
                except Exception:
                    pass
                break

        # Compute figure size so the image area preserves the encoder aspect ratio
        fig_width = 8.0
        img_aspect = float(self.img_height) / max(1, float(self.img_width))
        image_height_inches = fig_width * img_aspect
        waveform_h = 1.0
        bar_h = 0.6
        fig_height = waveform_h + image_height_inches + bar_h

        fig, (ax_wave, ax_image, ax_bar) = plt.subplots(3, 1, figsize=(fig_width, fig_height),
                                                       gridspec_kw={"height_ratios": [waveform_h, image_height_inches, bar_h]})

        x = np.linspace(-self.display_seconds, 0.0, self.display_samples)
        line, = ax_wave.plot(x, self.waveform, lw=0.6)
        ax_wave.set_ylim(-1.0, 1.0)
        ax_wave.set_xlim(-self.display_seconds, 0.0)
        ax_wave.set_xlabel('Time (s)')
        ax_wave.set_title('Microphone Waveform')

        # Image preview axis
        img_display = np.zeros((self.img_height, self.img_width, 3), dtype=np.float32)
        im = ax_image.imshow(img_display, vmin=0.0, vmax=1.0, interpolation='nearest', aspect='auto')
        ax_image.set_title('Live Decoded Image (approx)')
        ax_image.axis('off')

        bar = ax_bar.bar([0], [0.0], color='tab:green', width=0.4)
        ax_bar.set_xlim(-0.5, 0.5)
        ax_bar.set_ylim(0, 1.0)
        ax_bar.set_xticks([])
        ax_bar.set_title('Band Energy (normalized)')

        status_text = fig.text(0.02, 0.98, '', fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.6))

        last_update = time.time()

        def update(frame):
            nonlocal last_update
            # drain queue
            processed = None
            while True:
                try:
                    block = self.q.get_nowait()
                    processed = self.process_block(block[:, 0] if block.ndim > 1 else block)
                except queue.Empty:
                    break

            if processed is not None:
                # update waveform
                line.set_ydata(self.waveform)

                # --- Update live image accumulators ---
                if self.start_time is not None:
                    # determine which image line this frame maps to
                    elapsed = time.time() - self.start_time
                    line_idx = int(elapsed / self.duration_per_line)
                    if 0 <= line_idx < self.img_height:
                        # convert mags to arrays and normalize per-row
                        r_mags = np.asarray(processed.get('r_mags', np.zeros(self.img_width)), dtype=np.float32)
                        g_mags = np.asarray(processed.get('g_mags', np.zeros(self.img_width)), dtype=np.float32)
                        b_mags = np.asarray(processed.get('b_mags', np.zeros(self.img_width)), dtype=np.float32)
                        # simple percentile-based normalization per channel to reduce dynamic range
                        for arr in (r_mags, g_mags, b_mags):
                            p99 = np.percentile(arr, 99.0) if arr.size > 0 else 1.0
                            if p99 < 1e-12:
                                p99 = 1.0
                            arr /= p99

                        # accumulate into the image buffers
                        self.img_accumulated[line_idx, :, 0] += r_mags
                        self.img_accumulated[line_idx, :, 1] += g_mags
                        self.img_accumulated[line_idx, :, 2] += b_mags
                        self.img_counts[line_idx] += 1

                # compute display image from accumulators (average where possible)
                display_img = np.zeros_like(self.img_accumulated)
                for i in range(self.img_height):
                    if self.img_counts[i] > 0:
                        display_img[i, :, :] = self.img_accumulated[i, :, :] / float(self.img_counts[i])
                # Clip and simple row normalization to improve visibility
                display_img = np.clip(display_img, 0.0, None)
                # normalize each row by its 99th percentile to avoid saturation
                for i in range(self.img_height):
                    row = display_img[i]
                    nz = row[row > 1e-9]
                    if nz.size > 0:
                        p99 = np.percentile(nz, 99.0)
                        if p99 < 1e-9:
                            p99 = 1.0
                        display_img[i] = np.clip(row / p99, 0.0, 1.0)
                im.set_data(display_img)

                # normalized energy for bar
                # use sqrt of band energy to get comparable scale to RMS
                norm = np.sqrt(processed['band_energy'])
                # compute a dynamic normalization using recent noise floor
                nf = max(1e-12, processed['noise_floor'])
                norm_max = np.sqrt(nf) * self.energy_threshold_factor * 2.0
                val = float(min(1.0, norm / (norm_max + 1e-24)))
                bar[0].set_height(val)
                if processed['detected']:
                    bar[0].set_color('tab:red')
                else:
                    bar[0].set_color('tab:green')

                status = 'DETECTED' if processed['detected'] else 'quiet'
                status_text.set_text(f'Status: {status}   RMS: {processed["rms"]:.4f}   SNR: {processed["snr_db"]:.1f} dB')

            # throttle redraw
            if time.time() - last_update > 0.016:
                fig.canvas.draw_idle()
                last_update = time.time()

        ani = FuncAnimation(fig, update, interval=int(self.block_duration * 1000), cache_frame_data=False)

        plt.tight_layout()
        try:
            plt.show()
        except KeyboardInterrupt:
            pass

    def run(self):
        self.running = True
        try:
            self.start_stream()
        except Exception as e:
            print(f"Could not start audio input: {e}")
            return

        try:
            self.run_visualizer()
        finally:
            self.stop_stream()


def main(argv=None):
    p = argparse.ArgumentParser(description='Live microphone signal detector and visualizer')
    p.add_argument('--samplerate', '-r', type=int, default=DEFAULT_SAMPLE_RATE)
    p.add_argument('--block', '-b', type=float, default=DEFAULT_BLOCK_DURATION, help='block duration in seconds')
    p.add_argument('--freq-min', type=float, default=FREQ_MIN)
    p.add_argument('--freq-max', type=float, default=FREQ_MAX)
    p.add_argument('--display', type=float, default=WAVEFORM_DISPLAY_SECONDS, help='seconds of waveform to display')
    p.add_argument('--threshold', type=float, default=ENERGY_THRESHOLD_FACTOR, help='detection ratio above noise floor')
    p.add_argument('--meta', type=str, default='signal_color_meta.json', help='path to metadata JSON to auto-configure defaults')
    p.add_argument('--img-width', type=int, default=480, help='decoded image width in pixels')
    p.add_argument('--img-height', type=int, default=319, help='decoded image height in pixels')
    p.add_argument('--duration', type=float, default=0.1, help='duration per image line (s)')
    args = p.parse_args(argv)

    # If a metadata file is provided and exists, load it to override defaults
    if args.meta and os.path.exists(args.meta):
        try:
            with open(args.meta, 'r') as f:
                meta = json.load(f)
            # override CLI/defaults where available in the meta
            args.samplerate = int(meta.get('sample_rate', args.samplerate))
            args.freq_min = float(meta.get('freq_min', args.freq_min))
            args.freq_max = float(meta.get('freq_max', args.freq_max))
            args.img_width = int(meta.get('width', args.img_width))
            args.img_height = int(meta.get('height', args.img_height))
            args.duration = float(meta.get('duration_per_line', args.duration))
            print(f"Loaded meta from {args.meta}: samplerate={args.samplerate}, freq=({args.freq_min}-{args.freq_max}), img={args.img_width}x{args.img_height}, duration_per_line={args.duration}")
        except Exception as e:
            print(f"Warning: failed to read meta file {args.meta}: {e}")

    det = LiveSignalDetector(samplerate=args.samplerate, block_duration=args.block,
                             freq_min=args.freq_min, freq_max=args.freq_max,
                             display_seconds=args.display,
                             energy_threshold_factor=args.threshold,
                             img_width=args.img_width, img_height=args.img_height, duration_per_line=args.duration)

    print('Starting live signal detector')
    print(f'  samplerate={det.samplerate} block_size={det.block_size} freq_range=({det.freq_min}-{det.freq_max})')
    det.run()


if __name__ == '__main__':
    main()
