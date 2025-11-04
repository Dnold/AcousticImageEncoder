# ImageToAudio

A small research / demo toolkit that encodes images as audio signals and decodes them back. The project contains: an encoder that converts images into audio signals (per-line frequency synthesis), a decoder that converts recorded WAV back into images using STFT, and a small live microphone detector/preview that attempts to decode an image from a live microphone feed.

This repository is intended as an experimental playground — it's not production software, but it is convenient for testing acoustic image transmission (e.g. playing audio from a phone to a laptop microphone).

Contents
- `encoder_pretty.py` — Image -> audio encoder (color support). Adjusts frequency ranges for R/G/B and writes a WAV + metadata JSON.
- `decoder_live.py` — Offline WAV -> image decoder (stable color variant). Performs STFT, accumulates per-line spectra and reconstructs an image. Includes preprocessing and normalization improvements.
- `live_signal_detector.py` — Live microphone listener + visualizer. Shows waveform, band energy and a live decoding preview of the image as audio is received from the microphone. Reads `signal_color_meta.json` by default to match encoder settings.
- `signal_color_meta.json` — example encoder metadata describing width/height, duration per line, sample_rate and frequency band.

Quick start

1. Install dependencies (recommended into a venv):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Encode an image to audio (example):

```powershell
python encoder_pretty.py
# by default this will create signal_color.wav and signal_color_meta.json
```

3. Decode an audio file offline:

```powershell
python decoder_live.py
# or call decode_wav_to_image_live_color('signal_color.wav', meta_path='signal_color_meta.json')
```

4. Live detection and preview (hold your phone's speaker close to the microphone and play the encoded audio):

```powershell
python live_signal_detector.py
# By default the script will attempt to load signal_color_meta.json to match encoder settings
```

Notes & tips
- If your audio interface does not support the encoder's sample rate (e.g. 96000 Hz), run the live preview with a supported rate (44.1 kHz) and still use the encoder meta values for frequency ranges and duration per line.
- The live preview uses FFT-based per-column magnitudes and accumulates them into an image buffer. It works best when encoder and detector settings match (image width/height and duration per line).
- For better results over air: use a quiet environment, point the phone speaker to the microphone, and keep levels moderate (not clipping).

Troubleshooting
- "No audio devices" / sounddevice errors: check OS permissions and make sure a microphone is available.
- If the image looks vertically shifted: ensure the `duration_per_line` in the metadata matches the encoder and the live detector's `--duration` argument.
- If the image is noisy, try increasing the encoder duration per line or improving microphone placement.

License
- This is simple demo code. Use freely for experimentation. No warranty.

Contact
- For improvements or issues, modify the code locally and open PRs if you host this on GitHub.

Example Pictures:
![Preview1](./example/DogOriginal.png)
