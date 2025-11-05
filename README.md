# AcousticImageEncoder

A small research / demo toolkit that encodes images as audio signals and decodes them back. The project contains: an encoder that converts images into audio signals (per-line frequency synthesis), a decoder that converts recorded WAV back into images using STFT, and a small live microphone detector/preview that attempts to decode an image from a live microphone feed.

This repository is intended as an experimental playground eg. It's not production software, but it is convenient for testing acoustic image transmission (e.g. playing audio from a phone to a laptop microphone).

## Contents
- `signal_encoder.py` — Image -> audio encoder (color support). Adjusts frequency ranges for R/G/B and writes a WAV + metadata JSON.
- `signal_decoder.py` — Offline WAV -> image decoder (stable color variant). Performs STFT, accumulates per-line spectra and reconstructs an image. Includes preprocessing and normalization improvements.
- `live_signal_detector.py` — Live microphone listener + visualizer. Shows waveform, band energy and a live decoding preview of the image as audio is received from the microphone. Reads `signal_color_meta.json` by default to match encoder settings.
- `signal_color_meta.json` — example encoder metadata describing width/height, duration per line, sample_rate and frequency 


## Quick Start

1. Project Structure

The files are organized to separate source code, examples, and documentation for clear navigation:

AcousticImageEncoder/
│
├── .gitignore             # Ignores venv, pycache, and output files
├── LICENSE                # Defines usage rights (e.g., MIT)
├── README.md              # This document
├── requirements.txt       # Python dependencies
│
├── src/                   # Python source code
│   ├── encode.py          # Image -> WAV encoder
│   ├── decode_wav.py      # WAV -> Image offline decoder
│   └── live_detector.py   # Live microphone detector
│
├── ExampleImages/         # Images used in this README
│   ├── DogOriginal.png
│   └── DogDecoded.png
│
└── demo_files/            # Example files to get started
    └── Dog.png

2. Install dependencies (recommended into a venv):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Encode an image to audio (example):

```powershell
python encoder_pretty.py
# by default this will create signal_color.wav and signal_color_meta.json
```

4. Decode an audio file offline:

```powershell
python decoder_live.py
# or call decode_wav_to_image_live_color('signal_color.wav', meta_path='signal_color_meta.json')
```

5. Live detection and preview (hold your phone's speaker close to the microphone and play the encoded audio):

```powershell
python live_signal_detector.py
# By default the script will attempt to load signal_color_meta.json to match encoder settings
```

## Dependencies
Python Libraries (requirements.txt)

- numpy,Fundamental library for numerical computing and array manipulation.
- scipy,"Used for signal processing (STFT, filtering, resampling) and image filtering."
- Pillow (PIL),"Used for loading, resizing, and saving images."
- matplotlib,Powers the live preview visualizer.
- sounddevice,Used for playing audio and recording from the microphone.
- tqdm,Displays a clean progress bar during the encoding process.
- opencv-python (cv2),Required by decode_wav.py to write the .mp4 video preview file.

External Software (Optional: for Video Export)

FFmpeg

The offline decoder (decode_wav.py) can save a high-framerate .mp4 video. To automatically merge the audio from the WAV file into this video, the script requires ffmpeg to be installed and available in your system's PATH.

    If ffmpeg is NOT found, the script will successfully save a silent video file.

    If ffmpeg IS found, it creates a second, merged file (e.g., decoded_video_with_audio.mp4).

    Install ffmpeg from ffmpeg.org or via a package manager (e.g., brew install ffmpeg).

## Example Pictures:
Below is an image before and after being encoded and decoded by this toolkit.
<br>

Unprocessed Image
<br>
![Original Dog](./ExampleImages/DogOriginal.png)

Encoded then Decoded Image
<br>
![Decoded Dog](./ExampleImages/DogDecoded.png)
<br>

**Settings used for this example:**

- Duration per line: 0.18 s

- Sample Rate: 44100 Hz

- Frequency Range: 300 Hz - 18000 Hz

## Notes & tips
- If your audio interface does not support the encoder's sample rate (e.g. 96000 Hz), run the live preview with a supported rate (44.1 kHz) and still use the encoder meta values for frequency ranges and duration per line.
- The live preview uses FFT-based per-column magnitudes and accumulates them into an image buffer. It works best when encoder and detector settings match (image width/height and duration per line).
- For better results over air: use a quiet environment, point the phone speaker to the microphone, and keep levels moderate (not clipping).

## Troubleshooting
- "No audio devices" / sounddevice errors: check OS permissions and make sure a microphone is available.
- If the image looks vertically shifted: ensure the `duration_per_line` in the metadata matches the encoder and the live detector's `--duration` argument.
- If the image is noisy, try increasing the encoder duration per line or improving microphone placement.

## License
- This is simple demo code. Use freely for experimentation. No warranty.

## Contact
- For improvements or issues, modify the code locally and open PRs if you host this on GitHub.


