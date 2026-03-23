import argparse
import csv
import os
import sys
from typing import Dict, Iterable, List

import librosa
import numpy as np

SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".m4a", ".aac"}
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CAM_MAJOR = {
    "B": "1B",
    "F#": "2B",
    "C#": "3B",
    "G#": "4B",
    "D#": "5B",
    "A#": "6B",
    "F": "7B",
    "C": "8B",
    "G": "9B",
    "D": "10B",
    "A": "11B",
    "E": "12B",
}

CAM_MINOR = {
    "G#": "1A",
    "D#": "2A",
    "A#": "3A",
    "F": "4A",
    "C": "5A",
    "G": "6A",
    "D": "7A",
    "A": "8A",
    "E": "9A",
    "B": "10A",
    "F#": "11A",
    "C#": "12A",
}

MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float32,
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float32,
)


def list_songs(directory: str) -> List[str]:
    return [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, filename))
        and os.path.splitext(filename)[1].lower() in SUPPORTED_FORMATS
    ]


def estimate_key(chroma: np.ndarray) -> str:
    chroma_mean = chroma.mean(axis=1)
    chroma_norm = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-9)

    best_score = float("-inf")
    best_key = "Unknown"

    for index, key_name in enumerate(KEY_NAMES):
        major_score = np.dot(chroma_norm, np.roll(MAJOR_PROFILE, index))
        if major_score > best_score:
            best_score = major_score
            best_key = f"{key_name} major"

        minor_score = np.dot(chroma_norm, np.roll(MINOR_PROFILE, index))
        if minor_score > best_score:
            best_score = minor_score
            best_key = f"{key_name} minor"

    return best_key


def key_to_camelot(key: str) -> str:
    try:
        note, mode = key.split(" ", 1)
    except ValueError:
        return ""

    if mode == "major":
        return CAM_MAJOR.get(note, "")
    if mode == "minor":
        return CAM_MINOR.get(note, "")
    return ""


def _normalize_feature(value: float, low: float, high: float) -> float:
    """Normalize a feature into the 0-1 range with clipping."""
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _build_energy_excerpt(y: np.ndarray, sr: int) -> np.ndarray:
    """Pick the loudest sections of a track for energy analysis."""
    max_seconds = 60
    if y.size <= sr * max_seconds:
        return y

    # Compute coarse RMS over 2-second windows to find the loudest regions
    win = sr * 2
    hop = sr  # 1-second hop
    n_frames = max((y.size - win) // hop + 1, 1)
    frame_rms = np.array([
        float(np.sqrt(np.mean(y[i * hop: i * hop + win] ** 2)))
        for i in range(n_frames)
    ])

    # Select the top segments by loudness (non-overlapping)
    segment_seconds = 10
    seg_len = sr * segment_seconds
    n_needed = max_seconds // segment_seconds  # 6 segments of 10s = 60s
    used = set()
    segments = []

    for idx in np.argsort(frame_rms)[::-1]:
        start_sample = idx * hop
        # Skip if this region overlaps with one already selected
        if any(abs(start_sample - u) < seg_len for u in used):
            continue
        end_sample = min(start_sample + seg_len, y.size)
        segments.append(y[start_sample:end_sample])
        used.add(start_sample)
        if len(segments) >= n_needed:
            break

    return np.concatenate(segments) if segments else y[:sr * max_seconds]


def calculate_energy_score(y: np.ndarray, sr: int, tempo_bpm: float) -> int:
    """Estimate perceived song energy as a 0-100 score.

    Focuses on the loudest sections and weights peak loudness, bass
    content, and percussiveness heavily so that hard-hitting tracks
    (dubstep drops, EDM bangers) score near 100.
    """
    y_excerpt = _build_energy_excerpt(y, sr)

    # --- RMS loudness ---
    rms = librosa.feature.rms(y=y_excerpt)[0]
    rms_mean = float(np.mean(rms))
    rms_p95 = float(np.percentile(rms, 95))
    top_quarter = max(len(rms) // 4, 1)
    rms_peak_mean = float(np.mean(np.sort(rms)[::-1][:top_quarter]))

    # --- Onset / transient punch ---
    onset_env = librosa.onset.onset_strength(y=y_excerpt, sr=sr)
    onset_p90 = float(np.percentile(onset_env, 90)) if onset_env.size else 0.0

    # --- Spectral brightness ---
    spectral_centroid = librosa.feature.spectral_centroid(y=y_excerpt, sr=sr)[
        0]
    centroid_mean = float(np.mean(spectral_centroid))

    # --- Percussive content via HPSS on downsampled STFT ---
    y_ds = librosa.resample(y_excerpt, orig_sr=sr, target_sr=11025)
    s_mag = np.abs(librosa.stft(y_ds, n_fft=1024, hop_length=512))
    h_mag, p_mag = librosa.decompose.hpss(s_mag, kernel_size=(13, 31))
    harm_e = float(np.sum(h_mag * h_mag))
    perc_e = float(np.sum(p_mag * p_mag))
    percussive_ratio = perc_e / (perc_e + harm_e + 1e-9)

    # --- Bass energy ratio (below ~300 Hz vs total) ---
    S_full = np.abs(librosa.stft(y_excerpt, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bass_mask = freqs <= 300
    bass_energy = float(np.sum(S_full[bass_mask] ** 2))
    total_energy = float(np.sum(S_full ** 2)) + 1e-9
    bass_ratio = bass_energy / total_energy

    # --- Compression / sustained intensity ---
    compression_ratio = rms_mean / (rms_p95 + 1e-9)

    # --- Component scores (calibrated from real track data) ---
    loudness_component = (
        0.35 * _normalize_feature(rms_p95, 0.15, 0.48)
        + 0.35 * _normalize_feature(rms_peak_mean, 0.12, 0.45)
        + 0.30 * _normalize_feature(rms_mean, 0.08, 0.28)
    )
    onset_component = _normalize_feature(onset_p90, 2.0, 8.0)
    tempo_component = _normalize_feature(tempo_bpm, 70.0, 175.0)
    percussive_component = _normalize_feature(percussive_ratio, 0.05, 0.65)
    bass_component = _normalize_feature(bass_ratio, 0.10, 0.50)
    brightness_component = _normalize_feature(centroid_mean, 800.0, 3500.0)
    compression_component = _normalize_feature(compression_ratio, 0.55, 0.85)

    energy_0_to_1 = (
        0.22 * loudness_component
        + 0.22 * bass_component
        + 0.15 * compression_component
        + 0.15 * tempo_component
        + 0.11 * brightness_component
        + 0.10 * percussive_component
        + 0.05 * onset_component
    )

    return int(round(float(np.clip(energy_0_to_1, 0.0, 1.0)) * 100.0))


def analyze_song(path: str) -> Dict[str, str]:
    # Suppress low-level mpg123 C library stderr warnings for malformed MP3 tags
    stderr_fd = sys.stderr.fileno()
    with open(os.devnull, 'w') as devnull:
        old_stderr = os.dup(stderr_fd)
        os.dup2(devnull.fileno(), stderr_fd)
        try:
            y, sr = librosa.load(path, mono=True)
        finally:
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stderr)

    tempo = librosa.feature.tempo(y=y, sr=sr, aggregate=None)
    tempo_value = float(np.median(tempo)) if tempo.size else 0.0
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key = estimate_key(chroma)
    camelot_key = key_to_camelot(key)
    energy = calculate_energy_score(y, int(sr), tempo_value)

    return {
        "filename": os.path.basename(path),
        "tempo_bpm": str(int(round(tempo_value))),
        "camelot_key": camelot_key,
        "key": key,
        "energy": str(energy),
    }


def analyze_folder(directory: str) -> Iterable[Dict[str, str]]:
    songs = list_songs(directory)
    total = len(songs)

    for index, song_path in enumerate(songs, start=1):
        print(f"Analyzing {index}/{total}: {os.path.basename(song_path)}")
        try:
            yield analyze_song(song_path)
        except Exception as exc:
            yield {
                "filename": os.path.basename(song_path),
                "tempo_bpm": "",
                "camelot_key": "",
                "key": "",
                "energy": "",
                "error": str(exc),
            }


def write_csv(rows: Iterable[Dict[str, str]], output_path: str) -> None:
    fieldnames = ["filename", "tempo_bpm",
                  "camelot_key", "key", "energy", "error"]
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    src_dir = os.path.dirname(os.path.abspath(__file__))
    default_directory = os.path.normpath(os.path.join(src_dir, "..", "songs"))
    default_output = os.path.join(src_dir, "song_analysis.csv")

    parser = argparse.ArgumentParser(
        description="Analyze songs and extract tempo plus Camelot key."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=default_directory,
        help=f"Folder containing the audio files (default: {default_directory})",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help=f"CSV file path for results (default: {default_output})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = list(analyze_folder(args.directory))
    write_csv(results, args.output)
    print(f"Wrote {len(results)} rows to {args.output}")
