import argparse
import contextlib
import csv
import io
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

    return {
        "filename": os.path.basename(path),
        "tempo_bpm": str(int(round(tempo_value))),
        "camelot_key": camelot_key,
        "key": key,
    }


def analyze_folder(directory: str) -> Iterable[Dict[str, str]]:
    for song_path in list_songs(directory):
        try:
            yield analyze_song(song_path)
        except Exception as exc:
            yield {
                "filename": os.path.basename(song_path),
                "tempo_bpm": "",
                "camelot_key": "",
                "key": "",
                "error": str(exc),
            }


def write_csv(rows: Iterable[Dict[str, str]], output_path: str) -> None:
    fieldnames = ["filename", "tempo_bpm", "camelot_key", "key", "error"]
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
