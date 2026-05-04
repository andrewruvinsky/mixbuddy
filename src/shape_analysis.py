import os
import sys
from typing import Dict

import librosa
import numpy as np

from analyze_songs import calculate_energy_score


def _normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def load_audio_mono(path: str) -> tuple[np.ndarray, int]:
    """Load audio as mono while suppressing noisy MP3 decoder warnings."""
    stderr_fd = sys.stderr.fileno()
    with open(os.devnull, "w") as devnull:
        old_stderr = os.dup(stderr_fd)
        os.dup2(devnull.fileno(), stderr_fd)
        try:
            y, sr = librosa.load(path, mono=True)
        finally:
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stderr)
    return y, int(sr)


def calculate_dynamic_range_score(y: np.ndarray) -> int:
    """Estimate dynamic range as a 0-100 score.

    Higher values indicate larger contrast between quiet and loud passages.
    """
    rms = librosa.feature.rms(y=y)[0]
    if rms.size == 0:
        return 0

    p10 = float(np.percentile(rms, 10))
    p90 = float(np.percentile(rms, 90))
    p95 = float(np.percentile(rms, 95))
    mean_rms = float(np.mean(rms))

    # RMS spread in dB approximates perceived dynamic contrast.
    spread_db = 20.0 * np.log10((p90 + 1e-9) / (p10 + 1e-9))
    crest_like = p95 / (mean_rms + 1e-9)

    score_0_to_1 = (
        0.7 * _normalize(spread_db, 3.0, 24.0)
        + 0.3 * _normalize(crest_like, 1.1, 2.8)
    )
    return int(round(np.clip(score_0_to_1, 0.0, 1.0) * 100.0))


def calculate_pitch_range_score(y: np.ndarray, sr: int) -> int:
    """Estimate pitch range as a 0-100 score from voiced sections."""
    f0 = librosa.yin(
        y,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sr,
    )
    voiced = f0[np.isfinite(f0)]
    voiced = voiced[voiced > 0]

    if voiced.size < 8:
        return 0

    low_hz = float(np.percentile(voiced, 10))
    high_hz = float(np.percentile(voiced, 90))
    range_semitones = 12.0 * np.log2((high_hz + 1e-9) / (low_hz + 1e-9))

    # Typical range: narrow spoken-like melodic content (~4 st) to wide (~36 st).
    score_0_to_1 = _normalize(range_semitones, 4.0, 36.0)
    return int(round(score_0_to_1 * 100.0))


def calculate_vocal_presence_score(y: np.ndarray, sr: int) -> int:
    """Estimate vocal presence as a 0-100 score.

    Measures the proportion of spectral energy in vocal frequency range
    (200-4000 Hz) which is typical for speech and singing.
    """
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # Vocal range: 200-4000 Hz (typical human voice range)
    vocal_mask = (freqs >= 200.0) & (freqs <= 4000.0)
    vocal_energy = float(np.sum(S[vocal_mask] ** 2))
    total_energy = float(np.sum(S ** 2)) + 1e-9
    vocal_ratio = vocal_energy / total_energy

    # Typical vocal ratio ranges: 0.05 (instrumental/bass-heavy) to 0.45 (vocal-prominent)
    score_0_to_1 = _normalize(vocal_ratio, 0.05, 0.45)
    return int(round(np.clip(score_0_to_1, 0.0, 1.0) * 100.0))


def calculate_bass_prominence_score(y: np.ndarray, sr: int) -> int:
    """Estimate bass prominence as a 0-100 score.

    Measures the proportion of spectral energy in bass frequencies (≤300 Hz)
    relative to the full spectrum.
    """
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # Bass region: frequencies up to 300 Hz
    bass_mask = freqs <= 300.0
    bass_energy = float(np.sum(S[bass_mask] ** 2))
    total_energy = float(np.sum(S ** 2)) + 1e-9
    bass_ratio = bass_energy / total_energy

    # Realistic range: 0.15 (minimal bass) to 0.90 (extremely bass-heavy)
    score_0_to_1 = _normalize(bass_ratio, 0.15, 0.90)
    return int(round(np.clip(score_0_to_1, 0.0, 1.0) * 100.0))


def analyze_song_shape(path: str) -> Dict[str, int]:
    """Return radar metrics for one song: energy, dynamic range, pitch range, vocal presence, bass prominence."""
    y, sr = load_audio_mono(path)
    tempo = librosa.feature.tempo(y=y, sr=sr, aggregate=None)
    tempo_value = float(np.median(tempo)) if tempo.size else 0.0

    energy = calculate_energy_score(y, sr, tempo_value)
    dynamic_range = calculate_dynamic_range_score(y)
    pitch_range = calculate_pitch_range_score(y, sr)
    vocal_presence = calculate_vocal_presence_score(y, sr)
    bass_prominence = calculate_bass_prominence_score(y, sr)

    return {
        "energy": int(energy),
        "dynamic_range": int(dynamic_range),
        "pitch_range": int(pitch_range),
        "vocal_presence": int(vocal_presence),
        "bass_prominence": int(bass_prominence),
    }
