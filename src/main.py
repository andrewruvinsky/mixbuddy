import os
import csv
import random
import webbrowser
from flask import Flask, render_template, send_file, jsonify, request
from threading import Timer

from shape_analysis import analyze_song_shape

app = Flask(__name__)

# Get songs directory
src_dir = os.path.dirname(os.path.abspath(__file__))
MUSIC_DIR = os.path.normpath(os.path.join(src_dir, "..", "songs"))
CSV_PATH = os.path.join(src_dir, "song_analysis.csv")
shape_cache = {}


def load_song_data():
    """Load song analysis data from CSV"""
    song_data = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                song_data[row['filename']] = {
                    'tempo': row['tempo_bpm'],
                    'camelot_key': row['camelot_key'],
                    'key': row['key'],
                    'energy': row.get('energy', '')
                }
    return song_data


def camelot_distance(key1, key2):
    """Calculate compatibility between two Camelot keys.
    Returns a distance score where 0 is perfect match, lower is better."""
    if not key1 or not key2:
        return 999  # Invalid keys get worst score

    if key1 == key2:
        return 0  # Perfect match

    try:
        # Parse Camelot notation (e.g., "8A" -> number=8, letter='A')
        num1 = int(key1[:-1])
        letter1 = key1[-1]
        num2 = int(key2[:-1])
        letter2 = key2[-1]

        # Same number, different letter (e.g., 8A <-> 8B) - very compatible
        if num1 == num2 and letter1 != letter2:
            return 1

        # Adjacent numbers, same letter (e.g., 8A <-> 7A or 8A <-> 9A)
        if letter1 == letter2:
            diff = abs(num1 - num2)
            # Handle wraparound (12 to 1)
            diff = min(diff, 12 - diff)
            if diff == 1:
                return 2
            elif diff == 2:
                return 4
            else:
                return 6 + diff

        # Different number and letter - less compatible
        return 8
    except (ValueError, IndexError):
        return 999


def _mixability_from_score(score, max_score=35.0):
    mixability = max(0.0, 1.0 - score / max_score) * 100
    return int(round(mixability))


def _build_recommendation_entry(song, data, current_tempo, current_key, current_energy):
    song_tempo = data.get('tempo')
    song_key = data.get('camelot_key')

    if not song_tempo or not song_key:
        return None

    try:
        song_tempo = int(song_tempo)
    except (ValueError, TypeError):
        return None

    try:
        song_energy = int(data.get('energy', 0))
    except (ValueError, TypeError):
        song_energy = 0

    # Calculate tempo distance considering half-time and double-time mixing
    normal_diff = abs(current_tempo - song_tempo)
    half_time_diff = abs(current_tempo - song_tempo * 2)
    double_time_diff = abs(current_tempo - song_tempo / 2)
    tempo_distance = min(normal_diff, half_time_diff, double_time_diff)

    # Skip songs with tempo difference > 18 BPM
    if tempo_distance > 18:
        return None

    key_distance = camelot_distance(current_key, song_key)
    energy_distance = abs(current_energy - song_energy)

    # Weight key compatibility most heavily, then tempo proximity, then energy
    score = (key_distance * 3) + (tempo_distance * 0.9) + \
        (energy_distance * 0.1)

    return {
        'filename': song,
        'tempo': song_tempo,
        'camelot_key': song_key,
        'key': data.get('key', ''),
        'energy': data.get('energy', ''),
        'score': score,
        'tempo_diff': tempo_distance,
        'key_distance': key_distance,
        'energy_diff': energy_distance,
        'mixability': _mixability_from_score(score),
    }


def get_recommendations(current_song, all_songs_data):
    """Get top 10 song recommendations based on tempo and key similarity."""
    if current_song not in all_songs_data:
        return []

    current_data = all_songs_data[current_song]
    current_tempo = current_data.get('tempo')
    current_key = current_data.get('camelot_key')

    if not current_tempo or not current_key:
        return []

    try:
        current_tempo = int(current_tempo)
    except (ValueError, TypeError):
        return []

    try:
        current_energy = int(current_data.get('energy', 0))
    except (ValueError, TypeError):
        current_energy = 0

    # Score all other songs
    recommendations = []
    for song, data in all_songs_data.items():
        if song == current_song:
            continue  # Skip the current song

        entry = _build_recommendation_entry(
            song, data, current_tempo, current_key, current_energy)
        if entry is not None:
            recommendations.append(entry)

    # Sort by score (best first) and return top 10 (or fewer if not enough candidates)
    recommendations.sort(key=lambda x: x['score'])
    top = recommendations[:10]

    return top


def _to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _tempo_distance(current_tempo, song_tempo):
    normal_diff = abs(current_tempo - song_tempo)
    half_time_diff = abs(current_tempo - song_tempo * 2)
    double_time_diff = abs(current_tempo - song_tempo / 2)
    return min(normal_diff, half_time_diff, double_time_diff)


def _get_song_shape_metrics(song_name):
    if song_name in shape_cache:
        return shape_cache[song_name]

    song_path = os.path.join(MUSIC_DIR, song_name)
    if not os.path.exists(song_path) or not os.path.isfile(song_path):
        return None

    try:
        metrics = analyze_song_shape(song_path)
        shape_cache[song_name] = metrics
        return metrics
    except Exception:
        return None


def get_shuffled_recommendations(current_song, all_songs_data):
    """Shuffle only recommendations that satisfy strict compatibility constraints."""
    base_recommendations = get_recommendations(current_song, all_songs_data)
    if current_song not in all_songs_data:
        return []

    current_data = all_songs_data[current_song]
    current_key = current_data.get('camelot_key', '')
    current_tempo = _to_int(current_data.get('tempo'))
    current_energy = _to_int(current_data.get('energy'))
    current_shape = _get_song_shape_metrics(current_song)

    if not current_key or current_tempo <= 0 or current_shape is None:
        return []

    base_names = {rec.get('filename') for rec in base_recommendations}
    base_names.add(current_song)

    # Build shuffle pool from all songs, excluding base recommendation list.
    scored_candidates = []
    for song, data in all_songs_data.items():
        if song in base_names:
            continue

        entry = _build_recommendation_entry(
            song, data, current_tempo, current_key, current_energy)
        if entry is None:
            continue

        # Keep same-key compatibility for shuffled alternates.
        if entry.get('camelot_key', '') != current_key:
            continue

        # Keep energy profile compatibility tight.
        if entry.get('energy_diff', 999) > 15:
            continue

        candidate_shape = _get_song_shape_metrics(song)
        if candidate_shape is None:
            continue

        vocal_distance = abs(
            _to_int(candidate_shape.get('vocal_presence')) -
            _to_int(current_shape.get('vocal_presence'))
        )
        bass_distance = abs(
            _to_int(candidate_shape.get('bass_prominence')) -
            _to_int(current_shape.get('bass_prominence'))
        )

        scored_candidates.append({
            'entry': entry,
            'vocal_distance': vocal_distance,
            'bass_distance': bass_distance,
        })

    # Try strict profile first, then widen profile threshold to reach at least 5.
    compatible_entries = []
    for profile_threshold in (15, 20, 25, 30):
        compatible_entries = [
            item['entry']
            for item in scored_candidates
            if item['vocal_distance'] <= profile_threshold
            and item['bass_distance'] <= profile_threshold
        ]
        if len(compatible_entries) >= 5:
            break

    # Backfill with closest profile matches to hit at least 5 when possible.
    if len(compatible_entries) < 5:
        in_set = {rec.get('filename') for rec in compatible_entries}
        backfill = sorted(
            scored_candidates,
            key=lambda item: (
                item['vocal_distance'] + item['bass_distance'],
                item['entry'].get('score', 999),
            ),
        )
        for item in backfill:
            entry = item['entry']
            filename = entry.get('filename')
            if filename in in_set:
                continue
            compatible_entries.append(entry)
            in_set.add(filename)
            if len(compatible_entries) >= 5:
                break

    random.shuffle(compatible_entries)
    return compatible_entries[:10]


def get_songs(directory: str) -> list:
    """Get list of supported audio files from directory"""
    supported_formats = {'.mp3', '.wav', '.flac', '.m4a', '.aac'}

    if not os.path.exists(directory):
        return []

    songs = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        and os.path.splitext(f)[1].lower() in supported_formats
    ]
    return sorted(songs, key=lambda x: x.lower())


@app.route('/')
def index():
    """Main page showing song list"""
    songs = get_songs(MUSIC_DIR)
    song_data = load_song_data()
    return render_template('index.html', songs=songs, song_data=song_data)


@app.route('/play/<path:filename>')
def play_song(filename):
    """Serve audio file for playback"""
    song_path = os.path.join(MUSIC_DIR, filename)
    if os.path.exists(song_path):
        return send_file(song_path)
    return "File not found", 404


@app.route('/api/recommendations')
def api_recommendations():
    """Get song recommendations based on current song"""
    current_song = request.args.get('song')
    if not current_song:
        return jsonify({'error': 'No song specified'}), 400

    song_data = load_song_data()
    recommendations = get_recommendations(current_song, song_data)

    return jsonify({'recommendations': recommendations})


@app.route('/api/recommendations/shuffle')
def api_shuffle_recommendations():
    """Get shuffled recommendations within strict compatibility constraints."""
    current_song = request.args.get('song')
    if not current_song:
        return jsonify({'error': 'No song specified'}), 400

    song_data = load_song_data()
    recommendations = get_shuffled_recommendations(current_song, song_data)

    return jsonify({'recommendations': recommendations})


@app.route('/api/song-shape')
def api_song_shape():
    """Get energy/dynamic range/pitch range for one song."""
    song = request.args.get('song')
    if not song:
        return jsonify({'error': 'No song specified'}), 400

    song_path = os.path.join(MUSIC_DIR, song)
    if not os.path.exists(song_path) or not os.path.isfile(song_path):
        return jsonify({'error': 'Song not found'}), 404

    if song in shape_cache:
        return jsonify({'song': song, 'metrics': shape_cache[song]})

    try:
        metrics = analyze_song_shape(song_path)
        shape_cache[song] = metrics
        return jsonify({'song': song, 'metrics': metrics})
    except Exception as exc:
        return jsonify({'error': f'Unable to analyze song: {exc}'}), 500


def open_browser():
    """Open browser after a short delay"""
    webbrowser.open('http://127.0.0.1:8080')


if __name__ == "__main__":
    # Open browser automatically after starting server
    Timer(1, open_browser).start()

    print("Starting MixBuddy...")
    print("Opening browser at http://127.0.0.1:8080")
    print("Press Ctrl+C to quit")

    app.run(debug=False, port=8080)
