import os
import csv
import webbrowser
from flask import Flask, render_template, send_file, jsonify, request
from threading import Timer

app = Flask(__name__)

# Get songs directory
src_dir = os.path.dirname(os.path.abspath(__file__))
MUSIC_DIR = os.path.normpath(os.path.join(src_dir, "..", "songs"))
CSV_PATH = os.path.join(src_dir, "song_analysis.csv")


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
                    'mood': row.get('mood', '')
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
    
    # Score all other songs
    recommendations = []
    for song, data in all_songs_data.items():
        if song == current_song:
            continue  # Skip the current song
        
        song_tempo = data.get('tempo')
        song_key = data.get('camelot_key')
        
        if not song_tempo or not song_key:
            continue
        
        try:
            song_tempo = int(song_tempo)
        except (ValueError, TypeError):
            continue
        
        # Calculate tempo distance considering half-time and double-time mixing
        # Normal tempo difference
        normal_diff = abs(current_tempo - song_tempo)
        # Half-time: song played at 2x speed
        halfTime_diff = abs(current_tempo - song_tempo * 2)
        # Double-time: song played at 0.5x speed
        doubleTime_diff = abs(current_tempo - song_tempo / 2)
        
        # Use the best (smallest) difference
        tempo_distance = min(normal_diff, halfTime_diff, doubleTime_diff)
        
        # Skip songs with tempo difference > 12 BPM
        if tempo_distance > 12:
            continue
        
        # Calculate key compatibility
        key_distance = camelot_distance(current_key, song_key)
        
        # Combined score (lower is better)
        # Weight key compatibility more heavily for DJ mixing
        score = (key_distance * 3) + (tempo_distance * 0.2)
        
        recommendations.append({
            'filename': song,
            'tempo': song_tempo,
            'camelot_key': song_key,
            'key': data.get('key', ''),
            'mood': data.get('mood', ''),
            'score': score,
            'tempo_diff': tempo_distance,
            'key_distance': key_distance
        })
    
    # Sort by score (best first) and return top 10 (or fewer if not enough candidates)
    recommendations.sort(key=lambda x: x['score'])
    return recommendations[:10]


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