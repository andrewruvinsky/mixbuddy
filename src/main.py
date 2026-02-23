import os
import csv
import webbrowser
from flask import Flask, render_template, send_file
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
                    'key': row['key']
                }
    return song_data


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