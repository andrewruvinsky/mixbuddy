import os
import webbrowser
from flask import Flask, render_template_string, send_file
from threading import Timer

app = Flask(__name__)

# Get songs directory
src_dir = os.path.dirname(os.path.abspath(__file__))
MUSIC_DIR = os.path.normpath(os.path.join(src_dir, "..", "songs"))


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


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MixBuddy - Song Library</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            background-color: #2e2e2e;
            color: white;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 28px;
            font-weight: bold;
        }
        
        .song-list {
            background-color: #3a3a3a;
            border-radius: 8px;
            padding: 10px;
            max-height: 70vh;
            overflow-y: auto;
        }
        
        .song-item {
            display: flex;
            align-items: center;
            background-color: #4a4a4a;
            margin: 5px 0;
            padding: 12px 15px;
            border-radius: 5px;
            transition: background-color 0.2s;
        }
        
        .song-item:hover {
            background-color: #555555;
        }
        
        .play-btn {
            background-color: #5a5a5a;
            border: none;
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            font-size: 16px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
            transition: background-color 0.2s;
            flex-shrink: 0;
        }
        
        .play-btn:hover {
            background-color: #6a6a6a;
        }
        
        .play-btn:active {
            background-color: #4a4a4a;
        }
        
        .song-name {
            flex-grow: 1;
            font-size: 14px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .no-songs {
            text-align: center;
            padding: 40px;
            color: #aaa;
            font-size: 16px;
        }
        
        /* Scrollbar styling */
        .song-list::-webkit-scrollbar {
            width: 8px;
        }
        
        .song-list::-webkit-scrollbar-track {
            background: #3a3a3a;
        }
        
        .song-list::-webkit-scrollbar-thumb {
            background: #5a5a5a;
            border-radius: 4px;
        }
        
        .song-list::-webkit-scrollbar-thumb:hover {
            background: #6a6a6a;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>MixBuddy - Song Library</h1>
        <div class="song-list">
            {% if songs %}
                {% for song in songs %}
                <div class="song-item">
                    <button class="play-btn" onclick="togglePlay('{{ song }}', this)">▶</button>
                    <div class="song-name">{{ song }}</div>
                </div>
                {% endfor %}
            {% else %}
                <div class="no-songs">No songs found in songs folder</div>
            {% endif %}
        </div>
    </div>
    
    <audio id="audioPlayer" style="display: none;"></audio>
    
    <script>
        const audioPlayer = document.getElementById('audioPlayer');
        let currentButton = null;
        let currentSong = null;
        
        function togglePlay(songName, button) {
            if (currentSong === songName && !audioPlayer.paused) {
                // Pause current song
                audioPlayer.pause();
                button.textContent = '▶';
            } else {
                // Stop previous song if any
                if (currentButton && currentButton !== button) {
                    currentButton.textContent = '▶';
                }
                audioPlayer.pause();
                
                // Play new song
                audioPlayer.src = '/play/' + encodeURIComponent(songName);
                audioPlayer.play();
                button.textContent = '⏸';
                currentButton = button;
                currentSong = songName;
            }
        }
        
        // Reset button when audio ends
        audioPlayer.addEventListener('ended', function() {
            if (currentButton) {
                currentButton.textContent = '▶';
                currentButton = null;
                currentSong = null;
            }
        });
        
        // Handle errors
        audioPlayer.addEventListener('error', function() {
            console.error('Error playing audio');
            if (currentButton) {
                currentButton.textContent = '▶';
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Main page showing song list"""
    songs = get_songs(MUSIC_DIR)
    return render_template_string(HTML_TEMPLATE, songs=songs)


@app.route('/play/<path:filename>')
def play_song(filename):
    """Serve audio file for playback"""
    song_path = os.path.join(MUSIC_DIR, filename)
    if os.path.exists(song_path):
        return send_file(song_path)
    return "File not found", 404


def open_browser():
    """Open browser after a short delay"""
    webbrowser.open('http://127.0.0.1:5000')


if __name__ == "__main__":
    # Open browser automatically after starting server
    Timer(1, open_browser).start()
    
    print("Starting MixBuddy...")
    print("Opening browser at http://127.0.0.1:5000")
    print("Press Ctrl+C to quit")
    
    app.run(debug=False, port=5000)