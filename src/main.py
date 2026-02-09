import os
import random

def get_random_songs(directory: str, num_songs: int = 10) -> list:
    supported_formats = {'.mp3', '.wav', '.flac', '.m4a', '.aac'}
    
    songs = [
        os.path.join(directory, f) 
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) 
        and os.path.splitext(f)[1].lower() in supported_formats
    ]
    
    if len(songs) < num_songs:
        return songs
    
    return random.sample(songs, num_songs)

if __name__ == "__main__":
    music_dir = "/Users/andrewruvinsky/Desktop/Andrew's DJ Music"
    random_songs = get_random_songs(music_dir)

    for song in random_songs:
        print(os.path.basename(song))