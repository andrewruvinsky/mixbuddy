import os
import random

def get_songs(directory: str) -> list:
    supported_formats = {'.mp3', '.wav', '.flac', '.m4a', '.aac'}

    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        and os.path.splitext(f)[1].lower() in supported_formats
    ]


# TODO: Remove this function once established a more solid song recommendation system
def get_random_songs(songs: list, num_songs: int = 10) -> list:
    if len(songs) <= num_songs:
        return songs

    return random.sample(songs, num_songs)

if __name__ == "__main__":
    music_dir = "/Users/andrewruvinsky/Desktop/Andrew's DJ Music"
    songs = get_songs(music_dir)
    # Case insensitive search 
    query = input("Enter a song name: ").strip()
    query_lower = query.lower()

    # Proceed if input matches song in library
    if any(os.path.splitext(os.path.basename(song))[0].lower() == query_lower for song in songs):
        print(f"Songs based on {query}:")
        random_songs = get_random_songs(songs, num_songs=10)
        for song in random_songs:
            print(os.path.basename(song))
    else:
        print("Song not found in library.")