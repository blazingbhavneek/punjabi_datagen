import os
from pathlib import Path

from mutagen.mp3 import MP3


def get_total_mp3_duration(folder_path):
    total_duration = 0.0

    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist")
        return 0.0

    for file_path in Path(folder_path).glob("*.mp3"):
        try:
            audio = MP3(file_path)
            total_duration += audio.info.length
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)

    print(f"Total duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
    return total_duration


if __name__ == "__main__":
    folder_path = input("Enter the folder path containing MP3 files: ")
    total_seconds = get_total_mp3_duration(folder_path)
