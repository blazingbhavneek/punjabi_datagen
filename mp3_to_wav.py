import os
from pydub import AudioSegment

def convert_mp3_to_wav_and_delete_original(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.mp3'):
                mp3_path = os.path.join(subdir, file)
                wav_path = os.path.splitext(mp3_path)[0] + '.wav'
                audio = AudioSegment.from_mp3(mp3_path)
                audio.export(wav_path, format='wav')
                os.remove(mp3_path)
                print(f'Converted and deleted: {mp3_path} -> {wav_path}')

if __name__ == '__main__':
    repo_path = '.'
    convert_mp3_to_wav_and_delete_original(repo_path)
