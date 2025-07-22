import json
import os
import csv
from pydub import AudioSegment

script_dir = os.path.dirname(os.path.abspath(__file__))
punjabi_data_folder = os.path.join(script_dir, "..", "downloader")
processed_clips_folder = os.path.join(script_dir, "..", "data", "clips")
dataset_csv = os.path.join(script_dir, "..", "data", "dataset.csv")
video_list_file = os.path.join(punjabi_data_folder, "downloaded.txt")

os.makedirs(processed_clips_folder, exist_ok=True)

def trim_silence(audio):
    return audio.strip_silence(silence_len=500, silence_thresh=-40)


with open(video_list_file, 'r') as f:
    video_ids = [line.strip() for line in f]


clip_id = 1


with open(dataset_csv, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['id', 'text', 'source_video', 'duration', 'audio_file']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for video_id in video_ids:

        transcript_file = os.path.join(punjabi_data_folder, "subtitles", f"{video_id}.json")
        with open(transcript_file, 'r', encoding='utf-8') as f:
            snippets = json.load(f)

        audio_file = os.path.join(punjabi_data_folder, "audio", f"{video_id}.mp3")
        audio = AudioSegment.from_mp3(audio_file)

        for i in range(len(snippets) - 1):
            snippets[i]['end'] = snippets[i + 1]['start']
        snippets[-1]['end'] = '[EOA]'

        group_list = []
        current_group = []
        current_duration = 0
        for snippet in snippets:
            duration = snippet['duration']
            current_group.append(snippet)
            current_duration += duration
            if current_duration >= 10:
                group_list.append(current_group)
                current_group = []
                current_duration = 0
        if current_group:
            group_list.append(current_group)

        for group in group_list:
            start = group[0]['start']
            end = group[-1]['end']
            if end == '[EOA]':
                end = group[-1]['start'] + group[-1]['duration']
            text = " ".join([snippet['text'] for snippet in group])

            audio_segment = audio[start * 1000 : end * 1000]
            trimmed_audio = trim_silence(audio_segment)

            clip_file = os.path.join(processed_clips_folder, f"clip_{clip_id}.mp3")
            trimmed_audio.export(clip_file, format="mp3")

            duration = trimmed_audio.duration_seconds

            writer.writerow({
                'id': clip_id,
                'text': text,
                'source_video': video_id,
                'duration': duration,
                'audio_file': clip_file
            })

            clip_id += 1