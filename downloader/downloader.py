import yt_dlp
import os
import json
import time
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig

PROXY_USERNAME = "agvzlcps-rotate"
PROXY_PASSWORD = "4z9h4r1lnuaf"

def transcript_to_json(transcript, output_file):
    json_transcript = [
        {
            'start': snippet.start,
            'duration': snippet.duration,
            'text': snippet.text
        }
        for snippet in transcript
    ]
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_transcript, f, ensure_ascii=False, indent=4)

def downloader(channel_url):
    os.makedirs('audio', exist_ok=True)
    os.makedirs('subtitles', exist_ok=True)

    ydl_opts = {
        'extract_flat': 'in_playlist',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
        videos = info.get('entries', [info])

    ytt_api = YouTubeTranscriptApi(
        proxy_config=WebshareProxyConfig(
            proxy_username=PROXY_USERNAME,
            proxy_password=PROXY_PASSWORD
        )
    )

    for video in videos:
        video_id = video['id']
        print(f"\nProcessing video: {video_id}")

        for attempt in range(10):
            try:
                transcript_list = ytt_api.list(video_id)
                punjabi_transcript = transcript_list.find_generated_transcript(['pa'])
                if punjabi_transcript:
                    print(f"Found auto-generated Punjabi subtitles for {video_id}")
                    transcript = punjabi_transcript.fetch()
                    
                    json_file = os.path.join('subtitles', f"{video_id}.json")
                    transcript_to_json(transcript, json_file)
                    print(f"Saved subtitles to {json_file}")

                    audio_ydl_opts = {
                        'format': 'bestaudio/best',
                        'postprocessors': [{
                            'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3',
                            'preferredquality': '320',
                        }],
                        'outtmpl': os.path.join('audio', f'{video_id}.%(ext)s'),
                    }
                    with yt_dlp.YoutubeDL(audio_ydl_opts) as audio_ydl:
                        audio_ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
                    print(f"Downloaded audio to audio/{video_id}.mp3")

                    with open('downloaded.txt', 'a') as f:
                        f.write(f"{video_id}\n")
                    break
                else:
                    print(f"No auto-generated Punjabi subtitles for {video_id}")
                    break
            except Exception as e:
                print(f"Attempt {attempt+1}/10 failed for {video_id}: {e}")
                time.sleep(2)
        else:
            print(f"Failed to process {video_id} after 10 attempts.")

if __name__ == "__main__":
    channel_url = input("Enter the YouTube channel URL: ")
    downloader(channel_url)
    print("Data generation completed.")