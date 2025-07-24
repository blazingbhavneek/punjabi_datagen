import csv
import json
import os
import re
from pathlib import Path

import torch
import torchaudio
from cadence import PunctuationModel
from ctc_forced_aligner import (generate_emissions, get_alignments, get_spans,
                                load_alignment_model, load_audio,
                                postprocess_results, preprocess_text)
from huggingface_hub import snapshot_download
from pydub import AudioSegment

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
script_dir = os.path.dirname(os.path.abspath(__file__))
model_root = os.path.join(script_dir, "model")
if not os.path.exists(model_root):
    os.makedirs(model_root)
torchaudio.set_audio_backend('ffmpeg')

def clean_word(word):
    """Remove punctuation from a word for comparison"""
    return re.sub(r'[-.,?!;:\"\'…()।॥؟،٬᱾᱿]', '', word)

def download_cadence():
    repo_id = "ai4bharat/Cadence"
    print(f"Downloading {repo_id} to model folder...")
    repo_path = snapshot_download(
        repo_id=repo_id, cache_dir=model_root, local_files_only=False
    )
    print(f"Repository downloaded to: {repo_path}")

def build_words_info(segments):
    """Create a mapping between words and their segments"""
    words_info = []
    for seg_idx, segment in enumerate(segments):
        words = segment["text"].split()
        for word_idx, word in enumerate(words):
            words_info.append(
                {
                    "word": word,
                    "clean_word": clean_word(word),
                    "seg_idx": seg_idx,
                    "word_idx": word_idx,
                    "start": segment["start"],
                    "duration": segment["duration"],
                }
            )
    return words_info

def align_segment(alignment_model, alignment_tokenizer, audio_waveform, text_fragment, language):
    """Run forced alignment on an audio segment"""
    try:
        emissions, stride = generate_emissions(
            alignment_model, audio_waveform, batch_size=16
        )
        tokens_starred, text_starred = preprocess_text(
            text_fragment,
            romanize=False,
            language=language,
        )
        segments, scores, blank_token = get_alignments(
            emissions, tokens_starred, alignment_tokenizer
        )
        spans = get_spans(tokens_starred, segments, blank_token)
        return postprocess_results(text_starred, spans, stride, scores)
    except Exception as e:
        print(f"Alignment failed: {e}")
        return []

def process_file(input_dir, base_name, punct_model, alignment_model, alignment_tokenizer, output_dir="output"):
    audio_output_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "sentences.csv")
    csv_exists = os.path.exists(csv_path)
    
    with open(csv_path, "a", encoding="utf-8", newline="") as csvfile:
        fieldnames = ["base_name", "sentence_id", "sentence_text", "audio_path"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not csv_exists:
            writer.writeheader()

        json_path = os.path.join(input_dir, "subtitles", f"{base_name}.json")
        audio_path = os.path.join(input_dir, "audio", f"{base_name}.wav")
        
        with open(json_path, "r", encoding="utf-8") as f:
            segments = json.load(f)
        
        words_info = build_words_info(segments)
        original_text = " ".join(segment["text"] for segment in segments)
        punctuated_text = punct_model.punctuate([original_text])[0]
        sentences = []
        current_sentence = ""
        for char in punctuated_text:
            current_sentence += char
            if char == "।":
                sentences.append(current_sentence.strip())
                current_sentence = ""
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        audio_waveform = load_audio(audio_path, dtype=torch.float32, device=device)
        audio_pydub = AudioSegment.from_wav(audio_path)
        duration = len(audio_pydub) / 1000.0
        sample_rate = 16000
        
        word_ptr = 0
        limit=0
        for sent_idx, sentence in enumerate(sentences):
            clean_sentence = "".join(
                c for c in sentence if c not in ".,?!;:-\"'…()।॥؟،٬᱾᱿"
            )
            sentence_words = clean_sentence.split()
            n_words = len(sentence_words)
            
            if word_ptr + n_words > len(words_info):
                print(f"Skipping incomplete sentence: {sentence}")
                continue
                
            mismatch_found = False
            for i in range(n_words):
                expected_word = clean_word(sentence_words[i])
                actual_word = words_info[word_ptr + i]["clean_word"]
                if expected_word != actual_word:
                    print(f"Word mismatch: expected '{expected_word}', got '{actual_word}'")
                    mismatch_found = True
                    break
            
            if mismatch_found:
                next_ptr = word_ptr + 1
                while next_ptr < len(words_info) - n_words:
                    if clean_word(sentence_words[0]) == words_info[next_ptr]["clean_word"]:
                        print(f"Realigning to new start at word {next_ptr}")
                        word_ptr = next_ptr
                        break
                    next_ptr += 1
                else:
                    print(f"Cannot realign sentence: {sentence}")
                    word_ptr += n_words
                    continue

            start_seg_idx = words_info[word_ptr]["seg_idx"]
            end_seg_idx = words_info[word_ptr + n_words - 1]["seg_idx"]
            start_time = segments[start_seg_idx]["start"]
            
            last_seg = segments[end_seg_idx]
            last_seg_words = last_seg["text"].split()
            last_word_idx_in_seg = words_info[word_ptr + n_words - 1]["word_idx"]
            
            if last_word_idx_in_seg == len(last_seg_words) - 1:
                # end_time = last_seg["start"] + last_seg["duration"]
                end_time = segments[end_seg_idx + 1]["start"] if end_seg_idx + 1 < len(segments) else duration
            else:
                partial_text = " ".join(last_seg_words[:last_word_idx_in_seg + 1])
                seg_start = last_seg["start"]
                # seg_end = seg_start + last_seg["duration"]
                seg_end = segments[end_seg_idx + 1]["start"] if end_seg_idx + 1 < len(segments) else duration
                start_sample = int(seg_start * sample_rate)
                end_sample = int(seg_end * sample_rate)
                seg_audio = audio_waveform[start_sample:end_sample]
                word_timestamps = align_segment(
                    alignment_model, 
                    alignment_tokenizer,
                    seg_audio, 
                    partial_text, 
                    "pan"
                )
                end_time = seg_start + word_timestamps[-1]["end"] if word_timestamps else seg_end
            
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            clip = audio_pydub[start_ms:end_ms]
            audio_filename = f"{base_name}_sentence_{sent_idx}.mp3"
            audio_output_path = os.path.join(audio_output_dir, audio_filename)
            clip.export(audio_output_path, format="mp3")
            
            writer.writerow({
                "base_name": base_name,
                "sentence_id": sent_idx,
                "sentence_text": sentence,
                "audio_path": f"audio/{audio_filename}"
            })
            word_ptr += n_words
            limit += 1
            if limit > 1:
                break


def datagen(input_dir, output_dir="data"):
    print("Loading models...")
    punct_model = PunctuationModel(
        model_root,
        max_length=512,
        sliding_window=True,
        verbose=True
    )
    alignment_model, alignment_tokenizer = load_alignment_model(device, dtype=dtype)
    
    with open(os.path.join(input_dir, "download.txt"), "r") as f:
        for base_name in f:
            base_name = base_name.strip()
            if not base_name:
                continue

            print(f"Processing {base_name}...")
            process_file(
                input_dir, 
                base_name, 
                punct_model,
                alignment_model,
                alignment_tokenizer,
                output_dir=output_dir
            )
            print(f"Completed {base_name}")

if __name__ == "__main__":
    input_dir = "downloader"
    output_dir = "data"
    download_cadence()
    datagen(input_dir, output_dir)