import csv
import json
import os
import re
import subprocess
import shutil
import tempfile
from pathlib import Path

import torch
import torchaudio
from cadence import PunctuationModel
from textgrid import TextGrid
from huggingface_hub import snapshot_download
from pydub import AudioSegment

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
script_dir = os.path.dirname(os.path.abspath(__file__))
model_root = os.path.join(script_dir, "model")
mfa_model_root = os.path.join(script_dir, "mfa")  # MFA models directory
if not os.path.exists(model_root):
    os.makedirs(model_root)
torchaudio.set_audio_backend('ffmpeg')

def preprocess_punjabi_text(text):
    """
    Preprocess Punjabi text by separating each character, including punctuation,
    with spaces for MFA alignment.
    
    Args:
        text (str): Raw Punjabi transcript text.
        
    Returns:
        str: Preprocessed text with spaces between every grapheme.
    """
    # Normalize the text by stripping leading/trailing whitespace
    text = text.strip()
    
    # Define Punjabi punctuation marks you want to separate, e.g. comma, period, etc.
    punctuation = r'[।.,?!;:()]'
    
    # Separate punctuation by adding spaces around them
    text = re.sub(f'({punctuation})', r' \1 ', text)
    
    # Replace multiple spaces by single space (to avoid double spaces after above step)
    text = re.sub(r'\s+', ' ', text)
    
    # Split text into Unicode graphemes (characters including diacritics)
    # Here, we simply treat each Python character as a grapheme
    # For a more precise grapheme cluster splitting, use the 'regex' package with \X
    chars = list(text)
    
    # Join characters with space
    preprocessed_text = ' '.join(chars)
    
    # Clean multiple spaces again just in case, and strip spaces at ends
    preprocessed_text = re.sub(r'\s+', ' ', preprocessed_text).strip()
    
    return preprocessed_text


# Example:
punjabi_snippet = "ਚੌਥਾ ਪੱਪਾ ਹੈ ਪੰਚਾਇਤ ਪੱਕੀ ਫਸਲ ਜਿਵੇਂ ਗੜਿਆਂ ਨੇ ਝੰਬੀ ਹੋਵੇ ਇਹ ਇੰਝ ਮੇਰਾ ਨੁਕਸਾਨ ਕਰ ਜਾਂਦੀ ਏ ਸ਼ਿਕਵੇ ਸ਼ਿਕਾਇਤਾਂ ਤਾਨੇ ਮੇਹਣੇ ਸਭ ਸੁਣਨਾ ਪੈਂਦਾ ਹੈ ਪੰਚਾਇਤ ਤੋਂ ਇਹ ਹਰ ਰੋਜ਼ ਫੋਕੀ ਧਰਵਾਸ ਹੀ ਦਿੰਦੀ ਏ ਤੇ ਮੇਰੇ ਘਰ ਅਹਿਮਦ ਸ਼ਾਹ ਅਬਦਾਲੀ ਦੀ ਫੌਜ ਵਾਂਗ ਹਮਲੇ ਕਰਦੀ ਰਹਿੰਦੀ ਹੈ।"

print(preprocess_punjabi_text(punjabi_snippet))


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

def parse_textgrid(textgrid_path):
    """Parse MFA TextGrid output to get word timings"""
    try:
        tg = TextGrid.fromFile(textgrid_path)
        word_tier = None
        
        # Find the word tier (usually named "words" or first tier)
        for tier in tg:
            if tier.name.lower() == "words":
                word_tier = tier
                break
        
        if not word_tier and len(tg) > 0:
            word_tier = tg[0]  # Fallback to first tier
        
        if not word_tier:
            return []
        
        word_timings = []
        for interval in word_tier:
            if interval.mark.strip():  # Skip empty intervals
                word_timings.append({
                    'word': interval.mark,
                    'start': interval.minTime,
                    'end': interval.maxTime
                })
        return word_timings
    except Exception as e:
        print(f"Error parsing TextGrid: {str(e)}")
        return []

def run_mfa_align(audio_path, text, language, temp_dir):
    """Run MFA alignment on an audio snippet"""
    try:
        # Create input/output directories
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        mfa_temp = os.path.join(temp_dir, "mfa_temp")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(mfa_temp, exist_ok=True)
        
        # Prepare file paths
        base_name = os.path.basename(audio_path).rsplit('.', 1)[0]
        text_path = os.path.join(input_dir, f"{base_name}.txt")
        
        # Write text to file
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(preprocess_punjabi_text(text))
        
        # Copy audio to input dir
        audio_filename = f"{base_name}.wav"
        audio_dest = os.path.join(input_dir, audio_filename)
        
        if os.path.exists(audio_dest):
            os.remove(audio_dest)

        shutil.copy(audio_path, audio_dest)

        
        # Get MFA model paths
        dict_path = os.path.join(mfa_model_root, f"{language}_Dict.txt")
        acoustic_path = os.path.join(mfa_model_root, f"{language}_Acoustic_Model.zip")
        
        # Verify model files exist
        if not os.path.exists(dict_path):
            print(f"Dictionary file missing: {dict_path}")
            return []
        if not os.path.exists(acoustic_path):
            print(f"Acoustic model missing: {acoustic_path}")
            return []
        
        # Find MFA executable
        mfa_cmd = "mfa"
        if os.name == 'nt':  # Windows
            mfa_cmd = "mfa.bat"
        
        # Try to locate MFA in common paths
        if not any(os.access(os.path.join(path, mfa_cmd), os.X_OK) 
                   for path in os.environ["PATH"].split(os.pathsep)):
            # Check conda environment paths
            conda_prefix = os.environ.get('CONDA_PREFIX')
            if conda_prefix:
                conda_bin = os.path.join(conda_prefix, 'bin', mfa_cmd)
                if os.path.exists(conda_bin):
                    mfa_cmd = conda_bin
        
        # Run MFA command
        cmd = [
            mfa_cmd, "align", "--clean",
            input_dir,
            dict_path,
            acoustic_path,
            output_dir,
            "--temp_directory", mfa_temp
        ]
        
        print(f"Running MFA: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            check=True
        )
        
        # Parse output
        textgrid_path = os.path.join(output_dir, f"{base_name}.TextGrid")
        if os.path.exists(textgrid_path):
            return parse_textgrid(textgrid_path)
        else:
            print("MFA failed to create TextGrid")
            print("MFA stdout:", result.stdout)
            print("MFA stderr:", result.stderr)
            return []
    except subprocess.CalledProcessError as e:
        print(f"MFA alignment failed with return code {e.returncode}")
        print("Error output:", e.stderr)
        return []
    except Exception as e:
        print(f"MFA alignment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def process_file(input_dir, base_name, punct_model, output_dir="output"):
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
        
        audio_waveform, sample_rate = torchaudio.load(audio_path)
        audio_waveform = audio_waveform.to(device)
        audio_pydub = AudioSegment.from_wav(audio_path)
        duration = len(audio_pydub) / 1000.0
        
        word_ptr = 0
        last_sentence_end = 0.0  # Track end time of last sentence
        
        for sent_idx, sentence in enumerate(sentences):
            clean_sentence = "".join(
                c for c in sentence if c not in ".,?!;:-\"'…()।॥؟،٬᱾᱿"
            )
            sentence_words = clean_sentence.split()
            n_words = len(sentence_words)
            
            if word_ptr + n_words > len(words_info):
                print(f"Skipping incomplete sentence: {sentence}")
                continue
                
            # Verify word alignment
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

            # Calculate snippet boundaries (START from last_sentence_end)
            start_seg_idx = words_info[word_ptr]["seg_idx"]
            end_seg_idx = words_info[word_ptr + n_words - 1]["seg_idx"]
            
            # Start from max(last_sentence_end, segment start)
            start_time = max(last_sentence_end, segments[start_seg_idx]["start"])
            
            # End boundary calculation
            if end_seg_idx + 1 < len(segments):
                end_time = segments[end_seg_idx + 1]["start"]
            else:
                end_time = duration
            
            # Create audio snippet
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            snippet_audio = audio_waveform[:, start_sample:end_sample]
            
            # Create temp directory for MFA processing
            temp_dir = os.path.join(script_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
############################################
            # Save snippet to temp file
            snippet_path = os.path.join(temp_dir, "snippet.wav")
            print("snippet_audio.shape:", snippet_audio.shape)
            torchaudio.save(
                snippet_path, 
                snippet_audio.to(torch.float32).cpu(), 
                sample_rate
            )
            
            # Run MFA alignment
            word_timings = run_mfa_align(
                snippet_path, 
                text = sentence, 
                language = "Punjabi", 
                temp_dir = temp_dir
            )
            
            if word_timings:
                # Calculate absolute end time
                last_word_end = word_timings[-1]["end"]
                absolute_end = start_time + last_word_end
                
                # Update last_sentence_end for next iteration
                last_sentence_end = absolute_end
                
                # Trim audio using precise timing
                start_ms = int(start_time * 1000)
                end_ms = int(absolute_end * 1000)
                clip = audio_pydub[start_ms:end_ms]
                
                # Save audio
                audio_filename = f"{base_name}_sentence_{sent_idx}.mp3"
                audio_output_path = os.path.join(audio_output_dir, audio_filename)
                clip.export(audio_output_path, format="mp3")
                
                # Write to CSV
                writer.writerow({
                    "base_name": base_name,
                    "sentence_id": sent_idx,
                    "sentence_text": sentence,
                    "audio_path": f"audio/{audio_filename}"
                })
            else:
                # Fallback to original boundaries
                print(f"Using fallback boundaries for sentence {sent_idx}")
                last_sentence_end = end_time
                
                # Trim and save audio
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
############################################

            word_ptr += n_words

def datagen(input_dir, output_dir="data"):
    print("Loading models...")
    punct_model = PunctuationModel(
        model_root,
        max_length=512,
        sliding_window=True,
        verbose=True
    )
    
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
                output_dir=output_dir
            )
            print(f"Completed {base_name}")

if __name__ == "__main__":
    input_dir = "downloader"
    output_dir = "data"
    download_cadence()
    datagen(input_dir, output_dir)