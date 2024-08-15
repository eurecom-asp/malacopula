
"""
Authors: Massimiliano Todisco, Michele Panariello and chatGPT
Email: https://mailhide.io/e/Qk2FFM4a
Date: August 2024
"""

import random
import torchaudio
from torchaudio.transforms import Resample
import threading

audio_load_lock = threading.Lock()

def load_audio(file_path, target_sample_rate=16000):
    with audio_load_lock:
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != target_sample_rate:
            waveform = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)
    return waveform

def create_batches(protocol_a, protocol_b, target_attack, target_speaker, batch_size):
    if target_attack not in protocol_b:
        print(f"No entries found for attack {target_attack}")
        return None

    files_b_spoof = [file for speaker, file, label in protocol_b[target_attack] if speaker == target_speaker]
    if len(files_b_spoof) < batch_size:
        files_b_spoof = random.choices(files_b_spoof, k=batch_size * (batch_size // len(files_b_spoof) + 1))[:batch_size]
    else:
        files_b_spoof = random.sample(files_b_spoof, len(files_b_spoof))

    random.shuffle(files_b_spoof)
    batches = []

    for i in range(0, len(files_b_spoof), batch_size):
        batch_files_b = files_b_spoof[i:i + batch_size]
        if len(batch_files_b) < batch_size:
            batch_files_b += random.choices(files_b_spoof, k=batch_size - len(batch_files_b))

        files_a = random.choices(protocol_a[target_speaker], k=batch_size)
        batches.append((target_speaker, files_a, batch_files_b))

    return batches

def parse_protocol_a(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    protocol = {}
    for line in lines:
        parts = line.strip().split()
        protocol[parts[0]] = parts[1].split(',')
    return protocol


def parse_protocol_b(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    protocol = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 4:
            continue

        speaker = parts[0]
        file = parts[1]
        attack = parts[2]
        label = parts[3]

        if label != 'nontarget':
            if attack in protocol:
                protocol[attack].append((speaker, file, label))
            else:
                protocol[attack] = [(speaker, file, label)]
    return protocol
