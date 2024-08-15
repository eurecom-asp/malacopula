
"""
Authors: Massimiliano Todisco, Michele Panariello and chatGPT
Email: https://mailhide.io/e/Qk2FFM4a
Date: August 2024
"""

import os
import torch
import random
import torchaudio
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from models import load_models, Malacopula
from data_loader import parse_protocol_a, parse_protocol_b, create_batches, load_audio
from training import train
from validation import validate
from precompute_embeddings import precompute_embeddings
from validation import compute_embeddings, calculate_eer
from torch.nn.functional import cosine_similarity
import numpy as np
import torch.multiprocessing as mp
import concurrent.futures
import time
import configparser

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the conf.txt file
config.read('conf.ini')

# Access the values
TARGET_SAMPLE_RATE = int(config.get('DEFAULT', 'TARGET_SAMPLE_RATE'))
LEARNING_RATE = float(config.get('DEFAULT', 'LEARNING_RATE'))
NUM_EPOCHS = int(config.get('DEFAULT', 'NUM_EPOCHS'))
BATCH_SIZE = int(config.get('DEFAULT', 'BATCH_SIZE'))
VALIDATION_STEP = int(config.get('DEFAULT', 'VALIDATION_STEP'))
NUM_LAYERS = int(config.get('DEFAULT', 'NUM_LAYERS'))
KERNEL_SIZE = int(config.get('DEFAULT', 'KERNEL_SIZE'))
SPEAKERS_PER_GPU = int(config.get('DEFAULT', 'SPEAKERS_PER_GPU'))
AUDIO_FOLDER = config.get('DEFAULT', 'AUDIO_FOLDER')
OUTPUT_BASE_PATH = config.get('DEFAULT', 'OUTPUT_BASE_PATH')
PROTOCOL_A = config.get('DEFAULT', 'PROTOCOL_A')
PROTOCOL_B = config.get('DEFAULT', 'PROTOCOL_B')
TARGET_ATTACK = config.get('DEFAULT', 'TARGET_ATTACK')
TARGET_SPEAKER = config.get('DEFAULT', 'TARGET_SPEAKER')
f_A = config.get('DEFAULT', 'f_A')


def process_speaker(target_attack, target_speaker, protocol_a, protocol_b, device, progress_counter, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=0):
    # Extract the last part of AUDIO_FOLDER
    audio_folder_name = os.path.basename(os.path.normpath(AUDIO_FOLDER))

    # Create directories for saving models and logs
    save_dir = os.path.join(OUTPUT_BASE_PATH, f"{target_attack}_{NUM_LAYERS}_{KERNEL_SIZE}")
    model_save_dir = os.path.join(save_dir, f"{target_speaker}_{NUM_LAYERS}_{KERNEL_SIZE}")
    best_model_path = os.path.join(model_save_dir, f'best_model_{target_attack}_{target_speaker}.pth')

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    writer = SummaryWriter(log_dir=model_save_dir)  # Initialize the TensorBoard writer

    with torch.no_grad():
        model_ecapa, model_campp, feature_extractor_campp = load_models(device)

    malacopula = Malacopula(num_layers=NUM_LAYERS, kernel_size=KERNEL_SIZE).to(device)
    optimizer = optim.Adam(malacopula.parameters(), lr=LEARNING_RATE)

    # Precompute the enrollment and bonafide-target embeddings
    with torch.no_grad():
        enroll_ecapa_embedding, enroll_campp_embedding, bonafide_ecapa_target_scores, bonafide_campp_target_scores, enroll_dict_ecapa, enroll_dict_campp = precompute_embeddings(
            protocol_a, protocol_b, target_speaker, target_attack, AUDIO_FOLDER, device, writer, model_ecapa,
            model_campp, feature_extractor_campp, verbose)

    enroll_dict = []
    if f_A == 'ecapa':
        enroll_dict = enroll_dict_ecapa
    elif f_A == 'campp':
        enroll_dict = enroll_dict_campp

    best_w_distance = float('inf')  # Initialize with infinity

    for epoch in range(num_epochs):
        batches = create_batches(protocol_a, protocol_b, target_attack, target_speaker, batch_size=batch_size)

        if not batches:
            print("No suitable files found.")
            writer.close()
            progress_counter.value += 1
            return

        total_batches = len(batches)
        for batch_idx, batch in enumerate(batches):
            speaker, files_a, files_b = batch
            batch_audio_b = [load_audio(AUDIO_FOLDER + file_b + '.flac') for file_b in files_b]

            batch_embeddings_a = []

            for file_name in files_a:
                if file_name in enroll_dict:
                    batch_embeddings_a.append(enroll_dict[file_name])
                else:
                    print(f"Embedding for {file_name} not found in enroll_dict")

            loss = train(malacopula, batch_embeddings_a, batch_audio_b, optimizer, writer, epoch, batch_idx, total_batches, model_ecapa, feature_extractor_campp, model_campp, device, f_A)

            if verbose:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss:.6f}")

        if (epoch + 1) % VALIDATION_STEP == 0:
            with torch.no_grad():
                ecapa_eer, campp_eer, w_distance_ecapa, w_distance_campp = validate(
                malacopula, protocol_b, target_attack, target_speaker,
                enroll_ecapa_embedding, enroll_campp_embedding,
                bonafide_ecapa_target_scores,bonafide_campp_target_scores,
                AUDIO_FOLDER, device, model_ecapa, model_campp, feature_extractor_campp)

            writer.add_scalar('ECAPA_EER/validation', ecapa_eer, epoch + 1)
            writer.add_scalar('CAMpp_EER/validation', campp_eer, epoch + 1)
            writer.add_scalar('ECAPA_WassD/validation', w_distance_ecapa, epoch + 1)
            writer.add_scalar('CAMpp_WassD/validation', w_distance_campp, epoch + 1)

            if verbose:
                print(f"Epoch {epoch + 1}, VALIDATION: ECAPA_EER: {ecapa_eer:.4f}, CAMpp_EER: {campp_eer:.4f}, ECAPA_WassD: {w_distance_ecapa:.4f}, CAMpp_WassD: {w_distance_campp:.4f}")

            if f_A == 'ecapa':
                # Save the best model based on minimum w_distance_campp
                if w_distance_campp < best_w_distance:
                    best_w_distance = w_distance_campp
                    torch.save(malacopula.state_dict(), best_model_path)
                    if verbose:
                        print(f"Saved best model with w_distance_campp: {w_distance_campp:.4f}")
            elif f_A == 'campp':
                if w_distance_ecapa < best_w_distance:
                    best_w_distance = w_distance_ecapa
                    torch.save(malacopula.state_dict(), best_model_path)
                    if verbose:
                        print(f"Saved best model with w_distance_ecapa: {w_distance_ecapa:.4f}")

    # Load the best model for final processing
    malacopula.load_state_dict(torch.load(best_model_path))
    if verbose:
        print(f"Saving Malacopula processed spoof speech")

    selected_files = [file for speaker, file, label in protocol_b[target_attack] if speaker == target_speaker][:2]

    for i, file in enumerate(selected_files):
        audio = load_audio(AUDIO_FOLDER + file + '.flac')
        writer.add_audio(f'Spoof_{i + 1}', audio, sample_rate=TARGET_SAMPLE_RATE)
        with torch.no_grad():
            processed_audio = malacopula(audio.to(device))
        processed_audio = processed_audio.cpu()
        writer.add_audio(f'Malacopula_spoof_{i + 1}', processed_audio, sample_rate=TARGET_SAMPLE_RATE)

    # Add a random file from protocol_a for the target speaker
    random_file = random.choice(protocol_a[target_speaker])
    audio = load_audio(AUDIO_FOLDER + random_file + '.flac')
    writer.add_audio(f'Enrolment_bonafide', audio, sample_rate=TARGET_SAMPLE_RATE)

    # Save filter coefficients in the same directory as the best model
    malacopula.save_filter_coefficients(model_save_dir)

    # Save all the processed files in the specified format
    processed_files_dir = os.path.join(save_dir, f'{audio_folder_name}_{NUM_LAYERS}_{KERNEL_SIZE}')
    if not os.path.exists(processed_files_dir):
        os.makedirs(processed_files_dir)

    # Process and save each file in the target speaker's list
    for i, (speaker, file, label) in enumerate(protocol_b[target_attack]):
        if speaker == target_speaker:
            audio = load_audio(AUDIO_FOLDER + file + '.flac')
            with torch.no_grad():
                processed_audio = malacopula(audio.to(device))
            processed_audio = processed_audio.cpu()

            # Save the processed audio in FLAC format
            save_path = os.path.join(processed_files_dir, f'{file}.flac')
            torchaudio.save(save_path, processed_audio, TARGET_SAMPLE_RATE, bits_per_sample=16)

    writer.close()
    progress_counter.value += 1


def evaluate_speaker(speaker, protocol_a, protocol_b, target_attack, audio_folder, processed_audio_folder, device, model_ecapa, model_campp, feature_extractor_campp):
    enroll_files = protocol_a[speaker][:3]
    enroll_ecapa_embedding, enroll_campp_embedding = compute_embeddings(enroll_files, model_ecapa, model_campp,
                                                                        feature_extractor_campp, audio_folder,
                                                                        device)
    enroll_ecapa_embedding = torch.mean(enroll_ecapa_embedding, dim=0).unsqueeze(0)
    enroll_campp_embedding = torch.mean(enroll_campp_embedding, dim=0).unsqueeze(0)

    bonafide_target_files = [f for s, f, l in protocol_b['bonafide'] if s == speaker]
    bonafide_target_ecapa_embeddings, bonafide_target_campp_embeddings = compute_embeddings(
        bonafide_target_files, model_ecapa, model_campp, feature_extractor_campp, audio_folder, device)

    spoof_files = [f for s, f, l in protocol_b[target_attack] if s == speaker]

    # ECAPA embeddings
    spoof_ecapa_embeddings = []
    spoof_processed_ecapa_embeddings = []
    for file in spoof_files:
        processed_file_path = os.path.join(processed_audio_folder, file + '.flac')
        if os.path.exists(processed_file_path):
            try:
                audio = load_audio(os.path.join(audio_folder, file + '.flac')).to(device)
                embedding = model_ecapa.encode_batch(audio).detach()
                spoof_ecapa_embeddings.append(embedding)

                audio_processed = load_audio(processed_file_path).to(device)
                embedding_processed = model_ecapa.encode_batch(audio_processed).detach()
                spoof_processed_ecapa_embeddings.append(embedding_processed)
            except Exception as e:
                print(f"Error loading audio file: {e}")

    # ECAPA scores
    ecapa_scores = np.array([])
    ecapa_processed_scores = np.array([])
    labels_ecapa = np.array([])
    if spoof_ecapa_embeddings:
        spoof_ecapa_embedding = torch.cat(spoof_ecapa_embeddings, dim=0).squeeze(1)
        spoof_processed_ecapa_embedding = torch.cat(spoof_processed_ecapa_embeddings, dim=0).squeeze(1)
        bonafide_ecapa_target_scores = 1 - cosine_similarity(enroll_ecapa_embedding.cpu().detach(),
                                                             bonafide_target_ecapa_embeddings.cpu().detach())
        spoof_ecapa_scores = 1 - cosine_similarity(enroll_ecapa_embedding.cpu().detach(), spoof_ecapa_embedding.cpu().detach())
        ecapa_scores = np.concatenate([bonafide_ecapa_target_scores, spoof_ecapa_scores])
        spoof_processed_ecapa_scores = 1 - cosine_similarity(enroll_ecapa_embedding.cpu().detach(),
                                                             spoof_processed_ecapa_embedding.cpu().detach())
        ecapa_processed_scores = np.concatenate([bonafide_ecapa_target_scores, spoof_processed_ecapa_scores])

        labels_ecapa = np.concatenate(
            [np.zeros(len(bonafide_ecapa_target_scores)), np.ones(len(spoof_ecapa_scores))])

    # CAMPP embeddings
    spoof_campp_embeddings = []
    spoof_processed_campp_embeddings = []
    for file in spoof_files:
        processed_file_path = os.path.join(processed_audio_folder, file + '.flac')
        if os.path.exists(processed_file_path):
            try:
                audio = load_audio(os.path.join(audio_folder, file + '.flac')).to(device)
                mel = feature_extractor_campp(audio).unsqueeze(0).detach()
                embedding = model_campp(mel).unsqueeze(0).detach()
                spoof_campp_embeddings.append(embedding)

                audio_processed = load_audio(processed_file_path).to(device)
                mel = feature_extractor_campp(audio_processed).unsqueeze(0).detach()
                embedding_processed = model_campp(mel).unsqueeze(0).detach()
                spoof_processed_campp_embeddings.append(embedding_processed)
            except Exception as e:
                print(f"Error loading audio file: {e}")

    # CAMPP scores
    campp_scores = np.array([])
    campp_processed_scores = np.array([])
    labels_campp = np.array([])
    if spoof_campp_embeddings:
        spoof_campp_embedding = torch.cat(spoof_campp_embeddings, dim=0).squeeze(1)
        spoof_processed_campp_embedding = torch.cat(spoof_processed_campp_embeddings, dim=0).squeeze(1)
        bonafide_campp_target_scores = 1 - cosine_similarity(enroll_campp_embedding.cpu().detach(),
                                                             bonafide_target_campp_embeddings.cpu().detach())
        spoof_campp_scores = 1 - cosine_similarity(enroll_campp_embedding.cpu().detach(), spoof_campp_embedding.cpu().detach())
        campp_scores = np.concatenate([bonafide_campp_target_scores, spoof_campp_scores])
        spoof_processed_campp_scores = 1 - cosine_similarity(enroll_campp_embedding.cpu().detach(),
                                                             spoof_processed_campp_embedding.cpu().detach())
        campp_processed_scores = np.concatenate([bonafide_campp_target_scores, spoof_processed_campp_scores])

        labels_campp = np.concatenate(
            [np.zeros(len(bonafide_campp_target_scores)), np.ones(len(spoof_campp_scores))])

    # Clear cache to free up GPU memory
    torch.cuda.empty_cache()

    return ecapa_scores, ecapa_processed_scores, labels_ecapa, campp_scores, campp_processed_scores, labels_campp

def pooled_evaluation(target_attack, protocol_a, protocol_b, audio_folder, processed_audio_folder, device):
    with torch.no_grad():
        model_ecapa, model_campp, feature_extractor_campp = load_models(device)

        all_ecapa_scores = []
        all_ecapa_processed_scores = []
        all_campp_scores = []
        all_campp_processed_scores = []
        all_labels_ecapa = []
        all_labels_campp = []

        all_speakers = list(set(speaker for speaker, file, label in protocol_b[target_attack]))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_speaker = {executor.submit(evaluate_speaker, speaker, protocol_a, protocol_b, target_attack, audio_folder, processed_audio_folder, device, model_ecapa, model_campp, feature_extractor_campp): speaker for speaker in all_speakers}

            for future in tqdm(concurrent.futures.as_completed(future_to_speaker), desc="Attack Assessment", total=len(all_speakers)):
                speaker = future_to_speaker[future]
                try:
                    ecapa_scores, ecapa_processed_scores, labels_ecapa, campp_scores, campp_processed_scores, labels_campp = future.result()
                    if ecapa_scores.size > 0:
                        all_ecapa_scores.append(ecapa_scores)
                        all_ecapa_processed_scores.append(ecapa_processed_scores)
                        all_labels_ecapa.append(labels_ecapa)
                    if campp_scores.size > 0:
                        all_campp_scores.append(campp_scores)
                        all_campp_processed_scores.append(campp_processed_scores)
                        all_labels_campp.append(labels_campp)
                except Exception as exc:
                    print(f'Speaker {speaker} generated an exception: {exc}')

        if all_ecapa_scores:
            all_ecapa_scores = np.concatenate(all_ecapa_scores)
            all_ecapa_processed_scores = np.concatenate(all_ecapa_processed_scores)
            all_labels_ecapa = np.concatenate(all_labels_ecapa)
            pooled_ecapa_eer = calculate_eer(all_ecapa_scores, all_labels_ecapa)
            pooled_ecapa_processed_eer = calculate_eer(all_ecapa_processed_scores, all_labels_ecapa)
        else:
            pooled_ecapa_eer = None
            pooled_ecapa_processed_eer = None

        if all_campp_scores:
            all_campp_scores = np.concatenate(all_campp_scores)
            all_campp_processed_scores = np.concatenate(all_campp_processed_scores)
            all_labels_campp = np.concatenate(all_labels_campp)
            pooled_campp_eer = calculate_eer(all_campp_scores, all_labels_campp)
            pooled_campp_processed_eer = calculate_eer(all_campp_processed_scores, all_labels_campp)
        else:
            pooled_campp_eer = None
            pooled_campp_processed_eer = None

        return pooled_ecapa_eer, pooled_ecapa_processed_eer, pooled_campp_eer, pooled_campp_processed_eer


def main(target_attack, target_speaker=None, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, speakers_per_gpu=SPEAKERS_PER_GPU):

    protocol_a = parse_protocol_a(PROTOCOL_A)
    protocol_b = parse_protocol_b(PROTOCOL_B)

    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    num_gpus = len(devices)

    with mp.Manager() as manager:
        progress_counter = manager.Value('i', 0)  # Shared counter
        if target_speaker:
            # Process only the specified speaker
            process_speaker(target_attack, target_speaker, protocol_a, protocol_b, devices[0], progress_counter, num_epochs, batch_size, 1)
        else:
            # Process all speakers in protocol_b
            speakers = list(set(speaker for speaker, _, _ in protocol_b[target_attack]))
            num_speakers = len(speakers)

            # Precompute speaker lists for each GPU
            gpu_speaker_lists = {device: [] for device in devices}

            # Distribute speakers across the GPUs
            for idx, speaker in enumerate(speakers):
                device = devices[idx % num_gpus]
                gpu_speaker_lists[device].append(speaker)

            gpu_processes = {device: [] for device in devices}

            with tqdm(total=num_speakers, desc="Processing Speakers") as pbar:
                while any(gpu_speaker_lists[device] or gpu_processes[device] for device in devices):
                    for device in devices:
                        while len(gpu_processes[device]) < speakers_per_gpu and gpu_speaker_lists[device]:
                            speaker = gpu_speaker_lists[device].pop(0)
                            p = mp.Process(target=process_speakers_on_gpu, args=(
                            target_attack, [speaker], protocol_a, protocol_b, device, progress_counter,
                            num_epochs, batch_size))
                            p.start()
                            gpu_processes[device].append(p)

                        for p in gpu_processes[device]:
                            if not p.is_alive():
                                p.join()
                                gpu_processes[device].remove(p)
                                pbar.update(1)
                    time.sleep(1)

                pbar.n = progress_counter.value
                pbar.refresh()

    if not TARGET_SPEAKER and TARGET_ATTACK:
        audio_folder_name = os.path.basename(os.path.normpath(AUDIO_FOLDER))
        processed_audio_folder = os.path.join(OUTPUT_BASE_PATH, f"{target_attack}_{NUM_LAYERS}_{KERNEL_SIZE}", f"{audio_folder_name}_{NUM_LAYERS}_{KERNEL_SIZE}")

        pooled_ecapa_eer, pooled_ecapa_processed_eer, pooled_campp_eer, pooled_campp_processed_eer = pooled_evaluation(target_attack, protocol_a, protocol_b, AUDIO_FOLDER, processed_audio_folder, torch.device('cuda'))
        print(f"{target_attack}_{NUM_LAYERS}_{KERNEL_SIZE}")
        print(f"POOLED_ECAPA_EER: {pooled_ecapa_eer:.4f}, POOLED_MALACOPULA_ECAPA_EER: {pooled_ecapa_processed_eer:.4f}")
        print(f"POOLED_CAMpp_EER: {pooled_campp_eer:.4f}, POOLED_MALACOPULA_CAMpp_EER: {pooled_campp_processed_eer:.4f}")

def process_speakers_on_gpu(target_attack, speakers, protocol_a, protocol_b, device, progress_counter, num_epochs, batch_size):
    for speaker in speakers:
        process_speaker(target_attack, speaker, protocol_a, protocol_b, device, progress_counter, num_epochs, batch_size, 0)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    if TARGET_ATTACK:
        main(TARGET_ATTACK,TARGET_SPEAKER)
    else:
        for i in range(7, 20):
            attack_code = f"A{i:02}"
            print(f"Malacopula training for attack {attack_code}")
            main(attack_code,TARGET_SPEAKER)
