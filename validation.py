
"""
Authors: Massimiliano Todisco, Michele Panariello and chatGPT
Email: https://mailhide.io/e/Qk2FFM4a
Date: August 2024
"""

import torch
from torch.nn.functional import cosine_similarity
import numpy as np
from speechbrain.utils.metric_stats import EER
from scipy.stats import wasserstein_distance
from data_loader import load_audio


def signed_wasserstein_distance_median(A, B):
    distance = wasserstein_distance(A, B)
    median_A = np.median(A)
    median_B = np.median(B)

    if median_A > median_B:
        return -distance
    else:
        return distance


def validate(model, protocol_b, target_category, target_speaker,
             enroll_ecapa_embedding, enroll_campp_embedding,
             bonafide_ecapa_target_scores,bonafide_campp_target_scores,
                AUDIO_FOLDER, device, model_ecapa, model_campp, feature_extractor_campp):

    model.eval()

    with torch.no_grad():
        # Get spoof ECAPA embeddings
        spoof_files = [file for speaker, file, label in protocol_b[target_category] if speaker == target_speaker]

        # Get spoof ECAPA embeddings
        spoof_ecapa_embeddings = []
        for file in spoof_files:
            audio = load_audio(AUDIO_FOLDER + file + '.flac')
            processed_audio = model(audio.to(device).unsqueeze(0)).squeeze(0)
            embedding = model_ecapa.encode_batch(processed_audio)
            spoof_ecapa_embeddings.append(embedding)
        spoof_ecapa_embedding = torch.cat(spoof_ecapa_embeddings, dim=0).squeeze(1)

        # Calculate ECAPA scores
        spoof_ecapa_scores = 1 - cosine_similarity(enroll_ecapa_embedding.to(device), spoof_ecapa_embedding)
        ecapa_scores = torch.cat([bonafide_ecapa_target_scores.to(device), spoof_ecapa_scores]).cpu().numpy()
        labels = np.concatenate([np.zeros(len(bonafide_ecapa_target_scores.to(device))), np.ones(len(spoof_ecapa_scores))])
        ecapa_eer = calculate_eer(ecapa_scores, labels)

        # Get spoof CAM++ embeddings
        spoof_campp_embeddings = []
        for file in spoof_files:
            audio = load_audio(AUDIO_FOLDER + file + '.flac')
            processed_audio = model(audio.to(device).unsqueeze(0)).squeeze(0)
            mel = feature_extractor_campp(processed_audio).unsqueeze(0)
            embedding = model_campp(mel).unsqueeze(0)
            spoof_campp_embeddings.append(embedding)
        spoof_campp_embedding = torch.cat(spoof_campp_embeddings, dim=0).squeeze(1)

        # Calculate CAM++ scores
        spoof_campp_scores = 1 - cosine_similarity(enroll_campp_embedding.to(device), spoof_campp_embedding)
        campp_scores = torch.cat([bonafide_campp_target_scores.to(device), spoof_campp_scores]).cpu().numpy()
        campp_eer = calculate_eer(campp_scores, labels)

        # Calculate Wasserstein distances
        w_distance_ecapa = signed_wasserstein_distance_median(bonafide_ecapa_target_scores.cpu().numpy(),
                                                spoof_ecapa_scores.cpu().numpy())
        w_distance_campp = signed_wasserstein_distance_median(bonafide_campp_target_scores.cpu().numpy(),
                                                spoof_campp_scores.cpu().numpy())

        return ecapa_eer, campp_eer, w_distance_ecapa, w_distance_campp,


def compute_embeddings(files, model_ecapa, model_campp, feature_extractor_campp, audio_folder, device):
    ecapa_embeddings = []
    campp_embeddings = []

    for file in files:
        audio_path = audio_folder + file + '.flac'
        audio = load_audio(audio_path).to(device)

        with torch.no_grad():
            # Generate the embeddings
            ecapa_embedding = model_ecapa.encode_batch(audio)
            mel = feature_extractor_campp(audio).unsqueeze(0)
            campp_embedding = model_campp(mel).unsqueeze(0)

        ecapa_embeddings.append(ecapa_embedding.cpu().numpy())
        campp_embeddings.append(campp_embedding.cpu().numpy())

    ecapa_embeddings = np.concatenate(ecapa_embeddings, axis=0).squeeze(1)
    campp_embeddings = np.concatenate(campp_embeddings, axis=0).squeeze(1)

    return torch.tensor(ecapa_embeddings), torch.tensor(campp_embeddings)


def calculate_eer(scores, labels):
    # Separate scores based on labels
    positive_scores = [score for label, score in zip(labels, scores) if label == 1]
    negative_scores = [score for label, score in zip(labels, scores) if label == 0]

    # Convert lists to torch tensors
    positive_scores = torch.tensor(positive_scores)
    negative_scores = torch.tensor(negative_scores)

    # Calculate EER
    eer, thr = EER(positive_scores, negative_scores)
    return eer