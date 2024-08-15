
"""
Authors: Massimiliano Todisco, Michele Panariello and chatGPT
Email: https://mailhide.io/e/Qk2FFM4a
Date: August 2024
"""

import torch
from torch.nn.functional import cosine_similarity
import numpy as np

from validation import compute_embeddings, calculate_eer, signed_wasserstein_distance_median

def precompute_embeddings(protocol_a, protocol_b, target_speaker, target_category, audio_folder, device, writer, model_ecapa, model_campp, feature_extractor_campp, verbose):
    with torch.no_grad():
        enroll_files = protocol_a[target_speaker]
        bonafide_target_files = [file for speaker, file, label in protocol_b["bonafide"] if speaker == target_speaker]
        spoof_files = [file for speaker, file, label in protocol_b[target_category] if speaker == target_speaker]

        enroll_ecapa_embedding, enroll_campp_embedding = compute_embeddings(enroll_files, model_ecapa, model_campp, feature_extractor_campp, audio_folder, device)

        enroll_dict_ecapa = {enroll_files[i]: enroll_ecapa_embedding[i] for i in range(len(enroll_files))}
        enroll_dict_campp = {enroll_files[i]: enroll_campp_embedding[i] for i in range(len(enroll_files))}

        enroll_ecapa_embedding = torch.mean((enroll_ecapa_embedding), dim=0).unsqueeze(0)
        enroll_campp_embedding = torch.mean((enroll_campp_embedding), dim=0).unsqueeze(0)

        bonafide_target_ecapa_embedding, bonafide_target_campp_embedding = compute_embeddings(
            bonafide_target_files, model_ecapa, model_campp, feature_extractor_campp, audio_folder, device)

        spoof_ecapa_embedding, spoof_campp_embedding = compute_embeddings(
            spoof_files, model_ecapa, model_campp, feature_extractor_campp, audio_folder, device)

        bonafide_ecapa_target_scores = 1 - cosine_similarity(enroll_ecapa_embedding, bonafide_target_ecapa_embedding)
        spoof_ecapa_scores = 1 - cosine_similarity(enroll_ecapa_embedding, spoof_ecapa_embedding)
        ecapa_scores = torch.cat([bonafide_ecapa_target_scores, spoof_ecapa_scores]).cpu().numpy()
        labels = np.concatenate([np.zeros(len(bonafide_ecapa_target_scores)), np.ones(len(spoof_ecapa_scores))])
        ecapa_eer = calculate_eer(ecapa_scores, labels)

        bonafide_campp_target_scores = 1 - cosine_similarity(enroll_campp_embedding, bonafide_target_campp_embedding)
        spoof_campp_scores = 1 - cosine_similarity(enroll_campp_embedding, spoof_campp_embedding)
        campp_scores = torch.cat([bonafide_campp_target_scores, spoof_campp_scores]).cpu().numpy()
        campp_eer = calculate_eer(campp_scores, labels)

        w_distance_ecapa = signed_wasserstein_distance_median(bonafide_ecapa_target_scores.cpu().numpy(), spoof_ecapa_scores.cpu().numpy())
        w_distance_campp = signed_wasserstein_distance_median(bonafide_campp_target_scores.cpu().numpy(), spoof_campp_scores.cpu().numpy())

        writer.add_scalar('ECAPA_EER/validation', ecapa_eer, 0)
        writer.add_scalar('CAMpp_EER/validation', campp_eer, 0)
        writer.add_scalar('ECAPA_WassD/validation', w_distance_ecapa, 0)
        writer.add_scalar('CAMpp_WassD/validation', w_distance_campp, 0)

        if verbose:
            print(f"Epoch {0}, VALIDATION: ECAPA_EER: {ecapa_eer:.4f}, CAMpp_EER: {campp_eer:.4f}, ECAPA_WassD: {w_distance_ecapa:.4f}, CAMpp_WassD: {w_distance_campp:.4f}")

    return enroll_ecapa_embedding, enroll_campp_embedding, bonafide_ecapa_target_scores,bonafide_campp_target_scores, enroll_dict_ecapa, enroll_dict_campp
