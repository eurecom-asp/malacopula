
"""
Authors: Massimiliano Todisco, Michele Panariello and chatGPT
Email: https://mailhide.io/e/Qk2FFM4a
Date: August 2024
"""

import torch
from models import CosineDistanceLoss

def train(model, batch_a, batch_b, optimizer, writer, epoch, batch_idx, total_batches, model_ecapa, feature_extractor_campp, model_campp, device, f_A):
    model.train()
    optimizer.zero_grad()

    embedding_a = []
    embedding_b = []
    embeddings_a = []
    embeddings_b = []

    for audio_a, audio_b in zip(batch_a, batch_b):
        if f_A == 'ecapa':
            embedding_a = audio_a.unsqueeze(0).unsqueeze(0)
            processed_b = model(audio_b.to(device).unsqueeze(1)).squeeze(1)
            embedding_b = model_ecapa.encode_batch(processed_b)
        elif f_A == 'campp':
            embedding_a = audio_a.unsqueeze(0).unsqueeze(0)
            processed_audio = model(audio_b.to(device).unsqueeze(0)).squeeze(0)
            mel = feature_extractor_campp(processed_audio).unsqueeze(0)
            embedding_b = model_campp(mel).unsqueeze(0)

        embeddings_a.append(embedding_a)
        embeddings_b.append(embedding_b)

    embeddings_a = torch.cat(embeddings_a, dim=0).squeeze(1).to(device)
    embeddings_b = torch.cat(embeddings_b, dim=0).squeeze(1).to(device)

    loss_fn = CosineDistanceLoss()
    loss = loss_fn(embeddings_a, embeddings_b)

    loss.backward()
    optimizer.step()

    global_step = epoch * total_batches + batch_idx
    writer.add_scalar('Loss/train', loss.item(), global_step)

    return loss.item()
