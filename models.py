
"""
Authors: Massimiliano Todisco, Michele Panariello and chatGPT
Email: https://mailhide.io/e/Qk2FFM4a
Date: August 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import speechbrain as sb
from speakerlab.models.campplus.DTDNN import CAMPPlus
from speakerlab.process.processor import FBank
import numpy as np


class Malacopula(nn.Module):
    def __init__(self, num_layers=5, in_channels=1, out_channels=1, kernel_size=1025, padding='same', bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.convs = nn.ModuleList([
            nn.utils.parametrizations.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=bias))
            for _ in range(num_layers)
        ])
        self.bartlett_window = self.create_bartlett_window()
        self.apply_bartlett_window()

    def create_bartlett_window(self):
        bartlett_window = torch.bartlett_window(self.kernel_size)
        return bartlett_window.unsqueeze(0).unsqueeze(0)

    def apply_bartlett_window(self):
        for conv in self.convs:
            with torch.no_grad():
                bartlett_window = self.bartlett_window.to(conv.weight.device)
                conv.weight *= bartlett_window

    def save_filter_coefficients(self, directory_path):
        for i, conv in enumerate(self.convs, start=1):
            bartlett_window = self.bartlett_window.to(conv.weight.device)
            filter_weights = (conv.weight.data * bartlett_window).cpu().numpy()
            filter_weights = np.squeeze(filter_weights)

            filepath = f"{directory_path}/filter_{i}.txt"
            np.savetxt(filepath, filter_weights, fmt='%.6f', delimiter=' ')

    def forward(self, x):
        outputs = []
        for i, conv in enumerate(self.convs, start=1):
            powered_x = torch.pow(x, i)
            output = conv(powered_x)
            outputs.append(output)

        summed_output = torch.sum(torch.stack(outputs, dim=0), dim=0)
        max_abs_value = torch.max(torch.abs(summed_output))
        norm_output = summed_output / max_abs_value

        return norm_output

class CosineDistanceLoss(nn.Module):
    def __init__(self):
        super(CosineDistanceLoss, self).__init__()

    def forward(self, input, target):
        assert input.shape == target.shape, "Input and target must have the same shape"
        cosine_similarity = F.cosine_similarity(input, target, dim=-1)
        cosine_distance = 1 - cosine_similarity
        loss = cosine_distance.mean()
        return loss

def load_models(device):
    model_ecapa = sb.inference.speaker.EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":device})
    model_ecapa.eval()

    model_path = 'pretrained_models/campplus_voxceleb.bin'
    d = torch.load(model_path)
    model_campp = CAMPPlus().to(device)
    feature_extractor_campp = FBank(80, sample_rate=16000, mean_nor=True)
    model_campp.load_state_dict(d)
    model_campp.eval()

    return model_ecapa, model_campp, feature_extractor_campp
