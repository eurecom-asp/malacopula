
"""
Authors: Massimiliano Todisco, Michele Panariello and chatGPT
Email: https://mailhide.io/e/Qk2FFM4a
Date: August 2024
"""

import torch
import yaml
from torch.nn.functional import cosine_similarity
import numpy as np
import pandas as pd
import speechbrain as sb
from speakerlab.models.campplus.DTDNN import CAMPPlus
from speakerlab.process.processor import FBank
from speakerlab.models.eres2net.ERes2Net import ERes2Net
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_loader import parse_protocol_a, load_audio
from AASIST import Model as AS
import configparser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a ConfigParser object
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())

# Read the conf.txt file
config.read('conf.ini')

# Access the values
TARGET_SAMPLE_RATE = int(config.get('DEFAULT', 'TARGET_SAMPLE_RATE'))
NUM_LAYERS = int(config.get('DEFAULT', 'NUM_LAYERS'))
KERNEL_SIZE = int(config.get('DEFAULT', 'KERNEL_SIZE'))
AUDIO_FOLDER = config.get('DEFAULT', 'AUDIO_FOLDER')
PROTOCOL_A = config.get('DEFAULT', 'PROTOCOL_A')
PROTOCOL_B = config.get('DEFAULT', 'PROTOCOL_B')
AUDIO_FOLDER_MALAC = config.get('DEFAULT', 'AUDIO_FOLDER_MALAC')
SCORE_FILE = config.get('DEFAULT', 'SCORE_FILE')
RESULTS_FILE = config.get('DEFAULT', 'RESULTS_FILE')

print(f"Malacopula Evaluation | K={NUM_LAYERS} L={KERNEL_SIZE}")

def load_models(device):
    model_ecapa = sb.inference.speaker.EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                                      run_opts={"device": device})
    model_ecapa.eval()

    feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

    model_path = 'pretrained_models/campplus_voxceleb.bin'
    d = torch.load(model_path)
    model_campp = CAMPPlus().to(device)
    model_campp.load_state_dict(d)
    model_campp.eval()

    model_path = 'pretrained_models/pretrained_eres2net.ckpt'
    d = torch.load(model_path)
    model_ern = ERes2Net().to(device)
    model_ern.load_state_dict(d)
    model_ern.eval()

    with open('pretrained_models/AASIST.conf') as config_file:
        config = yaml.safe_load(config_file)

    aasist = AS(config['model_config'])
    aasist_weights = torch.load('pretrained_models/AASIST.pth')
    missing_keys, unexpected_keys = aasist.load_state_dict(aasist_weights)
    aasist.to(device)
    aasist.eval()

    return model_ecapa, model_campp, model_ern, feature_extractor, aasist


def parse_protocol_b0(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    protocol = []
    for line in lines:
        parts = line.strip().split()
        protocol.append(parts)
    return protocol


def compute_embedding(file, model_ecapa, model_campp, model_ern, feature_extractor, aasist, audio_folder, device):
    audio_path = audio_folder + file + '.flac'
    audio = load_audio(audio_path).to(device)

    with torch.no_grad():
        # Generate the embeddings
        ecapa_embedding = model_ecapa.encode_batch(audio).squeeze(1)
        mel = feature_extractor(audio).unsqueeze(0)
        campp_embedding = model_campp(mel)
        ern_embedding = model_ern(mel)

        if aasist:
            _, bf_logits = aasist(audio)
            bf_score = torch.nn.functional.softmax(bf_logits, dim=1)[:,0]
            # we take the 0th columns (i.e. the spoof score) because in the labeling of this script, 'spoof' == 1 and
            # 'bonafide' == 0
            return ecapa_embedding, campp_embedding, ern_embedding, bf_score.item()

    return ecapa_embedding, campp_embedding, ern_embedding


def process_enrollment_files(enroll_files, model_ecapa, model_campp, model_ern, feature_extractor, audio_folder, device):
    ecapa_embeddings = []
    campp_embeddings = []
    ern_embeddings = []

    for file in enroll_files:
        ecapa_embedding, campp_embedding, ern_embedding = compute_embedding(file, model_ecapa, model_campp, model_ern, feature_extractor,
                                                             None, audio_folder, device)  # leaving aasist as none here
        ecapa_embeddings.append(ecapa_embedding.cpu().numpy())
        campp_embeddings.append(campp_embedding.cpu().numpy())
        ern_embeddings.append(ern_embedding.cpu().numpy())

    # Concatenate embeddings along the first dimension
    enroll_ecapa_embedding = np.concatenate(ecapa_embeddings, axis=0)
    enroll_campp_embedding = np.concatenate(campp_embeddings, axis=0)
    enroll_ern_embedding = np.concatenate(ern_embeddings, axis=0)

    # Convert concatenated embeddings back to torch tensors
    enroll_ecapa_embedding = torch.tensor(enroll_ecapa_embedding).to(device)
    enroll_campp_embedding = torch.tensor(enroll_campp_embedding).to(device)
    enroll_ern_embedding = torch.tensor(enroll_ern_embedding).to(device)

    # Calculate the mean of embeddings along the 0th dimension
    enroll_ecapa_embedding = torch.mean(enroll_ecapa_embedding, dim=0).unsqueeze(0)
    enroll_campp_embedding = torch.mean(enroll_campp_embedding, dim=0).unsqueeze(0)
    enroll_ern_embedding = torch.mean(enroll_ern_embedding, dim=0).unsqueeze(0)

    return enroll_ecapa_embedding, enroll_campp_embedding, enroll_ern_embedding

def calculate_eer_by_group(df, score_column, label_column):
    results = {}
    attacks = df['Attack'].unique()

    # Calculate EER for pooled data
    labels = df[label_column].values
    scores = df[score_column].values
    eer, threshold = calculate_eer(scores, labels)
    results['pooled'] = {'EER': eer, 'Threshold': threshold}

    # Calculate EER for each attack type
    for attack in attacks:
        if attack == 'bonafide':
            continue
        # subset = df[(df['Attack'] == attack) | (df['Label'] == 'target')]
        subset = df[(df['Attack'] == attack) | (df['Label'].isin(['target', 'nontarget']))]
        if not subset.empty:
            labels = subset[label_column].values
            scores = subset[score_column].values
            if np.isnan(labels).any() or np.isnan(scores).any():
                print(f"NaN values found in attack {attack}, skipping.")
                continue
            eer, threshold = calculate_eer(scores, labels)
            results[attack] = {'EER': eer, 'Threshold': threshold}
    return results


def calculate_eer(scores, labels):
    # Separate scores based on labels
    positive_scores = [score for label, score in zip(labels, scores) if label == 1]
    negative_scores = [score for label, score in zip(labels, scores) if label == 0]

    # Convert lists to torch tensors
    positive_scores = torch.tensor(positive_scores)
    negative_scores = torch.tensor(negative_scores)

    # Calculate EER
    eer, thr = sb.utils.metric_stats.EER(positive_scores, negative_scores)
    return eer, thr


# Convert labels to binary format for 'spoof' vs 'target'
def label_to_binary_spoof_target(label):
    if label == 'spoof':
        return 1
    elif label == 'target':
        return 0
    else:
        return None


# Convert labels to binary format for 'nontarget' vs 'target'
def label_to_binary_nontarget_target(label):
    if label == 'nontarget':
        return 1
    elif label == 'target':
        return 0
    else:
        return None

def main():
    protocol_a = parse_protocol_a(PROTOCOL_A)
    protocol_b = parse_protocol_b0(PROTOCOL_B)

    # New dictionaries to store the results
    enroll_dict_ecapa = {}
    enroll_dict_campp = {}
    enroll_dict_ern = {}
    trial_dict_ecapa = {}
    trial_dict_campp = {}
    trial_dict_ern = {}
    trial_bf_score = {}

    with torch.no_grad():
        model_ecapa, model_campp, model_ern, feature_extractor, aasist = load_models(device)

        # Parallelize enrollment embedding extraction
        with ThreadPoolExecutor() as executor:
            futures = []
            for spkID, enroll_files in protocol_a.items():
                futures.append(executor.submit(process_enrollment_files, enroll_files, model_ecapa, model_campp, model_ern,
                                               feature_extractor, AUDIO_FOLDER, device))
            for spkID, future in zip(tqdm(protocol_a.keys(), desc="Extracting enrol embeddings"), futures):
                enroll_dict_ecapa[spkID], enroll_dict_campp[spkID], enroll_dict_ern[spkID] = future.result()

        # Get unique file names from protocol_b
        unique_files = set([parts[1] for parts in protocol_b])

        # Parallelize trial embedding extraction
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(compute_embedding, file_name, model_ecapa, model_campp, model_ern, feature_extractor, aasist,
                                       AUDIO_FOLDER_MALAC, device): file_name for file_name in unique_files}
            for future in tqdm(as_completed(futures), total=len(unique_files), desc="Extracting trial embeddings"):
                file_name = futures[future]
                trial_dict_ecapa[file_name], trial_dict_campp[file_name], trial_dict_ern[file_name], trial_bf_score[file_name] = future.result()

        # List to store the updated lines
        updated_lines = []

        for parts in tqdm(protocol_b, desc="Computing scores"):
            spkID = parts[0]
            file_name = parts[1]

            # Extract precomputed enrollment embedding for spkID
            if spkID in enroll_dict_ecapa:
                precomputed_embedding_ecapa = enroll_dict_ecapa[spkID]
                precomputed_embedding_campp = enroll_dict_campp[spkID]
                precomputed_embedding_ern = enroll_dict_ern[spkID]
            else:
                print(f"Embedding for {spkID} not found")
                continue

            # Extract trial embedding for the file
            ecapa_embedding = trial_dict_ecapa[file_name]
            campp_embedding = trial_dict_campp[file_name]
            ern_embedding = trial_dict_ern[file_name]

            # Compute cosine distances
            ecapa_scores = 1 - cosine_similarity(precomputed_embedding_ecapa.cpu().detach(),
                                                 ecapa_embedding.cpu().detach())
            campp_scores = 1 - cosine_similarity(precomputed_embedding_campp.cpu().detach(),
                                                 campp_embedding.cpu().detach())
            ern_scores = 1 - cosine_similarity(precomputed_embedding_ern.cpu().detach(),
                                                 ern_embedding.cpu().detach())

            bf_score = trial_bf_score[file_name]

            # Append the cosine distances to the line
            updated_line = ' '.join(parts) + f' {ecapa_scores.item()} {campp_scores.item()} {ern_scores.item()} {bf_score}\n'
            updated_lines.append(updated_line)

        # Save the updated protocol to a new file
        with open(SCORE_FILE, 'w') as f:
            f.writelines(updated_lines)

    print(f"Trial protocol with scores saved to {SCORE_FILE}")
    print(f"Computing EERs")

    # Read the file into a pandas DataFrame with unique column names
    df = pd.read_csv(SCORE_FILE, sep=" ", header=None,
                     names=["spkID", "fileID", "Attack", "Label", "Score ECAPA", "Score CAM++", "Score ERes2Net",
                            "Score AASIST"])

    # Convert scores to float
    df['Score ECAPA'] = df['Score ECAPA'].astype(float)
    df['Score CAM++'] = df['Score CAM++'].astype(float)
    df['Score ERes2Net'] = df['Score ERes2Net'].astype(float)
    df['Score AASIST'] = df['Score AASIST'].astype(float)
    df['Score_EACAPA+AASIST'] = df['Score ECAPA'] + df['Score AASIST']
    df['Score_CAM+++AASIST'] = df['Score CAM++'] + df['Score AASIST']
    df['Score_ERes2Net+AASIST'] = df['Score ERes2Net'] + df['Score AASIST']

    # Convert labels to a binary format for EER calculation
    df['Label_binary'] = df['Label'].apply(lambda x: 1 if x in ['spoof', 'nontarget'] else 0)

    results_ecapa_sasv = calculate_eer_by_group(df, 'Score ECAPA', 'Label_binary')
    results_campp_sasv = calculate_eer_by_group(df, 'Score CAM++', 'Label_binary')
    results_ern_sasv = calculate_eer_by_group(df, 'Score ERes2Net', 'Label_binary')
    results_ecapa_aasist_sasv = calculate_eer_by_group(df, 'Score_EACAPA+AASIST', 'Label_binary')
    results_campp_aasist_sasv = calculate_eer_by_group(df, 'Score_CAM+++AASIST', 'Label_binary')
    results_ern_aasist_sasv = calculate_eer_by_group(df, 'Score_ERes2Net+AASIST', 'Label_binary')

    df['Label_binary_spoof_target'] = df['Label'].apply(label_to_binary_spoof_target)
    df['Label_binary_nontarget_target'] = df['Label'].apply(label_to_binary_nontarget_target)

    # Filter out rows where 'Label_binary' is None
    df_filtered_spoof_target = df.dropna(subset=['Label_binary_spoof_target'])
    df_filtered_nontarget_target = df.dropna(subset=['Label_binary_nontarget_target'])

    # Calculate EER for each ASV system and combine the results
    results_ecapa_spf = calculate_eer_by_group(df_filtered_spoof_target, 'Score ECAPA', 'Label_binary_spoof_target')
    results_campp_spf = calculate_eer_by_group(df_filtered_spoof_target, 'Score CAM++', 'Label_binary_spoof_target')
    results_ern_spf = calculate_eer_by_group(df_filtered_spoof_target, 'Score ERes2Net', 'Label_binary_spoof_target')
    results_ecapa_sv = calculate_eer_by_group(df_filtered_nontarget_target, 'Score ECAPA',
                                              'Label_binary_nontarget_target')
    results_campp_sv = calculate_eer_by_group(df_filtered_nontarget_target, 'Score CAM++',
                                              'Label_binary_nontarget_target')
    results_ern_sv = calculate_eer_by_group(df_filtered_nontarget_target, 'Score ERes2Net',
                                            'Label_binary_nontarget_target')
    results_ecapa_aasist_spf = calculate_eer_by_group(df_filtered_spoof_target, 'Score_EACAPA+AASIST',
                                                      'Label_binary_spoof_target')
    results_campp_aasist_spf = calculate_eer_by_group(df_filtered_spoof_target, 'Score_CAM+++AASIST',
                                                      'Label_binary_spoof_target')
    results_ern_aasist_spf = calculate_eer_by_group(df_filtered_spoof_target, 'Score_ERes2Net+AASIST',
                                                    'Label_binary_spoof_target')
    results_ecapa_aasist_sv = calculate_eer_by_group(df_filtered_nontarget_target, 'Score_EACAPA+AASIST',
                                                     'Label_binary_nontarget_target')
    results_campp_aasist_sv = calculate_eer_by_group(df_filtered_nontarget_target, 'Score_CAM+++AASIST',
                                                     'Label_binary_nontarget_target')
    results_ern_aasist_sv = calculate_eer_by_group(df_filtered_nontarget_target, 'Score_ERes2Net+AASIST',
                                                   'Label_binary_nontarget_target')
    # results aasist spf
    results_aasist_spf = calculate_eer_by_group(df_filtered_spoof_target, 'Score AASIST',
                                                'Label_binary_spoof_target')

    # Convert results to DataFrames for display
    results_ecapa_df_sasv = pd.DataFrame(results_ecapa_sasv).T
    results_campp_df_sasv = pd.DataFrame(results_campp_sasv).T
    results_ern_df_sasv = pd.DataFrame(results_ern_sasv).T
    results_ecapa_aasist_df_sasv = pd.DataFrame(results_ecapa_aasist_sasv).T
    results_campp_aasist_df_sasv = pd.DataFrame(results_campp_aasist_sasv).T
    results_ern_aasist_df_sasv = pd.DataFrame(results_ern_aasist_sasv).T

    results_ecapa_df_spf = pd.DataFrame(results_ecapa_spf).T
    results_campp_df_spf = pd.DataFrame(results_campp_spf).T
    results_ern_df_spf = pd.DataFrame(results_ern_spf).T
    results_ecapa_aasist_df_spf = pd.DataFrame(results_ecapa_aasist_spf).T
    results_campp_aasist_df_spf = pd.DataFrame(results_campp_aasist_spf).T
    results_ern_aasist_df_spf = pd.DataFrame(results_ern_aasist_spf).T

    results_ecapa_df_sv = pd.DataFrame(results_ecapa_sv).T
    results_campp_df_sv = pd.DataFrame(results_campp_sv).T
    results_ern_df_sv = pd.DataFrame(results_ern_sv).T
    results_ecapa_aasist_df_sv = pd.DataFrame(results_ecapa_aasist_sv).T
    results_campp_aasist_df_sv = pd.DataFrame(results_campp_aasist_sv).T
    results_ern_aasist_df_sv = pd.DataFrame(results_ern_aasist_sv).T

    results_aasist_df_spf = pd.DataFrame(results_aasist_spf).T

    # Open a file to write the output
    with open(RESULTS_FILE, 'w') as f:
        # Redirect print statements to the file
        def print_to_file(*args, **kwargs):
            print(*args, **kwargs, file=f)

        print_to_file("ECAPA")
        print_to_file("SASV-EER")
        print_to_file(results_ecapa_df_sasv)
        print_to_file("SPF-EER")
        print_to_file(results_ecapa_df_spf)
        print_to_file("SV-EER")
        print_to_file(results_ecapa_df_sv)
        print_to_file("\nCAM++")
        print_to_file("SASV-EER")
        print_to_file(results_campp_df_sasv)
        print_to_file("SPF-EER")
        print_to_file(results_campp_df_spf)
        print_to_file("SV-EER")
        print_to_file(results_campp_df_sv)
        print_to_file("\nERes2Net")
        print_to_file("SASV-EER")
        print_to_file(results_ern_df_sasv)
        print_to_file("SPF-EER")
        print_to_file(results_ern_df_spf)
        print_to_file("SV-EER")
        print_to_file(results_ern_df_sv)
        print_to_file("\nAASIST")
        print_to_file("SPF-EER")
        print_to_file(results_aasist_df_spf)
        print_to_file("\nECAPA + AASIST")
        print_to_file("SASV-EER")
        print_to_file(results_ecapa_aasist_df_sasv)
        print_to_file("SPF-EER")
        print_to_file(results_ecapa_aasist_df_spf)
        print_to_file("SV-EER")
        print_to_file(results_ecapa_aasist_df_sv)
        print_to_file("\nCAM++ + AASIST")
        print_to_file("SASV-EER")
        print_to_file(results_campp_aasist_df_sasv)
        print_to_file("SPF-EER")
        print_to_file(results_campp_aasist_df_spf)
        print_to_file("SV-EER")
        print_to_file(results_campp_aasist_df_sv)
        print_to_file("\nERes2Net + AASIST")
        print_to_file("SASV-EER")
        print_to_file(results_ern_aasist_df_sasv)
        print_to_file("SPF-EER")
        print_to_file(results_ern_aasist_df_spf)
        print_to_file("SV-EER")
        print_to_file(results_ern_aasist_df_sv)

    print(f"Summary results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()