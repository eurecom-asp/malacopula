# Malacopula: Adversarial Automatic Speaker Verification Attacks Using a Neural-Based Generalised Hammerstein Model

## Description

Malacopula is a neural-based generalised Hammerstein model designed to enhance the effectiveness of spoofing attacks on Automatic Speaker Verification (ASV) systems. By introducing adversarial perturbations to spoofed speech utterances, Malacopula increases the vulnerability of ASV systems. This repository contains the implementation of the model, including the filter architecture and adversarial optimisation procedures.

## Features
- **Neural-based Generalised Hammerstein Model**: Non-linear learnable transformations applied to speech signals for adversarial perturbations.
- **Adversarial Optimisation**: Procedure for minimising cosine distance between spoofed and bona fide utterances.
- **Cross-System Evaluation**: Tested across multiple ASV architectures (CAM++, ECAPA, ERes2Net).
- **Impact Evaluation**: Includes assessments of spoofing and deepfake detection (AASIST).

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Experimentation and Results](#experimentation-and-results)
- [License](#license)
- [Citation](#citation)
- [Contact Information](#contact-information)
- [Contributing](#contributing)

## Installation
To set up the environment, you will need to install the dependencies listed in the `environment.yml` file using Conda. Follow the instructions below:

### Prerequisites
Ensure you have Conda installed. If not, you can download and install it from [here](https://docs.conda.io/en/latest/miniconda.html).

### Steps to Install
1. Clone the repository:
   ```bash
   git clone https://github.com/eurecom-asp/malacopula
   cd malacopula
2. Create the Conda environment using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
3. Activate the newly created environment:
   ```bash
   conda activate malac
This will set up the environment with all necessary dependencies, including CUDA, PyTorch, and other required libraries.

## Malacopula Filter Training and Evaluation

This repository allows you to train the Malacopula filter for a specific attack and speaker ID using `Train.py` and evaluate the Malacopula filter using `Eval.py`. Both scripts automatically parse the `conf.ini` file, which contains all the necessary parameters for training and evaluation.

### Configuration (`conf.ini`)

The `conf.ini` file contains parameters for both training and evaluation. Here is a breakdown of the key parameters:

```ini
[DEFAULT]
TARGET_SAMPLE_RATE=16000        # The target sample rate for audio processing, in Hz.
LEARNING_RATE=0.01              # The learning rate for the training algorithm.
NUM_EPOCHS=60                   # The number of epochs to train the model.
BATCH_SIZE=12                   # The number of samples in each batch during training.
VALIDATION_STEP=2               # The number of steps between validations during training.
NUM_LAYERS=3                    # The number of parallel branches in the Malacopula model [K].
KERNEL_SIZE=257                 # The length of the filter in the Malacopula model [L].
SPEAKERS_PER_GPU=5              # Number of speakers processed concurrently per GPU (optimizes speed but depends on available GPU memory).
AUDIO_FOLDER=path/to/ASVspoof2019/LA/ASVspoof2019_LA_eval/flac/  # Path to the ASVspoof2019_LA_eval audio files.
OUTPUT_BASE_PATH=path/to/output/folder/  # Base path to save the best model, Malacopula processed speech, TensorBoard files, and Malacopula filter coefficients during training.
PROTOCOL_A=path/to/protocols/ASVspoof2019.LA.asv.eval.gi.trn.txt  # Path to enrollment speech protocol.
PROTOCOL_B=path/to/protocols/ASVspoof2019.LA.asv.eval.gi.trl.v1.txt  # Path to trial speech protocol.
TARGET_CATEGORY=A17             # The target attack, e.g., A17 (leave it empty to process all the attacks).
TARGET_SPEAKER=LA_0001          # The target speaker ID, e.g., LA_0001 (leave it empty to process all the speakers).
f_A=campp                       # The type of embedding extractor being used for training (e.g., 'ecapa' or 'campp'). Once f_A is chosen, f_B is the other.

# Only for Eval.py
AUDIO_FOLDER_MALAC=${OUTPUT_BASE_PATH}Malacopula_${NUM_LAYERS}_${KERNEL_SIZE}/ # The folder where Malacopula processed utterances are saved.
SCORE_FILE=${OUTPUT_BASE_PATH}ASV2019_eval_scores_${NUM_LAYERS}_${KERNEL_SIZE}.txt # The file for scores with header: "spkID", "fileID", "attack", "label", "score ECAPA", "score CAM++", "score ERes2Net", "score AASIST".
RESULTS_FILE=${OUTPUT_BASE_PATH}ASV2019_eval_summary_${NUM_LAYERS}_${KERNEL_SIZE}.txt # The file summarizing the results in terms of EER.
```

### How to Use

### Training Malacopula Filter
Use the `Train.py` script to train the Malacopula filter for a specific attack and speaker ID. The `conf.ini` file is automatically parsed by the script.

```bash
python Train.py
```

### Creating Symbolic Links
After training, run the 'symbolic_malac_folder.py' script to create symbolic links in the specified folder for the trials, which include Malacopula-processed spoof utterances.

```bash
python symbolic_malac_folder.py
```
This script creates symbolic links for both target and non-target files, as well as Malacopula-processed spoof files, using the paths defined in the conf.ini file. It uses the PROTOCOL_B file to extract unique file names and labels.

### Evaluating Malacopula Filter
Once the symbolic links have been created, use the Eval.py script to evaluate the performance of the Malacopula filter. The conf.ini file is also parsed automatically by the evaluation script.

```bash
python Eval.py
```
