# Malacopula: adversarial automatic speaker verification attacks using a neural-based generalised Hammerstein model

## Description

Malacopula [[1](#ref-malacopula)] is a neural-based generalised Hammerstein model designed to enhance the effectiveness of spoofing attacks on Automatic Speaker Verification (ASV) systems. By introducing adversarial perturbations to spoofed speech utterances, Malacopula increases the vulnerability of ASV systems. The repository includes a complete implementation of the model, covering the filter architecture, adversarial optimisation procedures, and evaluation of ASV systems alongside antispoofing countermeasures, as detailed in the corresponding paper, available on [arXiv](https://arxiv.org/abs/2408.09300).

## Features
- **Neural-based Generalised Hammerstein Model**: Non-linear learnable transformations applied to speech signals for adversarial perturbations.
- **Adversarial Optimisation**: Procedure for minimising cosine distance between spoofed and bona fide utterances.
- **ASV Systems Evaluation**: Tested across multiple ASV architectures (CAM++ [[2](#ref-campp)], ECAPA [[3](#ref-ecapa)], ERes2Net [[4](#ref-eres2net)]).
- **Anti-spoofing Assessments**: Includes assessments of spoofing and deepfake detection (AASIST [[5](#ref-aasist)]).

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

## Dataset

The dataset used is ASVspoof2019, which can be found [here](https://datashare.ed.ac.uk/handle/10283/3336). You should download the dataset and place it into a folder of your choice. Afterward, update the configuration file (`conf.ini`) to reflect the correct path.

`AUDIO_FOLDER`: Path to the folder containing the evaluation `.flac` files `ASVspoof2019/LA/ASVspoof2019_LA_eval/flac`.

`PROTOCOL_A`: Path to enrollment protocol `ASVspoof2019.LA.asv.eval.gi.trn.txt`.

`PROTOCOL_B`: Path to trial protocol `ASVspoof2019.LA.asv.eval.gi.trl.v1.txt`.

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

## How to Use

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
This script creates symbolic links for both bona fide target and non-target files, as well as Malacopula-processed spoof files, using the paths defined in the conf.ini file. It uses the trial protocol file to extract unique file names and labels.

### Evaluating Malacopula Filter
Once the symbolic links have been created, use the `Eval.py` script to evaluate the performance of the Malacopula filter. The `conf.ini` file is also parsed automatically by the evaluation script.

```bash
python Eval.py
```

### Important Parameters in `conf.ini`

- **TARGET_CATEGORY**: Specifies the attack type, e.g., `A17`. Leave it empty to process all attacks.
- **TARGET_SPEAKER**: Specifies the speaker ID, e.g., `LA_0001`. Leave it empty to process all speakers.
- **OUTPUT_BASE_PATH**: Path where the best model, processed speech, and filter coefficients will be saved.
- **f_A**: The embedding extractor used for training (e.g., `ecapa` or `campp`). `f_B` will be automatically set to the other extractor.

### Evaluation-Specific Parameters

- **AUDIO_FOLDER_MALAC**: The folder where Malacopula-processed utterances are saved.
- **SCORE_FILE**: File containing the scores for evaluation, including speaker ID, attack type, and scores from multiple systems (ECAPA, CAM++, ERes2Net, AASIST).
- **RESULTS_FILE**: File summarizing the evaluation results in terms of SAVS[[6](#ref-savs)] Equal Error Rate (EER).

Both `Train.py` and `Eval.py` will automatically read the `conf.ini` file, so make sure to update the configuration as needed before running the scripts.

### Important

In order to run `symbolic_malac_folder.py` and `Eval.py`, ensure that you have processed **all the attacks and speakers**. This is necessary for accurate evaluation and proper creation of symbolic links for the trials.

## Malacopula Implementation

If you're here for the **Malacopula** implementation, you can find the PyTorch class inside `models.py`. Below is a brief overview of the class structure:

```python
import torch
import torch.nn as nn
import numpy as np

class Malacopula(nn.Module):
    def __init__(self, num_layers=5, in_channels=1, out_channels=1, kernel_size=1025, padding='same', bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.convs = nn.ModuleList([
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
            )
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
        self.apply_bartlett_window()
        for i, conv in enumerate(self.convs, start=1):
            powered_x = torch.pow(x, i)
            output = conv(powered_x)
            outputs.append(output)

        summed_output = torch.sum(torch.stack(outputs, dim=0), dim=0)
        max_abs_value = torch.max(torch.abs(summed_output))
        norm_output = summed_output / max_abs_value

        return norm_output
```

## Audio Examples

You can listen to some audio examples from ASVspoof 2019 LA [here](https://eurecom-asp.github.io/malacopula/). 

## Third-party Troubleshooting

Below are some common issues you might encounter when using this repository.

#### 1. Issue: `SpeechBrain.inference` not found

**Cause**: The `inference` module is not being properly imported in the `SpeechBrain` package.

**Solution**: 
To resolve this issue, add the following line to the `__init__.py` file of the `SpeechBrain` package:

1. Open the file located at `envs/malac/lib/python3.8/site-packages/speechbrain/__init__.py`.
2. Add this line of code:

   ```python
   from . import inference  # noqa
   ```

#### 2. Issue: `TypeError: 'type' object is not subscriptable` in `StreamingASR`

**Cause**: The code in `ASR.py` is using Python 3.9+ type hinting syntax that is not compatible with earlier Python versions, such as Python 3.8.

**Solution**: 
To fix this error, remove the type hint that causes the issue:

1. Open the file located at `/envs/malac/lib/python3.8/site-packages/speechbrain/inference/ASR.py`.
2. Locate line 790, which contains the following code:

   ```python
   ) -> tuple[list, list]:
3. Modify it by deleting the type hint:
   ```python
   ):
   ```

## How to Cite This Work

If you use this repository or the Malacopula model in your research, please cite the following paper:

```bibtex
@inproceedings{malacopula2024,
    author={Massimiliano Todisco and Michele Panariello and Xin Wang and Hector Delgado and Kong-Aik Lee and Nicholas Evans},
    title={Malacopula: adversarial automatic speaker verification attacks using a neural-based generalised Hammerstein model},
    booktitle={Proc. ASVspoof Workshop 2024},
    year={2024}
}
```

## References

For the speaker verification systems and models mentioned in this repository, please refer to the following sources:

1. **Malacopula: Adversarial Automatic Speaker Verification Attacks Using a Neural-Based Generalised Hammerstein Model**
   - Todisco, M., Panariello, M., Wang, X., Delgado, H., Lee, K.-A., & Evans, N. (2024). In *Proc. ASVspoof Workshop 2024*.
   - https://www.isca-archive.org/asvspoof_2024/todisco24_asvspoof.pdf
     <a name="ref-malacopula"></a>

2. **CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking**
   - Wang, H., Zheng, S., Chen, Y., Cheng, L., & Chen, Q. (2023). In *INTERSPEECH 2023*.
   - https://arxiv.org/pdf/2303.00332
   - [GitHub Repository](https://github.com/modelscope/3D-Speaker/tree/main?tab=readme-ov-file) 
     <a name="ref-campp"></a>

3. **ECAPA-TDNN: Emphasized Channel Attention, Propagation, and Aggregation in TDNN-based Speaker Verification**
   - Desplanques, B., Thienpondt, J., & Demuynck, K. (2020). In *INTERSPEECH 2020*, pages 3830-3834.
   - https://arxiv.org/pdf/2005.07143
   - [GitHub Repository](https://github.com/speechbrain/speechbrain)
     <a name="ref-ecapa"></a>

4. **ERes2NetV2: Boosting Short-Duration Speaker Verification Performance with Computational Efficiency**
   - Chen, Y., & others. (2024). *arXiv preprint arXiv:2406.02167*.
   - https://arxiv.org/pdf/2406.02167
   - [GitHub Repository](https://github.com/modelscope/3D-Speaker/tree/main?tab=readme-ov-file)
     <a name="ref-eres2net"></a>

5. **AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks**
   - Jung, J.-w., Heo, H.-S., Tak, H., Shim, H.-j., Chung, J.S., Lee, B.-J., & Yu, H.-J., Evans, N. (2022). In *Proc. ICASSP 2022*, pages 6367-6371.
   - https://arxiv.org/pdf/2110.01200
   - [GitHub Repository](https://github.com/clovaai/aasist)
     <a name="ref-aasist"></a>

6. **SASV 2022: The First Spoofing-Aware Speaker Verification Challenge**
   - Jung, J.-w., Tak, H., Shim, H.-j., Heo, H.-S., Lee, B.-J., Chung, S.-W., Yu, H.-J., Evans, N., & Kinnunen, T. (2022). In *Proc. INTERSPEECH 2022*, pages 2893-2897.
   - https://arxiv.org/pdf/2203.14732
   - [GitHub Repository](https://sasv-challenge.github.io)
     <a name="ref-savs"></a>


These references provide additional context for the models and methods used in this repository.

## Acknowledgements
This work is supported with funding received from the French Agence Nationale de la Recherche (ANR) via the BRUEL (ANR-22-CE39-0009) and COMPROMIS (ANR-22-PECY0011) projects.

