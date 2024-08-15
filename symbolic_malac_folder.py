
"""
Authors: Massimiliano Todisco, Michele Panariello and chatGPT
Email: https://mailhide.io/e/Qk2FFM4a
Date: August 2024
"""

import os
import configparser
from tqdm import tqdm  # Import tqdm for the progress bar

# Create a ConfigParser object
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())

# Read the conf.ini file
config.read('conf.ini')

# Access the values from the configuration file
NUM_LAYERS = int(config.get('DEFAULT', 'NUM_LAYERS'))
KERNEL_SIZE = int(config.get('DEFAULT', 'KERNEL_SIZE'))
AUDIO_FOLDER = config.get('DEFAULT', 'AUDIO_FOLDER')
OUTPUT_BASE_PATH = config.get('DEFAULT', 'OUTPUT_BASE_PATH')
PROTOCOL_B = config.get('DEFAULT', 'PROTOCOL_B')
AUDIO_FOLDER_MALAC = config.get('DEFAULT', 'AUDIO_FOLDER_MALAC')

# Create the virtual folder
os.makedirs(AUDIO_FOLDER_MALAC, exist_ok=True)

# Extract unique file names and labels from the protocol file
with open(PROTOCOL_B, 'r') as f:
    unique_files = set((line.split()[1], line.split()[2], line.split()[3]) for line in f)

# Count the total number of unique files
total_lines = len(unique_files)

# Initialize tqdm progress bar
with tqdm(total=total_lines, desc="Symbolic Copying of Files") as pbar:
    # Read the unique files list and process
    for columns in unique_files:
        file_name = f"{columns[0]}.flac"
        Axx = columns[1]
        label = columns[2]

        if label in ["target", "nontarget"]:
            # Symbolic link of target or nontarget files
            source_file = os.path.join(AUDIO_FOLDER, file_name)
            destination_symlink = os.path.join(AUDIO_FOLDER_MALAC, file_name)
            os.symlink(source_file, destination_symlink)
        elif label == "spoof":
            # Symbolic link of Malacopula processed spoof files
            spoof_dir = os.path.join(OUTPUT_BASE_PATH, f"{Axx}_{NUM_LAYERS}_{KERNEL_SIZE}", f"flac_{NUM_LAYERS}_{KERNEL_SIZE}")
            source_file = os.path.join(spoof_dir, file_name)
            destination_symlink = os.path.join(AUDIO_FOLDER_MALAC, file_name)
            os.symlink(source_file, destination_symlink)

        # Update the progress bar
        pbar.update(1)

# Final output message
print("\nFiles were copied as symbolic links into", AUDIO_FOLDER_MALAC)
