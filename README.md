# Malacopula: Adversarial Automatic Speaker Verification Attacks Using a Neural-Based Generalised Hammerstein Model

## Description

Malacopula is a neural-based Generalised Hammerstein model designed to enhance the effectiveness of spoofing attacks on Automatic Speaker Verification (ASV) systems. By introducing adversarial perturbations to spoofed speech utterances, Malacopula increases the vulnerability of ASV systems. This repository contains the implementation of the model, including the filter architecture and adversarial optimisation procedures.

## Features
- **Neural-based Generalised Hammerstein Model**: Non-linear transformations applied to speech signals for adversarial perturbations.
- **Adversarial Optimisation**: Procedure for minimising cosine distance between spoofed and bona fide utterances.
- **Cross-System Evaluation**: Tested across multiple ASV architectures (CAM++, ECAPA, ERes2Net).
- **Impact Evaluation**: Includes assessments of spoofing and deepfake detection (AASIST) and Mean Opinion Score (MOS) for speech quality.

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
