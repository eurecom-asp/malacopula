# Malacopula: Adversarial Automatic Speaker Verification Attacks Using a Neural-Based Generalised Hammerstein Model

## Description

Malacopula is a neural-based Generalised Hammerstein model designed to enhance the effectiveness of adversarial attacks on Automatic Speaker Verification (ASV) systems. By introducing adversarial perturbations to spoofed speech utterances, Malacopula increases the vulnerability of ASV systems while maintaining the quality of speech. This repository contains the implementation of the model, including the filter architecture and adversarial optimization procedures.

## Features
- **Generalised Hammerstein Model**: Non-linear transformations applied to speech signals for adversarial perturbations.
- **Adversarial Optimization**: Procedure for minimizing cosine distance between spoofed and bona fide utterances.
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

Clone the repository and install the dependencies:

```bash
git clone https://github.com/eurecom-asp/malacopula
cd malacopula
pip install -r requirements.txt
