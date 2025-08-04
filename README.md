# Causal Framework for Bias Mitigation in Biomedical NLP (CF-CKP)

This repository contains the implementation of the paper **CF-CKP: A Causal Framework for Bias Mitigation in Biomedical NLP**. The code is designed to help researchers and practitioners replicate the experiments and results described in the paper.

## File Overview

The following Python files are included in this repository:

- `CONFIG.py`: Configuration file containing hyperparameters and settings.
- `train.py`: Script to train the model. After training, the model is saved for later evaluation.
- `test.py`: Script to evaluate the trained model.
- `model.py`: Contains the model architecture and methods.
- `ysy_util.py`: Utility functions used in the project.

## Requirements

The following Python libraries are required to run the code:

- **PyTorch** (version 1.11.0)
- **Transformers** (version 4.16.0)

You can install these dependencies using the following commands:

```bash
conda install torch==1.11.0
conda install transformers==4.16.0

## Setup and Usage


