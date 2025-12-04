# Designing a Conditional Prior Distribution for Flow-Based Generative Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2502.09611-b31b1b.svg)](https://arxiv.org/abs/2502.09611)

This repository contains the official implementation of our paper "Designing a Conditional Prior Distribution for Flow-Based Generative Models" submitted to TMLR (Transactions on Machine Learning Research).

## Abstract

Flow-based generative models have recently shown impressive performance for conditional generation tasks, such as text-to-image generation. However, current methods transform a general unimodal noise distribution to a specific mode of the target data distribution. As such, every point in the initial source distribution can be mapped to every point in the target distribution, resulting in long average paths. To this end, in this work, we tap into a non-utilized property of conditional flow-based models: the ability to design a non-trivial prior distribution. Given an input condition, such as a text prompt, we first map it to a point lying in data space, representing an "average" data point with the minimal average distance to all data points of the same conditional mode (e.g., class). We then utilize the flow matching formulation to map samples from a parametric distribution centered around this point to the conditional target distribution. Experimentally, our method significantly improves training times and generation efficiency (FID, KID and CLIP alignment scores) compared to baselines, producing high quality samples using fewer sampling steps.

## Features

- **Conditional Flow Matching**: Implements flow matching for text-to-image generation
- **CLIP Integration**: Uses CLIP for text encoding
- **VQ-VAE Support**: Leverages VQ-VAE for efficient image representation
- **Distributed Training**: Supports multi-GPU training via Hugging Face Accelerate
- **Wandb Logging**: Integrated experiment tracking with Weights & Biases

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 1.12+

### Setup

1. Clone this repository:
```bash
git clone https://github.com/MoSalama98/conditional-prior-flow-matching.git
cd conditional-prior-flow-matching
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models:
   - VQ-VAE model weights should be placed in `models_weights/vq_model_weights/`
   - Pre-trained projector weights (if available) should be placed in `models_weights/projector_noise/`

## Usage

### Training the Projector/Decoder

Train the projector model that maps CLIP text embeddings to VQ-VAE latent space:

```bash
python train_mapper.py \
    --vqae_directory models_weights/vq_model_weights \
    --clip_model_name openai/clip-vit-base-patch32 \
    --output_dir models_weights/projector_no_noise \
    --num_train_epochs 20 \
    --batch_size 256 \
    --learning_rate 1e-4 \
    --image_size 256
```

### Training the Flow Matching Model

Train the main flow matching model for text-to-image generation:

```bash
python train.py \
    --output_save_dir models_weights \
    --output_dir models_weights/fm_noise_weights \
    --num_steps 10 \
    --dim_z 32 \
    --sigma 0.7 \
    --num_train_epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --weights_projector_path models_weights/projector_noise/decoder_weights.pth \
    --load_z models_weights/projector_noise/z_final.pth
```

### Key Arguments

**For `train_mapper.py`:**
- `--vqae_directory`: Directory containing VQ-VAE model weights
- `--clip_model_name`: CLIP model identifier (default: `openai/clip-vit-base-patch32`)
- `--output_dir`: Output directory for saved model weights
- `--num_train_epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--learning_rate`: Learning rate for optimizer
- `--image_size`: Image size (default: 256)

**For `train.py`:**
- `--output_save_dir`: Base directory for model weights
- `--output_dir`: Output directory for flow matching model weights
- `--num_steps`: Number of flow matching steps
- `--dim_z`: Dimension of Gaussian noise vector z
- `--sigma`: Sigma value for noise injection
- `--weights_projector_path`: Path to pre-trained projector weights
- `--load_z`: Path to z statistics file
- `--resume_from_checkpoint`: Epoch number to resume from (optional)

## Project Structure

```
flowmatching_TMLR/
├── train.py              # Main flow matching training script
├── train_mapper.py       # Projector/decoder training script
├── dataset.py            # Dataset loading utilities
├── neural_models.py      # Neural network model definitions
├── metrics.py            # Evaluation metrics
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{issachar2025designing,
  title={Designing a Conditional Prior Distribution for Flow-Based Generative Models},
  author={Issachar, Noam and Salama, Mohammad and Fattal, Raanan and Benaim, Sagie},
  journal={arXiv preprint arXiv:2502.09611},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the `diffusers` and `transformers` libraries
- OpenAI for CLIP
- The flow matching community for foundational work

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

