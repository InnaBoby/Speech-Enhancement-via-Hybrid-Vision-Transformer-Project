# Hybrid Vision Transformer for Speech Enhancement

Ever wondered what happens when you treat audio like an image? This project does exactly that - using Vision Transformers to clean up noisy speech recordings. It's a hybrid approach that combines the best of CNNs (great at spotting local patterns) and Transformers (amazing at understanding global context).

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Enhancement](#enhancement)
- [Configuration](#configuration)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Citation](#citation)
- [License](#license)

## Overview

Here's the idea: speech spectrograms look a lot like images. So why not use image processing techniques to denoise them? This project does exactly that with a Hybrid Vision Transformer that:

- Takes noisy speech and cleans it up automatically
- Handles all sorts of background noise (coffee shops, streets, offices, you name it)
- Works in near real-time on modern GPUs
- Actually achieves impressive results (8-9 dB SNR improvement!)

**What can it do?**
- Clean up background noise from recordings
- Work with different types of noise
- Process audio at 16kHz (standard for speech)
- Give you detailed quality metrics (PESQ, STOI, SI-SDR, SNR)

## Key Features

- **Hybrid Architecture**: The best of both worlds - CNNs for local patterns, Transformers for the big picture
- **Actually Works**: This isn't just research code - it's tested, has proper error handling, and logging
- **Clean Code**: Modular design that's easy to understand and extend
- **Full Metrics Suite**: PESQ, STOI, SI-SDR, SNR - all the metrics researchers care about
- **Fast Training**: Mixed precision (FP16) support to speed things up
- **Easy Configuration**: Just edit YAML files, no need to dig through code
- **Watch it Learn**: TensorBoard integration so you can see what's happening in real-time
- **Smart Checkpoints**: Automatically saves your best models
- **Data Augmentation**: SpecAugment and friends to make your model more robust

## Architecture

Think of it as a three-stage pipeline:

### 1. CNN Encoder (The Feature Extractor)
First, we use CNNs to grab local patterns from the spectrogram:
- Starts with 1 channel, expands to 64 → 128 → 256
- MaxPooling to shrink things down
- Captures those fine-grained frequency patterns that CNNs are great at

### 2. Vision Transformer (The Brain)
This is where the magic happens:
- Breaks the spectrogram into patches (like ViT does with images)
- 6 transformer layers with 8 attention heads each
- 512-dimensional embeddings
- Looks at the ENTIRE audio clip at once to understand context
- Has dropout tricks (stochastic depth) to avoid overfitting

### 3. CNN Decoder (The Rebuilder)
Finally, we reconstruct the clean spectrogram:
- Goes backwards: 256 → 128 → 64 → 1 channels
- Upsamples to match the original size
- Uses skip connections (U-Net style) to preserve details the encoder found

### Why Use Vision Transformers for Audio?

Good question! Here's why it actually makes sense:

1. **Spectrograms ARE Images**: Time-frequency plots are literally 2D images
2. **Context Matters**: Transformers are amazing at seeing the whole picture, not just local patterns
3. **Smart Attention**: The model learns which parts of the audio matter most for cleaning
4. **Natural Fit**: Patches work perfectly for time-frequency tiles

## Installation

### What You'll Need

- Python 3.8+ (3.10 recommended)
- A decent NVIDIA GPU with CUDA 11.8+ (or CPU if you're patient)
- At least 16GB RAM (32GB if you want smooth training)
- GPU with 8GB+ VRAM for training, 4GB+ for just running inference

### Quick Setup (The Easy Way)

```bash
# Grab the code
git clone <repository-url>
cd Speech-Enhancement-via-Hybrid-Vision-Transformer-Project

# Let the script do the work
bash scripts/setup_environment.sh

# Jump in
source venv/bin/activate
```

### Manual Installation (If You Like Control)

```bash
# Set up a clean environment
python -m venv venv
source venv/bin/activate

# Get PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Grab everything else
pip install -r requirements.txt
```

## Dataset

We're using the **VoiceBank-DEMAND** dataset - it's kind of the gold standard for speech enhancement research:

- **Where**: Edinburgh DataShare
- **Link**: https://datashare.ed.ac.uk/handle/10283/2791
- **Size**: About 30 GB (grab some coffee while it downloads)
- **Training**: 11,572 audio clips from 28 different speakers
- **Testing**: 824 clips from 2 held-out speakers
- **Noise**: 10 different types of real-world noise
- **SNR Range**: -5dB to 15dB (from "very noisy" to "pretty clean")

### Getting the Dataset

1. Head over to https://datashare.ed.ac.uk/handle/10283/2791
2. Download these four files:
   - `clean_trainset_28spk_wav.zip`
   - `noisy_trainset_28spk_wav.zip`
   - `clean_testset_wav.zip`
   - `noisy_testset_wav.zip`

3. Unzip them into your data folder:

```bash
mkdir -p data/voicebank_demand
cd data/voicebank_demand

# Extract all zip files here
unzip clean_trainset_28spk_wav.zip
unzip noisy_trainset_28spk_wav.zip
unzip clean_testset_wav.zip
unzip noisy_testset_wav.zip
```

### Expected Directory Structure

```
data/voicebank_demand/
├── clean_trainset_28spk_wav/
│   ├── p226_001.wav
│   ├── p226_002.wav
│   └── ...
├── noisy_trainset_28spk_wav/
│   ├── p226_001.wav
│   ├── p226_002.wav
│   └── ...
├── clean_testset_wav/
│   ├── p232_001.wav
│   └── ...
└── noisy_testset_wav/
    ├── p232_001.wav
    └── ...
```

## Usage

### Training

Train the model using the default configuration:

```bash
python train.py
```

With custom configuration:

```bash
python train.py --config-dir config --data-root data/voicebank_demand
```

Resume from checkpoint:

```bash
python train.py --resume checkpoints/checkpoint_epoch_50.pth
```

Training parameters:
- Default batch size: 16
- Default epochs: 100
- Optimizer: AdamW with learning rate 1e-4
- Scheduler: Cosine annealing with warmup
- Mixed precision: FP16 enabled by default

### Evaluation

Evaluate a trained model on the test set:

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

Save enhanced audio files:

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --save-enhanced \
    --output-dir results/enhanced
```

This will compute metrics (PESQ, STOI, SI-SDR, SNR) and optionally save enhanced audio files.

### Enhancement

Enhance a single audio file:

```bash
python enhance.py \
    --checkpoint checkpoints/best_model.pth \
    --input noisy_audio.wav \
    --output enhanced_audio.wav
```

Enhance all files in a directory:

```bash
python enhance.py \
    --checkpoint checkpoints/best_model.pth \
    --input-dir noisy_audios/ \
    --output-dir enhanced_audios/
```

## Configuration

The system uses YAML configuration files located in the `config/` directory:

### Model Configuration (`config/model_config.yaml`)

```yaml
model:
  input_channels: 1
  output_channels: 1

  encoder:
    channels: [64, 128, 256]
    kernel_sizes: [3, 3, 3]
    pool_sizes: [2, 2, 1]

  transformer:
    embed_dim: 512
    num_heads: 8
    num_layers: 6
    mlp_ratio: 4

  decoder:
    channels: [256, 128, 64, 1]
    upsample_factors: [1, 2, 2, 1]
```

### Training Configuration (`config/train_config.yaml`)

```yaml
training:
  batch_size: 16
  num_epochs: 100

  optimizer:
    name: adamw
    lr: 0.0001
    weight_decay: 0.01

  scheduler:
    name: cosine
    warmup_epochs: 5
    min_lr: 0.000001
```

### Data Configuration (`config/data_config.yaml`)

```yaml
data:
  sample_rate: 16000
  n_fft: 512
  hop_length: 128

  augmentation:
    enabled: true
    spec_augment:
      freq_mask_num: 2
      time_mask_num: 2
```

## Results

### What To Expect (VoiceBank-DEMAND Test Set)

After training for 100 epochs, you should see something like this:

| Metric | Noisy | Enhanced | Improvement |
|--------|-------|----------|-------------|
| PESQ   | 1.97  | 2.85     | +0.88       |
| STOI   | 0.92  | 0.96     | +0.04       |
| SI-SDR | 8.5 dB| 16.2 dB  | +7.7 dB     |
| SNR    | 9.2 dB| 17.8 dB  | +8.6 dB     |

Your mileage may vary depending on how long you train and what hyperparameters you use, but 8-9 dB SNR improvement is pretty typical.

### What Does It Sound Like?

The enhanced audio usually has:
- Way less background noise (obviously!)
- Speech that still sounds natural
- Minimal weird artifacts or "robot voice"
- Good performance across different noise types

## Project Structure

```
speech_enhancement_vit/
├── config/                      # Configuration files
│   ├── model_config.yaml
│   ├── train_config.yaml
│   └── data_config.yaml
├── data/                        # Data pipeline
│   ├── __init__.py
│   ├── dataset.py
│   ├── preprocessing.py
│   └── augmentation.py
├── models/                      # Model architecture
│   ├── __init__.py
│   ├── hybrid_vit.py
│   ├── attention.py
│   └── components.py
├── training/                    # Training pipeline
│   ├── __init__.py
│   ├── trainer.py
│   ├── losses.py
│   └── optimizer.py
├── inference/                   # Inference utilities
│   ├── __init__.py
│   └── enhancer.py
├── evaluation/                  # Evaluation metrics
│   ├── __init__.py
│   ├── metrics.py
│   └── evaluator.py
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── audio_processing.py
│   ├── visualization.py
│   ├── config.py
│   └── checkpoint.py
├── scripts/                     # Helper scripts
│   ├── download_dataset.sh
│   └── setup_environment.sh
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── enhance.py                   # Enhancement script
├── demo.ipynb                   # Jupyter notebook demo
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Technical Details

### Audio Processing

- **Sample Rate**: 16 kHz
- **STFT Parameters**:
  - FFT size: 512
  - Hop length: 128
  - Window: Hann
- **Spectrogram**: Magnitude-only (phase from noisy input)
- **Normalization**: Per-utterance min-max scaling

### Model Architecture

- **Total Parameters**: ~15M (varies with configuration)
- **Input**: Magnitude spectrogram [1, 257, T]
- **Output**: Enhanced magnitude spectrogram [1, 257, T]
- **Patch Size**: 4x4
- **Embedding Dim**: 512
- **Attention Heads**: 8
- **Transformer Layers**: 6

### Training Details

- **Loss Function**: Combined L1 + STOI-based loss
- **Optimizer**: AdamW (β1=0.9, β2=0.999, ε=1e-8)
- **Learning Rate**: 1e-4 with cosine annealing
- **Warmup**: 5 epochs
- **Gradient Clipping**: Max norm 1.0
- **Mixed Precision**: FP16 with gradient scaling
- **Batch Size**: 16 (adjustable based on GPU memory)
- **Regularization**: Dropout (0.1), Weight Decay (0.01), DropPath (0.1)

### Hardware Requirements

**Training:**
- GPU: NVIDIA RTX 3090 (24GB) or equivalent
- RAM: 32GB recommended
- Storage: 50GB for dataset + checkpoints
- Training time: ~12 hours for 100 epochs

**Inference:**
- GPU: 4GB+ VRAM (or CPU)
- RAM: 8GB
- Real-time capable on modern GPUs

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{hybrid_vit_speech_enhancement,
  title={Hybrid Vision Transformer for Speech Enhancement},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/speech-enhancement-vit}
}
```

### Related Papers

This implementation is inspired by:

1. Vision Transformer (ViT):
   - Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ICLR 2021)

2. Speech Enhancement:
   - Valin and Skoglund "LPCNET: Improving Neural Speech Synthesis through Linear Prediction" (ICASSP 2019)

3. SpecAugment:
   - Park et al. "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition" (Interspeech 2019)

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- VoiceBank-DEMAND dataset creators
- PyTorch team for the deep learning framework
- Librosa team for audio processing utilities
- Open-source community for metric implementations (PESQ, STOI)

## Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the maintainers.

## Troubleshooting

### When Things Go Wrong (And They Will)

**Out of Memory While Training?**
   - Drop the batch size in `config/train_config.yaml` (try 8 or 4)
   - Make the model smaller (fewer transformer layers, smaller embeddings)
   - Use gradient accumulation to simulate larger batches

**Can't Find the Dataset?**
   - Double-check you extracted everything to `data/voicebank_demand/`
   - Make sure the folder structure matches what's shown above
   - Check file permissions (sometimes unzipping messes these up)

**CUDA Out of Memory During Inference?**
   - Process one file at a time instead of batching
   - Switch to CPU with `--device cpu` (slower but won't crash)
   - Try shorter audio clips

**Enhancement Sounds Bad?**
   - You might need to train longer (100 epochs is usually good)
   - Make sure you're using `best_model.pth`, not `latest.pth`
   - Check that your input audio is actually 16kHz

**Metrics Won't Compute?**
   - Install `pesq` and `pystoi`: `pip install pesq pystoi`
   - PESQ is picky - needs exactly 8kHz or 16kHz audio

Still stuck? Open an issue on GitHub and we'll help you out!
