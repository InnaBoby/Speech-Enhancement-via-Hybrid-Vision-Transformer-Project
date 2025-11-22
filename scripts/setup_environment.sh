#!/bin/bash

# Environment Setup Script for Speech Enhancement Project

echo "================================================"
echo "Speech Enhancement Environment Setup"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "Python version $PYTHON_VERSION OK"
else
    echo "Error: Python 3.8 or higher is required"
    exit 1
fi
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (adjust for your CUDA version)
echo ""
echo "Installing PyTorch..."
echo "Please select PyTorch installation:"
echo "1) CUDA 11.8"
echo "2) CUDA 12.1"
echo "3) CPU only"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo "Installing PyTorch with CUDA 11.8..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    2)
        echo "Installing PyTorch with CUDA 12.1..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    3)
        echo "Installing PyTorch CPU-only version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ;;
    *)
        echo "Invalid choice. Installing CPU version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ;;
esac

# Install requirements
echo ""
echo "Installing project dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/voicebank_demand
mkdir -p checkpoints
mkdir -p logs
mkdir -p results

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. Download the VoiceBank-DEMAND dataset:"
echo "   bash scripts/download_dataset.sh"
echo ""
echo "2. Train the model:"
echo "   python train.py"
echo ""
echo "3. Evaluate the model:"
echo "   python evaluate.py --checkpoint checkpoints/best_model.pth"
echo ""
echo "4. Enhance audio:"
echo "   python enhance.py --checkpoint checkpoints/best_model.pth --input noisy.wav --output enhanced.wav"
echo ""
