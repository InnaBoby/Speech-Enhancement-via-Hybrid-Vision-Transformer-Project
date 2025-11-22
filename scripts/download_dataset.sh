#!/bin/bash

# Download VoiceBank-DEMAND Dataset
# This script downloads and extracts the VoiceBank-DEMAND dataset for speech enhancement

echo "================================================"
echo "VoiceBank-DEMAND Dataset Download Script"
echo "================================================"
echo ""
echo "This script will download the VoiceBank-DEMAND dataset."
echo "Dataset size: ~30 GB"
echo "Source: https://datashare.ed.ac.uk/handle/10283/2791"
echo ""

# Set download directory
DATA_DIR=${1:-"data/voicebank_demand"}
echo "Download directory: $DATA_DIR"
echo ""

# Create directory
mkdir -p "$DATA_DIR"

# Dataset URLs
CLEAN_TRAINSET_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip"
NOISY_TRAINSET_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip"
CLEAN_TESTSET_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip"
NOISY_TESTSET_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip"

# Function to download and extract
download_and_extract() {
    local url=$1
    local filename=$(basename "$url")
    local output_path="$DATA_DIR/$filename"

    echo "Downloading $filename..."

    # Download with wget or curl
    if command -v wget &> /dev/null; then
        wget -O "$output_path" "$url"
    elif command -v curl &> /dev/null; then
        curl -L -o "$output_path" "$url"
    else
        echo "Error: Neither wget nor curl is installed."
        exit 1
    fi

    echo "Extracting $filename..."
    unzip -q "$output_path" -d "$DATA_DIR"

    echo "Removing zip file..."
    rm "$output_path"

    echo "Done with $filename"
    echo ""
}

# Download all files
echo "Starting download..."
echo ""

# Note: The actual URLs may require manual download from the website
# Please visit: https://datashare.ed.ac.uk/handle/10283/2791

echo "================================================"
echo "IMPORTANT NOTICE"
echo "================================================"
echo ""
echo "The VoiceBank-DEMAND dataset requires manual download from:"
echo "https://datashare.ed.ac.uk/handle/10283/2791"
echo ""
echo "Please download the following files manually:"
echo "1. clean_trainset_28spk_wav.zip"
echo "2. noisy_trainset_28spk_wav.zip"
echo "3. clean_testset_wav.zip"
echo "4. noisy_testset_wav.zip"
echo ""
echo "Then extract them to: $DATA_DIR"
echo ""
echo "Expected directory structure:"
echo "$DATA_DIR/"
echo "├── clean_trainset_28spk_wav/"
echo "├── noisy_trainset_28spk_wav/"
echo "├── clean_testset_wav/"
echo "└── noisy_testset_wav/"
echo ""
echo "================================================"

# Create directory structure info file
cat > "$DATA_DIR/README.txt" << EOF
VoiceBank-DEMAND Dataset

Please download from: https://datashare.ed.ac.uk/handle/10283/2791

Required files:
- clean_trainset_28spk_wav.zip
- noisy_trainset_28spk_wav.zip
- clean_testset_wav.zip
- noisy_testset_wav.zip

Extract all files to this directory.

Expected structure:
voicebank_demand/
├── clean_trainset_28spk_wav/
├── noisy_trainset_28spk_wav/
├── clean_testset_wav/
└── noisy_testset_wav/
EOF

echo "Created README in $DATA_DIR with download instructions."
