#!/bin/bash
set -e

echo "🚀 Starting setup for ETH-Bot on new VM..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 could not be found. Please install Python 3."
    exit 1
fi

echo "📦 Creating virtual environment..."
# Remove existing venv if it exists (fresh start)
if [ -d "venv" ]; then
    rm -rf venv
fi
python3 -m venv venv

# Activate venv
source venv/bin/activate

echo "🔄 Upgrading pip..."
pip install --upgrade pip

echo "🔍 Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected. Installing standard Torch (with CUDA)..."
    pip install -r requirements.txt
else
    echo "⚠️  No NVIDIA GPU detected. Installing CPU-only Torch to save space..."
    # Install CPU-specific torch first to avoid downloading full CUDA version
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    # Install other requirements
    pip install -r requirements.txt
fi

echo "✅ Setup complete! To start the bot:"
echo "   source venv/bin/activate"
echo "   python main.py"
