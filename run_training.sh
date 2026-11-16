#!/bin/bash
# Script to run PyTorch ECG Model Training
echo "🚀 Starting ECG Hyperkalemia Model Training on Apple Silicon GPU..."
"/Users/shreybire/Desktop/Betic/ECG App/backend/.venv/bin/python" "/Users/shreybire/Desktop/Betic/ecgbyshrey2/main.py" --no_pretrained --epochs 15 --batch_size 32 --loss_type focal
