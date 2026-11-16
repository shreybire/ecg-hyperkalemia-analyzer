#!/bin/bash
# Script to predict on an ECG image (default: image.png or custom path argument)
TARGET_IMAGE="${1:-image.png}"
echo "🔮 Running Prediction & Grad-CAM on: $TARGET_IMAGE"
"/Users/shreybire/Desktop/Betic/ECG App/backend/.venv/bin/python" "/Users/shreybire/Desktop/Betic/ecgbyshrey2/predict.py" --image "$TARGET_IMAGE" --checkpoint "/Users/shreybire/Desktop/Betic/ecgbyshrey2/best_ecg_model.pth" --output "/Users/shreybire/Desktop/Betic/ecgbyshrey2/results/prediction_result.png"
