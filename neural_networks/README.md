# Phase 3: Neural Network Integration for Battlesnake AI

This directory contains the PyTorch-based neural network training pipeline and ONNX model management for the Battlesnake AI system.

## Architecture Overview

### Neural Network Types
1. **Position Evaluation Network** - Evaluates board positions for strategic value
2. **Move Prediction Network** - Predicts move probability distributions
3. **Game Outcome Network** - Predicts win probability from board state

### Training Pipeline Components
- Board state encoding and preprocessing
- Data collection from gameplay logs
- Training with supervised learning
- ONNX export for Rust integration
- Model validation and A/B testing

## Setup

```bash
# Install Python dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime numpy pandas matplotlib

# Install development dependencies
pip install pytest black flake8 jupyter
```

## Usage

### Training
```bash
# Train position evaluation network
python train_position_net.py

# Train move prediction network  
python train_move_net.py

# Train game outcome network
python train_outcome_net.py
```

### Data Collection
```bash
# Record gameplay data
python collect_data.py --mode training --games 1000

# Process and format data
python process_data.py --input raw_games/ --output processed/
```

### Model Export
```bash
# Export trained models to ONNX
python export_models.py --model position_net.pth --output models/
python export_models.py --model move_net.pth --output models/
python export_models.py --model outcome_net.pth --output models/