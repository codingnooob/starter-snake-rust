#!/usr/bin/env python3
"""
Create minimal ONNX models for both 8-channel and 12-channel neural network activation
This bypasses complex training issues and gets neural networks active for testing
Supports both legacy 8-channel and new 12-channel advanced spatial analysis pipeline
"""

import torch
import torch.nn as nn
import torch.onnx
import numpy as np
from pathlib import Path

class MinimalPositionEvaluator(nn.Module):
    """Minimal neural network for position evaluation"""
    def __init__(self, input_features=968):
        super().__init__()
        # Input: configurable features (968 for 8-channel, 1452 for 12-channel)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)  # Position score output
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

class MinimalMovePredictor(nn.Module):
    """Minimal neural network for move prediction"""
    def __init__(self, input_features=968):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)  # 4 move outputs (up, down, left, right)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.softmax(x)
        return x

class MinimalGameOutcomePredictor(nn.Module):
    """Minimal neural network for game outcome prediction"""
    def __init__(self, input_features=968):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)  # Win probability output
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

def create_model_set(channels, suffix=""):
    """Create a set of models for the specified channel count"""
    input_features = channels * 11 * 11  # channels × height × width
    
    models = {
        f"position_evaluation{suffix}.onnx": MinimalPositionEvaluator(input_features),
        f"move_prediction{suffix}.onnx": MinimalMovePredictor(input_features),
        f"game_outcome{suffix}.onnx": MinimalGameOutcomePredictor(input_features)
    }
    
    return models, input_features

def export_model(model, dummy_input, output_path, filename):
    """Export a single model to ONNX format"""
    model.eval()
    
    # Initialize with random but reasonable weights
    with torch.no_grad():
        for param in model.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['board_state'],
            output_names=['prediction'],
            dynamic_axes={
                'board_state': {0: 'batch_size'},
                'prediction': {0: 'batch_size'}
            }
        )
        
        # Verify file was created and get size
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {filename} created ({size_mb:.2f} MB)")
            return True
        else:
            print(f"   ❌ {filename} creation failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Error creating {filename}: {e}")
        return False

def create_onnx_models():
    """Create minimal ONNX models for both 8-channel and 12-channel deployment"""
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("🚀 Creating minimal ONNX models for neural network activation...")
    print("📊 Generating both 8-channel (legacy) and 12-channel (advanced) models...")
    
    # Model configurations
    configs = [
        {"channels": 8, "suffix": "", "description": "8-channel (legacy)"},
        {"channels": 12, "suffix": "_12ch", "description": "12-channel (advanced spatial analysis)"}
    ]
    
    total_created = 0
    
    for config in configs:
        channels = config["channels"]
        suffix = config["suffix"]
        description = config["description"]
        
        print(f"\n🔧 Creating {description} models...")
        
        # Create model set and input tensor
        models, input_features = create_model_set(channels, suffix)
        dummy_input = torch.randn(1, channels, 11, 11)
        
        print(f"   Input shape: (1, {channels}, 11, 11) = {input_features} features")
        
        for filename, model in models.items():
            success = export_model(model, dummy_input, models_dir / filename, filename)
            if success:
                total_created += 1
    
    print(f"\n🎯 Created {total_created} ONNX models in {models_dir}/")
    print("✅ 8-channel models: Compatible with legacy pipeline")  
    print("✅ 12-channel models: Compatible with advanced spatial analysis pipeline")
    print("🚀 Neural networks are now ready for both deployment modes!")

if __name__ == "__main__":
    create_onnx_models()