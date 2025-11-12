"""
ONNX Export Pipeline for Battlesnake Neural Networks

This module handles exporting trained PyTorch models to ONNX format
for integration with Rust inference systems.
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import argparse
from datetime import datetime
import logging

from neural_networks import create_position_network, create_move_network, create_outcome_network
from board_encoding import BoardStateEncoder

class ONNXModelExporter:
    """Export PyTorch models to ONNX format"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # ONNX runtime providers
        self.providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            self.providers.append('CUDAExecutionProvider')
    
    def export_model(self, 
                    model_type: str,
                    model_path: str,
                    model_name: Optional[str] = None,
                    input_shape: Optional[Tuple[int, int, int]] = None,
                    export_dir: Optional[str] = None) -> str:
        """
        Export a PyTorch model to ONNX format
        
        Args:
            model_type: Type of model ('position_evaluation', 'move_prediction', 'game_outcome')
            model_path: Path to PyTorch model checkpoint
            model_name: Name for exported ONNX model (auto-generated if None)
            input_shape: Input tensor shape (channels, height, width)
            export_dir: Directory to save exported model
            
        Returns:
            Path to exported ONNX model
        """
        if export_dir is None:
            export_dir = self.model_dir / "onnx"
        else:
            export_dir = Path(export_dir)
        export_dir.mkdir(exist_ok=True)
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_type}_{timestamp}"
        
        onnx_path = export_dir / f"{model_name}.onnx"
        
        self.logger.info(f"Exporting {model_type} model to ONNX")
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        model_state = checkpoint['model_state_dict']
        
        # Create model architecture
        if model_type == 'position_evaluation':
            model = create_position_network()
        elif model_type == 'move_prediction':
            model = create_move_network()
        elif model_type == 'game_outcome':
            model = create_outcome_network()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights
        model.load_state_dict(model_state)
        model.eval()
        
        # Define input shapes
        if input_shape is None:
            # Default input shape: (7, 20, 20) for board + (6,) for features
            input_channels, input_height, input_width = 7, 20, 20
            feature_dim = 6
        else:
            input_channels, input_height, input_width = input_shape
            feature_dim = 6
        
        # Create dummy inputs for export
        grid = torch.randn(1, input_channels, input_height, input_width)
        features = torch.randn(1, feature_dim)
        
        # Dynamic axis information for batch dimension
        dynamic_axes = {
            'grid': {0: 'batch_size'},
            'features': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        
        # Export to ONNX
        torch.onnx.export(
            model,
            (grid, features),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['grid', 'features'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=True
        )
        
        # Verify exported model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        self.logger.info(f"Model exported successfully: {onnx_path}")
        
        # Save model metadata
        self._save_model_metadata(onnx_path, model_type, model_name, input_shape)
        
        # Test inference
        self._test_onnx_inference(onnx_path, input_shape)
        
        return str(onnx_path)
    
    def _save_model_metadata(self, 
                           onnx_path: Path,
                           model_type: str,
                           model_name: str,
                           input_shape: Optional[Tuple[int, int, int]]):
        """Save model metadata for Rust integration"""
        metadata = {
            'model_name': model_name,
            'model_type': model_type,
            'input_shape': input_shape or (7, 20, 20),
            'feature_dim': 6,
            'output_shape': self._get_output_shape(model_type),
            'exported_at': datetime.now().isoformat(),
            'onnx_version': onnx.__version__,
            'torch_version': torch.__version__
        }
        
        metadata_path = onnx_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model metadata saved: {metadata_path}")
    
    def _get_output_shape(self, model_type: str) -> List[int]:
        """Get output shape for each model type"""
        if model_type == 'position_evaluation':
            return [1]  # Single position score
        elif model_type == 'move_prediction':
            return [4]  # 4 move probabilities
        elif model_type == 'game_outcome':
            return [1]  # Win probability
        else:
            return [1]
    
    def _test_onnx_inference(self, onnx_path: Path, input_shape: Optional[Tuple[int, int, int]]):
        """Test ONNX model inference"""
        self.logger.info("Testing ONNX model inference...")
        
        # Create ONNX session
        session = ort.InferenceSession(str(onnx_path), providers=self.providers)
        
        # Create test input
        if input_shape is None:
            input_channels, input_height, input_width = 7, 20, 20
        else:
            input_channels, input_height, input_width = input_shape
        
        grid = np.random.randn(1, input_channels, input_height, input_width).astype(np.float32)
        features = np.random.randn(1, 6).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {'grid': grid, 'features': features})
        
        self.logger.info(f"ONNX inference test successful. Output shape: {outputs[0].shape}")
        session = None  # Clean up

class ONNXInferenceTester:
    """Test ONNX model inference and performance"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.session = None
        self.providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            self.providers.append('CUDAExecutionProvider')
    
    def load_model(self):
        """Load ONNX model for inference"""
        self.session = ort.InferenceSession(str(self.model_path), providers=self.providers)
        self.logger.info(f"Loaded ONNX model: {self.model_path}")
    
    def inference(self, grid: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Run inference on ONNX model"""
        if self.session is None:
            self.load_model()
        
        inputs = {
            'grid': grid.astype(np.float32),
            'features': features.astype(np.float32)
        }
        
        outputs = self.session.run(None, inputs)
        return outputs[0]
    
    def benchmark_inference(self, num_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference performance"""
        if self.session is None:
            self.load_model()
        
        # Create test data
        grid = np.random.randn(1, 7, 20, 20).astype(np.float32)
        features = np.random.randn(1, 6).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            self.inference(grid, features)
        
        # Benchmark
        import time
        
        start_time = time.time()
        for _ in range(num_runs):
            self.inference(grid, features)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        
        return {
            'total_time': total_time,
            'avg_time': avg_time,
            'inference_per_second': 1.0 / avg_time,
            'num_runs': num_runs
        }
    
    @property
    def logger(self):
        return logging.getLogger(__name__)

class ModelVersioningSystem:
    """Manage model versions and deployments"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.versions_file = self.model_dir / "model_versions.json"
        self._ensure_version_file()
    
    def _ensure_version_file(self):
        """Ensure version tracking file exists"""
        if not self.versions_file.exists():
            version_data = {
                'models': {},
                'current_versions': {},
                'created_at': datetime.now().isoformat()
            }
            with open(self.versions_file, 'w') as f:
                json.dump(version_data, f, indent=2)
    
    def register_model(self, 
                      model_name: str,
                      model_type: str,
                      model_path: str,
                      version: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Register a model in the versioning system"""
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load existing versions
        with open(self.versions_file, 'r') as f:
            version_data = json.load(f)
        
        # Register new model
        model_key = f"{model_type}_{version}"
        version_data['models'][model_key] = {
            'model_name': model_name,
            'model_type': model_type,
            'model_path': str(model_path),
            'version': version,
            'registered_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Update current version for this model type
        version_data['current_versions'][model_type] = {
            'model_key': model_key,
            'version': version
        }
        
        # Save updated versions
        with open(self.versions_file, 'w') as f:
            json.dump(version_data, f, indent=2)
        
        return model_key
    
    def get_current_model(self, model_type: str) -> Optional[str]:
        """Get current model for a given type"""
        with open(self.versions_file, 'r') as f:
            version_data = json.load(f)
        
        current = version_data['current_versions'].get(model_type)
        if current:
            model_key = current['model_key']
            return version_data['models'][model_key]['model_path']
        return None
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered models, optionally filtered by type"""
        with open(self.versions_file, 'r') as f:
            version_data = json.load(f)
        
        models = list(version_data['models'].values())
        
        if model_type:
            models = [m for m in models if m['model_type'] == model_type]
        
        return models

def export_all_models(pytorch_dir: str = "models", output_dir: str = "models/onnx"):
    """Export all available PyTorch models to ONNX"""
    
    exporter = ONNXModelExporter(pytorch_dir)
    versioning = ModelVersioningSystem(pytorch_dir)
    
    model_types = ['position_evaluation', 'move_prediction', 'game_outcome']
    
    for model_type in model_types:
        # Look for best model for this type
        model_files = list(Path(pytorch_dir).glob(f"best_{model_type}.pth"))
        
        if model_files:
            model_path = model_files[0]  # Use first (or best) model
            self.logger.info(f"Found {model_type} model: {model_path}")
            
            # Export to ONNX
            onnx_path = exporter.export_model(
                model_type=model_type,
                model_path=str(model_path),
                export_dir=output_dir
            )
            
            # Register in versioning system
            versioning.register_model(
                model_name=f"{model_type}_v1",
                model_type=model_type,
                model_path=onnx_path
            )
            
            print(f"Exported {model_type} model: {onnx_path}")
        else:
            print(f"No {model_type} model found to export")

def main():
    """Main export script"""
    parser = argparse.ArgumentParser(description='Export PyTorch models to ONNX')
    parser.add_argument('--model_type', type=str,
                       choices=['position_evaluation', 'move_prediction', 'game_outcome'],
                       help='Model type to export')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to PyTorch model checkpoint')
    parser.add_argument('--output_dir', type=str, default='models/onnx',
                       help='Directory to save ONNX model')
    parser.add_argument('--export_all', action='store_true',
                       help='Export all available models')
    
    args = parser.parse_args()
    
    if args.export_all:
        export_all_models(output_dir=args.output_dir)
    else:
        if not args.model_type:
            print("Please specify --model_type or use --export_all")
            return
        
        exporter = ONNXModelExporter()
        onnx_path = exporter.export_model(
            model_type=args.model_type,
            model_path=args.model_path,
            export_dir=args.output_dir
        )
        
        print(f"Model exported to: {onnx_path}")

if __name__ == "__main__":
    # Test ONNX export
    print("Testing ONNX export functionality...")
    
    # Create test models directory
    Path("models").mkdir(exist_ok=True)
    
    # Test with a dummy model
    test_model_path = "models/test_model.pth"
    
    # Create a dummy model for testing
    model = create_position_network()
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': 0,
        'loss': 0.5
    }, test_model_path)
    
    print(f"Created test model: {test_model_path}")
    
    # Export to ONNX
    exporter = ONNXModelExporter()
    onnx_path = exporter.export_model(
        model_type='position_evaluation',
        model_path=test_model_path
    )
    
    print(f"ONNX model exported: {onnx_path}")
    
    # Test inference
    tester = ONNXInferenceTester(onnx_path)
    results = tester.benchmark_inference(10)
    print(f"Inference benchmark: {results}")