"""
Training Pipeline for Battlesnake Neural Networks

This module provides training scripts for the three neural network types:
1. Position Evaluation Network
2. Move Prediction Network  
3. Game Outcome Network
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Any
import argparse
import json
from pathlib import Path
from datetime import datetime
import logging
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from board_encoding import BoardState, BoardStateEncoder, TrainingSample
from neural_networks import create_position_network, create_move_network, create_outcome_network

class BattlesnakeDataset(Dataset):
    """PyTorch Dataset for Battlesnake training samples"""
    
    def __init__(self, samples: List[TrainingSample], task: str = 'position_evaluation'):
        self.samples = samples
        self.task = task
        self.encoder = BoardStateEncoder()
        
        # Pre-encode all samples for faster training
        self.encoded_samples = []
        for sample in samples:
            try:
                grid, features = self.encoder.normalize_input(sample.board_state)
                self.encoded_samples.append({
                    'grid': grid,
                    'features': features,
                    'target': sample.to_training_label(task)
                })
            except Exception as e:
                # Skip samples that can't be encoded
                continue
    
    def __len__(self):
        return len(self.encoded_samples)
    
    def __getitem__(self, idx):
        sample = self.encoded_samples[idx]
        target = sample['target']
        
        # Normalize game_outcome targets for BCELoss (must be between 0 and 1)
        if self.task == 'game_outcome':
            if isinstance(target, str):
                # Convert string outcomes to normalized floats
                outcome_map = {'loss': 0.0, 'draw': 0.5, 'win': 1.0}
                target = [outcome_map.get(target, 0.5)]  # Default to draw if unknown
            elif isinstance(target, (int, float)):
                # Convert numeric outcomes (-1, 0, 1) to (0, 0.5, 1)
                target = [(target + 1) / 2]
            target = torch.FloatTensor(target)
        else:
            target = torch.FloatTensor(target)
            
        return (
            torch.FloatTensor(sample['grid']),
            torch.FloatTensor(sample['features']),
            target
        )

class TrainingMetrics:
    """Track training metrics and progress"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.learning_rates = []
        
    def log_train_batch(self, loss: float, metric: float = None, lr: float = None):
        self.train_losses.append(loss)
        if metric is not None:
            self.train_metrics.append(metric)
        if lr is not None:
            self.learning_rates.append(lr)
    
    def log_validation(self, loss: float, metric: float = None):
        self.val_losses.append(loss)
        if metric is not None:
            self.val_metrics.append(metric)
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_epochs': len(self.train_losses),
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'best_train_loss': min(self.train_losses) if self.train_losses else None,
            'best_val_loss': min(self.val_losses) if self.val_losses else None,
            'total_batches': len(self.train_losses)
        }

class NeuralNetworkTrainer:
    """Unified trainer for all neural network types"""
    
    def __init__(self, device: str = 'auto'):
        self.device = self._setup_device(device)
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.metrics = TrainingMetrics()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup training device"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def create_model(self, model_type: str, **kwargs):
        """Create and configure model"""
        if model_type == 'position_evaluation':
            self.models[model_type] = create_position_network(**kwargs).to(self.device)
            self.optimizers[model_type] = optim.Adam(
                self.models[model_type].parameters(), 
                lr=0.001,
                weight_decay=1e-5
            )
            self.schedulers[model_type] = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers[model_type],
                mode='min',
                factor=0.5,
                patience=10
            )
            
        elif model_type == 'move_prediction':
            self.models[model_type] = create_move_network(**kwargs).to(self.device)
            self.optimizers[model_type] = optim.Adam(
                self.models[model_type].parameters(),
                lr=0.001,
                weight_decay=1e-5
            )
            self.schedulers[model_type] = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers[model_type],
                mode='min',
                factor=0.5,
                patience=10
            )
            
        elif model_type == 'game_outcome':
            self.models[model_type] = create_outcome_network(**kwargs).to(self.device)
            self.optimizers[model_type] = optim.Adam(
                self.models[model_type].parameters(),
                lr=0.001,
                weight_decay=1e-5
            )
            self.schedulers[model_type] = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers[model_type],
                mode='min',
                factor=0.5,
                patience=10
            )
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_model(self, 
                   model_type: str,
                   samples: List[TrainingSample],
                   epochs: int = 100,
                   batch_size: int = 32,
                   val_split: float = 0.2,
                   early_stopping_patience: int = 15,
                   save_dir: str = "models") -> Dict[str, Any]:
        """
        Train a neural network model
        
        Args:
            model_type: Type of model ('position_evaluation', 'move_prediction', 'game_outcome')
            samples: Training samples
            epochs: Number of training epochs
            batch_size: Training batch size
            val_split: Validation split ratio
            early_stopping_patience: Early stopping patience
            save_dir: Directory to save model checkpoints
            
        Returns:
            Training metrics dictionary
        """
        self.logger.info(f"Starting training for {model_type}")
        
        # Create model if it doesn't exist
        if model_type not in self.models:
            self.create_model(model_type)
        
        model = self.models[model_type]
        optimizer = self.optimizers[model_type]
        
        # Split data
        train_samples, val_samples = train_test_split(
            samples, 
            test_size=val_split, 
            random_state=42
        )
        
        self.logger.info(f"Training samples: {len(train_samples)}, Validation samples: {len(val_samples)}")
        
        # Create datasets and loaders
        train_dataset = BattlesnakeDataset(train_samples, model_type)
        val_dataset = BattlesnakeDataset(val_samples, model_type)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss function based on model type
        if model_type == 'position_evaluation':
            criterion = nn.MSELoss()
            metric_fn = self._position_metric
        elif model_type == 'move_prediction':
            criterion = nn.CrossEntropyLoss()
            metric_fn = self._move_metric
        elif model_type == 'game_outcome':
            criterion = nn.BCEWithLogitsLoss()
            metric_fn = self._outcome_metric
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_metric = 0.0
            
            for batch_idx, (grid, features, targets) in enumerate(train_loader):
                grid = grid.to(self.device)
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(grid, features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                if metric_fn:
                    train_metric += metric_fn(outputs, targets)
                
                # Log progress
                if batch_idx % 10 == 0:
                    self.logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_metric = 0.0
            
            with torch.no_grad():
                for grid, features, targets in val_loader:
                    grid = grid.to(self.device)
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = model(grid, features)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    if metric_fn:
                        val_metric += metric_fn(outputs, targets)
            
            # Calculate averages
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_metric = train_metric / len(train_loader) if train_metric > 0 else 0
            avg_val_metric = val_metric / len(val_loader) if val_metric > 0 else 0
            
            # Update learning rate
            self.schedulers[model_type].step(avg_val_loss)
            
            # Log metrics
            current_lr = optimizer.param_groups[0]['lr']
            self.logger.info(
                f'Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, '
                f'Val Loss: {avg_val_loss:.6f}, '
                f'LR: {current_lr:.6f}'
            )
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                model_path = save_dir / f"best_{model_type}.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': best_val_loss,
                    'model_config': model.__dict__
                }, model_path)
                
                self.logger.info(f"Saved best model to {model_path}")
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        self.logger.info(f"Training completed for {model_type}")
        return self.metrics.get_summary()
    
    def _position_metric(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate position evaluation metric"""
        # Pearson correlation coefficient
        with torch.no_grad():
            outputs_np = outputs.cpu().numpy().flatten()
            targets_np = targets.cpu().numpy().flatten()
            if len(outputs_np) > 1:
                corr = np.corrcoef(outputs_np, targets_np)[0, 1]
                return corr if not np.isnan(corr) else 0.0
        return 0.0
    
    def _move_metric(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate move prediction accuracy"""
        with torch.no_grad():
            predicted = torch.argmax(outputs, dim=1)
            actual = torch.argmax(targets, dim=1)
            accuracy = (predicted == actual).float().mean().item()
        return accuracy
    
    def _outcome_metric(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate game outcome accuracy"""
        with torch.no_grad():
            predicted = (outputs > 0.5).float()
            accuracy = (predicted == targets).float().mean().item()
        return accuracy

def train_position_network(samples: List[TrainingSample], **kwargs) -> Dict[str, Any]:
    """Train position evaluation network"""
    trainer = NeuralNetworkTrainer()
    return trainer.train_model('position_evaluation', samples, **kwargs)

def train_move_network(samples: List[TrainingSample], **kwargs) -> Dict[str, Any]:
    """Train move prediction network"""
    trainer = NeuralNetworkTrainer()
    return trainer.train_model('move_prediction', samples, **kwargs)

def train_outcome_network(samples: List[TrainingSample], **kwargs) -> Dict[str, Any]:
    """Train game outcome network"""
    trainer = NeuralNetworkTrainer()
    return trainer.train_model('game_outcome', samples, **kwargs)

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train Battlesnake Neural Networks')
    parser.add_argument('--model_type', type=str, default='position_evaluation',
                       choices=['position_evaluation', 'move_prediction', 'game_outcome'])
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to training data file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--device', type=str, default='auto',
                       help='Training device (auto, cpu, cuda)')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    # Generate fresh training data to avoid compatibility issues
    print("Generating training data with simulated game collection...")
    from data_collection import simulate_game_collection
    collector, samples = simulate_game_collection()
    print(f"Generated {len(samples)} training samples")
    
    # Train model
    trainer = NeuralNetworkTrainer(args.device)
    
    if args.model_type == 'position_evaluation':
        results = trainer.train_model(
            'position_evaluation', samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.val_split,
            save_dir=args.save_dir
        )
    elif args.model_type == 'move_prediction':
        results = trainer.train_model(
            'move_prediction', samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.val_split,
            save_dir=args.save_dir
        )
    elif args.model_type == 'game_outcome':
        results = trainer.train_model(
            'game_outcome', samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.val_split,
            save_dir=args.save_dir
        )
    
    print("\nTraining completed!")
    print("Results:", results)

if __name__ == "__main__":
    # Check if command line arguments are provided
    import sys
    if len(sys.argv) > 1:
        # Run main function with command line arguments
        main()
    else:
        # Test training with simulated data
        print("Testing training pipeline...")
        
        # Create test samples
        from data_collection import simulate_game_collection
        collector, samples = simulate_game_collection()
        
        print(f"Training with {len(samples)} samples...")
        
        # Test position network training (small epochs for testing)
        results = train_position_network(
            samples,
            epochs=5,
            batch_size=8,
            val_split=0.3
        )
        
        print("Position training results:", results)