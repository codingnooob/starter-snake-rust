#!/usr/bin/env python3
"""
Enhanced Training Pipeline with Self-Play Integration

This pipeline replaces synthetic training data with real game data from the 
self-play automation system, implementing the progressive training approach
with 5000+ games and enhanced convergence monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import argparse
import json
from pathlib import Path
from datetime import datetime
import logging
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import threading
import time

# Import self-play components
from self_play_data_manager import SelfPlayDataManager, SelfPlayTrainingSample
from self_play_automation import SelfPlayAutomationManager

# Import existing neural network components
from neural_networks.board_encoding import BoardState, BoardStateEncoder, TrainingSample
from neural_networks.neural_networks import create_position_network, create_move_network, create_outcome_network

class RealGameDataset(Dataset):
    """PyTorch Dataset for real self-play game data"""
    
    def __init__(self, samples: List[SelfPlayTrainingSample], task: str = 'position_evaluation'):
        self.samples = samples
        self.task = task
        self.encoder = BoardStateEncoder()
        
        # Pre-encode all samples for faster training
        self.encoded_samples = []
        failed_samples = 0
        
        for sample in samples:
            try:
                # Convert SelfPlayTrainingSample to format expected by encoder
                board_state = self._convert_sample_to_board_state(sample)
                grid, features = self.encoder.normalize_input(board_state)
                
                self.encoded_samples.append({
                    'grid': grid,
                    'features': features,
                    'target': self._get_target_from_sample(sample, task)
                })
            except Exception as e:
                failed_samples += 1
                continue
        
        logging.info(f"Successfully encoded {len(self.encoded_samples)} samples, failed: {failed_samples}")
    
    def _convert_sample_to_board_state(self, sample: SelfPlayTrainingSample) -> BoardState:
        """Convert SelfPlayTrainingSample to BoardState for encoder"""
        # This would need to be implemented based on the exact structure
        # of SelfPlayTrainingSample vs BoardState
        # For now, assume they have compatible structure
        return sample.board_state
    
    def _get_target_from_sample(self, sample: SelfPlayTrainingSample, task: str):
        """Extract training target based on task type"""
        if task == 'position_evaluation':
            return [sample.position_value]
        elif task == 'move_prediction':
            return sample.move_probabilities
        elif task == 'game_outcome':
            return [sample.game_outcome]
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def __len__(self):
        return len(self.encoded_samples)
    
    def __getitem__(self, idx):
        sample = self.encoded_samples[idx]
        target = sample['target']
        
        # Normalize game_outcome targets for BCELoss (must be between 0 and 1)
        if len(target) == 1 and isinstance(target[0], float) and -1 <= target[0] <= 1:
            # Convert from [-1, 1] to [0, 1] for game outcomes
            target = [(target[0] + 1) / 2]
        
        target = torch.FloatTensor(target)
            
        return (
            torch.FloatTensor(sample['grid']),
            torch.FloatTensor(sample['features']),
            target
        )

class EnhancedTrainingMetrics:
    """Enhanced metrics tracking with convergence monitoring"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.learning_rates = []
        self.convergence_improvements = []
        
        # Enhanced tracking for 5000+ game approach
        self.batch_milestones = []  # Track at 1000, 2000, 5000+ games
        self.performance_history = []
        
    def log_milestone(self, games_trained: int, performance_metrics: Dict[str, float]):
        """Log performance at major milestones"""
        self.batch_milestones.append({
            'games_trained': games_trained,
            'timestamp': datetime.now().isoformat(),
            'metrics': performance_metrics
        })
        
    def calculate_convergence_improvement(self, baseline_performance: Dict[str, float]) -> Dict[str, float]:
        """Calculate improvement over baseline performance"""
        if not self.batch_milestones:
            return {}
        
        latest_metrics = self.batch_milestones[-1]['metrics']
        improvements = {}
        
        for metric_name, baseline_value in baseline_performance.items():
            if metric_name in latest_metrics:
                current_value = latest_metrics[metric_name]
                improvement = ((current_value - baseline_value) / baseline_value) * 100
                improvements[f"{metric_name}_improvement_percent"] = improvement
        
        return improvements
    
    def meets_convergence_targets(self, targets: Dict[str, float]) -> Dict[str, bool]:
        """Check if convergence targets are met"""
        if not self.convergence_improvements:
            return {metric: False for metric in targets.keys()}
        
        latest_improvements = self.convergence_improvements[-1]
        return {
            metric: latest_improvements.get(metric, 0) >= target_value
            for metric, target_value in targets.items()
        }

class SelfPlayNeuralNetworkTrainer:
    """Enhanced trainer integrated with self-play data collection"""
    
    def __init__(self, device: str = 'auto'):
        self.device = self._setup_device(device)
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.metrics = EnhancedTrainingMetrics()
        
        # Self-play integration
        self.data_manager = None
        self.automation_manager = None
        
        # Progressive training targets (5000+ games approach)
        self.progressive_targets = {
            'bootstrap': 2000,
            'hybrid': 5000, 
            'advanced': 15000
        }
        
        # Convergence targets (enhanced from user requirements)
        self.convergence_targets = {
            'move_prediction_improvement_percent': 60.0,  # >60% improvement
            'game_outcome_improvement_percent': 80.0,     # >80% improvement
            'position_evaluation_improvement_percent': 40.0
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🚀 Enhanced Self-Play Neural Network Trainer initialized")
        self.logger.info(f"🎯 Progressive targets: {self.progressive_targets}")
        self.logger.info(f"📊 Convergence targets: {self.convergence_targets}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup training device"""
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device.type == 'cuda':
                logging.info(f"🎮 Using GPU: {torch.cuda.get_device_name()}")
            else:
                logging.info("💻 Using CPU for training")
            return device
        return torch.device(device)
    
    def initialize_self_play_integration(self):
        """Initialize integration with self-play system"""
        try:
            self.data_manager = SelfPlayDataManager()
            self.automation_manager = SelfPlayAutomationManager()
            
            self.logger.info("✅ Self-play integration initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize self-play integration: {e}")
            return False
    
    def collect_training_data(self, target_games: int, phase_name: str = "training") -> List[SelfPlayTrainingSample]:
        """Collect real training data from self-play automation"""
        self.logger.info(f"🎮 Collecting {target_games} games for {phase_name} phase...")
        
        if not self.automation_manager:
            raise RuntimeError("Self-play automation not initialized")
        
        # Start automation if not already running
        if not hasattr(self.automation_manager, '_running') or not self.automation_manager._running:
            success = self.automation_manager.start()
            if not success:
                raise RuntimeError("Failed to start self-play automation")
        
        # Collect games in batches for better progress tracking
        batch_size = 250  # Collect in batches of 250 games
        collected_samples = []
        
        for batch_start in range(0, target_games, batch_size):
            batch_games = min(batch_size, target_games - batch_start)
            self.logger.info(f"📊 Collecting batch: games {batch_start + 1}-{batch_start + batch_games}")
            
            # Run batch of games
            batch_success = self.automation_manager.run_batch(batch_games)
            if not batch_success:
                self.logger.warning(f"⚠️ Batch collection had issues, continuing...")
            
            # Extract data from completed games
            batch_samples = self.data_manager.get_training_samples(limit=batch_games)
            collected_samples.extend(batch_samples)
            
            # Progress update
            progress = (batch_start + batch_games) / target_games * 100
            self.logger.info(f"📈 Collection progress: {progress:.1f}% ({len(collected_samples)} samples)")
        
        self.logger.info(f"✅ Completed data collection: {len(collected_samples)} training samples from {target_games} games")
        return collected_samples
    
    def progressive_training_pipeline(self, baseline_performance: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Execute complete progressive training pipeline"""
        self.logger.info("🚀 Starting Progressive Training Pipeline (5000+ Games)")
        
        if not self.initialize_self_play_integration():
            raise RuntimeError("Failed to initialize self-play system")
        
        results = {
            'phases_completed': [],
            'convergence_analysis': {},
            'final_models': {},
            'performance_improvements': {}
        }
        
        # Phase 1: Bootstrap Training (2000 games)
        self.logger.info("📋 Phase 1: Bootstrap Training (2000 games)")
        bootstrap_samples = self.collect_training_data(
            self.progressive_targets['bootstrap'], 
            "bootstrap"
        )
        
        bootstrap_results = self._train_all_models(
            bootstrap_samples,
            phase_name="bootstrap",
            enhanced_training=True
        )
        
        results['phases_completed'].append('bootstrap')
        self.metrics.log_milestone(2000, bootstrap_results)
        
        # Phase 2: Hybrid Training (5000 games total - additional 3000)
        self.logger.info("📋 Phase 2: Hybrid Training (+3000 games = 5000 total)")
        additional_samples = self.collect_training_data(3000, "hybrid_additional")
        
        hybrid_samples = bootstrap_samples + additional_samples
        hybrid_results = self._train_all_models(
            hybrid_samples,
            phase_name="hybrid", 
            enhanced_training=True,
            previous_results=bootstrap_results
        )
        
        results['phases_completed'].append('hybrid')
        self.metrics.log_milestone(5000, hybrid_results)
        
        # Check if we need Phase 3 based on convergence
        convergence_check = self._check_convergence_status(hybrid_results, baseline_performance)
        
        if not convergence_check['meets_all_targets']:
            # Phase 3: Advanced Training (15000 games total - additional 10000)
            self.logger.info("📋 Phase 3: Advanced Training (+10000 games = 15000 total)")
            self.logger.info("🎯 Convergence targets not yet met, continuing to advanced phase...")
            
            advanced_additional = self.collect_training_data(10000, "advanced_additional")
            all_samples = hybrid_samples + advanced_additional
            
            advanced_results = self._train_all_models(
                all_samples,
                phase_name="advanced",
                enhanced_training=True,
                previous_results=hybrid_results
            )
            
            results['phases_completed'].append('advanced')
            self.metrics.log_milestone(15000, advanced_results)
            
            final_results = advanced_results
        else:
            self.logger.info("🎉 Convergence targets met at 5000 games!")
            final_results = hybrid_results
        
        # Final analysis
        results['convergence_analysis'] = self._analyze_final_convergence(
            final_results, baseline_performance
        )
        results['final_models'] = self._export_final_models()
        results['performance_improvements'] = self.metrics.calculate_convergence_improvement(
            baseline_performance or {}
        )
        
        self._log_pipeline_completion(results)
        return results
    
    def _train_all_models(self, 
                         samples: List[SelfPlayTrainingSample], 
                         phase_name: str,
                         enhanced_training: bool = True,
                         previous_results: Optional[Dict] = None) -> Dict[str, Any]:
        """Train all three neural network models with enhanced parameters"""
        
        self.logger.info(f"🧠 Training all models for {phase_name} phase with {len(samples)} samples")
        
        # Enhanced training parameters for 5000+ games
        training_config = {
            'bootstrap': {'epochs': 150, 'batch_size': 64, 'val_split': 0.15},
            'hybrid': {'epochs': 200, 'batch_size': 128, 'val_split': 0.15}, 
            'advanced': {'epochs': 300, 'batch_size': 256, 'val_split': 0.15}
        }
        
        config = training_config.get(phase_name, training_config['hybrid'])
        
        results = {}
        
        # Train Position Evaluation Network
        self.logger.info("🎯 Training Position Evaluation Network...")
        position_results = self._train_single_model(
            'position_evaluation', samples, **config
        )
        results['position_evaluation'] = position_results
        
        # Train Move Prediction Network (CRITICAL for addressing 3.5% issue)
        self.logger.info("🎯 Training Move Prediction Network (Enhanced for convergence)...")
        move_config = config.copy()
        move_config['epochs'] = int(config['epochs'] * 1.5)  # Extra epochs for move prediction
        move_results = self._train_single_model(
            'move_prediction', samples, **move_config
        )
        results['move_prediction'] = move_results
        
        # Train Game Outcome Network (CRITICAL for addressing 2.1% issue)  
        self.logger.info("🎯 Training Game Outcome Network (Enhanced for convergence)...")
        outcome_config = config.copy()
        outcome_config['epochs'] = int(config['epochs'] * 1.2)  # Extra epochs for outcome prediction
        outcome_results = self._train_single_model(
            'game_outcome', samples, **outcome_config
        )
        results['game_outcome'] = outcome_results
        
        # Calculate improvements if baseline provided
        if previous_results:
            improvements = self._calculate_phase_improvements(results, previous_results)
            results['improvements'] = improvements
            self.logger.info(f"📈 Phase improvements: {improvements}")
        
        return results
    
    def _train_single_model(self, model_type: str, samples: List[SelfPlayTrainingSample], **kwargs) -> Dict[str, Any]:
        """Train a single model with enhanced convergence monitoring"""
        
        # Create model if needed
        if model_type not in self.models:
            self._create_enhanced_model(model_type)
        
        model = self.models[model_type]
        optimizer = self.optimizers[model_type]
        
        # Create dataset with real game data
        dataset = RealGameDataset(samples, model_type)
        
        if len(dataset) == 0:
            raise ValueError(f"No valid samples for {model_type} training")
        
        # Split data
        train_size = int(len(dataset) * (1 - kwargs.get('val_split', 0.15)))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=kwargs.get('batch_size', 64), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=kwargs.get('batch_size', 64), shuffle=False)
        
        # Training loop with enhanced monitoring
        return self._enhanced_training_loop(
            model, optimizer, train_loader, val_loader, 
            model_type, kwargs.get('epochs', 150)
        )
    
    def _create_enhanced_model(self, model_type: str):
        """Create enhanced model with better architecture"""
        if model_type == 'position_evaluation':
            self.models[model_type] = create_position_network().to(self.device)
            self.optimizers[model_type] = optim.AdamW(
                self.models[model_type].parameters(),
                lr=0.001, weight_decay=0.01
            )
        elif model_type == 'move_prediction':
            self.models[model_type] = create_move_network().to(self.device)
            self.optimizers[model_type] = optim.AdamW(
                self.models[model_type].parameters(),
                lr=0.0005, weight_decay=0.01  # Lower LR for stability
            )
        elif model_type == 'game_outcome':
            self.models[model_type] = create_outcome_network().to(self.device)
            self.optimizers[model_type] = optim.AdamW(
                self.models[model_type].parameters(),
                lr=0.0008, weight_decay=0.01
            )
        
        # Enhanced schedulers for better convergence
        self.schedulers[model_type] = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizers[model_type], T_0=10, T_mult=2
        )
    
    def _enhanced_training_loop(self, model, optimizer, train_loader, val_loader, model_type: str, epochs: int) -> Dict[str, Any]:
        """Enhanced training loop with convergence monitoring"""
        
        best_val_loss = float('inf')
        patience_counter = 0
        convergence_history = []
        
        # Loss function selection
        if model_type == 'position_evaluation':
            criterion = nn.MSELoss()
        elif model_type == 'move_prediction':
            criterion = nn.CrossEntropyLoss()
        else:  # game_outcome
            criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for grid, features, targets in train_loader:
                grid = grid.to(self.device)
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(grid, features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_accuracy = 0.0
            
            with torch.no_grad():
                for grid, features, targets in val_loader:
                    grid = grid.to(self.device)
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = model(grid, features)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    # Calculate accuracy based on task
                    if model_type == 'move_prediction':
                        predicted = torch.argmax(outputs, dim=1)
                        actual = torch.argmax(targets, dim=1)
                        val_accuracy += (predicted == actual).float().mean().item()
                    elif model_type == 'game_outcome':
                        predicted = (torch.sigmoid(outputs) > 0.5).float()
                        val_accuracy += (predicted == targets).float().mean().item()
            
            # Calculate averages
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_accuracy = val_accuracy / len(val_loader)
            
            # Update learning rate
            self.schedulers[model_type].step()
            
            # Log progress
            if epoch % 20 == 0:
                self.logger.info(
                    f"  {model_type} Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {avg_val_loss:.6f}, Val Acc: {avg_val_accuracy:.3f}"
                )
            
            # Early stopping with enhanced patience
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                model_path = Path("models") / f"enhanced_best_{model_type}.pth"
                model_path.parent.mkdir(exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': best_val_loss,
                    'accuracy': avg_val_accuracy
                }, model_path)
            else:
                patience_counter += 1
            
            # Enhanced early stopping (higher patience for better convergence)
            if patience_counter >= 30:  # Increased patience
                self.logger.info(f"  Early stopping {model_type} after {epoch + 1} epochs")
                break
            
            # Track convergence
            convergence_history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': avg_val_accuracy
            })
        
        # Calculate final performance metrics
        final_performance = {
            'final_val_loss': best_val_loss,
            'final_val_accuracy': convergence_history[-1]['val_accuracy'] if convergence_history else 0.0,
            'epochs_trained': len(convergence_history),
            'convergence_achieved': patience_counter < 30
        }
        
        self.logger.info(f"✅ {model_type} training completed: {final_performance}")
        return final_performance
    
    def _check_convergence_status(self, results: Dict[str, Any], baseline: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Check if convergence targets are met"""
        if not baseline:
            return {'meets_all_targets': False, 'reason': 'No baseline provided'}
        
        improvements = {}
        targets_met = {}
        
        for model_type in ['move_prediction', 'game_outcome', 'position_evaluation']:
            if model_type in results:
                current_acc = results[model_type].get('final_val_accuracy', 0.0)
                baseline_acc = baseline.get(f'{model_type}_accuracy', 0.0)
                
                if baseline_acc > 0:
                    improvement = ((current_acc - baseline_acc) / baseline_acc) * 100
                    improvements[f'{model_type}_improvement_percent'] = improvement
                    
                    target_key = f'{model_type}_improvement_percent'
                    if target_key in self.convergence_targets:
                        targets_met[model_type] = improvement >= self.convergence_targets[target_key]
                    else:
                        targets_met[model_type] = improvement >= 40.0  # Default target
        
        all_targets_met = all(targets_met.values()) if targets_met else False
        
        return {
            'meets_all_targets': all_targets_met,
            'improvements': improvements,
            'targets_met': targets_met,
            'convergence_targets': self.convergence_targets
        }
    
    def _analyze_final_convergence(self, results: Dict[str, Any], baseline: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze final convergence and performance"""
        convergence_analysis = {
            'training_completed': True,
            'convergence_status': self._check_convergence_status(results, baseline),
            'model_performance': {},
            'recommendations': []
        }
        
        # Analyze each model
        for model_type, model_results in results.items():
            if isinstance(model_results, dict) and 'final_val_accuracy' in model_results:
                accuracy = model_results['final_val_accuracy']
                convergence_analysis['model_performance'][model_type] = {
                    'accuracy': accuracy,
                    'status': 'excellent' if accuracy > 0.8 else 'good' if accuracy > 0.6 else 'needs_improvement'
                }
        
        # Generate recommendations
        convergence_status = convergence_analysis['convergence_status']
        if not convergence_status['meets_all_targets']:
            convergence_analysis['recommendations'].append("Consider additional training data or architecture improvements")
        else:
            convergence_analysis['recommendations'].append("Models ready for production deployment")
        
        return convergence_analysis
    
    def _export_final_models(self) -> Dict[str, str]:
        """Export final models to ONNX format"""
        exported_models = {}
        
        for model_type, model in self.models.items():
            try:
                # Create dummy input for ONNX export
                dummy_grid = torch.randn(1, 12, 11, 11).to(self.device)  # 12-channel board
                dummy_features = torch.randn(1, 32).to(self.device)  # 32 features
                
                onnx_path = f"models/enhanced_{model_type}.onnx"
                torch.onnx.export(
                    model,
                    (dummy_grid, dummy_features),
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['board_grid', 'board_features'],
                    output_names=['prediction'],
                    dynamic_axes={'board_grid': {0: 'batch_size'}, 'board_features': {0: 'batch_size'}}
                )
                
                exported_models[model_type] = onnx_path
                self.logger.info(f"✅ Exported {model_type} to {onnx_path}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to export {model_type}: {e}")
                exported_models[model_type] = f"export_failed: {e}"
        
        return exported_models
    
    def _calculate_phase_improvements(self, current_results: Dict, previous_results: Dict) -> Dict[str, float]:
        """Calculate improvements between training phases"""
        improvements = {}
        
        for model_type in current_results:
            if model_type in previous_results:
                current_acc = current_results[model_type].get('final_val_accuracy', 0.0)
                previous_acc = previous_results[model_type].get('final_val_accuracy', 0.0)
                
                if previous_acc > 0:
                    improvement = ((current_acc - previous_acc) / previous_acc) * 100
                    improvements[f'{model_type}_phase_improvement'] = improvement
        
        return improvements
    
    def _log_pipeline_completion(self, results: Dict[str, Any]):
        """Log comprehensive pipeline completion results"""
        self.logger.info("=" * 80)
        self.logger.info("🎉 PROGRESSIVE TRAINING PIPELINE COMPLETED!")
        self.logger.info("=" * 80)
        
        phases_completed = results.get('phases_completed', [])
        self.logger.info(f"📋 Phases completed: {', '.join(phases_completed)}")
        
        convergence = results.get('convergence_analysis', {})
        if convergence.get('convergence_status', {}).get('meets_all_targets', False):
            self.logger.info("🎯 ✅ ALL CONVERGENCE TARGETS MET!")
        else:
            self.logger.info("🎯 ⚠️  Some convergence targets not yet met")
        
        improvements = results.get('performance_improvements', {})
        for metric, improvement in improvements.items():
            if 'improvement_percent' in metric:
                status = "✅" if improvement >= 40 else "⚠️"
                self.logger.info(f"📈 {status} {metric}: {improvement:.1f}%")
        
        models = results.get('final_models', {})
        self.logger.info(f"🔄 Exported models: {len([m for m in models.values() if 'export_failed' not in m])}/{len(models)}")
        
        self.logger.info("=" * 80)

def main():
    """Main function for enhanced training pipeline"""
    parser = argparse.ArgumentParser(description='Enhanced Self-Play Neural Network Training')
    parser.add_argument('--mode', type=str, default='progressive',
                       choices=['progressive', 'single_phase', 'test'],
                       help='Training mode')
    parser.add_argument('--target_games', type=int, default=5000,
                       help='Target number of games for data collection')
    parser.add_argument('--device', type=str, default='auto',
                       help='Training device (auto, cpu, cuda)')
    parser.add_argument('--baseline_file', type=str, default=None,
                       help='Baseline performance JSON file')
    
    args = parser.parse_args()
    
    # Load baseline performance if provided
    baseline_performance = None
    if args.baseline_file and Path(args.baseline_file).exists():
        with open(args.baseline_file, 'r') as f:
            baseline_performance = json.load(f)
    
    # Create trainer
    trainer = SelfPlayNeuralNetworkTrainer(args.device)
    
    if args.mode == 'progressive':
        # Run complete progressive training pipeline
        results = trainer.progressive_training_pipeline(baseline_performance)
        
        # Save results
        results_file = f"enhanced_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Progressive training completed! Results saved to {results_file}")
        
        # Check if targets were met
        convergence = results.get('convergence_analysis', {})
        if convergence.get('convergence_status', {}).get('meets_all_targets', False):
            print("🎉 ALL CONVERGENCE TARGETS MET! Neural networks ready for Phase 4 RL.")
        else:
            print("⚠️  Some targets not met. Consider additional training or architecture improvements.")
            
    elif args.mode == 'test':
        # Test mode with smaller dataset
        print("🧪 Running test mode with limited data...")
        if trainer.initialize_self_play_integration():
            test_samples = trainer.collect_training_data(100, "test")
            print(f"✅ Test data collection successful: {len(test_samples)} samples")
        else:
            print("❌ Test mode failed - could not initialize self-play integration")

if __name__ == "__main__":
    main()