"""
Model Evolution System for Self-Play Training Pipeline

Manages the complete model evolution lifecycle through multi-phase training:
Bootstrap (Heuristic Supervision) -> Hybrid -> Self-Play -> Continuous Learning

Integrates with Phase 9 advanced neural networks (CNN + Attention + Residual) and
provides sophisticated model tournament, versioning, and performance tracking.
"""

import os
import json
import logging
import pickle
import gzip
import shutil
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import numpy as np
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Import neural network models and configuration
from neural_networks.neural_models import (
    MultiTaskBattlesnakeNetwork, ModelConfig, create_multitask_network,
    get_model_size, test_model_inference_speed
)
from config.self_play_config import get_config, TrainingPhaseConfig, ModelTournamentConfig
from self_play_data_manager import SelfPlayDataManager, SelfPlayTrainingSample


@dataclass
class ModelMetrics:
    """Comprehensive metrics for model performance tracking"""
    model_version: str
    training_phase: str
    created_at: str
    
    # Training metrics
    train_loss: float
    validation_loss: float
    position_accuracy: float
    move_accuracy: float
    outcome_accuracy: float
    
    # Performance metrics
    win_rate_vs_heuristic: float
    win_rate_vs_previous: float
    strategic_quality_score: float
    
    # Technical metrics
    inference_time_ms: float
    memory_usage_mb: float
    model_size_mb: float
    
    # Tournament results
    tournament_games_played: int = 0
    tournament_wins: int = 0
    elo_rating: float = 1500.0
    
    def win_rate(self) -> float:
        """Calculate overall tournament win rate"""
        if self.tournament_games_played == 0:
            return 0.0
        return self.tournament_wins / self.tournament_games_played


@dataclass
class TrainingResult:
    """Result of a training session"""
    model_version: str
    training_phase: str
    success: bool
    final_metrics: ModelMetrics
    training_time_minutes: float
    samples_processed: int
    error_message: Optional[str] = None


class SelfPlayDataset(Dataset):
    """PyTorch dataset for self-play training samples"""
    
    def __init__(self, samples: List[SelfPlayTrainingSample], augment: bool = True):
        self.samples = samples
        self.augment = augment
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors
        board_state = torch.FloatTensor(sample.board_state)
        snake_features = torch.FloatTensor(sample.snake_features)
        game_context = torch.FloatTensor(sample.game_context)
        
        # Target labels
        target_move = torch.LongTensor([sample.target_move])
        position_value = torch.FloatTensor([sample.position_value])
        move_probs = torch.FloatTensor(sample.move_probabilities)
        game_outcome = torch.FloatTensor([sample.game_outcome])
        
        # Data augmentation (board rotations/flips with move consistency)
        if self.augment and np.random.random() < 0.5:
            board_state, target_move, move_probs = self._augment_sample(
                board_state, target_move, move_probs
            )
        
        return {
            'board_state': board_state,
            'snake_features': snake_features,
            'game_context': game_context,
            'target_move': target_move.squeeze(),
            'position_value': position_value.squeeze(),
            'move_probabilities': move_probs,
            'game_outcome': game_outcome.squeeze(),
            'quality_score': sample.quality_score
        }
    
    def _augment_sample(self, board_state, target_move, move_probs):
        """Apply data augmentation with move consistency"""
        # Simple rotation (90 degrees)
        if np.random.random() < 0.5:
            board_state = torch.rot90(board_state, k=1, dims=[1, 2])
            # Rotate move: up->right, right->down, down->left, left->up
            move_rotation = {0: 3, 1: 2, 2: 0, 3: 1}  # up, down, left, right
            old_move = target_move.item()
            target_move = torch.LongTensor([move_rotation.get(old_move, old_move)])
            # Rotate move probabilities accordingly
            move_probs = move_probs[[move_rotation.get(i, i) for i in range(4)]]
        
        return board_state, target_move, move_probs


class ModelTournament:
    """Tournament system for model evaluation and selection"""
    
    def __init__(self, tournament_config: ModelTournamentConfig):
        self.config = tournament_config
        self.logger = logging.getLogger(__name__)
        
        # Tournament tracking
        self.tournament_results = {}
        self.elo_ratings = {}
        
    def evaluate_model(self, model_version: str, model_path: str,
                      baseline_models: Dict[str, str]) -> Dict[str, float]:
        """Evaluate model against baseline opponents"""
        self.logger.info(f"Starting tournament evaluation for {model_version}")
        
        results = {}
        total_games = 0
        total_wins = 0
        
        for opponent_name, opponent_path in baseline_models.items():
            self.logger.info(f"Playing against {opponent_name}")
            
            # TODO: Interface with Phase 8 data collection system to run actual games
            # This would involve:
            # 1. Loading both neural network models
            # 2. Running Battlesnake CLI with both endpoints
            # 3. Collecting game results
            
            # For now, simulate tournament results
            wins, total = self._simulate_tournament_games(
                model_version, opponent_name, self.config.evaluation_games
            )
            
            win_rate = wins / total if total > 0 else 0.0
            results[f"vs_{opponent_name}"] = win_rate
            
            total_games += total
            total_wins += wins
            
        # Calculate overall tournament performance
        overall_win_rate = total_wins / total_games if total_games > 0 else 0.0
        results["overall_win_rate"] = overall_win_rate
        
        # Update ELO rating
        self._update_elo_rating(model_version, results)
        
        self.logger.info(f"Tournament complete: {model_version} achieved {overall_win_rate:.1%} win rate")
        return results
    
    def _simulate_tournament_games(self, model_a: str, opponent: str, 
                                 num_games: int) -> Tuple[int, int]:
        """Simulate tournament games (replace with actual game execution)"""
        
        # Simulate results based on model progression expectations
        if "bootstrap" in model_a.lower():
            base_win_rate = 0.55  # Should beat random, struggle with heuristic
        elif "hybrid" in model_a.lower():
            base_win_rate = 0.65  # Better against heuristic
        elif "self_play" in model_a.lower():
            base_win_rate = 0.75  # Strong performance
        else:  # continuous
            base_win_rate = 0.80  # Peak performance
        
        # Adjust based on opponent
        if opponent == "random":
            win_rate = base_win_rate + 0.2
        elif opponent == "heuristic":
            win_rate = base_win_rate
        else:  # previous_champion
            win_rate = base_win_rate - 0.1
        
        # Add some randomness
        win_rate = np.clip(win_rate + np.random.normal(0, 0.05), 0.1, 0.9)
        
        wins = np.random.binomial(num_games, win_rate)
        return wins, num_games
    
    def _update_elo_rating(self, model_version: str, results: Dict[str, float]):
        """Update ELO rating based on tournament results"""
        if model_version not in self.elo_ratings:
            self.elo_ratings[model_version] = 1500.0
        
        # Simple ELO update based on overall win rate
        overall_win_rate = results.get("overall_win_rate", 0.5)
        expected_score = 0.5  # Expected win rate against average opponent
        
        k_factor = 32  # Standard ELO K-factor
        rating_change = k_factor * (overall_win_rate - expected_score)
        
        self.elo_ratings[model_version] += rating_change
        
    def meets_promotion_criteria(self, results: Dict[str, float], 
                                metrics: ModelMetrics) -> bool:
        """Check if model meets promotion criteria"""
        criteria = [
            results.get("overall_win_rate", 0.0) >= self.config.min_win_rate_vs_heuristic,
            metrics.inference_time_ms <= self.config.max_inference_time_ms,
            metrics.memory_usage_mb <= self.config.max_memory_usage_mb,
            metrics.strategic_quality_score >= self.config.min_strategic_quality_score
        ]
        
        return all(criteria)


class ModelEvolutionSystem:
    """Main system for managing model evolution through training phases"""
    
    def __init__(self, config_manager=None):
        self.config = get_config() if config_manager is None else config_manager.load_config()
        self.training_config = self.config.training_pipeline
        
        # Setup directories
        self.models_dir = Path("models")
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.tournaments_dir = self.models_dir / "tournaments"
        
        for dir_path in [self.models_dir, self.checkpoints_dir, self.tournaments_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_manager = SelfPlayDataManager(config_manager)
        self.tournament_system = ModelTournament(self.training_config.tournament)
        
        # Model tracking database
        self.db_path = self.models_dir / "model_tracking.db"
        self._init_database()
        
        # Current models registry
        self.current_models = {}
        self.model_metrics = {}
        
        # Training device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
    def _init_database(self):
        """Initialize model tracking database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY,
                    version TEXT UNIQUE,
                    training_phase TEXT,
                    created_at TEXT,
                    model_path TEXT,
                    config_json TEXT,
                    metrics_json TEXT,
                    is_champion BOOLEAN DEFAULT FALSE,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY,
                    model_version TEXT,
                    training_phase TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    success BOOLEAN,
                    samples_processed INTEGER,
                    final_loss REAL,
                    duration_minutes REAL,
                    error_message TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tournament_results (
                    id INTEGER PRIMARY KEY,
                    model_version TEXT,
                    opponent TEXT,
                    games_played INTEGER,
                    games_won INTEGER,
                    win_rate REAL,
                    elo_rating REAL,
                    timestamp TEXT
                )
            """)
    
    def bootstrap_training_phase(self, force_retrain: bool = False) -> TrainingResult:
        """Execute bootstrap training phase with heuristic supervision"""
        self.logger.info("Starting Bootstrap Training Phase")
        
        phase_config = self.training_config.get_phase_config("bootstrap")
        if not phase_config:
            raise ValueError("Bootstrap phase configuration not found")
        
        # Check if bootstrap model already exists
        if not force_retrain and self._model_exists("bootstrap"):
            self.logger.info("Bootstrap model already exists, skipping training")
            existing_metrics = self._load_model_metrics("bootstrap")
            return TrainingResult(
                model_version="bootstrap",
                training_phase="bootstrap",
                success=True,
                final_metrics=existing_metrics,
                training_time_minutes=0,
                samples_processed=0
            )
        
        # Collect heuristic supervision data
        self.logger.info(f"Collecting {phase_config.games_required} heuristic supervision games")
        samples_collected = self.data_manager.collect_heuristic_supervision_data(
            phase_config.games_required, "heuristic_v1.0"
        )
        
        if samples_collected < phase_config.games_required * 0.8:
            raise RuntimeError(f"Insufficient training data: {samples_collected} < {phase_config.games_required * 0.8}")
        
        # Create training batch
        batch_file = self.data_manager.process_training_batch(phase_config, samples_collected)
        if not batch_file:
            raise RuntimeError("Failed to create training batch")
        
        # Train model
        model_version = f"bootstrap_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = self._train_model(model_version, "bootstrap", batch_file, phase_config)
        
        if result.success:
            # Mark as current bootstrap model
            self._set_current_model("bootstrap", model_version)
            self.logger.info(f"Bootstrap training completed successfully: {model_version}")
        
        return result
    
    def hybrid_training_phase(self, force_retrain: bool = False) -> TrainingResult:
        """Execute hybrid training phase with mixed supervision"""
        self.logger.info("Starting Hybrid Training Phase")
        
        phase_config = self.training_config.get_phase_config("hybrid")
        if not phase_config:
            raise ValueError("Hybrid phase configuration not found")
        
        # Ensure bootstrap model exists
        if not self._model_exists("bootstrap"):
            bootstrap_result = self.bootstrap_training_phase()
            if not bootstrap_result.success:
                raise RuntimeError("Bootstrap training failed, cannot proceed to hybrid phase")
        
        # Check if hybrid model already exists
        if not force_retrain and self._model_exists("hybrid"):
            self.logger.info("Hybrid model already exists, skipping training")
            existing_metrics = self._load_model_metrics("hybrid")
            return TrainingResult(
                model_version="hybrid",
                training_phase="hybrid",
                success=True,
                final_metrics=existing_metrics,
                training_time_minutes=0,
                samples_processed=0
            )
        
        # Collect mixed training data
        heuristic_games = int(phase_config.games_required * phase_config.heuristic_mix_ratio)
        self_play_games = phase_config.games_required - heuristic_games
        
        self.logger.info(f"Collecting {heuristic_games} heuristic + {self_play_games} self-play games")
        
        heuristic_samples = self.data_manager.collect_heuristic_supervision_data(
            heuristic_games, "heuristic_v1.0"
        )
        
        bootstrap_model = self._get_current_model("bootstrap")
        self_play_samples = self.data_manager.collect_self_play_data(
            self_play_games, bootstrap_model, bootstrap_model
        )
        
        total_samples = heuristic_samples + self_play_samples
        if total_samples < phase_config.games_required * 0.8:
            raise RuntimeError(f"Insufficient mixed training data: {total_samples}")
        
        # Create training batch
        batch_file = self.data_manager.process_training_batch(phase_config, total_samples)
        if not batch_file:
            raise RuntimeError("Failed to create hybrid training batch")
        
        # Train model
        model_version = f"hybrid_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = self._train_model(model_version, "hybrid", batch_file, phase_config)
        
        if result.success:
            self._set_current_model("hybrid", model_version)
            self.logger.info(f"Hybrid training completed successfully: {model_version}")
        
        return result
    
    def self_play_training_phase(self, force_retrain: bool = False) -> TrainingResult:
        """Execute self-play training phase with neural vs neural games"""
        self.logger.info("Starting Self-Play Training Phase")
        
        phase_config = self.training_config.get_phase_config("self_play")
        if not phase_config:
            raise ValueError("Self-play phase configuration not found")
        
        # Ensure hybrid model exists
        if not self._model_exists("hybrid"):
            hybrid_result = self.hybrid_training_phase()
            if not hybrid_result.success:
                raise RuntimeError("Hybrid training failed, cannot proceed to self-play phase")
        
        # Check if self-play model already exists
        if not force_retrain and self._model_exists("self_play"):
            self.logger.info("Self-play model already exists, skipping training")
            existing_metrics = self._load_model_metrics("self_play")
            return TrainingResult(
                model_version="self_play",
                training_phase="self_play",
                success=True,
                final_metrics=existing_metrics,
                training_time_minutes=0,
                samples_processed=0
            )
        
        # Collect self-play training data
        hybrid_model = self._get_current_model("hybrid")
        self.logger.info(f"Collecting {phase_config.games_required} pure self-play games")
        
        samples_collected = self.data_manager.collect_self_play_data(
            phase_config.games_required, hybrid_model, hybrid_model
        )
        
        if samples_collected < phase_config.games_required * 0.8:
            raise RuntimeError(f"Insufficient self-play training data: {samples_collected}")
        
        # Create training batch
        batch_file = self.data_manager.process_training_batch(phase_config, samples_collected)
        if not batch_file:
            raise RuntimeError("Failed to create self-play training batch")
        
        # Train model
        model_version = f"self_play_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = self._train_model(model_version, "self_play", batch_file, phase_config)
        
        if result.success:
            # Run tournament to validate improvement
            tournament_results = self._run_tournament_evaluation(model_version, result.final_metrics)
            
            if self.tournament_system.meets_promotion_criteria(tournament_results, result.final_metrics):
                self._set_current_model("self_play", model_version)
                self._set_champion_model(model_version)
                self.logger.info(f"Self-play training completed and promoted to champion: {model_version}")
            else:
                self.logger.warning(f"Self-play model did not meet promotion criteria")
        
        return result
    
    def continuous_training_cycle(self) -> TrainingResult:
        """Execute continuous learning training cycle"""
        self.logger.info("Starting Continuous Training Cycle")
        
        phase_config = self.training_config.get_phase_config("continuous")
        if not phase_config:
            raise ValueError("Continuous phase configuration not found")
        
        # Get current champion model
        champion_model = self._get_champion_model()
        if not champion_model:
            # Fall back to self-play model
            if not self._model_exists("self_play"):
                result = self.self_play_training_phase()
                if not result.success:
                    raise RuntimeError("No champion model available and self-play training failed")
            champion_model = self._get_current_model("self_play")
        
        # Collect fresh self-play data
        fresh_games = self.training_config.minimum_new_games_for_retraining
        self.logger.info(f"Collecting {fresh_games} fresh self-play games for continuous learning")
        
        samples_collected = self.data_manager.collect_self_play_data(
            fresh_games, champion_model, champion_model
        )
        
        if samples_collected < fresh_games * 0.8:
            self.logger.warning(f"Insufficient fresh data for continuous learning: {samples_collected}")
            return TrainingResult(
                model_version="continuous",
                training_phase="continuous",
                success=False,
                final_metrics=self._load_model_metrics(champion_model),
                training_time_minutes=0,
                samples_processed=0,
                error_message="Insufficient fresh training data"
            )
        
        # Create training batch
        batch_file = self.data_manager.process_training_batch(phase_config, samples_collected)
        
        # Train new model
        model_version = f"continuous_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = self._train_model(model_version, "continuous", batch_file, phase_config)
        
        if result.success:
            # Tournament evaluation against current champion
            tournament_results = self._run_tournament_evaluation(model_version, result.final_metrics)
            
            if tournament_results.get("overall_win_rate", 0) > 0.52:  # Small improvement threshold
                self._set_champion_model(model_version)
                self.logger.info(f"Continuous learning model promoted to champion: {model_version}")
            else:
                self.logger.info(f"Continuous learning model did not improve enough for promotion")
        
        return result
    
    def _train_model(self, model_version: str, training_phase: str, 
                    batch_file: str, phase_config: TrainingPhaseConfig) -> TrainingResult:
        """Train neural network model"""
        self.logger.info(f"Training model {model_version} for {training_phase} phase")
        
        start_time = datetime.now()
        
        try:
            # Create model configuration
            model_config = ModelConfig(
                board_channels=12,  # Enhanced encoding
                board_size=11,
                snake_features_dim=32,
                game_context_dim=16
            )
            
            # Create model
            model = create_multitask_network(model_config).to(self.device)
            
            # Load training data
            training_data = self._load_training_batch(batch_file)
            train_dataset = SelfPlayDataset(training_data, augment=True)
            
            # Split into train/validation
            val_size = int(len(train_dataset) * phase_config.validation_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=phase_config.batch_size,
                shuffle=True, 
                num_workers=4,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=phase_config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            # Setup optimizer and loss functions
            optimizer = optim.AdamW(
                model.parameters(),
                lr=phase_config.learning_rate,
                weight_decay=self.training_config.neural_network.weight_decay
            )
            
            # Multi-task loss weights
            loss_weights = self.training_config.neural_network.loss_weights
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(phase_config.epochs):
                # Training phase
                model.train()
                train_loss = self._train_epoch(model, train_loader, optimizer, loss_weights)
                
                # Validation phase
                model.eval()
                val_loss, val_metrics = self._validate_epoch(model, val_loader, loss_weights)
                
                self.logger.info(
                    f"Epoch {epoch+1}/{phase_config.epochs}: "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Move Acc: {val_metrics['move_accuracy']:.3f}"
                )
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model checkpoint
                    checkpoint_path = self.checkpoints_dir / f"{model_version}_best.pth"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'model_config': asdict(model_config)
                    }, checkpoint_path)
                else:
                    patience_counter += 1
                    if patience_counter >= phase_config.early_stopping_patience:
                        self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
            
            # Load best model for final evaluation
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Final evaluation
            model.eval()
            final_loss, final_metrics = self._validate_epoch(model, val_loader, loss_weights)
            
            # Save final model
            model_path = self.models_dir / f"{model_version}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': asdict(model_config),
                'training_phase': training_phase,
                'final_metrics': final_metrics,
                'created_at': datetime.now().isoformat()
            }, model_path)
            
            # Calculate technical metrics
            param_count, model_size_mb = get_model_size(model)
            inference_time = test_model_inference_speed(model, model_config, 50)
            
            # Create model metrics
            metrics = ModelMetrics(
                model_version=model_version,
                training_phase=training_phase,
                created_at=datetime.now().isoformat(),
                train_loss=train_loss,
                validation_loss=final_loss,
                position_accuracy=final_metrics['position_accuracy'],
                move_accuracy=final_metrics['move_accuracy'],
                outcome_accuracy=final_metrics['outcome_accuracy'],
                win_rate_vs_heuristic=0.0,  # Will be updated by tournament
                win_rate_vs_previous=0.0,
                strategic_quality_score=final_metrics['move_accuracy'] * 0.7 + final_metrics['position_accuracy'] * 0.3,
                inference_time_ms=inference_time,
                memory_usage_mb=model_size_mb,
                model_size_mb=model_size_mb
            )
            
            # Store in database
            self._store_model_record(model_version, str(model_path), model_config, metrics)
            
            training_time = (datetime.now() - start_time).total_seconds() / 60
            
            return TrainingResult(
                model_version=model_version,
                training_phase=training_phase,
                success=True,
                final_metrics=metrics,
                training_time_minutes=training_time,
                samples_processed=len(training_data)
            )
            
        except Exception as e:
            self.logger.error(f"Training failed for {model_version}: {e}")
            training_time = (datetime.now() - start_time).total_seconds() / 60
            
            return TrainingResult(
                model_version=model_version,
                training_phase=training_phase,
                success=False,
                final_metrics=ModelMetrics(
                    model_version=model_version,
                    training_phase=training_phase,
                    created_at=datetime.now().isoformat(),
                    train_loss=float('inf'),
                    validation_loss=float('inf'),
                    position_accuracy=0.0,
                    move_accuracy=0.0,
                    outcome_accuracy=0.0,
                    win_rate_vs_heuristic=0.0,
                    win_rate_vs_previous=0.0,
                    strategic_quality_score=0.0,
                    inference_time_ms=0.0,
                    memory_usage_mb=0.0,
                    model_size_mb=0.0
                ),
                training_time_minutes=training_time,
                samples_processed=0,
                error_message=str(e)
            )
    
    def _train_epoch(self, model, train_loader, optimizer, loss_weights) -> float:
        """Train model for one epoch"""
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                batch['board_state'].to(self.device),
                batch['snake_features'].to(self.device),
                batch['game_context'].to(self.device)
            )
            
            # Multi-task loss calculation
            position_loss = F.mse_loss(outputs['position_value'], batch['position_value'].to(self.device))
            move_loss = F.cross_entropy(outputs['move_logits'], batch['target_move'].to(self.device))
            outcome_loss = F.binary_cross_entropy(outputs['outcome_probability'], batch['game_outcome'].to(self.device))
            
            total_loss_batch = (
                loss_weights['position_evaluation'] * position_loss +
                loss_weights['move_prediction'] * move_loss +
                loss_weights['game_outcome'] * outcome_loss
            )
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.training_config.neural_network.gradient_clip_norm)
            
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(self, model, val_loader, loss_weights) -> Tuple[float, Dict[str, float]]:
        """Validate model for one epoch"""
        total_loss = 0.0
        position_errors = []
        move_correct = 0
        outcome_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    batch['board_state'].to(self.device),
                    batch['snake_features'].to(self.device),
                    batch['game_context'].to(self.device)
                )
                
                # Loss calculation
                position_loss = F.mse_loss(outputs['position_value'], batch['position_value'].to(self.device))
                move_loss = F.cross_entropy(outputs['move_logits'], batch['target_move'].to(self.device))
                outcome_loss = F.binary_cross_entropy(outputs['outcome_probability'], batch['game_outcome'].to(self.device))
                
                total_loss_batch = (
                    loss_weights['position_evaluation'] * position_loss +
                    loss_weights['move_prediction'] * move_loss +
                    loss_weights['game_outcome'] * outcome_loss
                )
                
                total_loss += total_loss_batch.item()
                
                # Accuracy calculations
                position_errors.extend(torch.abs(outputs['position_value'] - batch['position_value'].to(self.device)).cpu().numpy())
                
                move_pred = torch.argmax(outputs['move_logits'], dim=1)
                move_correct += (move_pred == batch['target_move'].to(self.device)).sum().item()
                
                outcome_pred = (outputs['outcome_probability'] > 0.5).float()
                outcome_correct += (outcome_pred == batch['game_outcome'].to(self.device)).sum().item()
                
                total_samples += batch['board_state'].size(0)
        
        avg_loss = total_loss / len(val_loader)
        position_accuracy = 1.0 - (np.mean(position_errors) / 50.0)  # Normalize by max position value
        move_accuracy = move_correct / total_samples
        outcome_accuracy = outcome_correct / total_samples
        
        return avg_loss, {
            'position_accuracy': position_accuracy,
            'move_accuracy': move_accuracy,
            'outcome_accuracy': outcome_accuracy
        }
    
    def _load_training_batch(self, batch_file: str) -> List[SelfPlayTrainingSample]:
        """Load training batch from file"""
        with gzip.open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)
        return batch_data['samples']
    
    def _run_tournament_evaluation(self, model_version: str, metrics: ModelMetrics) -> Dict[str, float]:
        """Run tournament evaluation for model"""
        model_path = str(self.models_dir / f"{model_version}.pth")
        
        baseline_models = {
            "random": "random_baseline",
            "heuristic": "heuristic_baseline"
        }
        
        # Add previous champion if exists
        champion = self._get_champion_model()
        if champion and champion != model_version:
            baseline_models["previous_champion"] = champion
        
        results = self.tournament_system.evaluate_model(model_version, model_path, baseline_models)
        
        # Update metrics with tournament results
        metrics.win_rate_vs_heuristic = results.get("vs_heuristic", 0.0)
        metrics.win_rate_vs_previous = results.get("vs_previous_champion", 0.0)
        metrics.tournament_games_played = self.training_config.tournament.evaluation_games * len(baseline_models)
        metrics.tournament_wins = int(results.get("overall_win_rate", 0.0) * metrics.tournament_games_played)
        metrics.elo_rating = self.tournament_system.elo_ratings.get(model_version, 1500.0)
        
        return results
    
    def _store_model_record(self, model_version: str, model_path: str, 
                           config: ModelConfig, metrics: ModelMetrics):
        """Store model record in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO models 
                (version, training_phase, created_at, model_path, config_json, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                model_version,
                metrics.training_phase,
                metrics.created_at,
                model_path,
                json.dumps(asdict(config)),
                json.dumps(asdict(metrics))
            ))
    
    def _model_exists(self, phase_or_version: str) -> bool:
        """Check if model exists for given phase or version"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM models 
                WHERE version = ? OR training_phase = ?
            """, (phase_or_version, phase_or_version))
            
            count = cursor.fetchone()[0]
            return count > 0
    
    def _load_model_metrics(self, model_version: str) -> ModelMetrics:
        """Load model metrics from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT metrics_json FROM models WHERE version = ?
            """, (model_version,))
            
            result = cursor.fetchone()
            if result:
                metrics_data = json.loads(result[0])
                return ModelMetrics(**metrics_data)
            else:
                raise ValueError(f"Model {model_version} not found")
    
    def _get_current_model(self, training_phase: str) -> str:
        """Get current model for training phase"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT version FROM models 
                WHERE training_phase = ? AND is_active = 1
                ORDER BY created_at DESC LIMIT 1
            """, (training_phase,))
            
            result = cursor.fetchone()
            return result[0] if result else None
    
    def _set_current_model(self, training_phase: str, model_version: str):
        """Set current model for training phase"""
        with sqlite3.connect(self.db_path) as conn:
            # Deactivate previous models in this phase
            conn.execute("""
                UPDATE models SET is_active = 0 
                WHERE training_phase = ?
            """, (training_phase,))
            
            # Activate new model
            conn.execute("""
                UPDATE models SET is_active = 1 
                WHERE version = ?
            """, (model_version,))
    
    def _get_champion_model(self) -> Optional[str]:
        """Get current champion model"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT version FROM models 
                WHERE is_champion = 1 
                ORDER BY created_at DESC LIMIT 1
            """)
            
            result = cursor.fetchone()
            return result[0] if result else None
    
    def _set_champion_model(self, model_version: str):
        """Set champion model"""
        with sqlite3.connect(self.db_path) as conn:
            # Remove champion status from previous models
            conn.execute("UPDATE models SET is_champion = 0")
            
            # Set new champion
            conn.execute("""
                UPDATE models SET is_champion = 1 
                WHERE version = ?
            """, (model_version,))
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status"""
        status = {
            'phases_completed': [],
            'current_champion': self._get_champion_model(),
            'models_trained': 0,
            'latest_models': {}
        }
        
        with sqlite3.connect(self.db_path) as conn:
            # Count total models
            cursor = conn.execute("SELECT COUNT(*) FROM models")
            status['models_trained'] = cursor.fetchone()[0]
            
            # Get latest model for each phase
            for phase in ['bootstrap', 'hybrid', 'self_play', 'continuous']:
                cursor = conn.execute("""
                    SELECT version, created_at FROM models 
                    WHERE training_phase = ? 
                    ORDER BY created_at DESC LIMIT 1
                """, (phase,))
                
                result = cursor.fetchone()
                if result:
                    status['phases_completed'].append(phase)
                    status['latest_models'][phase] = {
                        'version': result[0],
                        'created_at': result[1]
                    }
        
        return status


# Convenience functions for pipeline integration
def train_all_phases(force_retrain: bool = False) -> Dict[str, TrainingResult]:
    """Train all phases in sequence"""
    evolution_system = ModelEvolutionSystem()
    results = {}
    
    # Bootstrap phase
    results['bootstrap'] = evolution_system.bootstrap_training_phase(force_retrain)
    if not results['bootstrap'].success:
        return results
    
    # Hybrid phase
    results['hybrid'] = evolution_system.hybrid_training_phase(force_retrain)
    if not results['hybrid'].success:
        return results
    
    # Self-play phase
    results['self_play'] = evolution_system.self_play_training_phase(force_retrain)
    
    return results


if __name__ == "__main__":
    # Test the model evolution system
    logging.basicConfig(level=logging.INFO)
    
    evolution_system = ModelEvolutionSystem()
    
    # Show current status
    status = evolution_system.get_evolution_status()
    print("=== Model Evolution System Status ===")
    print(json.dumps(status, indent=2, default=str))
    
    # Test bootstrap training
    print("\n=== Testing Bootstrap Training ===")
    result = evolution_system.bootstrap_training_phase(force_retrain=False)
    print(f"Bootstrap training result: {result.success}")
    
    if result.success:
        print(f"  Model version: {result.model_version}")
        print(f"  Training time: {result.training_time_minutes:.1f} minutes")
        print(f"  Samples processed: {result.samples_processed}")
        print(f"  Move accuracy: {result.final_metrics.move_accuracy:.3f}")
        print(f"  Inference time: {result.final_metrics.inference_time_ms:.1f} ms")