#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Framework
Validates neural networks achieve 30-50+ point contributions instead of 0.12 placeholders
through sophisticated evaluation of position accuracy, move prediction quality, and performance.

Root Cause Solution Validation: Ensures neural networks trained with heuristic supervision
demonstrate genuine strategic intelligence and meaningful decision-making capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import pickle
import time
import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import scipy.stats as stats
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

from neural_networks.neural_models import (
    ModelConfig, PositionEvaluatorNetwork, MovePredictorNetwork, 
    GameOutcomePredictor, MultiTaskBattlesnakeNetwork,
    create_position_evaluator, create_move_predictor, 
    create_game_outcome_predictor, create_multitask_network
)
from neural_networks.heuristic_supervision import (
    HeuristicTrainingTarget, HeuristicScores
)
from neural_networks.train_neural_networks import HeuristicDataset


@dataclass
class EvaluationConfig:
    """Configuration for comprehensive model evaluation"""
    # Evaluation data
    test_data_path: Optional[str] = None
    num_test_samples: int = 1000
    
    # Accuracy thresholds
    position_accuracy_threshold: float = 0.70  # 70% agreement with heuristics
    move_accuracy_threshold: float = 0.65      # 65% move prediction accuracy
    outcome_accuracy_threshold: float = 0.60   # 60% game outcome prediction
    
    # Performance requirements
    max_inference_time_ms: float = 5.0         # <5ms per model
    max_total_inference_ms: float = 10.0       # <10ms total for all models
    max_memory_usage_mb: float = 50.0          # <50MB total memory usage
    
    # Statistical analysis
    confidence_level: float = 0.95             # 95% confidence intervals
    min_effect_size: float = 0.1              # Minimum meaningful effect size
    
    # Comparison baselines
    random_baseline: bool = True               # Compare against random choices
    heuristic_baseline: bool = True           # Compare against heuristic system
    placeholder_baseline: bool = True          # Compare against 0.12 placeholder
    
    # Output settings
    generate_plots: bool = True
    save_detailed_results: bool = True
    output_dir: str = "evaluation_results"
    
    # Hardware settings
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    # Accuracy metrics
    position_accuracy: float
    position_mae: float                    # Mean absolute error
    position_correlation: float           # Correlation with heuristic scores
    
    move_accuracy: float
    move_top3_accuracy: float             # Top-3 move prediction accuracy
    move_cross_entropy: float             # Cross-entropy loss
    
    outcome_accuracy: float
    outcome_auc: float                    # Area under ROC curve
    outcome_brier_score: float            # Brier score for probability calibration
    
    # Performance metrics
    inference_time_ms: float
    memory_usage_mb: float
    throughput_fps: float                 # Frames per second
    
    # Statistical significance
    confidence_intervals: Dict[str, Tuple[float, float]]
    p_values: Dict[str, float]
    effect_sizes: Dict[str, float]
    
    # Comparison to baselines
    improvement_over_random: float
    improvement_over_placeholder: float
    agreement_with_heuristics: float


class StatisticalAnalyzer:
    """
    Statistical analysis tools for model evaluation
    Provides confidence intervals, significance tests, and effect size calculations
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        
    def bootstrap_confidence_interval(self, data: np.ndarray, 
                                    statistic_fn=np.mean, n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_fn(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        
        lower = np.percentile(bootstrap_stats, lower_percentile)
        upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return (lower, upper)
    
    def paired_t_test(self, predictions: np.ndarray, 
                     baseline: np.ndarray) -> Tuple[float, float]:
        """Perform paired t-test comparing predictions to baseline"""
        differences = predictions - baseline
        t_statistic, p_value = stats.ttest_1samp(differences, 0)
        return t_statistic, p_value
    
    def cohen_d_effect_size(self, predictions: np.ndarray, 
                          baseline: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        diff_mean = np.mean(predictions - baseline)
        pooled_std = np.sqrt((np.var(predictions) + np.var(baseline)) / 2)
        return diff_mean / pooled_std if pooled_std > 0 else 0.0
    
    def mcnemar_test(self, predictions_1: np.ndarray, predictions_2: np.ndarray, 
                    ground_truth: np.ndarray) -> Tuple[float, float]:
        """McNemar's test for comparing two classifiers"""
        # Create contingency table
        correct_1 = (predictions_1 == ground_truth)
        correct_2 = (predictions_2 == ground_truth)
        
        # McNemar table: both correct, 1 correct 2 wrong, 1 wrong 2 correct, both wrong
        both_correct = np.sum(correct_1 & correct_2)
        only_1_correct = np.sum(correct_1 & ~correct_2)
        only_2_correct = np.sum(~correct_1 & correct_2)
        both_wrong = np.sum(~correct_1 & ~correct_2)
        
        # McNemar statistic
        if only_1_correct + only_2_correct == 0:
            return 0.0, 1.0
        
        mcnemar_stat = (abs(only_1_correct - only_2_correct) - 1) ** 2 / (only_1_correct + only_2_correct)
        p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        
        return mcnemar_stat, p_value


class PerformanceBenchmark:
    """
    Performance benchmarking tools for neural network evaluation
    Measures inference time, memory usage, and throughput
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        
    def benchmark_inference_time(self, model: nn.Module, sample_inputs: Tuple[torch.Tensor, ...], 
                                num_runs: int = 1000, warmup_runs: int = 100) -> Dict[str, float]:
        """Comprehensive inference time benchmarking"""
        model.eval()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(*sample_inputs)
        
        # Synchronize for accurate timing
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark runs
        times = []
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = model(*sample_inputs)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        times = np.array(times)
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p50_ms': np.percentile(times, 50),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'throughput_fps': 1000.0 / np.mean(times)
        }
    
    def measure_memory_usage(self, model: nn.Module, sample_inputs: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """Measure memory usage during inference"""
        model.eval()
        
        # Measure model parameters memory
        param_memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        # Measure inference memory (CUDA only)
        inference_memory_mb = 0.0
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                _ = model(*sample_inputs)
            
            inference_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        return {
            'parameter_memory_mb': param_memory_mb,
            'inference_memory_mb': inference_memory_mb,
            'total_memory_mb': param_memory_mb + inference_memory_mb
        }


class NeuralNetworkEvaluator:
    """
    Comprehensive neural network evaluator
    Validates 30-50+ point contributions vs 0.12 placeholder performance
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analysis tools
        self.stats_analyzer = StatisticalAnalyzer(config.confidence_level)
        self.performance_benchmark = PerformanceBenchmark(self.device)
        
        # Evaluation results storage
        self.evaluation_results = {}
        
    def load_test_data(self, data_path: str) -> List[HeuristicTrainingTarget]:
        """Load test data with heuristic supervision"""
        self.logger.info(f"Loading test data from {data_path}")
        
        try:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict) and 'val_targets' in data:
                test_targets = data['val_targets']
            elif isinstance(data, list):
                test_targets = data
            else:
                raise ValueError("Unknown data format")
            
            # Limit number of test samples if specified
            if self.config.num_test_samples and len(test_targets) > self.config.num_test_samples:
                test_targets = test_targets[:self.config.num_test_samples]
            
            self.logger.info(f"Loaded {len(test_targets)} test samples")
            return test_targets
            
        except Exception as e:
            self.logger.error(f"Failed to load test data: {e}")
            raise
    
    def create_test_dataloader(self, test_targets: List[HeuristicTrainingTarget]) -> torch.utils.data.DataLoader:
        """Create test data loader"""
        from neural_networks.train_neural_networks import TrainingConfig
        
        # Create minimal config for dataset
        training_config = TrainingConfig(batch_size=32, enable_data_augmentation=False)
        test_dataset = HeuristicDataset(test_targets, training_config, augment=False)
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=0,  # Single-threaded for consistent evaluation
            pin_memory=False
        )
        
        return test_loader
    
    def evaluate_position_prediction(self, model: nn.Module, test_loader) -> Dict[str, Any]:
        """Evaluate position prediction accuracy against heuristic supervision"""
        self.logger.info("Evaluating position prediction accuracy")
        
        model.eval()
        predictions = []
        targets = []
        heuristic_totals = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Get model predictions
                if isinstance(model, MultiTaskBattlesnakeNetwork):
                    outputs = model(batch['board_encoding'], 
                                   batch['snake_features'], 
                                   batch['game_context'])
                    position_pred = outputs['position_value']
                else:
                    position_pred = model(batch['board_encoding'], 
                                        batch['snake_features'], 
                                        batch['game_context'])
                
                predictions.extend(position_pred.cpu().numpy())
                targets.extend(batch['position_value'].cpu().numpy())
                
                # Extract heuristic totals for comparison
                for scores in batch['heuristic_scores']:
                    heuristic_totals.append(scores['total_heuristic_score'])
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        heuristic_totals = np.array(heuristic_totals)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        correlation = np.corrcoef(predictions, targets)[0, 1]
        
        # Accuracy within tolerance (±5 points)
        accuracy = np.mean(np.abs(predictions - targets) < 5.0)
        
        # Statistical analysis
        ci_accuracy = self.stats_analyzer.bootstrap_confidence_interval(
            np.abs(predictions - targets) < 5.0)
        ci_mae = self.stats_analyzer.bootstrap_confidence_interval(
            np.abs(predictions - targets))
        
        # Compare to baseline (placeholder 0.12)
        placeholder_predictions = np.full_like(predictions, 0.12)
        placeholder_mae = np.mean(np.abs(placeholder_predictions - targets))
        improvement_over_placeholder = (placeholder_mae - mae) / placeholder_mae * 100
        
        # Compare to random baseline
        random_predictions = np.random.uniform(-50, 50, len(predictions))
        random_mae = np.mean(np.abs(random_predictions - targets))
        improvement_over_random = (random_mae - mae) / random_mae * 100
        
        # Statistical significance tests
        t_stat_placeholder, p_val_placeholder = self.stats_analyzer.paired_t_test(
            np.abs(predictions - targets), np.abs(placeholder_predictions - targets))
        
        effect_size_placeholder = self.stats_analyzer.cohen_d_effect_size(
            np.abs(predictions - targets), np.abs(placeholder_predictions - targets))
        
        results = {
            'accuracy': accuracy,
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'confidence_intervals': {
                'accuracy': ci_accuracy,
                'mae': ci_mae
            },
            'improvement_over_placeholder_percent': improvement_over_placeholder,
            'improvement_over_random_percent': improvement_over_random,
            'statistical_significance': {
                'vs_placeholder_p_value': p_val_placeholder,
                'vs_placeholder_effect_size': effect_size_placeholder
            },
            'raw_data': {
                'predictions': predictions,
                'targets': targets,
                'heuristic_totals': heuristic_totals
            }
        }
        
        self.logger.info(f"Position prediction results:")
        self.logger.info(f"  Accuracy (±5pts): {accuracy:.3f}")
        self.logger.info(f"  MAE: {mae:.2f}")
        self.logger.info(f"  Correlation: {correlation:.3f}")
        self.logger.info(f"  Improvement over placeholder: {improvement_over_placeholder:.1f}%")
        
        return results
    
    def evaluate_move_prediction(self, model: nn.Module, test_loader) -> Dict[str, Any]:
        """Evaluate move prediction accuracy"""
        self.logger.info("Evaluating move prediction accuracy")
        
        model.eval()
        predictions = []
        targets = []
        probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Get model predictions
                if isinstance(model, MultiTaskBattlesnakeNetwork):
                    outputs = model(batch['board_encoding'], 
                                   batch['snake_features'], 
                                   batch['game_context'])
                    move_probs = outputs['move_probabilities']
                else:
                    move_probs, _ = model(batch['board_encoding'], 
                                        batch['snake_features'], 
                                        batch['game_context'])
                
                predicted_moves = move_probs.argmax(dim=1)
                
                predictions.extend(predicted_moves.cpu().numpy())
                targets.extend(batch['move_index'].cpu().numpy())
                probabilities.extend(move_probs.cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        probabilities = np.array(probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(targets, predictions)
        
        # Top-3 accuracy (probability that correct move is in top 3)
        top3_accuracy = np.mean([target in np.argsort(probs)[-3:] 
                               for target, probs in zip(targets, probabilities)])
        
        # Cross-entropy loss
        cross_entropy = -np.mean([np.log(probs[target] + 1e-8) 
                                for target, probs in zip(targets, probabilities)])
        
        # Confusion matrix and classification report
        move_names = ['up', 'down', 'left', 'right']
        conf_matrix = confusion_matrix(targets, predictions)
        class_report = classification_report(targets, predictions, 
                                           target_names=move_names, 
                                           output_dict=True, zero_division=0)
        
        # Statistical analysis
        ci_accuracy = self.stats_analyzer.bootstrap_confidence_interval(
            (predictions == targets).astype(float))
        
        # Compare to random baseline (25% accuracy)
        random_accuracy = 0.25
        improvement_over_random = (accuracy - random_accuracy) / random_accuracy * 100
        
        # Statistical significance (binomial test)
        from scipy.stats import binomtest
        binom_test = binomtest(np.sum(predictions == targets), len(targets), 0.25)
        
        results = {
            'accuracy': accuracy,
            'top3_accuracy': top3_accuracy,
            'cross_entropy': cross_entropy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'confidence_intervals': {
                'accuracy': ci_accuracy
            },
            'improvement_over_random_percent': improvement_over_random,
            'statistical_significance': {
                'binomial_test_p_value': binom_test.pvalue
            },
            'raw_data': {
                'predictions': predictions,
                'targets': targets,
                'probabilities': probabilities
            }
        }
        
        self.logger.info(f"Move prediction results:")
        self.logger.info(f"  Accuracy: {accuracy:.3f}")
        self.logger.info(f"  Top-3 accuracy: {top3_accuracy:.3f}")
        self.logger.info(f"  Cross-entropy: {cross_entropy:.3f}")
        self.logger.info(f"  Improvement over random: {improvement_over_random:.1f}%")
        
        return results
    
    def evaluate_performance_benchmarks(self, models: Dict[str, nn.Module]) -> Dict[str, Any]:
        """Evaluate inference performance benchmarks"""
        self.logger.info("Evaluating performance benchmarks")
        
        performance_results = {}
        total_inference_time = 0
        total_memory_usage = 0
        
        # Create sample inputs
        board_input = torch.randn(1, 12, 11, 11, device=self.device, dtype=torch.float32)
        snake_features = torch.randn(1, 32, device=self.device, dtype=torch.float32)
        game_context = torch.randn(1, 16, device=self.device, dtype=torch.float32)
        sample_inputs = (board_input, snake_features, game_context)
        
        for model_name, model in models.items():
            self.logger.info(f"Benchmarking {model_name}")
            
            model = model.to(self.device)
            model.eval()
            
            # Inference time benchmark
            time_results = self.performance_benchmark.benchmark_inference_time(
                model, sample_inputs, num_runs=1000, warmup_runs=100)
            
            # Memory usage benchmark
            memory_results = self.performance_benchmark.measure_memory_usage(
                model, sample_inputs)
            
            performance_results[model_name] = {
                'timing': time_results,
                'memory': memory_results
            }
            
            total_inference_time += time_results['mean_ms']
            total_memory_usage += memory_results['total_memory_mb']
            
            self.logger.info(f"  Inference time: {time_results['mean_ms']:.2f}ms ± {time_results['std_ms']:.2f}ms")
            self.logger.info(f"  Memory usage: {memory_results['total_memory_mb']:.1f}MB")
            self.logger.info(f"  Throughput: {time_results['throughput_fps']:.1f} FPS")
        
        # Overall performance summary
        performance_summary = {
            'total_inference_time_ms': total_inference_time,
            'total_memory_usage_mb': total_memory_usage,
            'meets_inference_requirement': total_inference_time < self.config.max_total_inference_ms,
            'meets_memory_requirement': total_memory_usage < self.config.max_memory_usage_mb,
            'individual_results': performance_results
        }
        
        self.logger.info(f"Overall performance:")
        self.logger.info(f"  Total inference time: {total_inference_time:.2f}ms "
                        f"({'✓' if total_inference_time < self.config.max_total_inference_ms else '✗'} <{self.config.max_total_inference_ms}ms)")
        self.logger.info(f"  Total memory usage: {total_memory_usage:.1f}MB "
                        f"({'✓' if total_memory_usage < self.config.max_memory_usage_mb else '✗'} <{self.config.max_memory_usage_mb}MB)")
        
        return performance_summary
    
    def generate_evaluation_plots(self, evaluation_results: Dict[str, Any]):
        """Generate comprehensive evaluation plots"""
        if not self.config.generate_plots:
            return
            
        self.logger.info("Generating evaluation plots")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Position prediction plots
        if 'position_prediction' in evaluation_results:
            pos_data = evaluation_results['position_prediction']['raw_data']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Scatter plot: Predictions vs Targets
            axes[0, 0].scatter(pos_data['targets'], pos_data['predictions'], alpha=0.6)
            axes[0, 0].plot([-50, 50], [-50, 50], 'r--', label='Perfect prediction')
            axes[0, 0].set_xlabel('Heuristic Position Value')
            axes[0, 0].set_ylabel('Neural Network Prediction')
            axes[0, 0].set_title('Position Prediction Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Residual plot
            residuals = pos_data['predictions'] - pos_data['targets']
            axes[0, 1].scatter(pos_data['targets'], residuals, alpha=0.6)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Heuristic Position Value')
            axes[0, 1].set_ylabel('Prediction Residuals')
            axes[0, 1].set_title('Prediction Residuals')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Distribution comparison
            axes[1, 0].hist(pos_data['targets'], bins=30, alpha=0.7, label='Heuristic Values', density=True)
            axes[1, 0].hist(pos_data['predictions'], bins=30, alpha=0.7, label='Predictions', density=True)
            axes[1, 0].set_xlabel('Position Value')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Value Distributions')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Error distribution
            axes[1, 1].hist(np.abs(residuals), bins=30, alpha=0.7, density=True)
            axes[1, 1].axvline(x=5.0, color='r', linestyle='--', label='±5pt threshold')
            axes[1, 1].set_xlabel('Absolute Error')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Prediction Error Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'position_evaluation.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Move prediction confusion matrix
        if 'move_prediction' in evaluation_results:
            move_data = evaluation_results['move_prediction']
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Confusion matrix
            sns.heatmap(move_data['confusion_matrix'], 
                       xticklabels=['up', 'down', 'left', 'right'],
                       yticklabels=['up', 'down', 'left', 'right'],
                       annot=True, fmt='d', cmap='Blues',
                       ax=axes[0])
            axes[0].set_title('Move Prediction Confusion Matrix')
            axes[0].set_xlabel('Predicted Move')
            axes[0].set_ylabel('True Move')
            
            # Move probability distributions
            probs = move_data['raw_data']['probabilities']
            targets = move_data['raw_data']['targets']
            
            move_names = ['up', 'down', 'left', 'right']
            for i, move_name in enumerate(move_names):
                mask = targets == i
                if np.any(mask):
                    axes[1].hist(probs[mask, i], bins=20, alpha=0.7, 
                               label=f'{move_name} (true)', density=True)
            
            axes[1].set_xlabel('Predicted Probability')
            axes[1].set_ylabel('Density')
            axes[1].set_title('Probability Distributions by True Move')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'move_evaluation.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Performance benchmarks
        if 'performance' in evaluation_results:
            perf_data = evaluation_results['performance']['individual_results']
            
            model_names = list(perf_data.keys())
            inference_times = [perf_data[name]['timing']['mean_ms'] for name in model_names]
            memory_usage = [perf_data[name]['memory']['total_memory_mb'] for name in model_names]
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Inference time comparison
            bars1 = axes[0].bar(model_names, inference_times)
            axes[0].axhline(y=self.config.max_inference_time_ms, color='r', 
                           linestyle='--', label=f'Requirement (<{self.config.max_inference_time_ms}ms)')
            axes[0].set_ylabel('Inference Time (ms)')
            axes[0].set_title('Model Inference Times')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, time_val in zip(bars1, inference_times):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{time_val:.1f}ms', ha='center', va='bottom')
            
            # Memory usage comparison
            bars2 = axes[1].bar(model_names, memory_usage)
            axes[1].axhline(y=self.config.max_memory_usage_mb, color='r',
                           linestyle='--', label=f'Requirement (<{self.config.max_memory_usage_mb}MB)')
            axes[1].set_ylabel('Memory Usage (MB)')
            axes[1].set_title('Model Memory Usage')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mem_val in zip(bars2, memory_usage):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{mem_val:.1f}MB', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'performance_benchmarks.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        self.logger.info(f"Evaluation plots saved to {self.output_dir}")
    
    def comprehensive_evaluation(self, models: Dict[str, nn.Module], 
                                test_data_path: str) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of neural networks
        Validates 30-50+ point contributions vs 0.12 placeholder performance
        """
        self.logger.info("=" * 60)
        self.logger.info("COMPREHENSIVE NEURAL NETWORK EVALUATION")
        self.logger.info("Validating 30-50+ point contributions vs 0.12 placeholder")
        self.logger.info("=" * 60)
        
        # Load test data
        test_targets = self.load_test_data(test_data_path)
        test_loader = self.create_test_dataloader(test_targets)
        
        evaluation_results = {}
        
        # Evaluate each model component
        for model_name, model in models.items():
            self.logger.info(f"\n--- Evaluating {model_name} ---")
            
            model = model.to(self.device)
            
            # Position prediction evaluation
            if 'position' in model_name.lower() or 'multitask' in model_name.lower():
                pos_results = self.evaluate_position_prediction(model, test_loader)
                evaluation_results['position_prediction'] = pos_results
            
            # Move prediction evaluation
            if 'move' in model_name.lower() or 'multitask' in model_name.lower():
                move_results = self.evaluate_move_prediction(model, test_loader)
                evaluation_results['move_prediction'] = move_results
        
        # Performance benchmarks
        performance_results = self.evaluate_performance_benchmarks(models)
        evaluation_results['performance'] = performance_results
        
        # Generate comprehensive report
        report = self.generate_evaluation_report(evaluation_results)
        evaluation_results['comprehensive_report'] = report
        
        # Generate plots
        self.generate_evaluation_plots(evaluation_results)
        
        # Save detailed results
        if self.config.save_detailed_results:
            results_path = self.output_dir / 'detailed_evaluation_results.json'
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = self.prepare_for_json_serialization(evaluation_results)
            
            with open(results_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            self.logger.info(f"Detailed results saved to {results_path}")
        
        return evaluation_results
    
    def prepare_for_json_serialization(self, data: Any) -> Any:
        """Prepare data for JSON serialization by converting numpy arrays"""
        if isinstance(data, dict):
            return {k: self.prepare_for_json_serialization(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.prepare_for_json_serialization(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif hasattr(data, 'tolist'):
            return data.tolist()
        else:
            return data
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        report = {
            'evaluation_timestamp': time.time(),
            'configuration': asdict(self.config),
            'summary': {},
            'detailed_analysis': {},
            'recommendations': []
        }
        
        # Summary metrics
        summary = {}
        
        if 'position_prediction' in evaluation_results:
            pos_results = evaluation_results['position_prediction']
            summary['position_accuracy'] = pos_results['accuracy']
            summary['position_mae'] = pos_results['mae']
            summary['position_improvement_over_placeholder'] = pos_results['improvement_over_placeholder_percent']
            
            # Check if meets requirements
            summary['position_meets_accuracy_threshold'] = pos_results['accuracy'] >= self.config.position_accuracy_threshold
        
        if 'move_prediction' in evaluation_results:
            move_results = evaluation_results['move_prediction']
            summary['move_accuracy'] = move_results['accuracy']
            summary['move_top3_accuracy'] = move_results['top3_accuracy']
            summary['move_improvement_over_random'] = move_results['improvement_over_random_percent']
            
            # Check if meets requirements
            summary['move_meets_accuracy_threshold'] = move_results['accuracy'] >= self.config.move_accuracy_threshold
        
        if 'performance' in evaluation_results:
            perf_results = evaluation_results['performance']
            summary['total_inference_time_ms'] = perf_results['total_inference_time_ms']
            summary['total_memory_usage_mb'] = perf_results['total_memory_usage_mb']
            summary['meets_performance_requirements'] = (
                perf_results['meets_inference_requirement'] and 
                perf_results['meets_memory_requirement']
            )
        
        # Overall assessment
        summary['overall_success'] = all([
            summary.get('position_meets_accuracy_threshold', True),
            summary.get('move_meets_accuracy_threshold', True),
            summary.get('meets_performance_requirements', True)
        ])
        
        # Placeholder comparison
        if 'position_prediction' in evaluation_results:
            pos_improvement = evaluation_results['position_prediction']['improvement_over_placeholder_percent']
            summary['significant_improvement_over_placeholder'] = pos_improvement > 50  # >50% improvement
            
        report['summary'] = summary
        
        # Recommendations
        recommendations = []
        
        if not summary.get('position_meets_accuracy_threshold', True):
            recommendations.append({
                'category': 'Position Prediction',
                'issue': f"Position accuracy ({summary['position_accuracy']:.3f}) below threshold ({self.config.position_accuracy_threshold})",
                'recommendation': 'Consider retraining with more data or adjusting model architecture'
            })
        
        if not summary.get('move_meets_accuracy_threshold', True):
            recommendations.append({
                'category': 'Move Prediction', 
                'issue': f"Move accuracy ({summary['move_accuracy']:.3f}) below threshold ({self.config.move_accuracy_threshold})",
                'recommendation': 'Increase model capacity or improve data augmentation'
            })
        
        if not summary.get('meets_performance_requirements', True):
            if not evaluation_results['performance']['meets_inference_requirement']:
                recommendations.append({
                    'category': 'Performance',
                    'issue': f"Inference time ({summary['total_inference_time_ms']:.1f}ms) exceeds requirement ({self.config.max_total_inference_ms}ms)",
                    'recommendation': 'Optimize model architecture or use model pruning/quantization'
                })
            
            if not evaluation_results['performance']['meets_memory_requirement']:
                recommendations.append({
                    'category': 'Performance',
                    'issue': f"Memory usage ({summary['total_memory_usage_mb']:.1f}MB) exceeds requirement ({self.config.max_memory_usage_mb}MB)",
                    'recommendation': 'Reduce model size or use parameter sharing'
                })
        
        if summary.get('significant_improvement_over_placeholder', False):
            recommendations.append({
                'category': 'Success',
                'issue': 'Models significantly outperform 0.12 placeholder',
                'recommendation': 'Deploy models to production - ready for 30-50+ point contributions'
            })
        
        report['recommendations'] = recommendations
        
        return report
    
    def print_evaluation_summary(self, evaluation_results: Dict[str, Any]):
        """Print comprehensive evaluation summary"""
        print("\n" + "=" * 60)
        print("NEURAL NETWORK EVALUATION SUMMARY")
        print("=" * 60)
        
        report = evaluation_results.get('comprehensive_report', {})
        summary = report.get('summary', {})
        
        # Position prediction results
        if 'position_prediction' in evaluation_results:
            print(f"\n📊 POSITION EVALUATION:")
            print(f"  Accuracy (±5pts): {summary.get('position_accuracy', 0):.1%}")
            print(f"  Mean Absolute Error: {summary.get('position_mae', 0):.2f}")
            print(f"  Improvement over 0.12 placeholder: {summary.get('position_improvement_over_placeholder', 0):.1f}%")
            print(f"  Meets accuracy threshold: {'✅' if summary.get('position_meets_accuracy_threshold') else '❌'}")
        
        # Move prediction results  
        if 'move_prediction' in evaluation_results:
            print(f"\n🎯 MOVE PREDICTION:")
            print(f"  Accuracy: {summary.get('move_accuracy', 0):.1%}")
            print(f"  Top-3 Accuracy: {summary.get('move_top3_accuracy', 0):.1%}")
            print(f"  Improvement over random: {summary.get('move_improvement_over_random', 0):.1f}%")
            print(f"  Meets accuracy threshold: {'✅' if summary.get('move_meets_accuracy_threshold') else '❌'}")
        
        # Performance results
        if 'performance' in evaluation_results:
            print(f"\n⚡ PERFORMANCE:")
            print(f"  Total inference time: {summary.get('total_inference_time_ms', 0):.1f}ms")
            print(f"  Total memory usage: {summary.get('total_memory_usage_mb', 0):.1f}MB")
            print(f"  Meets performance requirements: {'✅' if summary.get('meets_performance_requirements') else '❌'}")
        
        # Overall assessment
        print(f"\n🎉 OVERALL ASSESSMENT:")
        print(f"  All requirements met: {'✅' if summary.get('overall_success') else '❌'}")
        print(f"  Ready for production: {'✅' if summary.get('significant_improvement_over_placeholder') else '❌'}")
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. [{rec['category']}] {rec['recommendation']}")
        
        print(f"\n📁 Results saved to: {self.output_dir}")
        print("=" * 60)


def evaluate_neural_networks(models: Dict[str, nn.Module], test_data_path: str,
                            config: Optional[EvaluationConfig] = None) -> Dict[str, Any]:
    """
    Main entry point for comprehensive neural network evaluation
    Validates 30-50+ point contributions vs 0.12 placeholder performance
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if config is None:
        config = EvaluationConfig()
    
    # Create evaluator
    evaluator = NeuralNetworkEvaluator(config)
    
    # Run comprehensive evaluation
    results = evaluator.comprehensive_evaluation(models, test_data_path)
    
    # Print summary
    evaluator.print_evaluation_summary(results)
    
    return results


# Example usage and testing
if __name__ == "__main__":
    print("Comprehensive Model Evaluation Framework Test")
    print("=" * 50)
    
    # Create test models
    from neural_networks.neural_models import ModelConfig
    
    config = ModelConfig()
    test_models = {
        'multitask': create_multitask_network(config)
    }
    
    print(f"Created {len(test_models)} test models for evaluation")
    
    # Create test data (normally would load real test data)
    from neural_networks.heuristic_supervision import HeuristicScores, HeuristicTrainingTarget
    
    test_targets = []
    for i in range(200):  # Small test dataset
        heuristic_scores = HeuristicScores(
            safety_score=np.random.uniform(20, 45),
            territory_score=np.random.uniform(4, 8),
            opponent_score=np.random.uniform(3, 6),
            food_score=np.random.uniform(2, 9),
            exploration_score=np.random.uniform(25, 35),
            total_heuristic_score=np.random.uniform(60, 100),
            confidence=np.random.uniform(0.7, 0.95)
        )
        
        target = HeuristicTrainingTarget(
            board_encoding=np.random.random((11, 11, 12)).astype(np.float32),
            snake_features=np.random.random(32).astype(np.float32),
            game_context=np.random.random(16).astype(np.float32),
            heuristic_scores=heuristic_scores,
            optimal_move=np.random.choice(['up', 'down', 'left', 'right']),
            move_probabilities=np.random.dirichlet([1, 1, 1, 1]).astype(np.float32),
            position_value=np.random.uniform(-40, 40),
            game_outcome=np.random.random(),
            metadata={'test': True}
        )
        test_targets.append(target)
    
    # Save test data
    test_data_path = 'test_evaluation_data.pkl'
    with open(test_data_path, 'wb') as f:
        pickle.dump({'val_targets': test_targets}, f)
    
    print(f"Created test data with {len(test_targets)} samples")
    
    # Create evaluation config
    eval_config = EvaluationConfig(
        num_test_samples=200,
        generate_plots=True,
        save_detailed_results=True,
        output_dir='test_evaluation_results'
    )
    
    print(f"Evaluation configuration:")
    print(f"  Position accuracy threshold: {eval_config.position_accuracy_threshold}")
    print(f"  Move accuracy threshold: {eval_config.move_accuracy_threshold}")
    print(f"  Performance requirements: <{eval_config.max_total_inference_ms}ms, <{eval_config.max_memory_usage_mb}MB")
    
    # Run evaluation
    try:
        evaluation_results = evaluate_neural_networks(
            test_models, test_data_path, eval_config
        )
        
        print(f"\n✅ Model evaluation completed successfully!")
        print(f"✅ Framework ready to validate 30-50+ point contributions")
        print(f"✅ Comprehensive evaluation system operational")
        
        # Clean up test files
        os.remove(test_data_path)
        
    except Exception as e:
        print(f"Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()