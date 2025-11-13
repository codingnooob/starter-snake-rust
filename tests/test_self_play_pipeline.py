"""
Comprehensive Test Suite for Self-Play Training Pipeline

Production-grade test coverage for all self-play training components including:
- Configuration system validation
- Data manager functionality
- Model evolution training phases
- Performance evaluation accuracy
- Training orchestrator coordination
- Automated runner scheduling and triggers

Provides both unit tests and integration tests with realistic data scenarios.
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import sqlite3
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components to test
from config.self_play_config import (
    SelfPlayConfig, TrainingPhaseConfig, ModelTournamentConfig, 
    NeuralNetworkConfig, TrainingPipelineConfig, get_config
)
from self_play_data_manager import (
    SelfPlayDataManager, GameDataProcessor, ExperienceReplayBuffer,
    TrainingDataPipeline
)
from model_evolution import (
    ModelEvolutionSystem, TournamentManager, ModelRegistry,
    TrainingPhase, EvolutionStatus
)
from model_performance_evaluator import (
    ModelPerformanceEvaluator, StatisticalValidator, 
    PerformanceBenchmark, EvaluationResult
)
from self_play_training_pipeline import (
    SelfPlayTrainingPipeline, TrainingConfiguration, 
    PipelineStatus, PipelineState
)
from automated_training_runner import (
    AutomatedTrainingRunner, TrainingSchedule, TriggerCondition,
    TriggerType, RunnerStatus, ResourceManager, ConditionChecker
)


class TestSelfPlayConfig(unittest.TestCase):
    """Test configuration system"""
    
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_file = self.test_dir / "test_config.json"
    
    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_config_creation_and_validation(self):
        """Test configuration creation and validation"""
        config = SelfPlayConfig()
        
        # Test default values
        self.assertIsInstance(config.neural_network, NeuralNetworkConfig)
        self.assertIsInstance(config.training_pipeline, TrainingPipelineConfig)
        self.assertIsInstance(config.tournament, ModelTournamentConfig)
        
        # Test phase configurations
        self.assertIn("bootstrap", config.training_phases)
        self.assertIn("hybrid", config.training_phases)
        self.assertIn("self_play", config.training_phases)
        self.assertIn("continuous", config.training_phases)
        
        # Validate learning rates
        self.assertGreater(config.training_phases["bootstrap"].learning_rate, 0)
        self.assertGreater(config.training_phases["hybrid"].learning_rate, 0)
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization"""
        config = SelfPlayConfig()
        
        # Save to file
        config.save_to_file(str(self.config_file))
        self.assertTrue(self.config_file.exists())
        
        # Load from file
        loaded_config = SelfPlayConfig.load_from_file(str(self.config_file))
        
        # Compare key values
        self.assertEqual(
            config.neural_network.hidden_layers,
            loaded_config.neural_network.hidden_layers
        )
        self.assertEqual(
            config.tournament.games_per_matchup,
            loaded_config.tournament.games_per_matchup
        )
    
    def test_config_validation_errors(self):
        """Test configuration validation catches errors"""
        config = SelfPlayConfig()
        
        # Test invalid learning rate
        config.training_phases["bootstrap"].learning_rate = -0.01
        with self.assertRaises(ValueError):
            config.validate()
        
        # Test invalid batch size
        config.training_phases["bootstrap"].learning_rate = 0.001
        config.training_phases["bootstrap"].batch_size = 0
        with self.assertRaises(ValueError):
            config.validate()


class TestSelfPlayDataManager(unittest.TestCase):
    """Test data management functionality"""
    
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = SelfPlayConfig()
        self.data_manager = SelfPlayDataManager(
            str(self.test_dir), 
            config=self.config
        )
    
    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_experience_replay_buffer(self):
        """Test experience replay buffer functionality"""
        buffer = ExperienceReplayBuffer(max_size=1000, quality_threshold=0.5)
        
        # Add sample experiences
        for i in range(100):
            experience = {
                'board_state': np.random.rand(12, 11, 11),
                'action': np.random.randint(0, 4),
                'reward': np.random.rand(),
                'next_state': np.random.rand(12, 11, 11),
                'quality_score': np.random.rand()
            }
            buffer.add_experience(experience)
        
        self.assertEqual(len(buffer), 100)
        
        # Test sampling
        batch = buffer.sample_batch(32)
        self.assertEqual(len(batch), 32)
        
        # Test quality filtering
        buffer.quality_threshold = 0.8
        high_quality_batch = buffer.sample_batch(32)
        self.assertEqual(len(high_quality_batch), 32)
    
    def test_game_data_processing(self):
        """Test game data processing"""
        processor = GameDataProcessor(self.config)
        
        # Create mock game data
        game_data = {
            'id': 'test-game-123',
            'turns': [
                {
                    'turn': 0,
                    'board': {
                        'width': 11,
                        'height': 11,
                        'food': [{'x': 5, 'y': 5}],
                        'snakes': [
                            {
                                'id': 'snake1',
                                'head': {'x': 1, 'y': 1},
                                'body': [{'x': 1, 'y': 1}, {'x': 1, 'y': 2}],
                                'health': 100
                            }
                        ]
                    },
                    'you': {
                        'id': 'snake1',
                        'head': {'x': 1, 'y': 1},
                        'body': [{'x': 1, 'y': 1}, {'x': 1, 'y': 2}],
                        'health': 100
                    },
                    'move': 'up'
                }
            ]
        }
        
        # Process game data
        processed_samples = processor.process_game_data([game_data])
        
        self.assertGreater(len(processed_samples), 0)
        self.assertIn('board_encoding', processed_samples[0])
        self.assertIn('action', processed_samples[0])
        self.assertEqual(processed_samples[0]['board_encoding'].shape, (12, 11, 11))
    
    def test_data_pipeline_integration(self):
        """Test complete data pipeline"""
        pipeline = TrainingDataPipeline(self.data_manager)
        
        # Test pipeline initialization
        self.assertIsNotNone(pipeline.processor)
        self.assertIsNotNone(pipeline.experience_buffer)
        
        # Test data statistics
        stats = self.data_manager.get_data_statistics()
        self.assertIn('total_games', stats)
        self.assertIn('processing_metrics', stats)


class TestModelEvolution(unittest.TestCase):
    """Test model evolution system"""
    
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = SelfPlayConfig()
        self.evolution_system = ModelEvolutionSystem(
            str(self.test_dir),
            config=self.config
        )
    
    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_mock_model(self, name: str = "test_model"):
        """Create mock neural network model"""
        model = nn.Sequential(
            nn.Linear(12 * 11 * 11, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 4 actions: up, down, left, right
        )
        return model
    
    def test_model_registry(self):
        """Test model registry functionality"""
        registry = ModelRegistry(str(self.test_dir / "registry"))
        
        # Register a model
        model = self.create_mock_model()
        model_info = {
            'name': 'test_model_v1',
            'architecture': 'CNN_Attention',
            'training_phase': TrainingPhase.BOOTSTRAP,
            'performance_metrics': {'win_rate': 0.65, 'avg_length': 45.2}
        }
        
        model_id = registry.register_model(model, model_info)
        self.assertIsNotNone(model_id)
        
        # Retrieve model
        retrieved_model, retrieved_info = registry.get_model(model_id)
        self.assertIsNotNone(retrieved_model)
        self.assertEqual(retrieved_info['name'], 'test_model_v1')
        
        # List models
        models = registry.list_models()
        self.assertGreater(len(models), 0)
        self.assertEqual(models[0]['model_id'], model_id)
    
    def test_tournament_manager(self):
        """Test tournament management"""
        tournament = TournamentManager(self.config)
        
        # Create test models
        models = {}
        for i in range(4):
            model = self.create_mock_model(f"model_{i}")
            models[f"model_{i}"] = {
                'model': model,
                'info': {'name': f'model_{i}', 'win_rate': 0.5 + i * 0.1}
            }
        
        # Mock tournament execution
        with patch.object(tournament, '_run_game_simulation') as mock_game:
            mock_game.return_value = {'winner': 'model_1', 'length': 50, 'reason': 'food'}
            
            # Run tournament
            results = tournament.run_tournament(models, games_per_matchup=10)
            
            self.assertIn('rankings', results)
            self.assertIn('match_results', results)
            self.assertGreater(len(results['rankings']), 0)
    
    def test_training_phases(self):
        """Test different training phases"""
        # Test bootstrap phase
        with patch.object(self.evolution_system, '_collect_heuristic_data') as mock_data:
            mock_data.return_value = [
                {'board_state': np.random.rand(12, 11, 11), 'action': 0, 'reward': 1.0}
                for _ in range(100)
            ]
            
            model = self.evolution_system.bootstrap_training_phase(
                target_games=100,
                training_epochs=5
            )
            
            self.assertIsNotNone(model)
    
    def test_evolution_status_tracking(self):
        """Test evolution status tracking"""
        status = self.evolution_system.get_evolution_status()
        
        expected_keys = [
            'current_phase', 'total_models_trained', 'best_win_rate',
            'champion_model', 'generation_history', 'training_progress'
        ]
        
        for key in expected_keys:
            self.assertIn(key, status)


class TestModelPerformanceEvaluator(unittest.TestCase):
    """Test model performance evaluation"""
    
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = SelfPlayConfig()
        self.evaluator = ModelPerformanceEvaluator(
            str(self.test_dir),
            config=self.config
        )
    
    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_mock_game_results(self, num_games: int = 100) -> List[Dict]:
        """Create mock game results for testing"""
        results = []
        for i in range(num_games):
            results.append({
                'game_id': f'game_{i}',
                'winner': np.random.choice(['model_a', 'model_b']),
                'length': np.random.randint(10, 200),
                'final_scores': {
                    'model_a': np.random.randint(0, 100),
                    'model_b': np.random.randint(0, 100)
                },
                'termination_reason': np.random.choice(['collision', 'timeout', 'food']),
                'performance_metrics': {
                    'avg_health': np.random.uniform(50, 100),
                    'food_collected': np.random.randint(0, 10),
                    'strategic_score': np.random.uniform(0, 100)
                }
            })
        return results
    
    def test_statistical_validator(self):
        """Test statistical validation functionality"""
        validator = StatisticalValidator()
        
        # Test win rate calculation
        game_results = self.create_mock_game_results(100)
        win_rate, confidence_interval = validator.calculate_win_rate_with_confidence(
            game_results, 'model_a'
        )
        
        self.assertGreaterEqual(win_rate, 0.0)
        self.assertLessEqual(win_rate, 1.0)
        self.assertEqual(len(confidence_interval), 2)
        self.assertLessEqual(confidence_interval[0], win_rate)
        self.assertGreaterEqual(confidence_interval[1], win_rate)
        
        # Test significance testing
        model_a_wins = [1 if r['winner'] == 'model_a' else 0 for r in game_results]
        model_b_wins = [1 if r['winner'] == 'model_b' else 0 for r in game_results]
        
        significant, p_value = validator.test_statistical_significance(
            model_a_wins, model_b_wins
        )
        
        self.assertIsInstance(significant, bool)
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)
    
    def test_performance_benchmarking(self):
        """Test performance benchmarking"""
        benchmark = PerformanceBenchmark()
        
        # Create mock model
        model = nn.Sequential(
            nn.Linear(12 * 11 * 11, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        
        # Test inference speed
        inference_time = benchmark.measure_inference_speed(model, batch_size=32)
        self.assertGreater(inference_time, 0)
        
        # Test memory usage
        memory_usage = benchmark.measure_memory_usage(model)
        self.assertGreater(memory_usage, 0)
        
        # Test production readiness
        is_ready, metrics = benchmark.validate_production_readiness(
            model, 
            max_inference_time_ms=10,
            max_memory_mb=50
        )
        
        self.assertIsInstance(is_ready, bool)
        self.assertIn('inference_time_ms', metrics)
        self.assertIn('memory_usage_mb', metrics)
    
    def test_comprehensive_evaluation(self):
        """Test comprehensive model evaluation"""
        # Create mock models
        models = {
            'champion': nn.Sequential(nn.Linear(1452, 256), nn.ReLU(), nn.Linear(256, 4)),
            'challenger': nn.Sequential(nn.Linear(1452, 256), nn.ReLU(), nn.Linear(256, 4))
        }
        
        # Mock evaluation data
        with patch.object(self.evaluator, '_run_evaluation_games') as mock_games:
            mock_games.return_value = self.create_mock_game_results(200)
            
            result = self.evaluator.evaluate_model_comprehensive(
                models['challenger'],
                champion_model=models['champion'],
                evaluation_games=200
            )
            
            self.assertIsInstance(result, EvaluationResult)
            self.assertIn('win_rate', result.performance_metrics)
            self.assertIn('statistical_significance', result.performance_metrics)
            self.assertIn('strategic_analysis', result.detailed_analysis)


class TestSelfPlayTrainingPipeline(unittest.TestCase):
    """Test main training pipeline orchestration"""
    
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Mock config manager
        self.mock_config_manager = Mock()
        self.mock_config_manager.load_config.return_value = SelfPlayConfig()
        
        self.pipeline = SelfPlayTrainingPipeline(self.mock_config_manager)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        self.assertIsNotNone(self.pipeline.config)
        self.assertIsNotNone(self.pipeline.data_manager)
        self.assertIsNotNone(self.pipeline.evolution_system)
        self.assertIsNotNone(self.pipeline.performance_evaluator)
        
        # Test initial status
        status = self.pipeline.get_pipeline_status()
        self.assertEqual(status.status, PipelineStatus.IDLE)
        self.assertEqual(status.current_phase, TrainingPhase.BOOTSTRAP)
    
    def test_training_configuration_validation(self):
        """Test training configuration validation"""
        # Valid configuration
        valid_config = TrainingConfiguration(
            target_phases=["bootstrap", "hybrid"],
            continuous_learning_enabled=True,
            max_training_time_hours=8
        )
        
        self.assertTrue(self.pipeline._validate_training_configuration(valid_config))
        
        # Invalid configuration
        invalid_config = TrainingConfiguration(
            target_phases=["invalid_phase"],
            max_training_time_hours=-1
        )
        
        self.assertFalse(self.pipeline._validate_training_configuration(invalid_config))
    
    def test_pipeline_state_management(self):
        """Test pipeline state management"""
        # Test state transitions
        self.pipeline._update_pipeline_state(
            PipelineStatus.TRAINING,
            current_phase=TrainingPhase.HYBRID,
            total_models_trained=5
        )
        
        status = self.pipeline.get_pipeline_status()
        self.assertEqual(status.status, PipelineStatus.TRAINING)
        self.assertEqual(status.current_phase, TrainingPhase.HYBRID)
        self.assertEqual(status.total_models_trained, 5)
        
        # Test state persistence
        self.pipeline._save_pipeline_state()
        self.assertTrue(self.pipeline.state_file.exists())
        
        # Test state loading
        self.pipeline._load_pipeline_state()
        reloaded_status = self.pipeline.get_pipeline_status()
        self.assertEqual(reloaded_status.status, PipelineStatus.TRAINING)
    
    def test_pipeline_monitoring(self):
        """Test pipeline monitoring functionality"""
        # Start monitoring
        self.pipeline.start_monitoring()
        
        # Check monitoring thread
        self.assertIsNotNone(self.pipeline.monitor_thread)
        self.assertTrue(self.pipeline.monitoring_active)
        
        # Stop monitoring
        self.pipeline.stop_monitoring()
        self.assertFalse(self.pipeline.monitoring_active)


class TestAutomatedTrainingRunner(unittest.TestCase):
    """Test automated training runner"""
    
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Patch the runner to use test directory
        with patch('automated_training_runner.Path') as mock_path:
            mock_path.return_value = self.test_dir
            self.runner = AutomatedTrainingRunner()
    
    def tearDown(self):
        if hasattr(self, 'runner'):
            self.runner.stop_automation()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_resource_manager(self):
        """Test resource management"""
        resource_manager = ResourceManager()
        
        # Test resource availability check
        available, message = resource_manager.check_resource_availability({
            'min_free_memory_gb': 0.1,  # Very low requirement for testing
            'min_free_disk_gb': 0.1,
            'max_cpu_usage_percent': 99
        })
        
        self.assertTrue(available)
        
        # Test execution lock
        self.assertTrue(resource_manager.acquire_execution_lock('test_schedule'))
        self.assertFalse(resource_manager.acquire_execution_lock('another_schedule'))
        
        # Test lock release
        resource_manager.release_execution_lock()
        self.assertTrue(resource_manager.acquire_execution_lock('another_schedule'))
    
    def test_schedule_management(self):
        """Test schedule management"""
        # Create test schedule
        from automated_training_runner import TrainingConfiguration
        
        config = TrainingConfiguration(
            continuous_learning_enabled=True,
            max_training_time_hours=2
        )
        
        schedule = TrainingSchedule(
            name="test_daily",
            cron_expression="0 2 * * *",
            training_config=config,
            min_data_threshold=100
        )
        
        # Add schedule
        self.runner.add_schedule(schedule)
        self.assertIn("test_daily", self.runner.schedules)
        
        # Test invalid cron expression
        invalid_schedule = TrainingSchedule(
            name="invalid",
            cron_expression="invalid cron",
            training_config=config
        )
        
        with self.assertRaises(ValueError):
            self.runner.add_schedule(invalid_schedule)
    
    def test_trigger_conditions(self):
        """Test trigger condition management"""
        trigger = TriggerCondition(
            name="test_trigger",
            trigger_type=TriggerType.DATA_DRIVEN,
            condition_function="check_new_data_threshold",
            parameters={"min_games": 1000},
            cooldown_hours=6
        )
        
        # Add trigger
        self.runner.add_trigger(trigger)
        self.assertIn("test_trigger", self.runner.triggers)
    
    def test_condition_checker(self):
        """Test condition checking functionality"""
        # Mock dependencies
        mock_data_manager = Mock()
        mock_evolution_system = Mock()
        mock_performance_evaluator = Mock()
        
        checker = ConditionChecker(
            mock_data_manager,
            mock_evolution_system,
            mock_performance_evaluator
        )
        
        # Test data threshold check
        mock_data_manager.get_data_statistics.return_value = {
            'processing_metrics': {'samples_generated': 2000}
        }
        
        result = checker.check_new_data_threshold({'min_games': 1000})
        self.assertTrue(result)
        
        # Test manual trigger file
        trigger_file = self.test_dir / "trigger_training.txt"
        trigger_file.touch()
        
        result = checker.check_manual_trigger_file({
            'trigger_file': str(trigger_file)
        })
        self.assertTrue(result)
        self.assertFalse(trigger_file.exists())  # Should be removed after detection
    
    def test_runner_status_and_control(self):
        """Test runner status and control"""
        # Test initial status
        status = self.runner.get_runner_status()
        self.assertEqual(status['status'], RunnerStatus.IDLE.value)
        self.assertFalse(status['running'])
        
        # Test automation start/stop (mock to avoid actual threading)
        with patch.object(self.runner, '_monitoring_loop'), \
             patch.object(self.runner, '_scheduler_loop'):
            
            self.runner.start_automation()
            self.assertTrue(self.runner.running)
            self.assertEqual(self.runner.status, RunnerStatus.MONITORING)
            
            self.runner.stop_automation()
            self.assertFalse(self.runner.running)
            self.assertEqual(self.runner.status, RunnerStatus.IDLE)


class TestIntegration(unittest.TestCase):
    """Integration tests across components"""
    
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        logging.disable(logging.CRITICAL)  # Disable logging for tests
    
    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        logging.disable(logging.NOTSET)  # Re-enable logging
    
    def test_end_to_end_pipeline_flow(self):
        """Test complete pipeline flow integration"""
        # This would be a comprehensive integration test
        # For now, we test component interfaces are compatible
        
        config = SelfPlayConfig()
        
        # Test configuration propagation
        data_manager = SelfPlayDataManager(str(self.test_dir), config=config)
        evolution_system = ModelEvolutionSystem(str(self.test_dir), config=config)
        evaluator = ModelPerformanceEvaluator(str(self.test_dir), config=config)
        
        # Test component compatibility
        self.assertEqual(config.neural_network.input_channels, 12)
        self.assertIsNotNone(data_manager.config)
        self.assertIsNotNone(evolution_system.config)
        self.assertIsNotNone(evaluator.config)
    
    def test_data_flow_compatibility(self):
        """Test data flow between components"""
        config = SelfPlayConfig()
        
        # Create mock game data
        game_data = [{
            'id': 'test-game',
            'turns': [{
                'turn': 0,
                'board': {
                    'width': 11, 'height': 11,
                    'food': [{'x': 5, 'y': 5}],
                    'snakes': [{'id': 'snake1', 'head': {'x': 1, 'y': 1}, 'body': [{'x': 1, 'y': 1}], 'health': 100}]
                },
                'you': {'id': 'snake1', 'head': {'x': 1, 'y': 1}, 'body': [{'x': 1, 'y': 1}], 'health': 100},
                'move': 'up'
            }]
        }]
        
        # Process data
        processor = GameDataProcessor(config)
        samples = processor.process_game_data(game_data)
        
        # Verify data format compatibility
        self.assertGreater(len(samples), 0)
        sample = samples[0]
        
        # Check format matches neural network expectations
        self.assertEqual(sample['board_encoding'].shape, (12, 11, 11))
        self.assertIn(sample['action'], [0, 1, 2, 3])  # Valid action space


def create_test_suite():
    """Create comprehensive test suite"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestSelfPlayConfig,
        TestSelfPlayDataManager,
        TestModelEvolution,
        TestModelPerformanceEvaluator,
        TestSelfPlayTrainingPipeline,
        TestAutomatedTrainingRunner,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def run_performance_benchmarks():
    """Run performance benchmarks for critical components"""
    print("\n=== Performance Benchmarks ===")
    
    # Benchmark data processing
    config = SelfPlayConfig()
    processor = GameDataProcessor(config)
    
    # Create large mock dataset
    large_game_data = []
    for i in range(100):  # 100 games
        game_data = {
            'id': f'benchmark-game-{i}',
            'turns': []
        }
        
        for turn in range(50):  # 50 turns per game
            game_data['turns'].append({
                'turn': turn,
                'board': {
                    'width': 11, 'height': 11,
                    'food': [{'x': np.random.randint(0, 11), 'y': np.random.randint(0, 11)}],
                    'snakes': [{
                        'id': 'snake1',
                        'head': {'x': np.random.randint(0, 11), 'y': np.random.randint(0, 11)},
                        'body': [{'x': np.random.randint(0, 11), 'y': np.random.randint(0, 11)}],
                        'health': np.random.randint(1, 100)
                    }]
                },
                'you': {
                    'id': 'snake1',
                    'head': {'x': np.random.randint(0, 11), 'y': np.random.randint(0, 11)},
                    'body': [{'x': np.random.randint(0, 11), 'y': np.random.randint(0, 11)}],
                    'health': np.random.randint(1, 100)
                },
                'move': np.random.choice(['up', 'down', 'left', 'right'])
            })
        
        large_game_data.append(game_data)
    
    # Benchmark processing time
    import time
    
    start_time = time.time()
    processed_samples = processor.process_game_data(large_game_data)
    processing_time = time.time() - start_time
    
    print(f"Data Processing Benchmark:")
    print(f"  Games processed: {len(large_game_data)}")
    print(f"  Samples generated: {len(processed_samples)}")
    print(f"  Processing time: {processing_time:.2f} seconds")
    print(f"  Samples per second: {len(processed_samples)/processing_time:.1f}")
    
    # Memory usage benchmark
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    print(f"  Memory usage: {memory_usage:.1f} MB")


if __name__ == '__main__':
    # Setup test environment
    print("=== Self-Play Training Pipeline Test Suite ===")
    
    # Run unit tests
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    
    print("\n=== Running Unit Tests ===")
    result = runner.run(suite)
    
    # Run performance benchmarks
    run_performance_benchmarks()
    
    # Print summary
    print(f"\n=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    # Exit with error code if tests failed
    if result.failures or result.errors:
        sys.exit(1)
    else:
        print(f"\n✓ All tests passed! Pipeline ready for production deployment.")
        sys.exit(0)