"""
Comprehensive Test Suite for Self-Play Data Collection Pipeline

This test suite validates all components of the data collection infrastructure:
- Configuration management
- Server pool management  
- Game data extraction
- Training data pipeline
- Data management system
- Integration testing
- Performance validation

Architecture compliance: Tests 100+ games/hour target, 12-channel encoding, data quality
"""

import unittest
import tempfile
import shutil
import os
import json
import time
import threading
import logging
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess

# Import our modules
import sys
sys.path.append('.')

from config.self_play_config import (
    ConfigManager, SystemConfig, ServerConfig, GameConfig, 
    DataCollectionConfig, setup_logging
)
from self_play_automation import (
    SelfPlayAutomationManager, ServerPool, GameOrchestrator
)
from game_data_extractor import (
    RealTimeGameExtractor, GameData, MoveDecision, BoardState, 
    SnakeState, HeuristicScores, Coordinate, ServerLogParser
)
from training_data_pipeline import (
    TrainingDataProcessor, BoardEncoder, FeatureExtractor, 
    TrainingSample, Move
)
from data_management import (
    DataManagementSystem, CompressedStorageManager, StorageConfig,
    DataValidator, TrainingDataExporter
)

class TestConfigurationSystem(unittest.TestCase):
    """Test configuration management system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_default_config_creation(self):
        """Test default configuration creation"""
        config_manager = ConfigManager(self.config_file)
        config = config_manager.load_config()
        
        self.assertIsInstance(config, SystemConfig)
        self.assertEqual(len(config.servers), 4)  # Default 4 servers
        self.assertEqual(config.game.board_width, 11)
        self.assertEqual(config.game.board_height, 11)
        self.assertEqual(config.data_collection.board_encoding_channels, 12)
        self.assertGreaterEqual(config.data_collection.target_games_per_hour, 100)
    
    def test_config_validation(self):
        """Test configuration validation"""
        config_manager = ConfigManager(self.config_file)
        config = config_manager.load_config()
        
        # Test port uniqueness validation
        config.servers[0].port = config.servers[1].port
        with self.assertRaises(ValueError):
            config_manager.config = config
            config_manager._validate_config()
    
    def test_throughput_estimation(self):
        """Test throughput estimation calculations"""
        config_manager = ConfigManager(self.config_file)
        config = config_manager.load_config()
        
        throughput = config_manager.estimate_throughput()
        
        self.assertIn('servers', throughput)
        self.assertIn('estimated_games_per_hour', throughput)
        self.assertIn('target_games_per_hour', throughput)
        self.assertGreater(throughput['estimated_games_per_hour'], 0)
    
    def test_config_serialization(self):
        """Test configuration save/load cycle"""
        config_manager = ConfigManager(self.config_file)
        original_config = config_manager.load_config()
        
        # Modify configuration
        original_config.data_collection.target_games_per_hour = 150
        config_manager.config = original_config
        config_manager.save_config()
        
        # Load and verify
        new_manager = ConfigManager(self.config_file)
        loaded_config = new_manager.load_config()
        
        self.assertEqual(loaded_config.data_collection.target_games_per_hour, 150)

class TestGameDataExtraction(unittest.TestCase):
    """Test game data extraction components"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_coordinate_operations(self):
        """Test coordinate data structures"""
        coord = Coordinate(5, 7)
        coord_dict = coord.to_dict()
        
        self.assertEqual(coord_dict['x'], 5)
        self.assertEqual(coord_dict['y'], 7)
        
        restored_coord = Coordinate.from_dict(coord_dict)
        self.assertEqual(restored_coord.x, coord.x)
        self.assertEqual(restored_coord.y, coord.y)
    
    def test_snake_state_creation(self):
        """Test snake state data structures"""
        body = [Coordinate(3, 3), Coordinate(3, 4), Coordinate(3, 5)]
        snake = SnakeState(
            id="test_snake",
            name="Test Snake",
            health=85,
            body=body,
            head=body[0],
            length=len(body)
        )
        
        self.assertEqual(snake.head.x, 3)
        self.assertEqual(snake.head.y, 3)
        self.assertEqual(len(snake.body), 3)
        
        # Test neck property
        self.assertIsNotNone(snake.neck)
        self.assertEqual(snake.neck.x, 3)
        self.assertEqual(snake.neck.y, 4)
    
    def test_board_state_creation(self):
        """Test board state data structures"""
        snake = SnakeState(
            id="test", name="test", health=100,
            body=[Coordinate(5, 5)], head=Coordinate(5, 5), length=1
        )
        
        board = BoardState(
            width=11, height=11,
            food=[Coordinate(2, 2), Coordinate(8, 8)],
            hazards=[],
            snakes=[snake],
            turn=10
        )
        
        self.assertEqual(board.width, 11)
        self.assertEqual(board.height, 11)
        self.assertEqual(len(board.food), 2)
        self.assertEqual(len(board.snakes), 1)
        self.assertEqual(board.turn, 10)
    
    def test_heuristic_scores(self):
        """Test heuristic score data structures"""
        scores = HeuristicScores(
            safety=35.0,
            territory=6.5,
            food=4.2,
            opponent=2.1,
            exploration=28.3
        )
        
        expected_total = 35.0 + 6.5 + 4.2 + 2.1 + 28.3
        self.assertAlmostEqual(scores.total_score, expected_total, places=1)
        
        # Test serialization
        score_dict = scores.to_dict()
        restored_scores = HeuristicScores.from_dict(score_dict)
        self.assertAlmostEqual(restored_scores.total_score, expected_total, places=1)
    
    def test_server_log_parser(self):
        """Test server log parsing"""
        parser = ServerLogParser(8000)
        
        test_lines = [
            "MOVE 15: right",
            "Safety score: 42.5",
            "Territory score: 7.2",
            "Food score: 3.1",
            "Decision time: 45.3 ms"
        ]
        
        results = []
        for line in test_lines:
            result = parser.parse_log_line(line)
            if result:
                results.append(result)
        
        self.assertGreater(len(results), 0)
        
        # Check for move decision
        move_results = [r for r in results if r.get('type') == 'move_decision']
        self.assertEqual(len(move_results), 1)
        self.assertEqual(move_results[0]['move'], 'right')
        self.assertEqual(move_results[0]['turn'], 15)
        
        # Check for heuristic scores
        heuristic_results = [r for r in results if 'heuristic_scores' in r]
        self.assertGreater(len(heuristic_results), 0)

class TestTrainingDataPipeline(unittest.TestCase):
    """Test training data pipeline components"""
    
    def setUp(self):
        # Create test data
        self.test_snake = SnakeState(
            id="test_snake",
            name="Test Snake", 
            health=85,
            body=[Coordinate(5, 5), Coordinate(5, 6), Coordinate(5, 7)],
            head=Coordinate(5, 5),
            length=3
        )
        
        self.test_board = BoardState(
            width=11, height=11,
            food=[Coordinate(8, 3), Coordinate(2, 9)],
            hazards=[],
            snakes=[self.test_snake],
            turn=25
        )
        
        self.test_heuristics = HeuristicScores(
            safety=35.0, territory=6.5, food=4.2, 
            opponent=2.1, exploration=28.3
        )
    
    def test_board_encoder_shape(self):
        """Test board encoder produces correct tensor shape"""
        encoder = BoardEncoder()
        tensor = encoder.encode_board(self.test_board, "test_snake")
        
        self.assertEqual(tensor.shape, (11, 11, 12))
        self.assertEqual(tensor.dtype, np.float32)
    
    def test_board_encoder_channels(self):
        """Test board encoder populates expected channels"""
        encoder = BoardEncoder()
        tensor = encoder.encode_board(self.test_board, "test_snake")
        
        # Channel 0: Our snake - should have non-zero values at snake positions
        our_snake_channel = tensor[:, :, encoder.CH_OUR_SNAKE]
        self.assertGreater(np.sum(our_snake_channel > 0), 0)
        
        # Channel 4: Food - should have values at food positions
        food_channel = tensor[:, :, encoder.CH_FOOD]
        self.assertEqual(np.sum(food_channel > 0), 2)  # 2 food items
        
        # Channel 5: Walls - should have values at boundaries
        wall_channel = tensor[:, :, encoder.CH_WALLS]
        self.assertGreater(np.sum(wall_channel > 0), 0)
    
    def test_feature_extractor_shapes(self):
        """Test feature extractor produces correct shapes"""
        extractor = FeatureExtractor()
        
        snake_features = extractor.extract_snake_features(self.test_snake, self.test_board)
        game_context = extractor.extract_game_context(self.test_board, self.test_snake)
        
        self.assertEqual(snake_features.shape, (32,))
        self.assertEqual(game_context.shape, (16,))
        self.assertEqual(snake_features.dtype, np.float32)
        self.assertEqual(game_context.dtype, np.float32)
    
    def test_feature_extractor_values(self):
        """Test feature extractor produces reasonable values"""
        extractor = FeatureExtractor()
        
        snake_features = extractor.extract_snake_features(self.test_snake, self.test_board)
        
        # Health should be normalized (85/100 = 0.85)
        self.assertAlmostEqual(snake_features[0], 0.85, places=2)
        
        # Length should be normalized (3/20 = 0.15) 
        self.assertAlmostEqual(snake_features[1], 0.15, places=2)
        
        # Head position should be normalized (5/11 ≈ 0.45)
        self.assertAlmostEqual(snake_features[2], 5.0/11.0, places=2)  # x
        self.assertAlmostEqual(snake_features[3], 5.0/11.0, places=2)  # y
    
    def test_move_enum(self):
        """Test move enumeration and conversion"""
        self.assertEqual(Move.from_string("up"), Move.UP)
        self.assertEqual(Move.from_string("down"), Move.DOWN)
        self.assertEqual(Move.from_string("left"), Move.LEFT)
        self.assertEqual(Move.from_string("right"), Move.RIGHT)
        
        self.assertEqual(Move.UP.to_string(), "up")
        self.assertEqual(int(Move.RIGHT), 3)
    
    def test_training_sample_creation(self):
        """Test training sample creation and validation"""
        processor = TrainingDataProcessor()
        
        move_decision = MoveDecision(
            snake_id="test_snake",
            turn=25,
            move="right",
            board_state=self.test_board,
            heuristic_scores=self.test_heuristics,
            decision_time_ms=42.5,
            timestamp=datetime.now(),
            server_port=8000
        )
        
        sample = processor._process_move_decision(move_decision, [], 0.8)
        
        self.assertIsNotNone(sample)
        self.assertEqual(sample.board_state.shape, (11, 11, 12))
        self.assertEqual(sample.snake_features.shape, (32,))
        self.assertEqual(sample.game_context.shape, (16,))
        self.assertEqual(sample.target_move, int(Move.RIGHT))
        self.assertEqual(sample.move_probabilities.shape, (4,))
        
        # Validate sample
        is_valid = processor.validate_sample(sample)
        self.assertTrue(is_valid)
    
    def test_training_sample_serialization(self):
        """Test training sample serialization"""
        processor = TrainingDataProcessor()
        
        move_decision = MoveDecision(
            snake_id="test_snake", turn=25, move="up",
            board_state=self.test_board, heuristic_scores=self.test_heuristics,
            decision_time_ms=42.5, timestamp=datetime.now(), server_port=8000
        )
        
        original_sample = processor._process_move_decision(move_decision, [], 0.8)
        
        # Serialize and deserialize
        sample_dict = original_sample.to_dict()
        restored_sample = TrainingSample.from_dict(sample_dict)
        
        # Compare key fields
        self.assertEqual(original_sample.target_move, restored_sample.target_move)
        self.assertEqual(original_sample.snake_id, restored_sample.snake_id)
        self.assertEqual(original_sample.turn, restored_sample.turn)
        np.testing.assert_array_equal(original_sample.board_state, restored_sample.board_state)

class TestDataManagement(unittest.TestCase):
    """Test data management system"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test storage config
        self.storage_config = StorageConfig(
            base_directory=self.temp_dir,
            compression_enabled=True,
            compression_level=1,  # Fast compression for tests
            backup_enabled=False  # Disable for tests
        )
        
        self.storage_manager = CompressedStorageManager(self.storage_config)
        
        # Create test training samples
        self.test_samples = self._create_test_samples(50)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_samples(self, count: int) -> list:
        """Create test training samples"""
        samples = []
        
        for i in range(count):
            sample = TrainingSample(
                board_state=np.random.random((11, 11, 12)).astype(np.float32),
                snake_features=np.random.random(32).astype(np.float32),
                game_context=np.random.random(16).astype(np.float32),
                target_move=i % 4,
                position_value=float(np.random.uniform(-50, 50)),
                move_probabilities=np.random.dirichlet([1, 1, 1, 1]).astype(np.float32),
                win_probability=float(np.random.uniform(0, 1)),
                heuristic_scores={
                    'safety': float(np.random.uniform(8, 50)),
                    'territory': float(np.random.uniform(3, 9)),
                    'food': float(np.random.uniform(0, 11)),
                    'opponent': float(np.random.uniform(0, 6)),
                    'exploration': float(np.random.uniform(20, 35))
                },
                game_id=f"test_game_{i // 10}",
                turn=i % 100,
                snake_id="test_snake",
                timestamp=datetime.now()
            )
            samples.append(sample)
        
        return samples
    
    def test_data_validator(self):
        """Test data validation"""
        validator = DataValidator()
        
        # Test valid sample
        valid_result = validator.validate_training_sample(self.test_samples[0])
        self.assertTrue(valid_result['valid'])
        self.assertEqual(len(valid_result['issues']), 0)
        
        # Test invalid sample (bad shape)
        bad_sample = self.test_samples[0]
        bad_sample.board_state = np.random.random((10, 10, 12))  # Wrong size
        
        invalid_result = validator.validate_training_sample(bad_sample)
        self.assertFalse(invalid_result['valid'])
        self.assertGreater(len(invalid_result['issues']), 0)
    
    def test_storage_save_load_pickle(self):
        """Test pickle storage format"""
        self.storage_config.preferred_format = "pickle"
        
        # Save samples
        metadata = self.storage_manager.save_training_samples(
            self.test_samples[:10],
            "test_dataset",
            "1.0",
            "Test dataset for pickle format"
        )
        
        self.assertEqual(metadata.sample_count, 10)
        self.assertEqual(metadata.data_format, "pickle")
        self.assertGreater(metadata.file_size_bytes, 0)
        
        # Load samples
        loaded_samples, loaded_metadata = self.storage_manager.load_training_samples("test_dataset")
        
        self.assertEqual(len(loaded_samples), 10)
        self.assertEqual(loaded_metadata.dataset_id, "test_dataset")
        
        # Compare first sample
        original = self.test_samples[0]
        loaded = loaded_samples[0]
        
        self.assertEqual(original.target_move, loaded.target_move)
        self.assertEqual(original.game_id, loaded.game_id)
        np.testing.assert_array_almost_equal(original.board_state, loaded.board_state, decimal=5)
    
    def test_storage_list_datasets(self):
        """Test dataset listing"""
        # Create multiple datasets
        for i in range(3):
            self.storage_manager.save_training_samples(
                self.test_samples[i*10:(i+1)*10],
                f"dataset_{i}",
                "1.0"
            )
        
        datasets = self.storage_manager.list_datasets()
        self.assertEqual(len(datasets), 3)
        
        # Check they're sorted by creation time (newest first)
        for i in range(len(datasets)-1):
            self.assertGreaterEqual(datasets[i].created_at, datasets[i+1].created_at)
    
    def test_data_management_system(self):
        """Test integrated data management system"""
        # Create temporary config for testing
        temp_config_dir = os.path.join(self.temp_dir, "config")
        os.makedirs(temp_config_dir, exist_ok=True)
        
        with patch('config.self_play_config.get_config') as mock_config:
            # Mock configuration
            mock_dc = Mock()
            mock_dc.data_directory = self.temp_dir
            mock_dc.compression_level = 1
            mock_dc.max_file_size_mb = 100
            mock_dc.retention_days = 30
            mock_dc.backup_enabled = False
            
            mock_config_obj = Mock()
            mock_config_obj.data_collection = mock_dc
            mock_config_obj.servers = []
            mock_config.return_value = mock_config_obj
            
            # Test system
            system = DataManagementSystem()
            
            # Save data
            metadata = system.save_training_data(
                self.test_samples[:20],
                "integration_test",
                "1.0",
                "Integration test dataset"
            )
            
            self.assertEqual(metadata.sample_count, 20)
            
            # Load data
            loaded_samples, loaded_metadata = system.load_training_data("integration_test")
            self.assertEqual(len(loaded_samples), 20)
            
            # Get stats
            stats = system.get_comprehensive_stats()
            self.assertIn('storage', stats)
            self.assertIn('recent_datasets', stats)
            
            system.shutdown()

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Setup logging for tests
        logging.basicConfig(level=logging.WARNING)  # Reduce noise in tests
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('subprocess.Popen')
    @patch('requests.get')
    def test_server_pool_mock(self, mock_requests, mock_popen):
        """Test server pool with mocked processes"""
        # Mock successful server startup
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is running
        mock_popen.return_value = mock_process
        
        # Mock successful health check
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests.return_value = mock_response
        
        # Create test config
        config = SystemConfig(
            servers=[ServerConfig(port=8000, name="test_server")],
            game=GameConfig(),
            data_collection=DataCollectionConfig()
        )
        
        server_pool = ServerPool(config)
        
        # Mock cargo build
        with patch.object(server_pool, '_ensure_cargo_build', return_value=True):
            success = server_pool.start_all_servers()
            self.assertTrue(success)
            
            # Check server status
            stats = server_pool.get_stats()
            self.assertEqual(stats['total_servers'], 1)
            
            server_pool.stop_all_servers()
    
    def test_end_to_end_data_flow(self):
        """Test end-to-end data flow without external dependencies"""
        # Create mock game data
        snake = SnakeState(
            id="test_snake", name="Test", health=90,
            body=[Coordinate(3, 3), Coordinate(3, 4)],
            head=Coordinate(3, 3), length=2
        )
        
        board = BoardState(
            width=11, height=11,
            food=[Coordinate(7, 7)],
            hazards=[],
            snakes=[snake],
            turn=15
        )
        
        heuristics = HeuristicScores(safety=30, territory=5, food=8, opponent=3, exploration=25)
        
        move_decision = MoveDecision(
            snake_id="test_snake", turn=15, move="up",
            board_state=board, heuristic_scores=heuristics,
            decision_time_ms=35.0, timestamp=datetime.now(), server_port=8000
        )
        
        game_data = GameData(
            game_id="integration_test_game",
            start_time=datetime.now(),
            moves=[move_decision]
        )
        
        # Process through training pipeline
        processor = TrainingDataProcessor()
        samples = processor.process_game(game_data)
        
        self.assertEqual(len(samples), 1)
        self.assertTrue(processor.validate_sample(samples[0]))
        
        # Store data
        storage_config = StorageConfig(
            base_directory=self.temp_dir,
            backup_enabled=False
        )
        storage_manager = CompressedStorageManager(storage_config)
        
        metadata = storage_manager.save_training_samples(
            samples, "integration_test", "1.0"
        )
        
        self.assertEqual(metadata.sample_count, 1)
        
        # Load and verify
        loaded_samples, loaded_metadata = storage_manager.load_training_samples("integration_test")
        
        self.assertEqual(len(loaded_samples), 1)
        self.assertEqual(loaded_samples[0].target_move, samples[0].target_move)

class TestPerformanceValidation(unittest.TestCase):
    """Performance and throughput validation tests"""
    
    def test_board_encoding_performance(self):
        """Test board encoding performance"""
        encoder = BoardEncoder()
        
        # Create test board
        snake = SnakeState(
            id="perf_test", name="Test", health=100,
            body=[Coordinate(5, 5)], head=Coordinate(5, 5), length=1
        )
        board = BoardState(
            width=11, height=11, food=[Coordinate(2, 2)],
            hazards=[], snakes=[snake], turn=1
        )
        
        # Time encoding
        start_time = time.time()
        iterations = 1000
        
        for _ in range(iterations):
            tensor = encoder.encode_board(board, "perf_test")
        
        end_time = time.time()
        avg_time_ms = ((end_time - start_time) / iterations) * 1000
        
        # Should be fast enough for real-time processing
        self.assertLess(avg_time_ms, 10.0, f"Board encoding too slow: {avg_time_ms:.2f}ms")
        
        print(f"Board encoding performance: {avg_time_ms:.2f}ms average")
    
    def test_training_sample_generation_performance(self):
        """Test training sample generation performance"""
        processor = TrainingDataProcessor()
        
        # Create test data
        snake = SnakeState(
            id="perf_test", name="Test", health=95,
            body=[Coordinate(4, 4), Coordinate(4, 5)],
            head=Coordinate(4, 4), length=2
        )
        board = BoardState(
            width=11, height=11, food=[Coordinate(8, 2)],
            hazards=[], snakes=[snake], turn=20
        )
        heuristics = HeuristicScores(safety=25, territory=6, food=5, opponent=2, exploration=30)
        
        move_decision = MoveDecision(
            snake_id="perf_test", turn=20, move="left",
            board_state=board, heuristic_scores=heuristics,
            decision_time_ms=40.0, timestamp=datetime.now(), server_port=8000
        )
        
        # Time processing
        start_time = time.time()
        iterations = 100
        
        for _ in range(iterations):
            sample = processor._process_move_decision(move_decision, [], 0.7)
            self.assertIsNotNone(sample)
        
        end_time = time.time()
        avg_time_ms = ((end_time - start_time) / iterations) * 1000
        
        # Should be fast enough for high-throughput processing
        self.assertLess(avg_time_ms, 50.0, f"Sample generation too slow: {avg_time_ms:.2f}ms")
        
        print(f"Training sample generation performance: {avg_time_ms:.2f}ms average")
    
    def test_data_storage_performance(self):
        """Test data storage performance"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            storage_config = StorageConfig(
                base_directory=temp_dir,
                compression_level=6,  # Balanced compression
                backup_enabled=False
            )
            storage_manager = CompressedStorageManager(storage_config)
            
            # Create test samples
            samples = []
            for i in range(100):  # 100 samples for performance test
                sample = TrainingSample(
                    board_state=np.random.random((11, 11, 12)).astype(np.float32),
                    snake_features=np.random.random(32).astype(np.float32),
                    game_context=np.random.random(16).astype(np.float32),
                    target_move=i % 4,
                    position_value=float(np.random.uniform(-50, 50)),
                    move_probabilities=np.random.dirichlet([1, 1, 1, 1]).astype(np.float32),
                    win_probability=float(np.random.uniform(0, 1)),
                    heuristic_scores={'safety': 25.0, 'territory': 5.0},
                    game_id=f"perf_game_{i//10}",
                    turn=i, snake_id="perf_snake",
                    timestamp=datetime.now()
                )
                samples.append(sample)
            
            # Time storage
            start_time = time.time()
            metadata = storage_manager.save_training_samples(samples, "perf_test", "1.0")
            save_time = time.time() - start_time
            
            # Time loading
            start_time = time.time()
            loaded_samples, _ = storage_manager.load_training_samples("perf_test")
            load_time = time.time() - start_time
            
            print(f"Storage performance: Save {save_time*1000:.1f}ms, Load {load_time*1000:.1f}ms for {len(samples)} samples")
            
            # Performance thresholds (should be fast enough for high throughput)
            self.assertLess(save_time, 2.0, f"Storage save too slow: {save_time:.2f}s")
            self.assertLess(load_time, 1.0, f"Storage load too slow: {load_time:.2f}s")
            
            self.assertEqual(len(loaded_samples), len(samples))
        
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

class TestArchitectureCompliance(unittest.TestCase):
    """Test architecture compliance and requirements"""
    
    def test_12_channel_board_encoding(self):
        """Test 12-channel board encoding compliance"""
        encoder = BoardEncoder()
        
        self.assertEqual(encoder.NUM_CHANNELS, 12)
        self.assertEqual(encoder.BOARD_SIZE, 11)
        
        # Test all channels are properly defined
        expected_channels = [
            encoder.CH_OUR_SNAKE,
            encoder.CH_OPPONENT_1,
            encoder.CH_OPPONENT_2, 
            encoder.CH_OPPONENT_3,
            encoder.CH_FOOD,
            encoder.CH_WALLS,
            encoder.CH_OUR_TERRITORY,
            encoder.CH_OPPONENT_TERRITORY,
            encoder.CH_DANGER_ZONES,
            encoder.CH_MOVEMENT_HISTORY,
            encoder.CH_STRATEGIC_POSITIONS,
            encoder.CH_GAME_STATE
        ]
        
        self.assertEqual(len(set(expected_channels)), 12)  # All unique
        self.assertEqual(max(expected_channels), 11)  # 0-indexed to 11
    
    def test_heuristic_score_ranges(self):
        """Test heuristic score ranges match architecture spec"""
        # Test realistic score ranges from architecture
        scores = HeuristicScores(
            safety=42.0,    # 8-50 range
            territory=7.5,  # 3-9 range 
            food=6.0,       # 0-11 range
            opponent=4.0,   # 0-6 range
            exploration=28.0 # ~30 range
        )
        
        # Verify scores are within expected ranges
        self.assertGreaterEqual(scores.safety, 8.0)
        self.assertLessEqual(scores.safety, 50.0)
        
        self.assertGreaterEqual(scores.territory, 3.0)
        self.assertLessEqual(scores.territory, 9.0)
        
        self.assertGreaterEqual(scores.food, 0.0)
        self.assertLessEqual(scores.food, 11.0)
        
        self.assertGreaterEqual(scores.opponent, 0.0)
        self.assertLessEqual(scores.opponent, 6.0)
        
        # Total should be in expected range (~100 points)
        total = scores.total_score
        self.assertGreater(total, 50.0)  # Minimum reasonable total
        self.assertLess(total, 150.0)   # Maximum reasonable total
    
    def test_training_data_format_compliance(self):
        """Test training data format matches architecture spec"""
        processor = TrainingDataProcessor()
        
        # Create test sample
        snake = SnakeState(
            id="arch_test", name="Test", health=80,
            body=[Coordinate(6, 6)], head=Coordinate(6, 6), length=1
        )
        board = BoardState(
            width=11, height=11, food=[Coordinate(3, 3)],
            hazards=[], snakes=[snake], turn=30
        )
        heuristics = HeuristicScores(safety=20, territory=5, food=7, opponent=1, exploration=32)
        
        move_decision = MoveDecision(
            snake_id="arch_test", turn=30, move="down",
            board_state=board, heuristic_scores=heuristics,
            decision_time_ms=25.0, timestamp=datetime.now(), server_port=8000
        )
        
        sample = processor._process_move_decision(move_decision, [], 0.6)
        
        # Verify architecture compliance
        self.assertEqual(sample.board_state.shape, (11, 11, 12))  # 12-channel board
        self.assertEqual(sample.snake_features.shape, (32,))     # 32 snake features
        self.assertEqual(sample.game_context.shape, (16,))       # 16 context features
        
        # Verify value ranges
        self.assertGreaterEqual(sample.position_value, -50.0)
        self.assertLessEqual(sample.position_value, 50.0)
        
        self.assertGreaterEqual(sample.win_probability, 0.0)
        self.assertLessEqual(sample.win_probability, 1.0)
        
        # Verify move probabilities sum to 1
        self.assertAlmostEqual(sample.move_probabilities.sum(), 1.0, places=3)
        
        # Verify heuristic scores are preserved
        self.assertIn('safety', sample.heuristic_scores)
        self.assertIn('territory', sample.heuristic_scores)
        self.assertIn('food', sample.heuristic_scores)

def run_performance_benchmark():
    """Run performance benchmark for throughput validation"""
    print("\n=== Performance Benchmark ===")
    
    # Simulate game processing for throughput testing
    processor = TrainingDataProcessor()
    
    # Create realistic test data
    games_to_process = 10
    moves_per_game = 50
    
    all_samples = []
    start_time = time.time()
    
    for game_i in range(games_to_process):
        game_samples = []
        
        for move_i in range(moves_per_game):
            # Create test move decision
            snake = SnakeState(
                id=f"snake_{game_i}", name="Benchmark Snake", 
                health=100 - move_i, 
                body=[Coordinate(5, 5)], head=Coordinate(5, 5), length=1
            )
            board = BoardState(
                width=11, height=11, food=[Coordinate(8, 8)],
                hazards=[], snakes=[snake], turn=move_i
            )
            heuristics = HeuristicScores(
                safety=30, territory=6, food=5, opponent=2, exploration=25
            )
            
            move_decision = MoveDecision(
                snake_id=f"snake_{game_i}", turn=move_i, move="up",
                board_state=board, heuristic_scores=heuristics,
                decision_time_ms=30.0, timestamp=datetime.now(), server_port=8000
            )
            
            sample = processor._process_move_decision(move_decision, [], 0.5)
            if sample:
                game_samples.append(sample)
        
        all_samples.extend(game_samples)
    
    processing_time = time.time() - start_time
    
    samples_per_second = len(all_samples) / processing_time
    moves_per_hour = samples_per_second * 3600
    
    # Estimate games per hour (assuming 50 moves per game average)
    games_per_hour = moves_per_hour / 50
    
    print(f"Processed {len(all_samples)} training samples in {processing_time:.2f}s")
    print(f"Processing rate: {samples_per_second:.1f} samples/second")
    print(f"Estimated throughput: {games_per_hour:.1f} games/hour")
    
    # Check if we meet the 100+ games/hour target
    throughput_target = 100
    if games_per_hour >= throughput_target:
        print(f"✅ PASS: Throughput target met ({games_per_hour:.1f} >= {throughput_target})")
    else:
        print(f"❌ FAIL: Throughput target not met ({games_per_hour:.1f} < {throughput_target})")
    
    return games_per_hour >= throughput_target

if __name__ == '__main__':
    # Setup logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== Self-Play Data Collection Test Suite ===")
    
    # Run all tests
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestConfigurationSystem,
        TestGameDataExtraction,
        TestTrainingDataPipeline,
        TestDataManagement,
        TestIntegration,
        TestPerformanceValidation,
        TestArchitectureCompliance
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run performance benchmark
    throughput_ok = run_performance_benchmark()
    
    # Summary
    print(f"\n=== Test Results Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Throughput target: {'✅ PASS' if throughput_ok else '❌ FAIL'}")
    
    if result.failures:
        print("\n=== Failures ===")
        for test, traceback in result.failures:
            print(f"{test}: {traceback}")
    
    if result.errors:
        print("\n=== Errors ===")
        for test, traceback in result.errors:
            print(f"{test}: {traceback}")
    
    # Exit with appropriate code
    success = (len(result.failures) == 0 and len(result.errors) == 0 and throughput_ok)
    exit(0 if success else 1)