"""
Self-Play Training Data Manager

Dynamic training data pipeline for the self-play training system. Handles data collection
from multiple sources, quality control, experience replay, and training sample generation
for the multi-phase training pipeline (Bootstrap -> Hybrid -> Self-Play -> Continuous).

Integrates with Phase 8 data collection (332K games/hour) and Phase 9 neural networks.
"""

import os
import json
import pickle
import gzip
import sqlite3
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Iterator
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import queue
import random
from concurrent.futures import ThreadPoolExecutor

# Import existing infrastructure
from neural_networks.data_collection import GameDataCollector, GameRecord, TrainingSample
from neural_networks.board_encoding import BoardStateEncoder
from config.self_play_config import get_config, TrainingPhaseConfig


@dataclass
class SelfPlayTrainingSample:
    """Enhanced training sample for self-play training pipeline"""
    # Core data (12-channel board encoding)
    board_state: np.ndarray  # Shape: (11, 11, 12)
    snake_features: np.ndarray  # Shape: (32,)
    game_context: np.ndarray  # Shape: (16,)
    
    # Target labels
    target_move: int  # 0=up, 1=down, 2=left, 3=right
    position_value: float  # Actual game outcome value (-1 to +1)
    move_probabilities: np.ndarray  # Shape: (4,) - move distribution
    game_outcome: float  # Final game result (-1=loss, 0=draw, +1=win)
    
    # Metadata for training pipeline
    model_version: str  # Model that generated this decision
    opponent_strength: float  # Opponent capability rating (0-1)
    training_phase: str  # 'bootstrap', 'hybrid', 'self_play', 'continuous'
    quality_score: float  # Data quality metric (0-1)
    
    # Game context
    turn: int
    game_id: str
    timestamp: str
    
    # Heuristic supervision (for bootstrap and hybrid phases)
    heuristic_scores: Optional[Dict[str, float]] = None
    heuristic_move_ranking: Optional[List[int]] = None


class ExperienceReplayBuffer:
    """High-performance experience replay buffer for training optimization"""
    
    def __init__(self, max_capacity: int = 100000, min_quality_threshold: float = 0.3):
        self.max_capacity = max_capacity
        self.min_quality_threshold = min_quality_threshold
        
        self.samples: deque = deque(maxlen=max_capacity)
        self.quality_index = {}  # quality_score -> list of indices
        self.phase_index = defaultdict(list)  # training_phase -> list of indices
        
        self.lock = threading.RLock()
        
    def add_sample(self, sample: SelfPlayTrainingSample):
        """Add sample to replay buffer with quality filtering"""
        with self.lock:
            if sample.quality_score < self.min_quality_threshold:
                return False
            
            # Add to main buffer
            index = len(self.samples)
            self.samples.append(sample)
            
            # Update indices
            quality_bucket = int(sample.quality_score * 10) / 10
            if quality_bucket not in self.quality_index:
                self.quality_index[quality_bucket] = []
            self.quality_index[quality_bucket].append(index)
            
            self.phase_index[sample.training_phase].append(index)
            
            return True
    
    def sample_batch(self, batch_size: int, phase_filter: Optional[str] = None,
                    quality_bias: float = 0.7) -> List[SelfPlayTrainingSample]:
        """Sample batch with quality biasing and phase filtering"""
        with self.lock:
            if len(self.samples) == 0:
                return []
            
            # Apply phase filter
            if phase_filter and phase_filter in self.phase_index:
                available_indices = self.phase_index[phase_filter]
            else:
                available_indices = list(range(len(self.samples)))
            
            if not available_indices:
                return []
            
            # Quality-biased sampling
            if quality_bias > 0:
                weights = []
                for idx in available_indices:
                    sample = self.samples[idx]
                    weight = 1.0 + (sample.quality_score * quality_bias)
                    weights.append(weight)
                
                # Weighted random selection
                selected_indices = np.random.choice(
                    available_indices,
                    size=min(batch_size, len(available_indices)),
                    replace=False,
                    p=np.array(weights) / sum(weights)
                )
            else:
                # Uniform sampling
                selected_indices = random.sample(
                    available_indices,
                    min(batch_size, len(available_indices))
                )
            
            return [self.samples[idx] for idx in selected_indices]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            if len(self.samples) == 0:
                return {'size': 0, 'phases': {}, 'quality_distribution': {}}
            
            phase_counts = {phase: len(indices) for phase, indices in self.phase_index.items()}
            quality_dist = {}
            
            for sample in self.samples:
                quality_bucket = int(sample.quality_score * 10) / 10
                quality_dist[quality_bucket] = quality_dist.get(quality_bucket, 0) + 1
            
            return {
                'size': len(self.samples),
                'capacity': self.max_capacity,
                'phases': phase_counts,
                'quality_distribution': quality_dist,
                'avg_quality': np.mean([s.quality_score for s in self.samples])
            }


class SelfPlayDataManager:
    """Main data manager for self-play training pipeline"""
    
    def __init__(self, config_manager=None):
        self.config = get_config() if config_manager is None else config_manager.load_config()
        self.data_config = self.config.data_collection
        
        # Setup directories
        self.base_dir = Path(self.data_config.data_directory)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.training_dir = self.base_dir / "training"
        self.metadata_dir = self.base_dir / "metadata"
        
        for dir_path in [self.raw_dir, self.processed_dir, self.training_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.board_encoder = BoardStateEncoder()
        self.experience_buffer = ExperienceReplayBuffer(
            max_capacity=50000,  # Configurable based on system memory
            min_quality_threshold=0.3
        )
        
        # Data tracking database
        self.db_path = self.base_dir / "data_tracking.db"
        self._init_database()
        
        # Performance metrics
        self.metrics = {
            'samples_processed': 0,
            'samples_generated': 0,
            'quality_filtered': 0,
            'heuristic_samples': 0,
            'self_play_samples': 0,
            'start_time': datetime.now()
        }
        
        # Processing threads
        self.processing_pool = ThreadPoolExecutor(max_workers=4)
        self.processing_queue = queue.Queue(maxsize=1000)
        self.shutdown_event = threading.Event()
    
    def _init_database(self):
        """Initialize SQLite database for data tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_samples (
                    id INTEGER PRIMARY KEY,
                    game_id TEXT,
                    turn INTEGER,
                    training_phase TEXT,
                    model_version TEXT,
                    quality_score REAL,
                    position_value REAL,
                    game_outcome REAL,
                    timestamp TEXT,
                    file_path TEXT,
                    processed BOOLEAN DEFAULT FALSE
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_batches (
                    id INTEGER PRIMARY KEY,
                    batch_name TEXT UNIQUE,
                    training_phase TEXT,
                    sample_count INTEGER,
                    avg_quality REAL,
                    created_at TEXT,
                    file_path TEXT
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_training_phase ON training_samples(training_phase)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quality_score ON training_samples(quality_score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_processed ON training_samples(processed)")
    
    def collect_heuristic_supervision_data(self, num_games: int, 
                                         model_version: str = "heuristic_v1.0") -> int:
        """Collect training data with heuristic supervision for bootstrap phase"""
        self.logger.info(f"Starting heuristic supervision data collection: {num_games} games")
        
        collected_samples = 0
        
        # Use existing GameDataCollector as the base
        collector = GameDataCollector(str(self.raw_dir))
        
        # TODO: Interface with Phase 8 high-performance data collection system
        # This is where we would orchestrate the Rust servers to generate games
        # and extract heuristic supervision data from the sophisticated system
        
        # For now, simulate the interface - in production this would be replaced
        # with actual integration to Phase 8 system
        collected_samples = self._simulate_heuristic_collection(num_games, model_version, collector)
        
        self.logger.info(f"Collected {collected_samples} heuristic supervision samples")
        return collected_samples
    
    def collect_self_play_data(self, num_games: int, model_a_version: str, 
                              model_b_version: str = None) -> int:
        """Collect self-play training data from neural network vs neural network games"""
        if model_b_version is None:
            model_b_version = model_a_version
        
        self.logger.info(f"Starting self-play data collection: {num_games} games "
                        f"({model_a_version} vs {model_b_version})")
        
        collected_samples = 0
        
        # TODO: Interface with Phase 8 data collection to orchestrate neural vs neural games
        # This would involve:
        # 1. Loading neural network models for both players
        # 2. Running Battlesnake CLI with neural network endpoints
        # 3. Extracting decision data and game outcomes
        # 4. Converting to training samples
        
        collected_samples = self._simulate_self_play_collection(
            num_games, model_a_version, model_b_version
        )
        
        self.logger.info(f"Collected {collected_samples} self-play samples")
        return collected_samples
    
    def process_training_batch(self, phase_config: TrainingPhaseConfig, 
                              batch_size: int = 1000) -> Optional[str]:
        """Process a batch of training data for specific training phase"""
        self.logger.info(f"Processing training batch for {phase_config.name} phase")
        
        # Sample data based on phase requirements
        if phase_config.data_source == "heuristic_supervision":
            samples = self._get_heuristic_samples(batch_size)
        elif phase_config.data_source == "mixed":
            heuristic_count = int(batch_size * phase_config.heuristic_mix_ratio)
            self_play_count = batch_size - heuristic_count
            samples = (self._get_heuristic_samples(heuristic_count) + 
                      self._get_self_play_samples(self_play_count))
        elif phase_config.data_source == "self_play":
            samples = self._get_self_play_samples(batch_size)
        else:  # continuous
            samples = self.experience_buffer.sample_batch(batch_size)
        
        if not samples:
            self.logger.warning(f"No samples available for {phase_config.name} phase")
            return None
        
        # Create training batch
        batch_name = f"{phase_config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_file = self._create_training_batch(samples, batch_name, phase_config.name)
        
        self.logger.info(f"Created training batch: {batch_file} ({len(samples)} samples)")
        return batch_file
    
    def _create_training_batch(self, samples: List[SelfPlayTrainingSample], 
                              batch_name: str, training_phase: str) -> str:
        """Create compressed training batch file"""
        
        # Convert samples to training format
        training_data = {
            'batch_name': batch_name,
            'training_phase': training_phase,
            'created_at': datetime.now().isoformat(),
            'sample_count': len(samples),
            'samples': samples
        }
        
        # Save compressed batch
        batch_file = self.training_dir / f"{batch_name}.pkl.gz"
        with gzip.open(batch_file, 'wb') as f:
            pickle.dump(training_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Update database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO data_batches 
                (batch_name, training_phase, sample_count, avg_quality, created_at, file_path)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                batch_name,
                training_phase,
                len(samples),
                np.mean([s.quality_score for s in samples]),
                datetime.now().isoformat(),
                str(batch_file)
            ))
        
        return str(batch_file)
    
    def get_training_iterator(self, training_phase: str, batch_size: int = 64,
                             shuffle: bool = True) -> Iterator[List[SelfPlayTrainingSample]]:
        """Get iterator for training data batches"""
        
        # Get all available batches for this phase
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT file_path, sample_count FROM data_batches 
                WHERE training_phase = ? 
                ORDER BY created_at DESC
            """, (training_phase,))
            
            batch_files = cursor.fetchall()
        
        if not batch_files:
            self.logger.warning(f"No training batches found for phase: {training_phase}")
            return
        
        self.logger.info(f"Found {len(batch_files)} training batches for {training_phase}")
        
        # Load and yield batches
        for file_path, _ in batch_files:
            try:
                with gzip.open(file_path, 'rb') as f:
                    batch_data = pickle.load(f)
                
                samples = batch_data['samples']
                if shuffle:
                    random.shuffle(samples)
                
                # Yield in mini-batches
                for i in range(0, len(samples), batch_size):
                    mini_batch = samples[i:i + batch_size]
                    yield mini_batch
                    
            except Exception as e:
                self.logger.error(f"Error loading batch {file_path}: {e}")
                continue
    
    def _simulate_heuristic_collection(self, num_games: int, model_version: str, 
                                     collector: GameDataCollector) -> int:
        """Simulate heuristic supervision data collection (replace with Phase 8 integration)"""
        
        samples_created = 0
        
        for game_idx in range(num_games):
            # Create enhanced mock game with heuristic supervision
            mock_game = self._create_mock_game_with_heuristics(game_idx)
            
            # Convert to SelfPlayTrainingSample format
            for turn in range(len(mock_game['moves'])):
                sample = self._create_heuristic_training_sample(mock_game, turn, model_version)
                if sample:
                    self.experience_buffer.add_sample(sample)
                    samples_created += 1
        
        return samples_created
    
    def _simulate_self_play_collection(self, num_games: int, model_a: str, model_b: str) -> int:
        """Simulate self-play data collection (replace with Phase 8 integration)"""
        
        samples_created = 0
        
        for game_idx in range(num_games):
            # Create mock self-play game
            mock_game = self._create_mock_self_play_game(game_idx, model_a, model_b)
            
            # Convert to training samples
            for turn in range(len(mock_game['moves'])):
                sample = self._create_self_play_training_sample(mock_game, turn, model_a)
                if sample:
                    self.experience_buffer.add_sample(sample)
                    samples_created += 1
        
        return samples_created
    
    def _create_mock_game_with_heuristics(self, game_idx: int) -> Dict[str, Any]:
        """Create mock game data with heuristic supervision"""
        return {
            'game_id': f'heuristic_game_{game_idx}',
            'board_state': np.random.rand(11, 11, 12),  # Mock 12-channel encoding
            'snake_features': np.random.rand(32),
            'game_context': np.random.rand(16),
            'moves': ['up', 'right', 'down', 'left'] * 25,  # Mock game
            'outcome': random.choice(['win', 'loss', 'draw']),
            'heuristic_scores': {
                'safety': random.uniform(8, 50),
                'territory': random.uniform(3, 9),
                'food': random.uniform(0, 11),
                'opponent': random.uniform(3, 6),
                'exploration': random.uniform(20, 40)
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_mock_self_play_game(self, game_idx: int, model_a: str, model_b: str) -> Dict[str, Any]:
        """Create mock self-play game data"""
        return {
            'game_id': f'selfplay_game_{game_idx}',
            'board_state': np.random.rand(11, 11, 12),
            'snake_features': np.random.rand(32),
            'game_context': np.random.rand(16),
            'moves': ['up', 'right', 'down', 'left'] * 30,
            'outcome': random.choice(['win', 'loss', 'draw']),
            'model_a': model_a,
            'model_b': model_b,
            'move_probabilities': np.random.dirichlet([1, 1, 1, 1]),
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_heuristic_training_sample(self, mock_game: Dict[str, Any], 
                                        turn: int, model_version: str) -> SelfPlayTrainingSample:
        """Create training sample with heuristic supervision"""
        
        move_map = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        outcome_map = {'win': 1.0, 'loss': -1.0, 'draw': 0.0}
        
        # Calculate position value from heuristic scores
        heuristic_total = sum(mock_game['heuristic_scores'].values())
        position_value = np.tanh(heuristic_total / 50.0)  # Normalize to [-1, 1]
        
        # Quality score based on heuristic completeness and game outcome
        quality_score = 0.8 + (0.2 * random.random())  # High quality for heuristic data
        
        return SelfPlayTrainingSample(
            board_state=mock_game['board_state'],
            snake_features=mock_game['snake_features'],
            game_context=mock_game['game_context'],
            target_move=move_map[mock_game['moves'][turn]],
            position_value=position_value,
            move_probabilities=np.array([0.25, 0.25, 0.25, 0.25]),  # Uniform default
            game_outcome=outcome_map[mock_game['outcome']],
            model_version=model_version,
            opponent_strength=0.7,  # Heuristic opponent strength
            training_phase='bootstrap',
            quality_score=quality_score,
            turn=turn,
            game_id=mock_game['game_id'],
            timestamp=mock_game['timestamp'],
            heuristic_scores=mock_game['heuristic_scores']
        )
    
    def _create_self_play_training_sample(self, mock_game: Dict[str, Any], 
                                        turn: int, model_version: str) -> SelfPlayTrainingSample:
        """Create training sample from self-play game"""
        
        move_map = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        outcome_map = {'win': 1.0, 'loss': -1.0, 'draw': 0.0}
        
        # Position value from actual game outcome
        game_outcome_val = outcome_map[mock_game['outcome']]
        position_value = game_outcome_val * (1.0 - turn / len(mock_game['moves']))
        
        # Quality score based on game characteristics
        quality_score = 0.6 + (0.3 * random.random())  # Medium-high for self-play
        
        return SelfPlayTrainingSample(
            board_state=mock_game['board_state'],
            snake_features=mock_game['snake_features'],
            game_context=mock_game['game_context'],
            target_move=move_map[mock_game['moves'][turn]],
            position_value=position_value,
            move_probabilities=mock_game.get('move_probabilities', np.array([0.25, 0.25, 0.25, 0.25])),
            game_outcome=game_outcome_val,
            model_version=model_version,
            opponent_strength=0.8,  # Neural opponent strength
            training_phase='self_play',
            quality_score=quality_score,
            turn=turn,
            game_id=mock_game['game_id'],
            timestamp=mock_game['timestamp']
        )
    
    def _get_heuristic_samples(self, count: int) -> List[SelfPlayTrainingSample]:
        """Get heuristic supervision samples from buffer"""
        return self.experience_buffer.sample_batch(count, phase_filter='bootstrap')
    
    def _get_self_play_samples(self, count: int) -> List[SelfPlayTrainingSample]:
        """Get self-play samples from buffer"""
        return self.experience_buffer.sample_batch(count, phase_filter='self_play')
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get comprehensive data statistics"""
        buffer_stats = self.experience_buffer.get_statistics()
        
        # Database statistics
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT training_phase, COUNT(*), AVG(quality_score)
                FROM training_samples 
                GROUP BY training_phase
            """)
            phase_stats = {phase: {'count': count, 'avg_quality': avg_q} 
                          for phase, count, avg_q in cursor.fetchall()}
            
            cursor = conn.execute("SELECT COUNT(*) FROM data_batches")
            batch_count = cursor.fetchone()[0]
        
        return {
            'experience_buffer': buffer_stats,
            'training_phases': phase_stats,
            'batch_count': batch_count,
            'processing_metrics': self.metrics,
            'uptime_hours': (datetime.now() - self.metrics['start_time']).total_seconds() / 3600
        }
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old training data beyond retention period"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get old batch files to delete
            cursor = conn.execute("""
                SELECT file_path FROM data_batches 
                WHERE created_at < ?
            """, (cutoff_date.isoformat(),))
            
            old_files = [row[0] for row in cursor.fetchall()]
            
            # Delete files
            deleted_count = 0
            for file_path in old_files:
                try:
                    Path(file_path).unlink(missing_ok=True)
                    deleted_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete {file_path}: {e}")
            
            # Clean database records
            conn.execute("""
                DELETE FROM data_batches WHERE created_at < ?
            """, (cutoff_date.isoformat(),))
            
            conn.execute("""
                DELETE FROM training_samples WHERE timestamp < ?
            """, (cutoff_date.isoformat(),))
        
        self.logger.info(f"Cleaned up {deleted_count} old data files")


# Convenience functions for easy integration
def create_training_batch(phase_name: str, num_games: int = 1000) -> Optional[str]:
    """Create a training batch for specified phase"""
    from config.self_play_config import get_config
    
    config = get_config()
    phase_config = config.training_pipeline.get_phase_config(phase_name)
    
    if not phase_config:
        logging.error(f"Unknown training phase: {phase_name}")
        return None
    
    manager = SelfPlayDataManager()
    
    # Collect data based on phase requirements
    if phase_config.data_source == "heuristic_supervision":
        manager.collect_heuristic_supervision_data(num_games)
    elif phase_config.data_source == "self_play":
        manager.collect_self_play_data(num_games, "current_model")
    
    # Create training batch
    return manager.process_training_batch(phase_config, num_games)


if __name__ == "__main__":
    # Test the data manager
    logging.basicConfig(level=logging.INFO)
    
    manager = SelfPlayDataManager()
    
    # Test heuristic data collection
    print("Testing heuristic supervision data collection...")
    manager.collect_heuristic_supervision_data(50)
    
    # Test self-play data collection  
    print("Testing self-play data collection...")
    manager.collect_self_play_data(30, "test_model_v1", "test_model_v2")
    
    # Create training batches
    from config.self_play_config import get_config
    config = get_config()
    
    for phase in config.training_pipeline.phases:
        print(f"Creating training batch for {phase.name} phase...")
        batch_file = manager.process_training_batch(phase, 100)
        print(f"Created: {batch_file}")
    
    # Print statistics
    stats = manager.get_data_statistics()
    print("\n=== Data Manager Statistics ===")
    print(json.dumps(stats, indent=2, default=str))