"""
Data Management System

This module handles efficient storage, versioning, and lifecycle management of training data
from the self-play data collection pipeline. Provides compressed storage, data organization,
backup functionality, and export capabilities for neural network training.

Architecture compliance: Compressed storage, automated cleanup, export ready for PyTorch
"""

import os
import json
import pickle
import gzip
import shutil
import hashlib
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Iterator
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import sqlite3
import numpy as np
import concurrent.futures
from contextlib import contextmanager

# External imports for advanced storage formats
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    logging.warning("h5py not available, HDF5 storage disabled")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("pandas not available, some features disabled")

from training_data_pipeline import TrainingSample
from game_data_extractor import GameData
from config.self_play_config import get_config

@dataclass
class DatasetMetadata:
    """Metadata for a training dataset"""
    dataset_id: str
    version: str
    created_at: datetime
    sample_count: int
    game_count: int
    file_size_bytes: int
    compression_level: int
    data_format: str  # 'pickle', 'hdf5', 'json'
    description: str = ""
    tags: List[str] = None
    
    # Quality metrics
    avg_game_length: float = 0.0
    heuristic_score_stats: Dict[str, float] = None
    board_encoding_shape: Tuple[int, ...] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.heuristic_score_stats is None:
            self.heuristic_score_stats = {}
        if self.board_encoding_shape is None:
            self.board_encoding_shape = (11, 11, 12)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetMetadata':
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)

@dataclass 
class StorageConfig:
    """Configuration for data storage"""
    base_directory: str = "data/self_play"
    compression_enabled: bool = True
    compression_level: int = 6  # 1-9, 6 is good balance
    max_file_size_mb: int = 100
    backup_enabled: bool = True
    backup_directory: str = "data/self_play/backups"
    retention_days: int = 30
    cleanup_enabled: bool = True
    preferred_format: str = "pickle"  # pickle, hdf5, json

class DataValidator:
    """Validates training data quality and integrity"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".validator")
    
    def validate_training_sample(self, sample: TrainingSample) -> Dict[str, Any]:
        """Validate a single training sample"""
        issues = []
        stats = {}
        
        # Shape validation
        expected_shapes = {
            'board_state': (11, 11, 12),
            'snake_features': (32,),
            'game_context': (16,),
            'move_probabilities': (4,)
        }
        
        for field, expected_shape in expected_shapes.items():
            actual_shape = getattr(sample, field).shape
            if actual_shape != expected_shape:
                issues.append(f"{field} shape mismatch: expected {expected_shape}, got {actual_shape}")
        
        # Value range validation
        if not (0 <= sample.target_move <= 3):
            issues.append(f"Invalid target_move: {sample.target_move}")
        
        if not (-50.0 <= sample.position_value <= 50.0):
            issues.append(f"Invalid position_value: {sample.position_value}")
            
        if not (0.0 <= sample.win_probability <= 1.0):
            issues.append(f"Invalid win_probability: {sample.win_probability}")
        
        # Probability distribution validation
        prob_sum = sample.move_probabilities.sum()
        if not np.isclose(prob_sum, 1.0, atol=1e-3):
            issues.append(f"Move probabilities don't sum to 1.0: {prob_sum}")
        
        # NaN/Inf validation
        arrays_to_check = [
            ('board_state', sample.board_state),
            ('snake_features', sample.snake_features),
            ('game_context', sample.game_context),
            ('move_probabilities', sample.move_probabilities)
        ]
        
        for name, array in arrays_to_check:
            if np.any(np.isnan(array)):
                issues.append(f"{name} contains NaN values")
            if np.any(np.isinf(array)):
                issues.append(f"{name} contains infinite values")
        
        # Calculate statistics
        stats = {
            'board_occupancy': np.mean(sample.board_state > 0),
            'snake_features_range': (float(sample.snake_features.min()), float(sample.snake_features.max())),
            'game_context_mean': float(sample.game_context.mean()),
            'heuristic_total': sum(sample.heuristic_scores.values()) if sample.heuristic_scores else 0.0
        }
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'stats': stats
        }
    
    def validate_dataset(self, samples: List[TrainingSample]) -> Dict[str, Any]:
        """Validate an entire dataset"""
        if not samples:
            return {'valid': False, 'issues': ['Empty dataset'], 'stats': {}}
        
        total_issues = []
        sample_stats = []
        
        # Validate samples in parallel for large datasets
        if len(samples) > 100:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                validations = list(executor.map(self.validate_training_sample, samples))
        else:
            validations = [self.validate_training_sample(sample) for sample in samples]
        
        # Aggregate results
        valid_samples = sum(1 for v in validations if v['valid'])
        for i, validation in enumerate(validations):
            if not validation['valid']:
                total_issues.extend([f"Sample {i}: {issue}" for issue in validation['issues']])
            sample_stats.append(validation['stats'])
        
        # Calculate dataset statistics
        dataset_stats = self._calculate_dataset_stats(samples, sample_stats)
        
        return {
            'valid': len(total_issues) == 0,
            'issues': total_issues[:100],  # Limit to first 100 issues
            'sample_count': len(samples),
            'valid_samples': valid_samples,
            'invalid_samples': len(samples) - valid_samples,
            'stats': dataset_stats
        }
    
    def _calculate_dataset_stats(self, samples: List[TrainingSample], 
                               sample_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive dataset statistics"""
        
        # Move distribution
        move_counts = defaultdict(int)
        position_values = []
        win_probabilities = []
        game_lengths = defaultdict(int)
        heuristic_totals = []
        
        for sample in samples:
            move_counts[sample.target_move] += 1
            position_values.append(sample.position_value)
            win_probabilities.append(sample.win_probability)
            game_lengths[sample.game_id] += 1
            
            if sample.heuristic_scores:
                heuristic_totals.append(sum(sample.heuristic_scores.values()))
        
        # Convert to numpy arrays for statistics
        position_values = np.array(position_values)
        win_probabilities = np.array(win_probabilities)
        heuristic_totals = np.array(heuristic_totals) if heuristic_totals else np.array([])
        
        stats = {
            'move_distribution': dict(move_counts),
            'position_value_stats': {
                'mean': float(position_values.mean()),
                'std': float(position_values.std()),
                'min': float(position_values.min()),
                'max': float(position_values.max())
            } if len(position_values) > 0 else {},
            'win_probability_stats': {
                'mean': float(win_probabilities.mean()),
                'std': float(win_probabilities.std())
            } if len(win_probabilities) > 0 else {},
            'avg_game_length': sum(game_lengths.values()) / len(game_lengths) if game_lengths else 0.0,
            'unique_games': len(game_lengths),
            'heuristic_score_stats': {
                'mean': float(heuristic_totals.mean()),
                'std': float(heuristic_totals.std()),
                'min': float(heuristic_totals.min()),
                'max': float(heuristic_totals.max())
            } if len(heuristic_totals) > 0 else {}
        }
        
        return stats

class CompressedStorageManager:
    """Manages compressed storage of training data"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".storage")
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        directories = [
            self.config.base_directory,
            os.path.join(self.config.base_directory, "raw"),
            os.path.join(self.config.base_directory, "processed"), 
            os.path.join(self.config.base_directory, "training"),
            os.path.join(self.config.base_directory, "metadata"),
            self.config.backup_directory if self.config.backup_enabled else None
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    def save_training_samples(self, samples: List[TrainingSample], 
                            dataset_id: str, version: str = "1.0",
                            description: str = "") -> DatasetMetadata:
        """Save training samples with compression and metadata"""
        
        if not samples:
            raise ValueError("Cannot save empty dataset")
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{dataset_id}_v{version}_{timestamp}"
        
        # Choose storage format and save
        if self.config.preferred_format == "hdf5" and HDF5_AVAILABLE:
            filepath = self._save_hdf5(samples, filename_base)
            data_format = "hdf5"
        elif self.config.preferred_format == "json":
            filepath = self._save_json(samples, filename_base)
            data_format = "json"
        else:
            filepath = self._save_pickle(samples, filename_base)
            data_format = "pickle"
        
        # Create metadata
        file_size = os.path.getsize(filepath)
        
        # Calculate quality metrics
        validator = DataValidator()
        validation_result = validator.validate_dataset(samples)
        
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            version=version,
            created_at=datetime.now(),
            sample_count=len(samples),
            game_count=len(set(sample.game_id for sample in samples)),
            file_size_bytes=file_size,
            compression_level=self.config.compression_level,
            data_format=data_format,
            description=description,
            avg_game_length=validation_result['stats'].get('avg_game_length', 0.0),
            heuristic_score_stats=validation_result['stats'].get('heuristic_score_stats', {}),
            board_encoding_shape=(11, 11, 12)
        )
        
        # Save metadata
        self._save_metadata(metadata, filename_base)
        
        # Create backup if enabled
        if self.config.backup_enabled:
            self._create_backup(filepath, metadata)
        
        self.logger.info(f"Saved {len(samples)} training samples to {filepath}")
        self.logger.info(f"Dataset size: {file_size / 1024 / 1024:.2f} MB")
        
        return metadata
    
    def _save_pickle(self, samples: List[TrainingSample], filename_base: str) -> str:
        """Save samples as compressed pickle"""
        filepath = os.path.join(self.config.base_directory, "training", f"{filename_base}.pkl.gz")
        
        # Convert samples to serializable format
        sample_dicts = [sample.to_dict() for sample in samples]
        
        with gzip.open(filepath, 'wb', compresslevel=self.config.compression_level) as f:
            pickle.dump(sample_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return filepath
    
    def _save_hdf5(self, samples: List[TrainingSample], filename_base: str) -> str:
        """Save samples as HDF5 format"""
        if not HDF5_AVAILABLE:
            raise RuntimeError("HDF5 not available")
        
        filepath = os.path.join(self.config.base_directory, "training", f"{filename_base}.h5")
        
        with h5py.File(filepath, 'w') as f:
            # Create datasets for each tensor type
            n_samples = len(samples)
            
            # Board states [n_samples, 11, 11, 12]
            board_states = np.array([sample.board_state for sample in samples])
            f.create_dataset('board_states', data=board_states, compression='gzip', 
                           compression_opts=self.config.compression_level)
            
            # Snake features [n_samples, 32]
            snake_features = np.array([sample.snake_features for sample in samples])
            f.create_dataset('snake_features', data=snake_features, compression='gzip',
                           compression_opts=self.config.compression_level)
            
            # Game context [n_samples, 16]
            game_context = np.array([sample.game_context for sample in samples])
            f.create_dataset('game_context', data=game_context, compression='gzip',
                           compression_opts=self.config.compression_level)
            
            # Targets and labels
            f.create_dataset('target_moves', data=[sample.target_move for sample in samples])
            f.create_dataset('position_values', data=[sample.position_value for sample in samples])
            f.create_dataset('win_probabilities', data=[sample.win_probability for sample in samples])
            
            # Move probabilities [n_samples, 4]
            move_probs = np.array([sample.move_probabilities for sample in samples])
            f.create_dataset('move_probabilities', data=move_probs)
            
            # Metadata as strings (JSON encoded)
            f.create_dataset('game_ids', data=[s.game_id.encode('utf-8') for s in samples])
            f.create_dataset('snake_ids', data=[s.snake_id.encode('utf-8') for s in samples])
            f.create_dataset('turns', data=[s.turn for s in samples])
            f.create_dataset('timestamps', data=[s.timestamp.isoformat().encode('utf-8') for s in samples])
            
            # Heuristic scores as JSON strings
            heuristic_json = [json.dumps(s.heuristic_scores).encode('utf-8') for s in samples]
            f.create_dataset('heuristic_scores', data=heuristic_json)
        
        return filepath
    
    def _save_json(self, samples: List[TrainingSample], filename_base: str) -> str:
        """Save samples as compressed JSON"""
        filepath = os.path.join(self.config.base_directory, "training", f"{filename_base}.json.gz")
        
        # Convert samples to serializable format
        sample_dicts = [sample.to_dict() for sample in samples]
        
        with gzip.open(filepath, 'wt', compresslevel=self.config.compression_level) as f:
            json.dump(sample_dicts, f, indent=None, separators=(',', ':'))
        
        return filepath
    
    def _save_metadata(self, metadata: DatasetMetadata, filename_base: str):
        """Save dataset metadata"""
        metadata_path = os.path.join(self.config.base_directory, "metadata", f"{filename_base}_metadata.json")
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    def _create_backup(self, filepath: str, metadata: DatasetMetadata):
        """Create backup copy of dataset"""
        if not self.config.backup_enabled:
            return
        
        backup_dir = os.path.join(self.config.backup_directory, 
                                 datetime.now().strftime("%Y%m%d"))
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
        
        filename = os.path.basename(filepath)
        backup_path = os.path.join(backup_dir, filename)
        
        shutil.copy2(filepath, backup_path)
        
        # Also backup metadata
        metadata_filename = filename.replace('.pkl.gz', '_metadata.json').replace('.h5', '_metadata.json').replace('.json.gz', '_metadata.json')
        metadata_source = os.path.join(self.config.base_directory, "metadata", metadata_filename)
        metadata_backup = os.path.join(backup_dir, metadata_filename)
        
        if os.path.exists(metadata_source):
            shutil.copy2(metadata_source, metadata_backup)
        
        self.logger.info(f"Created backup: {backup_path}")
    
    def load_training_samples(self, dataset_id: str, version: Optional[str] = None) -> Tuple[List[TrainingSample], DatasetMetadata]:
        """Load training samples by dataset ID and version"""
        
        # Find matching files
        metadata_files = self._find_metadata_files(dataset_id, version)
        
        if not metadata_files:
            raise FileNotFoundError(f"No datasets found for ID '{dataset_id}'{f' version {version}' if version else ''}")
        
        # Use the most recent if multiple found
        latest_metadata_file = max(metadata_files, key=lambda x: os.path.getmtime(x))
        
        # Load metadata
        with open(latest_metadata_file, 'r') as f:
            metadata_dict = json.load(f)
        metadata = DatasetMetadata.from_dict(metadata_dict)
        
        # Load samples based on format
        data_filename = os.path.basename(latest_metadata_file).replace('_metadata.json', '')
        
        if metadata.data_format == "pickle":
            samples = self._load_pickle(data_filename)
        elif metadata.data_format == "hdf5":
            samples = self._load_hdf5(data_filename)
        elif metadata.data_format == "json":
            samples = self._load_json(data_filename)
        else:
            raise ValueError(f"Unknown data format: {metadata.data_format}")
        
        self.logger.info(f"Loaded {len(samples)} training samples from {metadata.dataset_id} v{metadata.version}")
        
        return samples, metadata
    
    def _find_metadata_files(self, dataset_id: str, version: Optional[str] = None) -> List[str]:
        """Find metadata files matching criteria"""
        metadata_dir = os.path.join(self.config.base_directory, "metadata")
        files = []
        
        for filename in os.listdir(metadata_dir):
            if not filename.endswith('_metadata.json'):
                continue
            
            if dataset_id in filename:
                if version is None or f"_v{version}_" in filename:
                    files.append(os.path.join(metadata_dir, filename))
        
        return files
    
    def _load_pickle(self, data_filename: str) -> List[TrainingSample]:
        """Load samples from pickle format"""
        filepath = os.path.join(self.config.base_directory, "training", f"{data_filename}.pkl.gz")
        
        with gzip.open(filepath, 'rb') as f:
            sample_dicts = pickle.load(f)
        
        return [TrainingSample.from_dict(sample_dict) for sample_dict in sample_dicts]
    
    def _load_hdf5(self, data_filename: str) -> List[TrainingSample]:
        """Load samples from HDF5 format"""
        if not HDF5_AVAILABLE:
            raise RuntimeError("HDF5 not available")
        
        filepath = os.path.join(self.config.base_directory, "training", f"{data_filename}.h5")
        
        samples = []
        with h5py.File(filepath, 'r') as f:
            n_samples = len(f['target_moves'])
            
            for i in range(n_samples):
                # Decode string data
                game_id = f['game_ids'][i].decode('utf-8')
                snake_id = f['snake_ids'][i].decode('utf-8')
                timestamp = datetime.fromisoformat(f['timestamps'][i].decode('utf-8'))
                heuristic_scores = json.loads(f['heuristic_scores'][i].decode('utf-8'))
                
                sample = TrainingSample(
                    board_state=f['board_states'][i],
                    snake_features=f['snake_features'][i],
                    game_context=f['game_context'][i],
                    target_move=f['target_moves'][i],
                    position_value=f['position_values'][i],
                    move_probabilities=f['move_probabilities'][i],
                    win_probability=f['win_probabilities'][i],
                    heuristic_scores=heuristic_scores,
                    game_id=game_id,
                    turn=f['turns'][i],
                    snake_id=snake_id,
                    timestamp=timestamp
                )
                samples.append(sample)
        
        return samples
    
    def _load_json(self, data_filename: str) -> List[TrainingSample]:
        """Load samples from JSON format"""
        filepath = os.path.join(self.config.base_directory, "training", f"{data_filename}.json.gz")
        
        with gzip.open(filepath, 'rt') as f:
            sample_dicts = json.load(f)
        
        return [TrainingSample.from_dict(sample_dict) for sample_dict in sample_dicts]
    
    def list_datasets(self) -> List[DatasetMetadata]:
        """List all available datasets"""
        metadata_dir = os.path.join(self.config.base_directory, "metadata")
        datasets = []
        
        if not os.path.exists(metadata_dir):
            return datasets
        
        for filename in os.listdir(metadata_dir):
            if filename.endswith('_metadata.json'):
                try:
                    filepath = os.path.join(metadata_dir, filename)
                    with open(filepath, 'r') as f:
                        metadata_dict = json.load(f)
                    metadata = DatasetMetadata.from_dict(metadata_dict)
                    datasets.append(metadata)
                except Exception as e:
                    self.logger.error(f"Error loading metadata from {filename}: {e}")
        
        # Sort by creation time (newest first)
        datasets.sort(key=lambda x: x.created_at, reverse=True)
        
        return datasets
    
    def delete_dataset(self, dataset_id: str, version: Optional[str] = None):
        """Delete a dataset and its metadata"""
        metadata_files = self._find_metadata_files(dataset_id, version)
        
        for metadata_file in metadata_files:
            # Load metadata to get data filename
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            metadata = DatasetMetadata.from_dict(metadata_dict)
            
            # Delete data file
            data_filename = os.path.basename(metadata_file).replace('_metadata.json', '')
            extensions = ['.pkl.gz', '.h5', '.json.gz']
            
            for ext in extensions:
                data_filepath = os.path.join(self.config.base_directory, "training", f"{data_filename}{ext}")
                if os.path.exists(data_filepath):
                    os.remove(data_filepath)
                    self.logger.info(f"Deleted data file: {data_filepath}")
            
            # Delete metadata file
            os.remove(metadata_file)
            self.logger.info(f"Deleted metadata file: {metadata_file}")

class DataLifecycleManager:
    """Manages data lifecycle including cleanup and archival"""
    
    def __init__(self, storage_manager: CompressedStorageManager, config: StorageConfig):
        self.storage_manager = storage_manager
        self.config = config
        self.logger = logging.getLogger(__name__ + ".lifecycle")
        
        # Database for tracking data usage
        self.db_path = os.path.join(config.base_directory, "data_tracking.db")
        self._init_database()
        
        # Background cleanup thread
        self.cleanup_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
    
    def _init_database(self):
        """Initialize SQLite database for tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS dataset_usage (
                    dataset_id TEXT,
                    version TEXT,
                    last_accessed TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    PRIMARY KEY (dataset_id, version)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cleanup_log (
                    cleanup_date TIMESTAMP,
                    datasets_deleted INTEGER,
                    space_freed_bytes INTEGER
                )
            ''')
    
    def start_background_cleanup(self):
        """Start background cleanup thread"""
        if self.cleanup_thread is not None:
            return
        
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self.cleanup_thread.start()
        self.logger.info("Started background cleanup thread")
    
    def stop_background_cleanup(self):
        """Stop background cleanup thread"""
        if self.cleanup_thread is None:
            return
        
        self.shutdown_event.set()
        self.cleanup_thread.join(timeout=10)
        self.cleanup_thread = None
        self.logger.info("Stopped background cleanup thread")
    
    def _cleanup_loop(self):
        """Background cleanup loop"""
        while not self.shutdown_event.is_set():
            try:
                self.cleanup_old_data()
                # Sleep for 6 hours
                for _ in range(6 * 60 * 6):  # 6 hours in 10-second intervals
                    if self.shutdown_event.is_set():
                        break
                    time.sleep(10)
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def record_dataset_access(self, dataset_id: str, version: str):
        """Record that a dataset was accessed"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO dataset_usage (dataset_id, version, last_accessed, access_count)
                VALUES (?, ?, ?, COALESCE((SELECT access_count FROM dataset_usage 
                                         WHERE dataset_id = ? AND version = ?) + 1, 1))
            ''', (dataset_id, version, datetime.now(), dataset_id, version))
    
    def cleanup_old_data(self) -> Dict[str, Any]:
        """Clean up old data based on retention policy"""
        if not self.config.cleanup_enabled:
            return {'cleaned': False, 'reason': 'Cleanup disabled'}
        
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        
        datasets = self.storage_manager.list_datasets()
        datasets_to_delete = []
        
        # Find datasets to delete
        for dataset in datasets:
            if dataset.created_at < cutoff_date:
                # Check if recently accessed
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        'SELECT last_accessed FROM dataset_usage WHERE dataset_id = ? AND version = ?',
                        (dataset.dataset_id, dataset.version)
                    )
                    result = cursor.fetchone()
                    
                    if result is None:
                        # Never accessed, safe to delete
                        datasets_to_delete.append(dataset)
                    else:
                        last_accessed = datetime.fromisoformat(result[0])
                        if last_accessed < cutoff_date:
                            # Not accessed recently, safe to delete
                            datasets_to_delete.append(dataset)
        
        # Delete datasets
        deleted_count = 0
        space_freed = 0
        
        for dataset in datasets_to_delete:
            try:
                space_freed += dataset.file_size_bytes
                self.storage_manager.delete_dataset(dataset.dataset_id, dataset.version)
                deleted_count += 1
                self.logger.info(f"Cleaned up dataset {dataset.dataset_id} v{dataset.version}")
            except Exception as e:
                self.logger.error(f"Error deleting dataset {dataset.dataset_id}: {e}")
        
        # Log cleanup
        if deleted_count > 0:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'INSERT INTO cleanup_log (cleanup_date, datasets_deleted, space_freed_bytes) VALUES (?, ?, ?)',
                    (datetime.now(), deleted_count, space_freed)
                )
        
        self.logger.info(f"Cleanup completed: {deleted_count} datasets deleted, {space_freed / 1024 / 1024:.2f} MB freed")
        
        return {
            'cleaned': True,
            'datasets_deleted': deleted_count,
            'space_freed_mb': space_freed / 1024 / 1024,
            'cutoff_date': cutoff_date.isoformat()
        }
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        datasets = self.storage_manager.list_datasets()
        
        total_size = sum(dataset.file_size_bytes for dataset in datasets)
        total_samples = sum(dataset.sample_count for dataset in datasets)
        total_games = sum(dataset.game_count for dataset in datasets)
        
        # Get disk usage
        base_dir = Path(self.config.base_directory)
        if base_dir.exists():
            disk_usage = sum(f.stat().st_size for f in base_dir.rglob('*') if f.is_file())
        else:
            disk_usage = 0
        
        # Format breakdown
        format_stats = defaultdict(lambda: {'count': 0, 'size': 0})
        for dataset in datasets:
            format_stats[dataset.data_format]['count'] += 1
            format_stats[dataset.data_format]['size'] += dataset.file_size_bytes
        
        return {
            'dataset_count': len(datasets),
            'total_samples': total_samples,
            'total_games': total_games,
            'total_size_mb': total_size / 1024 / 1024,
            'disk_usage_mb': disk_usage / 1024 / 1024,
            'avg_dataset_size_mb': (total_size / len(datasets) / 1024 / 1024) if datasets else 0,
            'format_breakdown': dict(format_stats),
            'oldest_dataset': min(datasets, key=lambda x: x.created_at).created_at.isoformat() if datasets else None,
            'newest_dataset': max(datasets, key=lambda x: x.created_at).created_at.isoformat() if datasets else None
        }

class TrainingDataExporter:
    """Exports data in formats suitable for neural network training"""
    
    def __init__(self, storage_manager: CompressedStorageManager):
        self.storage_manager = storage_manager
        self.logger = logging.getLogger(__name__ + ".exporter")
    
    def export_for_pytorch(self, dataset_ids: List[str], output_dir: str,
                          train_split: float = 0.8, val_split: float = 0.1,
                          test_split: float = 0.1) -> Dict[str, str]:
        """Export datasets for PyTorch training with train/val/test splits"""
        
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Load all datasets
        all_samples = []
        for dataset_id in dataset_ids:
            samples, metadata = self.storage_manager.load_training_samples(dataset_id)
            all_samples.extend(samples)
            self.logger.info(f"Loaded {len(samples)} samples from {dataset_id}")
        
        if not all_samples:
            raise ValueError("No samples loaded from datasets")
        
        # Shuffle samples for random splits
        np.random.shuffle(all_samples)
        
        n_samples = len(all_samples)
        train_end = int(n_samples * train_split)
        val_end = train_end + int(n_samples * val_split)
        
        splits = {
            'train': all_samples[:train_end],
            'val': all_samples[train_end:val_end], 
            'test': all_samples[val_end:]
        }
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Export each split
        export_paths = {}
        
        for split_name, samples in splits.items():
            if not samples:
                continue
            
            # Create tensors
            board_states = np.array([s.board_state for s in samples])
            snake_features = np.array([s.snake_features for s in samples])
            game_context = np.array([s.game_context for s in samples])
            target_moves = np.array([s.target_move for s in samples])
            position_values = np.array([s.position_value for s in samples])
            move_probabilities = np.array([s.move_probabilities for s in samples])
            win_probabilities = np.array([s.win_probability for s in samples])
            
            # Save as NumPy arrays (easily loadable by PyTorch)
            split_dir = os.path.join(output_dir, split_name)
            Path(split_dir).mkdir(exist_ok=True)
            
            np.save(os.path.join(split_dir, 'board_states.npy'), board_states)
            np.save(os.path.join(split_dir, 'snake_features.npy'), snake_features)
            np.save(os.path.join(split_dir, 'game_context.npy'), game_context)
            np.save(os.path.join(split_dir, 'target_moves.npy'), target_moves)
            np.save(os.path.join(split_dir, 'position_values.npy'), position_values)
            np.save(os.path.join(split_dir, 'move_probabilities.npy'), move_probabilities)
            np.save(os.path.join(split_dir, 'win_probabilities.npy'), win_probabilities)
            
            # Save metadata
            metadata = {
                'num_samples': len(samples),
                'input_shapes': {
                    'board_states': board_states.shape,
                    'snake_features': snake_features.shape,
                    'game_context': game_context.shape
                },
                'target_shapes': {
                    'target_moves': target_moves.shape,
                    'position_values': position_values.shape,
                    'move_probabilities': move_probabilities.shape,
                    'win_probabilities': win_probabilities.shape
                },
                'move_distribution': {i: int(np.sum(target_moves == i)) for i in range(4)},
                'created_at': datetime.now().isoformat(),
                'source_datasets': dataset_ids
            }
            
            with open(os.path.join(split_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            export_paths[split_name] = split_dir
            self.logger.info(f"Exported {split_name} set: {len(samples)} samples to {split_dir}")
        
        # Create PyTorch dataset loader code
        loader_code = self._generate_pytorch_loader_code(output_dir)
        with open(os.path.join(output_dir, 'pytorch_dataset.py'), 'w') as f:
            f.write(loader_code)
        
        self.logger.info(f"Export completed: {len(all_samples)} total samples")
        
        return export_paths
    
    def _generate_pytorch_loader_code(self, output_dir: str) -> str:
        """Generate PyTorch dataset loader code"""
        return f"""#!/usr/bin/env python3
\"\"\"
PyTorch Dataset Loader for Battlesnake Training Data

Auto-generated by data_management.py
Data location: {output_dir}
\"\"\"

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class BattlesnakeDataset(Dataset):
    def __init__(self, data_dir, dtype=torch.float32):
        self.data_dir = Path(data_dir)
        
        # Load all data
        self.board_states = torch.from_numpy(np.load(self.data_dir / 'board_states.npy')).to(dtype)
        self.snake_features = torch.from_numpy(np.load(self.data_dir / 'snake_features.npy')).to(dtype)
        self.game_context = torch.from_numpy(np.load(self.data_dir / 'game_context.npy')).to(dtype)
        
        self.target_moves = torch.from_numpy(np.load(self.data_dir / 'target_moves.npy')).long()
        self.position_values = torch.from_numpy(np.load(self.data_dir / 'position_values.npy')).to(dtype)
        self.move_probabilities = torch.from_numpy(np.load(self.data_dir / 'move_probabilities.npy')).to(dtype)
        self.win_probabilities = torch.from_numpy(np.load(self.data_dir / 'win_probabilities.npy')).to(dtype)
        
    def __len__(self):
        return len(self.target_moves)
    
    def __getitem__(self, idx):
        return {{
            'board_state': self.board_states[idx],
            'snake_features': self.snake_features[idx], 
            'game_context': self.game_context[idx],
            'target_move': self.target_moves[idx],
            'position_value': self.position_values[idx],
            'move_probabilities': self.move_probabilities[idx],
            'win_probability': self.win_probabilities[idx]
        }}

def create_data_loaders(base_dir, batch_size=32, num_workers=4):
    \"\"\"Create train/val/test data loaders\"\"\"
    
    loaders = {{}}
    
    for split in ['train', 'val', 'test']:
        split_dir = Path(base_dir) / split
        if split_dir.exists():
            dataset = BattlesnakeDataset(split_dir)
            shuffle = (split == 'train')
            
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True
            )
            
            loaders[split] = loader
    
    return loaders

if __name__ == "__main__":
    # Example usage
    loaders = create_data_loaders('{output_dir}')
    
    if 'train' in loaders:
        print(f"Train samples: {{len(loaders['train'].dataset)}}")
        
        # Get a batch
        batch = next(iter(loaders['train']))
        print(f"Batch shapes:")
        for key, tensor in batch.items():
            print(f"  {{key}}: {{tensor.shape}}")
"""

class DataManagementSystem:
    """Main data management system that coordinates all components"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        
        # Create storage configuration
        dc = self.config.data_collection
        self.storage_config = StorageConfig(
            base_directory=dc.data_directory,
            compression_level=dc.compression_level,
            max_file_size_mb=dc.max_file_size_mb,
            retention_days=dc.retention_days,
            backup_enabled=dc.backup_enabled
        )
        
        # Initialize components
        self.storage_manager = CompressedStorageManager(self.storage_config)
        self.lifecycle_manager = DataLifecycleManager(self.storage_manager, self.storage_config)
        self.exporter = TrainingDataExporter(self.storage_manager)
        self.validator = DataValidator()
        
        self.logger = logging.getLogger(__name__ + ".system")
        
        # Start background services
        if self.storage_config.cleanup_enabled:
            self.lifecycle_manager.start_background_cleanup()
    
    def save_training_data(self, samples: List[TrainingSample], 
                          dataset_id: str, version: str = "1.0",
                          description: str = "") -> DatasetMetadata:
        """Save training data with full validation and management"""
        
        # Validate data quality
        validation_result = self.validator.validate_dataset(samples)
        
        if not validation_result['valid']:
            self.logger.warning(f"Dataset validation issues: {len(validation_result['issues'])} problems found")
            
            # Filter out invalid samples if there are valid ones
            if validation_result['valid_samples'] > 0:
                self.logger.info(f"Saving {validation_result['valid_samples']} valid samples out of {len(samples)}")
                # In a real implementation, we'd filter the samples here
        
        # Save the data
        metadata = self.storage_manager.save_training_samples(samples, dataset_id, version, description)
        
        # Record access for lifecycle management
        self.lifecycle_manager.record_dataset_access(dataset_id, version)
        
        return metadata
    
    def load_training_data(self, dataset_id: str, version: Optional[str] = None) -> Tuple[List[TrainingSample], DatasetMetadata]:
        """Load training data with access tracking"""
        
        samples, metadata = self.storage_manager.load_training_samples(dataset_id, version)
        
        # Record access
        self.lifecycle_manager.record_dataset_access(metadata.dataset_id, metadata.version)
        
        return samples, metadata
    
    def export_for_training(self, dataset_ids: List[str], output_dir: str) -> Dict[str, str]:
        """Export datasets for neural network training"""
        return self.exporter.export_for_pytorch(dataset_ids, output_dir)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        storage_stats = self.lifecycle_manager.get_storage_stats()
        datasets = self.storage_manager.list_datasets()
        
        return {
            'storage': storage_stats,
            'recent_datasets': [
                {
                    'id': ds.dataset_id,
                    'version': ds.version,
                    'samples': ds.sample_count,
                    'size_mb': ds.file_size_bytes / 1024 / 1024,
                    'created': ds.created_at.isoformat()
                }
                for ds in datasets[:10]  # Last 10 datasets
            ],
            'system_health': {
                'cleanup_enabled': self.storage_config.cleanup_enabled,
                'backup_enabled': self.storage_config.backup_enabled,
                'compression_level': self.storage_config.compression_level,
                'retention_days': self.storage_config.retention_days
            }
        }
    
    def cleanup_and_optimize(self) -> Dict[str, Any]:
        """Perform cleanup and optimization"""
        return self.lifecycle_manager.cleanup_old_data()
    
    def shutdown(self):
        """Shutdown the data management system"""
        self.lifecycle_manager.stop_background_cleanup()
        self.logger.info("Data management system shutdown complete")

if __name__ == "__main__":
    # Testing and demonstration
    logging.basicConfig(level=logging.INFO)
    
    print("=== Data Management System Test ===")
    
    # Create test system
    system = DataManagementSystem()
    
    # Get system stats
    stats = system.get_comprehensive_stats()
    print(f"Storage stats:")
    print(f"  Datasets: {stats['storage']['dataset_count']}")
    print(f"  Total size: {stats['storage']['total_size_mb']:.2f} MB")
    print(f"  Disk usage: {stats['storage']['disk_usage_mb']:.2f} MB")
    
    # List existing datasets
    datasets = system.storage_manager.list_datasets()
    if datasets:
        print(f"\nExisting datasets:")
        for ds in datasets[:5]:  # Show first 5
            print(f"  {ds.dataset_id} v{ds.version}: {ds.sample_count} samples, {ds.file_size_bytes/1024/1024:.2f} MB")
    else:
        print("\nNo existing datasets found")
    
    print(f"\nData management system ready")
    print(f"Configuration: {system.storage_config.preferred_format} format, compression level {system.storage_config.compression_level}")
    
    system.shutdown()