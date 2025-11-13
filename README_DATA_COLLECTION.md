# Battlesnake Self-Play Data Collection Infrastructure

A high-throughput automated data collection system for generating neural network training data from Battlesnake games. This system implements Phase 1 of the self-play training architecture, targeting 100+ games/hour data generation with sophisticated heuristic supervision.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [System Architecture](#system-architecture)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Data Pipeline](#data-pipeline)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Integration](#integration)
- [API Reference](#api-reference)

## Overview

### What This System Does

The self-play data collection infrastructure automates the generation of high-quality training data for Battlesnake neural networks by:

1. **Orchestrating Multiple Servers**: Manages 4 concurrent Battlesnake servers on ports 8000-8003
2. **Automating Game Execution**: Uses Battlesnake CLI to run games continuously
3. **Extracting Decision Data**: Captures sophisticated heuristic scores (Safety: 8-50pts, Territory: 3-9pts, etc.)
4. **Encoding Board States**: Converts games into 12-channel neural network input format
5. **Managing Data Lifecycle**: Compresses, stores, and exports training data efficiently

### Key Features

- **High Throughput**: 100+ games/hour target performance
- **12-Channel Board Encoding**: Architecture-compliant neural network input format
- **Heuristic Supervision**: Extracts sophisticated decision scores from existing Rust system
- **Compressed Storage**: Efficient data management with versioning and cleanup
- **PyTorch Integration**: Direct export to PyTorch-compatible format
- **Production Quality**: Robust error handling, monitoring, and recovery

## Quick Start

### Prerequisites

```bash
# Required software
- Rust (cargo) for Battlesnake server
- Python 3.8+ with pip
- Battlesnake CLI: `npm install -g @battlesnake/cli`

# Optional (for advanced features)
- HDF5 libraries: `apt-get install libhdf5-dev` (Ubuntu)
```

### 30-Second Demo

```bash
# 1. Install Python dependencies
pip install numpy requests psutil

# 2. Run a quick test
python -c "
from config.self_play_config import get_config
config = get_config()
print(f'System ready: {len(config.servers)} servers configured')
"

# 3. Start data collection (10 games)
python self_play_automation.py --batch 10

# 4. Check generated data
ls -la data/self_play/training/
```

## Installation

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential python3-dev libhdf5-dev

# macOS
brew install hdf5

# Install Battlesnake CLI globally
npm install -g @battlesnake/cli
```

### Python Dependencies

```bash
# Core dependencies (required)
pip install numpy requests psutil

# Optional dependencies (recommended)
pip install h5py pandas  # For HDF5 storage and analysis
pip install torch        # For PyTorch integration
```

### Verify Installation

```bash
# Test Battlesnake CLI
battlesnake version

# Test Rust compilation
cargo build --release

# Test Python modules
python -c "import numpy, requests, psutil; print('✅ Core dependencies OK')"
```

## System Architecture

### Component Overview

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Configuration     │    │   Server Pool        │    │   Game Orchestrator │
│   Management        │───▶│   Management         │───▶│   (Battlesnake CLI) │
│                     │    │                      │    │                     │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
                                       │
                                       ▼
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Data Management   │◀───│   Training Data      │◀───│   Game Data         │
│   System            │    │   Pipeline           │    │   Extractor         │
│                     │    │                      │    │                     │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

### Data Flow

1. **Game Orchestration**: Multiple Rust servers (ports 8000-8003) serve Battlesnake API
2. **CLI Automation**: Battlesnake CLI runs automated games against these servers
3. **Real-time Extraction**: Log parsers capture game states and heuristic scores
4. **Neural Encoding**: Board states converted to 12-channel tensors + feature vectors
5. **Storage**: Compressed pickle/HDF5 storage with metadata and versioning
6. **Export**: PyTorch-ready datasets with train/val/test splits

## Configuration

### Basic Configuration

The system uses `config/self_play_config.py` for centralized configuration:

```python
# Default configuration automatically created on first run
python -c "
from config.self_play_config import get_config
config = get_config()
print(f'Target: {config.data_collection.target_games_per_hour} games/hour')
print(f'Servers: {len(config.servers)} concurrent servers')
print(f'Storage: {config.data_collection.data_directory}')
"
```

### Advanced Configuration

Create `config/self_play_settings.json` to override defaults:

```json
{
  "servers": [
    {"port": 8000, "name": "battlesnake-server-8000", "timeout_ms": 500},
    {"port": 8001, "name": "battlesnake-server-8001", "timeout_ms": 500},
    {"port": 8002, "name": "battlesnake-server-8002", "timeout_ms": 500},
    {"port": 8003, "name": "battlesnake-server-8003", "timeout_ms": 500}
  ],
  "game": {
    "board_width": 11,
    "board_height": 11,
    "timeout_ms": 500,
    "turn_limit": 500
  },
  "data_collection": {
    "target_games_per_hour": 120,
    "concurrent_servers": 4,
    "board_encoding_channels": 12,
    "data_directory": "data/self_play",
    "compression_level": 6,
    "retention_days": 30
  }
}
```

### Performance Tuning

```python
# High-throughput configuration
{
  "data_collection": {
    "target_games_per_hour": 200,        # Aggressive target
    "max_games_per_batch": 50,           # Larger batches
    "compression_level": 3,              # Faster compression
    "backup_enabled": false              # Disable for max speed
  },
  "game": {
    "timeout_ms": 300,                   # Faster games
    "turn_limit": 300                    # Shorter games
  }
}
```

## Usage Guide

### Basic Usage

#### 1. Run a Batch of Games

```bash
# Run 25 games and stop
python self_play_automation.py --batch 25

# Run with specific configuration
python self_play_automation.py --batch 50 --config config/custom_settings.json
```

#### 2. Continuous Data Collection

```bash
# Run continuously (until stopped with Ctrl+C)
python self_play_automation.py --continuous

# Run for specific duration
python self_play_automation.py --continuous 2.5  # 2.5 hours
```

#### 3. Monitor System Performance

```bash
# Real-time monitoring
python -c "
from self_play_automation import SelfPlayAutomationManager
manager = SelfPlayAutomationManager()
if manager.start():
    import time
    time.sleep(10)
    stats = manager.get_comprehensive_stats()
    print(f'Games/hour: {stats[\"system\"][\"actual_games_per_hour\"]:.1f}')
    manager.stop()
"
```

### Advanced Usage

#### 1. Custom Data Processing

```python
from training_data_pipeline import TrainingDataProcessor
from game_data_extractor import RealTimeGameExtractor
from data_management import DataManagementSystem

# Initialize components
extractor = RealTimeGameExtractor()
processor = TrainingDataProcessor()
data_system = DataManagementSystem()

# Process games and store data
extractor.start_monitoring()
# ... run games ...
completed_games = extractor.get_completed_games()

for game in completed_games:
    samples = processor.process_game(game)
    if samples:
        metadata = data_system.save_training_data(
            samples, f"custom_run_{game.game_id}", "1.0"
        )
        print(f"Saved {len(samples)} samples: {metadata.dataset_id}")

extractor.stop_monitoring()
```

#### 2. Export for Neural Network Training

```python
from data_management import DataManagementSystem

system = DataManagementSystem()

# List available datasets
datasets = system.storage_manager.list_datasets()
print(f"Available datasets: {len(datasets)}")
for ds in datasets[:5]:
    print(f"  {ds.dataset_id} v{ds.version}: {ds.sample_count} samples")

# Export multiple datasets for training
dataset_ids = ["game_batch_1", "game_batch_2", "continuous_run_1"]
export_paths = system.export_for_training(dataset_ids, "export/pytorch_data")

print(f"Exported to: {export_paths}")
# Use the generated pytorch_dataset.py file for training
```

#### 3. Data Quality Analysis

```python
from data_management import DataValidator

validator = DataValidator()

# Load and validate a dataset
system = DataManagementSystem()
samples, metadata = system.load_training_data("your_dataset_id")

validation_result = validator.validate_dataset(samples)
print(f"Valid samples: {validation_result['valid_samples']}/{validation_result['sample_count']}")
print(f"Data quality: {validation_result['valid']}")

if validation_result['stats']:
    stats = validation_result['stats']
    print(f"Move distribution: {stats['move_distribution']}")
    print(f"Avg game length: {stats['avg_game_length']:.1f} turns")
```

## Data Pipeline

### Training Sample Format

Each training sample contains:

```python
TrainingSample(
    # Input tensors for neural network
    board_state=np.ndarray(shape=(11, 11, 12), dtype=float32),    # 12-channel board
    snake_features=np.ndarray(shape=(32,), dtype=float32),        # Snake features
    game_context=np.ndarray(shape=(16,), dtype=float32),          # Game context
    
    # Target outputs for supervised learning
    target_move=int,                    # 0=up, 1=down, 2=left, 3=right
    position_value=float,               # -50 to +50 position evaluation
    move_probabilities=np.ndarray(4),   # Move preference distribution
    win_probability=float,              # 0-1 game outcome prediction
    
    # Supervision from heuristic system
    heuristic_scores={
        'safety': float,      # 8-50 points from collision detection
        'territory': float,   # 3-9 points from Voronoi control
        'food': float,        # 0-11 points from A* pathfinding
        'opponent': float,    # 0-6 points from opponent modeling
        'exploration': float  # ~30 points from exploration
    },
    
    # Metadata
    game_id=str, turn=int, snake_id=str, timestamp=datetime
)
```

### 12-Channel Board Encoding

| Channel | Content | Description |
|---------|---------|-------------|
| 0 | Our Snake | Head=1.0, body segments=0.9,0.8,0.7... |
| 1-3 | Opponent Snakes | Same encoding for up to 3 opponents |
| 4 | Food | Food positions = 1.0 |
| 5 | Walls | Board boundaries = 1.0 |
| 6 | Our Territory | Voronoi territory control (distance-based) |
| 7 | Opponent Territory | Combined opponent territory |
| 8 | Danger Zones | Collision risks (1.0=deadly, 0.6=risky, 0.3=minor) |
| 9 | Movement History | Recent positions with decay |
| 10 | Strategic Positions | Cutting points, tactical positions |
| 11 | Game State | Turn urgency, health pressure, food scarcity |

### Storage Formats

#### Pickle (Default)
- **File**: `dataset_v1.0_20240315_143022.pkl.gz`
- **Pros**: Fast, preserves Python objects exactly
- **Cons**: Python-specific, larger files
- **Use**: Development, rapid prototyping

#### HDF5 (Recommended for Production)
- **File**: `dataset_v1.0_20240315_143022.h5`
- **Pros**: Cross-platform, efficient arrays, metadata
- **Cons**: Requires h5py installation
- **Use**: Large datasets, production systems

#### JSON (Human-Readable)
- **File**: `dataset_v1.0_20240315_143022.json.gz`
- **Pros**: Human-readable, debugging-friendly
- **Cons**: Slower, larger files
- **Use**: Debugging, data inspection

## Performance Tuning

### Achieving 100+ Games/Hour

#### 1. System Configuration
```bash
# Optimize for throughput
export RUST_LOG=error  # Reduce log noise
ulimit -n 8192        # Increase file descriptors
```

#### 2. Configuration Tuning
```python
# High-performance config
{
  "game": {
    "timeout_ms": 300,    # Faster decision making
    "turn_limit": 250     # Shorter games
  },
  "data_collection": {
    "compression_level": 3,           # Faster compression
    "max_games_per_batch": 40,        # Larger batches
    "backup_enabled": False           # Skip backup overhead
  }
}
```

#### 3. Hardware Recommendations
- **CPU**: 4+ cores (one per server + processing)
- **RAM**: 8GB+ (board encodings are memory-intensive)
- **Disk**: SSD recommended for compressed data I/O
- **Network**: Local only (no external dependencies)

### Performance Monitoring

```python
# Monitor performance in real-time
from self_play_automation import SelfPlayAutomationManager
import time

manager = SelfPlayAutomationManager()
manager.start()

for i in range(10):
    time.sleep(30)  # Check every 30 seconds
    stats = manager.get_comprehensive_stats()
    
    current_rate = stats['system']['actual_games_per_hour']
    target_rate = stats['system']['target_games_per_hour']
    efficiency = stats['system']['efficiency_percent']
    
    print(f"Minute {i+1}: {current_rate:.1f} games/hour ({efficiency:.1f}% efficiency)")
    
    if current_rate < target_rate * 0.8:
        print("⚠️  Performance below target - check system resources")

manager.stop()
```

## Troubleshooting

### Common Issues

#### 1. "Battlesnake CLI not found"
```bash
# Install Battlesnake CLI
npm install -g @battlesnake/cli

# Verify installation
battlesnake version

# Check PATH
echo $PATH | grep npm
```

#### 2. "Port already in use"
```bash
# Check what's using the port
lsof -i :8000

# Kill conflicting processes
pkill -f "cargo run"

# Or change ports in config
vim config/self_play_settings.json
```

#### 3. "Server failed to start"
```bash
# Check Rust compilation
cargo build --release

# Check server logs
RUST_LOG=debug cargo run

# Verify PORT environment variable
echo $PORT
export PORT=8000
```

#### 4. "Low throughput performance"
```bash
# Check system resources
top
df -h

# Optimize configuration
# Reduce compression: compression_level: 1
# Reduce game length: turn_limit: 200
# Disable backup: backup_enabled: false
```

#### 5. "Data validation errors"
```python
# Debug data quality issues
from data_management import DataValidator

validator = DataValidator()
# ... load problematic samples ...
for i, sample in enumerate(bad_samples):
    result = validator.validate_training_sample(sample)
    if not result['valid']:
        print(f"Sample {i} issues: {result['issues']}")
```

### Debug Mode

```bash
# Enable verbose logging
python self_play_automation.py --batch 5 --log-level DEBUG

# Monitor detailed server output
tail -f logs/self_play_collection.log

# Check individual game logs
ls logs/games/
cat logs/games/game_20240315_143022_0001.log
```

### System Health Checks

```python
# Comprehensive health check
from self_play_automation import SelfPlayAutomationManager
from data_management import DataManagementSystem

def health_check():
    print("=== System Health Check ===")
    
    # 1. Configuration
    try:
        from config.self_play_config import get_config
        config = get_config()
        print("✅ Configuration: OK")
    except Exception as e:
        print(f"❌ Configuration: {e}")
        return False
    
    # 2. Dependencies
    try:
        import numpy, requests, psutil
        print("✅ Dependencies: OK")
    except Exception as e:
        print(f"❌ Dependencies: {e}")
        return False
    
    # 3. Storage
    try:
        system = DataManagementSystem()
        stats = system.get_comprehensive_stats()
        print(f"✅ Storage: {stats['storage']['dataset_count']} datasets")
    except Exception as e:
        print(f"❌ Storage: {e}")
        return False
    
    # 4. Battlesnake CLI
    try:
        import subprocess
        result = subprocess.run(['battlesnake', 'version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Battlesnake CLI: OK")
        else:
            print("❌ Battlesnake CLI: Not working")
            return False
    except Exception as e:
        print(f"❌ Battlesnake CLI: {e}")
        return False
    
    print("✅ All systems operational")
    return True

if __name__ == "__main__":
    health_check()
```

## Integration

### Neural Network Training Integration

#### 1. Export Data for PyTorch

```python
from data_management import DataManagementSystem

# Export training data
system = DataManagementSystem()
dataset_ids = ["batch_1", "batch_2", "continuous_1"]  # Your dataset IDs
paths = system.export_for_training(dataset_ids, "neural_training/data")

print(f"Training data exported to: {paths}")
# Creates: neural_training/data/train/, val/, test/ with .npy files
# Also creates: neural_training/data/pytorch_dataset.py for easy loading
```

#### 2. Use Generated PyTorch Dataset

```python
# Use the auto-generated dataset loader
import sys
sys.path.append('neural_training/data')
from pytorch_dataset import create_data_loaders

# Create data loaders
loaders = create_data_loaders('neural_training/data', batch_size=64)

# Training loop example
for batch in loaders['train']:
    board_states = batch['board_state']      # [batch, 11, 11, 12]
    snake_features = batch['snake_features'] # [batch, 32]
    game_context = batch['game_context']     # [batch, 16]
    
    target_moves = batch['target_move']      # [batch] - classification target
    position_values = batch['position_value'] # [batch] - regression target
    # ... training code ...
```

#### 3. Continuous Training Pipeline

```python
# Production training pipeline with continuous data collection

import threading
import time
from pathlib import Path

class ContinuousTrainingPipeline:
    def __init__(self):
        self.data_system = DataManagementSystem()
        self.automation = SelfPlayAutomationManager()
        
    def start_data_collection(self):
        """Start background data collection"""
        def collect_data():
            self.automation.start()
            self.automation.run_continuous()  # Run indefinitely
        
        self.collection_thread = threading.Thread(target=collect_data, daemon=True)
        self.collection_thread.start()
        
    def periodic_training_update(self, model, optimizer, export_dir="training_data"):
        """Periodically retrain model with new data"""
        
        while True:
            # Wait for new data (e.g., every hour)
            time.sleep(3600)  
            
            # Get recent datasets
            datasets = self.data_system.storage_manager.list_datasets()
            recent_datasets = [ds.dataset_id for ds in datasets[:5]]  # Last 5 datasets
            
            if recent_datasets:
                # Export new training data
                paths = self.data_system.export_for_training(recent_datasets, export_dir)
                
                # Retrain model (your training code here)
                self.retrain_model(model, optimizer, paths)
                print(f"Model updated with {len(recent_datasets)} new datasets")

# Usage
pipeline = ContinuousTrainingPipeline()
pipeline.start_data_collection()
# pipeline.periodic_training_update(your_model, your_optimizer)
```

### API Integration

#### REST API for External Systems

```python
# Simple Flask API to expose data collection
from flask import Flask, jsonify, request
from data_management import DataManagementSystem

app = Flask(__name__)
data_system = DataManagementSystem()

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    datasets = data_system.storage_manager.list_datasets()
    return jsonify([{
        'id': ds.dataset_id,
        'version': ds.version,
        'samples': ds.sample_count,
        'created': ds.created_at.isoformat()
    } for ds in datasets])

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify(data_system.get_comprehensive_stats())

@app.route('/api/export', methods=['POST'])
def export_datasets():
    dataset_ids = request.json.get('dataset_ids', [])
    output_dir = request.json.get('output_dir', 'export')
    
    paths = data_system.export_for_training(dataset_ids, output_dir)
    return jsonify({'export_paths': paths})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## API Reference

### Key Classes

#### `SelfPlayAutomationManager`
Main orchestration class for the entire system.

```python
manager = SelfPlayAutomationManager(config_file=None)
success = manager.start()                    # Start all systems
success = manager.run_batch(num_games=50)    # Run specific number of games
manager.run_continuous(duration_hours=2.0)  # Run for specified duration
stats = manager.get_comprehensive_stats()    # Get performance metrics
manager.stop()                               # Clean shutdown
```

#### `DataManagementSystem`
Handles all data storage, versioning, and export operations.

```python
system = DataManagementSystem(config=None)

# Save training data
metadata = system.save_training_data(samples, dataset_id, version="1.0", description="")

# Load training data
samples, metadata = system.load_training_data(dataset_id, version=None)

# Export for neural training
paths = system.export_for_training(dataset_ids, output_dir)

# System statistics
stats = system.get_comprehensive_stats()
cleanup_result = system.cleanup_and_optimize()
system.shutdown()
```

#### `TrainingDataProcessor`
Converts game data to neural network training format.

```python
processor = TrainingDataProcessor(config=None)

# Process individual games
samples = processor.process_game(game_data)

# Process multiple games in parallel
all_samples = processor.process_games_batch(games_list)

# Validate samples
is_valid = processor.validate_sample(sample)

# Get processing statistics
stats = processor.get_processing_stats()
```

### Configuration Reference

```python
# Complete configuration structure
{
  "servers": [
    {
      "port": int,           # Server port (8000-8003)
      "name": str,           # Server identifier
      "timeout_ms": int,     # Decision timeout (500)
      "max_memory_mb": int   # Memory limit (256)
    }
  ],
  "game": {
    "board_width": int,      # Board width (11)
    "board_height": int,     # Board height (11)
    "game_mode": str,        # Game mode ("solo")
    "timeout_ms": int,       # Game timeout (500)
    "turn_limit": int,       # Max turns (500)
    "food_spawn_chance": int, # Food spawn % (15)
    "minimum_food": int      # Min food count (1)
  },
  "data_collection": {
    "target_games_per_hour": int,     # Performance target (100)
    "concurrent_servers": int,        # Parallel servers (4)
    "max_games_per_batch": int,       # Batch size (25)
    "board_encoding_channels": int,   # Tensor channels (12)
    "feature_vector_size": int,       # Feature size (32)
    "context_vector_size": int,       # Context size (16)
    "data_directory": str,            # Storage path
    "compression_level": int,         # Compression (1-9)
    "max_file_size_mb": int,         # File size limit (100)
    "retention_days": int,            # Data retention (30)
    "backup_enabled": bool           # Enable backups (true)
  }
}
```

---

## Support and Contributing

### Getting Help
- **Issues**: Create GitHub issues for bugs or feature requests
- **Performance**: Use the troubleshooting section and health checks
- **Integration**: See the integration examples and API reference

### Performance Targets
- **Throughput**: 100+ games/hour (validated in tests)
- **Data Quality**: >90% valid training samples
- **Architecture Compliance**: 12-channel board encoding, heuristic supervision
- **Storage Efficiency**: Compressed format with automated lifecycle management

### System Requirements Met
✅ **100+ games/hour throughput**  
✅ **12-channel board encoding**  
✅ **Heuristic score extraction**  
✅ **Compressed storage with versioning**  
✅ **PyTorch integration**  
✅ **Production-quality error handling**  
✅ **Comprehensive testing and validation**