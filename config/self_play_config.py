"""
Self-Play Training Pipeline Configuration Management

This module provides centralized configuration for the Battlesnake self-play training pipeline.
Handles data collection, neural network training, model evolution, and autonomous operation.

Architecture compliance: 332K games/hour throughput, 12-channel board encoding, 
CNN + Attention + Residual neural networks, autonomous training loops
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path

@dataclass
class ServerConfig:
    """Configuration for individual Battlesnake server instances"""
    port: int
    name: str
    timeout_ms: int = 500
    max_memory_mb: int = 256
    
    def get_env_vars(self) -> Dict[str, str]:
        """Get environment variables for this server instance"""
        return {
            'PORT': str(self.port),
            'RUST_LOG': 'info',
            'BATTLESNAKE_SERVER_TIMEOUT': str(self.timeout_ms)
        }

@dataclass  
class GameConfig:
    """Configuration for Battlesnake CLI game parameters"""
    board_width: int = 11
    board_height: int = 11
    game_mode: str = "solo"  # solo, duo, squad
    timeout_ms: int = 500
    turn_limit: int = 500
    food_spawn_chance: int = 15
    minimum_food: int = 1
    hazard_damage: int = 0
    
    def get_cli_args(self) -> List[str]:
        """Get Battlesnake CLI arguments from configuration"""
        return [
            "-W", str(self.board_width),
            "-H", str(self.board_height), 
            "-g", self.game_mode,
            "--timeout", str(self.timeout_ms),
            "--turn-timeout", str(self.timeout_ms),
            "--food-spawn-chance", str(self.food_spawn_chance),
            "--minimum-food", str(self.minimum_food)
        ]

@dataclass
class DataCollectionConfig:
    """Configuration for data collection and processing"""
    # Performance targets
    target_games_per_hour: int = 100
    concurrent_servers: int = 4
    max_games_per_batch: int = 25
    
    # Data format specifications  
    board_encoding_channels: int = 12
    feature_vector_size: int = 32
    context_vector_size: int = 16
    
    # Storage configuration
    data_directory: str = "data/self_play"
    compression_level: int = 6
    max_file_size_mb: int = 100
    retention_days: int = 30
    backup_enabled: bool = True
    
    # Quality assurance
    min_game_turns: int = 5
    max_invalid_samples_percent: float = 10.0
    required_heuristic_components: List[str] = None
    
    def __post_init__(self):
        if self.required_heuristic_components is None:
            self.required_heuristic_components = [
                'safety', 'territory', 'opponent', 'food', 'exploration'
            ]

@dataclass
class TrainingPhaseConfig:
    """Configuration for individual training phases"""
    name: str
    games_required: int
    learning_rate: float
    batch_size: int
    epochs: int
    data_source: str  # 'heuristic_supervision', 'mixed', 'self_play', 'continuous'
    heuristic_mix_ratio: float = 0.0  # For hybrid phase
    validation_split: float = 0.15
    early_stopping_patience: int = 10

@dataclass
class ModelTournamentConfig:
    """Configuration for model performance tournaments"""
    evaluation_games: int = 1000
    confidence_level: float = 0.95
    minimum_improvement: float = 0.05  # 5% win rate improvement required
    
    # Promotion criteria
    min_win_rate_vs_heuristic: float = 0.65
    max_inference_time_ms: int = 10
    max_memory_usage_mb: int = 50
    min_strategic_quality_score: float = 0.70
    
    # Opponent selection for tournaments
    baseline_opponents: List[str] = None
    champion_retention_games: int = 500  # Games to keep previous champion
    
    def __post_init__(self):
        if self.baseline_opponents is None:
            self.baseline_opponents = ['random', 'heuristic', 'previous_champion']

@dataclass 
class NeuralNetworkConfig:
    """Configuration for neural network training"""
    # Model architecture (using Phase 9 advanced networks)
    model_type: str = "MultiTaskBattlesnakeNetwork"  # From neural_models.py
    
    # Training parameters
    optimizer: str = "AdamW"
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    loss_weights: Dict[str, float] = None
    
    # Regularization
    dropout_rate: float = 0.1
    label_smoothing: float = 0.1
    
    # Data augmentation
    use_data_augmentation: bool = True
    augmentation_probability: float = 0.5
    
    # Model constraints (Phase 9 compliance)
    max_model_size_mb: int = 50
    max_inference_time_ms: int = 10
    target_accuracy_threshold: float = 0.70
    
    # ONNX export settings
    onnx_optimization_level: str = "all"  # "basic", "extended", "all"
    onnx_validation_tolerance: float = 1e-5
    
    def __post_init__(self):
        if self.loss_weights is None:
            self.loss_weights = {
                'position_evaluation': 0.4,
                'move_prediction': 0.4,
                'game_outcome': 0.2
            }

@dataclass
class TrainingPipelineConfig:
    """Configuration for the complete training pipeline"""
    
    # Training phases (Bootstrap -> Hybrid -> Self-Play -> Continuous)
    phases: List[TrainingPhaseConfig] = None
    
    # Model management
    tournament: ModelTournamentConfig = None
    neural_network: NeuralNetworkConfig = None
    
    # Continuous learning settings
    continuous_training_enabled: bool = True
    retraining_interval_hours: int = 24
    minimum_new_games_for_retraining: int = 1000
    
    # Model versioning
    model_version_format: str = "v{major}.{minor}_{timestamp}"
    keep_model_history: int = 10  # Number of previous models to retain
    
    # Performance monitoring  
    performance_tracking_enabled: bool = True
    metrics_collection_interval_s: int = 60
    training_progress_checkpoint_interval: int = 100  # batches
    
    # Resource management
    max_training_time_hours: int = 4
    gpu_memory_fraction: float = 0.8
    cpu_workers: int = 4
    
    # Safety and rollback
    enable_model_rollback: bool = True
    rollback_performance_threshold: float = 0.9  # 90% of previous performance
    max_consecutive_failures: int = 3
    
    def __post_init__(self):
        if self.phases is None:
            self.phases = self._create_default_phases()
        if self.tournament is None:
            self.tournament = ModelTournamentConfig()
        if self.neural_network is None:
            self.neural_network = NeuralNetworkConfig()
    
    def _create_default_phases(self) -> List[TrainingPhaseConfig]:
        """Create default training phase configuration"""
        return [
            TrainingPhaseConfig(
                name="bootstrap",
                games_required=1000,
                learning_rate=0.001,
                batch_size=64,
                epochs=50,
                data_source="heuristic_supervision"
            ),
            TrainingPhaseConfig(
                name="hybrid", 
                games_required=5000,
                learning_rate=0.0005,
                batch_size=128,
                epochs=100,
                data_source="mixed",
                heuristic_mix_ratio=0.7
            ),
            TrainingPhaseConfig(
                name="self_play",
                games_required=10000,
                learning_rate=0.0001, 
                batch_size=256,
                epochs=200,
                data_source="self_play"
            ),
            TrainingPhaseConfig(
                name="continuous",
                games_required=-1,  # Ongoing
                learning_rate=0.00005,
                batch_size=256,
                epochs=50,
                data_source="continuous"
            )
        ]
    
    def get_phase_config(self, phase_name: str) -> Optional[TrainingPhaseConfig]:
        """Get configuration for specific training phase"""
        for phase in self.phases:
            if phase.name == phase_name:
                return phase
        return None

@dataclass
class SystemConfig:
    """Main system configuration combining all components"""
    servers: List[ServerConfig]
    game: GameConfig
    data_collection: DataCollectionConfig
    
    # Training pipeline configuration
    training_pipeline: TrainingPipelineConfig = None
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/self_play_training.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Monitoring and health checks
    health_check_interval_s: int = 30
    performance_monitoring_enabled: bool = True
    error_retry_attempts: int = 3
    error_retry_delay_s: int = 5

    def __post_init__(self):
        if self.training_pipeline is None:
            self.training_pipeline = TrainingPipelineConfig()

class ConfigManager:
    """Manages configuration loading, validation, and port allocation"""
    
    DEFAULT_CONFIG_FILE = "config/self_play_settings.json"
    DEFAULT_PORTS = [8000, 8001, 8002, 8003]
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self.DEFAULT_CONFIG_FILE
        self.config: Optional[SystemConfig] = None
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        directories = [
            "config", "data/self_play", "logs", "data/self_play/raw",
            "data/self_play/processed", "data/self_play/training", 
            "data/self_play/backups", "data/self_play/metadata",
            "models", "models/checkpoints", "models/tournaments"
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> SystemConfig:
        """Load configuration from file or create default"""
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                self.config = self._dict_to_config(data)
                logging.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logging.warning(f"Failed to load config from {self.config_file}: {e}")
                self.config = self._create_default_config()
        else:
            self.config = self._create_default_config()
            self.save_config()
            
        self._validate_config()
        return self.config
    
    def _create_default_config(self) -> SystemConfig:
        """Create default configuration"""
        servers = [
            ServerConfig(port=port, name=f"battlesnake-server-{port}")
            for port in self.DEFAULT_PORTS
        ]
        
        return SystemConfig(
            servers=servers,
            game=GameConfig(),
            data_collection=DataCollectionConfig(concurrent_servers=len(servers)),
            training_pipeline=TrainingPipelineConfig()
        )
    
    def _dict_to_config(self, data: Dict[str, Any]) -> SystemConfig:
        """Convert dictionary to SystemConfig object"""
        servers = [ServerConfig(**server) for server in data['servers']]
        game = GameConfig(**data['game'])
        data_collection = DataCollectionConfig(**data['data_collection'])
        
        # Handle training pipeline configuration
        training_pipeline = None
        if 'training_pipeline' in data:
            tp_data = data['training_pipeline']
            
            # Convert phases
            phases = None
            if 'phases' in tp_data:
                phases = [TrainingPhaseConfig(**phase) for phase in tp_data['phases']]
            
            # Convert tournament config
            tournament = None
            if 'tournament' in tp_data:
                tournament = ModelTournamentConfig(**tp_data['tournament'])
            
            # Convert neural network config
            neural_network = None
            if 'neural_network' in tp_data:
                neural_network = NeuralNetworkConfig(**tp_data['neural_network'])
            
            # Create training pipeline config
            tp_base = {k: v for k, v in tp_data.items() 
                      if k not in ['phases', 'tournament', 'neural_network']}
            training_pipeline = TrainingPipelineConfig(
                phases=phases,
                tournament=tournament, 
                neural_network=neural_network,
                **tp_base
            )
        
        system_data = {k: v for k, v in data.items() 
                      if k not in ['servers', 'game', 'data_collection', 'training_pipeline']}
        
        return SystemConfig(
            servers=servers,
            game=game, 
            data_collection=data_collection,
            training_pipeline=training_pipeline,
            **system_data
        )
    
    def save_config(self):
        """Save current configuration to file"""
        if self.config is None:
            raise ValueError("No configuration loaded to save")
            
        config_dict = asdict(self.config)
        
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logging.info(f"Configuration saved to {self.config_file}")
    
    def _validate_config(self):
        """Validate configuration for common issues"""
        if self.config is None:
            raise ValueError("No configuration to validate")
            
        # Validate port uniqueness
        ports = [server.port for server in self.config.servers]
        if len(ports) != len(set(ports)):
            raise ValueError("Duplicate ports detected in server configuration")
        
        # Validate port range 
        for port in ports:
            if not (1024 <= port <= 65535):
                raise ValueError(f"Invalid port {port}: must be between 1024-65535")
        
        # Validate performance targets
        dc = self.config.data_collection
        if dc.concurrent_servers != len(self.config.servers):
            logging.warning(
                f"concurrent_servers ({dc.concurrent_servers}) != "
                f"actual servers ({len(self.config.servers)}), adjusting"
            )
            dc.concurrent_servers = len(self.config.servers)
        
        # Validate data collection parameters
        if dc.board_encoding_channels != 12:
            raise ValueError("board_encoding_channels must be 12 per architecture spec")
        
        if dc.target_games_per_hour < 10:
            raise ValueError("target_games_per_hour too low, minimum 10")
        
        # Validate training pipeline configuration
        if self.config.training_pipeline:
            tp = self.config.training_pipeline
            
            # Validate phase progression
            expected_phases = ["bootstrap", "hybrid", "self_play", "continuous"]
            actual_phases = [phase.name for phase in tp.phases]
            if actual_phases != expected_phases:
                logging.warning(f"Phase order unusual: {actual_phases}, expected: {expected_phases}")
            
            # Validate neural network constraints
            nn = tp.neural_network
            if nn.max_inference_time_ms > 15:
                logging.warning(f"Inference time target {nn.max_inference_time_ms}ms may be too slow")
            
            if nn.max_model_size_mb > 50:
                raise ValueError("Model size must be ≤50MB per deployment constraints")
            
            logging.info("Training pipeline configuration validation passed")
        
        logging.info("Configuration validation passed")
    
    def get_available_ports(self) -> List[int]:
        """Get list of available ports from configuration"""
        return [server.port for server in self.config.servers]
    
    def allocate_port(self) -> int:
        """Allocate next available port (simple round-robin)"""
        if self.config is None:
            raise ValueError("No configuration loaded")
        return self.config.servers[0].port  # For now, return first port
    
    def get_server_config(self, port: int) -> Optional[ServerConfig]:
        """Get server configuration for specific port"""
        if self.config is None:
            return None
        
        for server in self.config.servers:
            if server.port == port:
                return server
        return None
    
    def estimate_throughput(self) -> Dict[str, float]:
        """Estimate system throughput based on configuration"""
        if self.config is None:
            raise ValueError("No configuration loaded")
            
        # Basic throughput estimation
        servers = len(self.config.servers)
        game_duration_estimate_minutes = 3.0  # Average game length
        games_per_server_per_hour = 60 / game_duration_estimate_minutes
        total_games_per_hour = servers * games_per_server_per_hour
        
        return {
            'servers': servers,
            'estimated_games_per_hour': total_games_per_hour,
            'target_games_per_hour': self.config.data_collection.target_games_per_hour,
            'utilization_percent': (self.config.data_collection.target_games_per_hour / 
                                   total_games_per_hour) * 100
        }

    def estimate_training_time(self) -> Dict[str, Any]:
        """Estimate training time for each phase"""
        if self.config is None or self.config.training_pipeline is None:
            raise ValueError("No training configuration loaded")
        
        tp = self.config.training_pipeline
        estimates = {}
        
        # Estimate data collection time based on throughput
        throughput = self.estimate_throughput()
        games_per_hour = throughput['estimated_games_per_hour']
        
        total_training_time_hours = 0
        
        for phase in tp.phases:
            if phase.games_required <= 0:  # Continuous phase
                estimates[phase.name] = {
                    'data_collection_hours': 'ongoing',
                    'training_time_hours': 'periodic',
                    'total_time_hours': 'continuous'
                }
                continue
            
            # Data collection time
            data_collection_hours = phase.games_required / games_per_hour
            
            # Training time estimate (rough approximation)
            # Assume ~1 hour per 1000 games of training data
            training_time_hours = (phase.games_required / 1000) * 1.0
            
            total_phase_time = data_collection_hours + training_time_hours
            total_training_time_hours += total_phase_time
            
            estimates[phase.name] = {
                'games_required': phase.games_required,
                'data_collection_hours': round(data_collection_hours, 2),
                'training_time_hours': round(training_time_hours, 2), 
                'total_time_hours': round(total_phase_time, 2)
            }
        
        estimates['total_pipeline_time_hours'] = round(total_training_time_hours, 2)
        estimates['estimated_games_per_hour'] = games_per_hour
        
        return estimates
    
    def get_training_config(self) -> Optional[TrainingPipelineConfig]:
        """Get training pipeline configuration"""
        if self.config is None:
            return None
        return self.config.training_pipeline
    
    def get_current_phase_config(self, phase_name: str) -> Optional[TrainingPhaseConfig]:
        """Get configuration for specific training phase"""
        tp = self.get_training_config()
        return tp.get_phase_config(phase_name) if tp else None

def setup_logging(config: SystemConfig):
    """Setup logging based on configuration"""
    log_level = getattr(logging, config.log_level.upper())
    
    # Create logs directory if it doesn't exist
    if config.log_file:
        Path(config.log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format=config.log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.log_file) if config.log_file else logging.NullHandler()
        ]
    )
    
    # Reduce noise from some libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

# Global configuration instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config() -> SystemConfig:
    """Get loaded system configuration"""
    return get_config_manager().load_config()

if __name__ == "__main__":
    # Configuration testing and setup
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    print("=== Self-Play Training Pipeline Configuration ===")
    print(f"Servers: {len(config.servers)}")
    for server in config.servers:
        print(f"  - {server.name} on port {server.port}")
    
    print(f"\nGame Configuration:")
    print(f"  - Board: {config.game.board_width}x{config.game.board_height}")
    print(f"  - Mode: {config.game.game_mode}")
    print(f"  - Timeout: {config.game.timeout_ms}ms")
    
    print(f"\nData Collection:")
    print(f"  - Target: {config.data_collection.target_games_per_hour} games/hour")
    print(f"  - Channels: {config.data_collection.board_encoding_channels}")
    print(f"  - Storage: {config.data_collection.data_directory}")
    
    print(f"\nTraining Pipeline:")
    tp = config.training_pipeline
    print(f"  - Phases: {len(tp.phases)}")
    for phase in tp.phases:
        games_str = f"{phase.games_required} games" if phase.games_required > 0 else "ongoing"
        print(f"    * {phase.name}: {games_str}, lr={phase.learning_rate}")
    
    print(f"  - Tournament games: {tp.tournament.evaluation_games}")
    print(f"  - Min improvement: {tp.tournament.minimum_improvement*100:.1f}%")
    print(f"  - Model constraints: {tp.neural_network.max_inference_time_ms}ms, {tp.neural_network.max_model_size_mb}MB")
    
    throughput = config_manager.estimate_throughput()
    print(f"\nThroughput Estimate:")
    print(f"  - Estimated: {throughput['estimated_games_per_hour']:.1f} games/hour")
    print(f"  - Target: {throughput['target_games_per_hour']} games/hour") 
    print(f"  - Utilization: {throughput['utilization_percent']:.1f}%")
    
    training_time = config_manager.estimate_training_time()
    print(f"\nTraining Time Estimates:")
    for phase_name, estimate in training_time.items():
        if phase_name == 'total_pipeline_time_hours' or phase_name == 'estimated_games_per_hour':
            continue
        if isinstance(estimate, dict) and 'total_time_hours' in estimate:
            print(f"  - {phase_name}: {estimate['total_time_hours']} hours total")
    print(f"  - Total pipeline: {training_time['total_pipeline_time_hours']} hours")
    
    setup_logging(config)
    logging.info("Self-play training pipeline configuration system initialized successfully")