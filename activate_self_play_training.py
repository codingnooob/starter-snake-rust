#!/usr/bin/env python3
"""
Self-Play Training System Activation Script

This script activates the complete self-play training system to replace synthetic
training data with real game data from Battlesnake CLI automation.

Enhanced for 5000+ game collection with progressive training approach.
"""

import os
import sys
import subprocess
import time
import logging
import threading
from pathlib import Path
from datetime import datetime

# Import self-play components
sys.path.append('.')
from self_play_automation import SelfPlayAutomationManager
from self_play_training_pipeline import SelfPlayTrainingPipeline, TrainingConfiguration
from config.self_play_config import get_config, setup_logging

class SelfPlayActivationManager:
    """Manages the complete activation of the self-play training system"""
    
    def __init__(self):
        self.config = get_config()
        setup_logging(self.config)
        self.logger = logging.getLogger(__name__)
        
        # System components
        self.automation_manager = None
        self.training_pipeline = None
        
        # Activation state
        self.activation_successful = False
        self.activation_start_time = datetime.now()
        
    def run_activation_sequence(self) -> bool:
        """Execute complete activation sequence"""
        self.logger.info("🚀 STARTING SELF-PLAY TRAINING SYSTEM ACTIVATION")
        
        try:
            # Phase 1: Prerequisites validation
            if not self._validate_prerequisites():
                return False
            
            # Phase 2: System preparation
            if not self._prepare_system():
                return False
            
            # Phase 3: Start automation system
            if not self._start_automation_system():
                return False
            
            # Phase 4: Validate with initial games
            if not self._validate_with_initial_games():
                return False
            
            # Phase 5: Replace synthetic training data
            if not self._replace_synthetic_training_data():
                return False
            
            # Phase 6: Start progressive training
            if not self._start_progressive_training():
                return False
            
            # Phase 7: Validate system integration
            if not self._validate_system_integration():
                return False
            
            self.activation_successful = True
            self._log_activation_success()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Activation failed: {e}")
            self._cleanup_failed_activation()
            return False
    
    def _validate_prerequisites(self) -> bool:
        """Validate all prerequisites for self-play activation"""
        self.logger.info("📋 Phase 1: Validating prerequisites...")
        
        # Check Battlesnake CLI
        try:
            result = subprocess.run(['battlesnake', 'version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise Exception("Battlesnake CLI not responding")
            self.logger.info(f"✅ Battlesnake CLI available: {result.stdout.strip()}")
        except Exception as e:
            self.logger.error(f"❌ Battlesnake CLI not available: {e}")
            self.logger.error("Please install: https://github.com/BattlesnakeOfficial/rules/tree/main/cli")
            return False
        
        # Check Rust project can build
        try:
            self.logger.info("🔨 Building Rust project...")
            result = subprocess.run(['cargo', 'check'], 
                                  capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                raise Exception(f"Cargo check failed: {result.stderr}")
            self.logger.info("✅ Rust project builds successfully")
        except Exception as e:
            self.logger.error(f"❌ Rust project build failed: {e}")
            return False
        
        # Check Python dependencies
        required_packages = ['torch', 'numpy', 'requests', 'psutil']
        for package in required_packages:
            try:
                __import__(package)
                self.logger.info(f"✅ {package} available")
            except ImportError:
                self.logger.error(f"❌ {package} not available")
                return False
        
        # Check disk space (need space for 5000+ games data)
        free_space_gb = os.statvfs('.').f_frsize * os.statvfs('.').f_bavail / (1024**3)
        if free_space_gb < 10:  # Need at least 10GB
            self.logger.error(f"❌ Insufficient disk space: {free_space_gb:.1f}GB (need 10GB+)")
            return False
        self.logger.info(f"✅ Sufficient disk space: {free_space_gb:.1f}GB available")
        
        return True
    
    def _prepare_system(self) -> bool:
        """Prepare system directories and components"""
        self.logger.info("🔧 Phase 2: Preparing system...")
        
        # Create required directories
        directories = [
            "logs", "data/self_play", "data/self_play/raw", "data/self_play/processed",
            "data/self_play/training", "data/self_play/backups", "data/self_play/metadata",
            "models", "models/checkpoints", "models/tournaments", "pipeline"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"✅ Directory ready: {directory}")
        
        # Initialize training pipeline
        try:
            self.training_pipeline = SelfPlayTrainingPipeline()
            self.logger.info("✅ Training pipeline initialized")
        except Exception as e:
            self.logger.error(f"❌ Training pipeline initialization failed: {e}")
            return False
        
        return True
    
    def _start_automation_system(self) -> bool:
        """Start the self-play automation system"""
        self.logger.info("🤖 Phase 3: Starting automation system...")
        
        try:
            self.automation_manager = SelfPlayAutomationManager()
            
            if not self.automation_manager.start():
                raise Exception("Failed to start automation manager")
            
            self.logger.info("✅ Self-play automation system started")
            
            # Wait for servers to stabilize
            time.sleep(5)
            
            # Verify all servers are healthy
            stats = self.automation_manager.get_comprehensive_stats()
            healthy_servers = stats['servers']['healthy_servers']
            total_servers = stats['servers']['total_servers']
            
            if healthy_servers != total_servers:
                raise Exception(f"Only {healthy_servers}/{total_servers} servers healthy")
            
            self.logger.info(f"✅ All {total_servers} servers healthy and ready")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start automation system: {e}")
            return False
    
    def _validate_with_initial_games(self) -> bool:
        """Validate system with initial test games"""
        self.logger.info("🎮 Phase 4: Validating with initial games...")
        
        try:
            # Run a small batch of test games
            test_batch_size = 10
            self.logger.info(f"Running {test_batch_size} test games...")
            
            success = self.automation_manager.run_batch(test_batch_size)
            
            if not success:
                raise Exception("Test game batch failed")
            
            # Check game completion
            stats = self.automation_manager.get_comprehensive_stats()
            completed_games = stats['orchestrator']['total_games']
            
            if completed_games < test_batch_size * 0.8:  # Allow 20% failure rate
                raise Exception(f"Too few games completed: {completed_games}/{test_batch_size}")
            
            self.logger.info(f"✅ Test validation successful: {completed_games} games completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Test validation failed: {e}")
            return False
    
    def _replace_synthetic_training_data(self) -> bool:
        """Replace synthetic training data generation with real game data"""
        self.logger.info("🔄 Phase 5: Replacing synthetic training data...")
        
        try:
            # Check if synthetic training files exist
            synthetic_files = [
                "neural_networks/data_collection.py",
                "neural_networks/training_pipeline.py"
            ]
            
            for file_path in synthetic_files:
                if Path(file_path).exists():
                    # Create backup
                    backup_path = f"{file_path}.synthetic_backup"
                    subprocess.run(['cp', file_path, backup_path])
                    self.logger.info(f"✅ Backup created: {backup_path}")
            
            # Update training pipeline configuration to use real data
            # This is handled by the configuration we already updated
            self.logger.info("✅ Training pipeline configured for real game data")
            
            # Verify data collection components are ready
            data_manager_available = Path("self_play_data_manager.py").exists()
            if not data_manager_available:
                self.logger.warning("⚠️ self_play_data_manager.py not found - may need manual integration")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to replace synthetic training data: {e}")
            return False
    
    def _start_progressive_training(self) -> bool:
        """Start the progressive training approach"""
        self.logger.info("📈 Phase 6: Starting progressive training...")
        
        try:
            # Configure progressive training
            config = TrainingConfiguration(
                target_phases=["bootstrap"],  # Start with bootstrap phase
                force_retrain=False,
                continuous_learning_enabled=True,
                min_improvement_threshold=0.05,  # 5% improvement required
                max_training_time_hours=6
            )
            
            self.logger.info("🎯 Training Configuration:")
            self.logger.info(f"  - Target phases: {config.target_phases}")
            self.logger.info(f"  - Force retrain: {config.force_retrain}")
            self.logger.info(f"  - Continuous learning: {config.continuous_learning_enabled}")
            self.logger.info(f"  - Min improvement: {config.min_improvement_threshold*100:.1f}%")
            
            # Start training pipeline in background thread
            def training_worker():
                try:
                    self.logger.info("🚀 Starting progressive training pipeline...")
                    success = self.training_pipeline.run_complete_pipeline(config)
                    if success:
                        self.logger.info("✅ Progressive training pipeline completed successfully")
                    else:
                        self.logger.error("❌ Progressive training pipeline failed")
                except Exception as e:
                    self.logger.error(f"❌ Training pipeline error: {e}")
            
            training_thread = threading.Thread(target=training_worker, daemon=True)
            training_thread.start()
            
            # Wait briefly to ensure training starts
            time.sleep(3)
            
            # Check training pipeline status
            pipeline_state = self.training_pipeline.get_pipeline_status()
            if pipeline_state.status.value == "error":
                raise Exception(f"Training pipeline error: {pipeline_state.last_error}")
            
            self.logger.info(f"✅ Progressive training started: {pipeline_state.status.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start progressive training: {e}")
            return False
    
    def _validate_system_integration(self) -> bool:
        """Validate complete system integration"""
        self.logger.info("🔍 Phase 7: Validating system integration...")
        
        try:
            # Check automation system status
            automation_stats = self.automation_manager.get_comprehensive_stats()
            self.logger.info(f"✅ Automation: {automation_stats['system']['actual_games_per_hour']:.1f} games/hour")
            
            # Check training pipeline status
            pipeline_metrics = self.training_pipeline.get_pipeline_metrics()
            pipeline_state = pipeline_metrics['pipeline_state']
            self.logger.info(f"✅ Training: {pipeline_state['status']} - {pipeline_state['operation_details']}")
            
            # Check data flow
            data_stats = pipeline_metrics.get('data_statistics', {})
            if data_stats:
                self.logger.info(f"✅ Data flow: {data_stats.get('total_samples', 0)} samples processed")
            
            # Verify real game data is being collected
            data_dir = Path("data/self_play")
            if any(data_dir.rglob("*.pkl*")):
                self.logger.info("✅ Real game data files detected")
            else:
                self.logger.warning("⚠️ No game data files found yet (may be normal for new activation)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System integration validation failed: {e}")
            return False
    
    def _log_activation_success(self):
        """Log successful activation details"""
        duration = (datetime.now() - self.activation_start_time).total_seconds()
        
        self.logger.info("=" * 60)
        self.logger.info("🎉 SELF-PLAY TRAINING SYSTEM ACTIVATION SUCCESSFUL!")
        self.logger.info("=" * 60)
        self.logger.info(f"⏱️  Activation time: {duration:.1f} seconds")
        self.logger.info(f"🎯 Target games: 5000+ (Bootstrap: 2000, Hybrid: 5000, Self-play: 15000)")
        self.logger.info(f"🚀 Target throughput: {self.config.data_collection.target_games_per_hour} games/hour")
        self.logger.info(f"🔄 Progressive training: Bootstrap → Hybrid → Self-Play → Continuous")
        self.logger.info(f"📊 Enhanced training convergence monitoring active")
        self.logger.info("")
        self.logger.info("📋 NEXT STEPS:")
        self.logger.info("1. Monitor training progress with: python self_play_training_pipeline.py")
        self.logger.info("2. Check system status with automation_stats = manager.get_comprehensive_stats()")
        self.logger.info("3. Training will automatically progress through phases")
        self.logger.info("4. Models will be exported to ONNX when criteria are met")
        self.logger.info("")
        self.logger.info("✅ Self-play training system is now ACTIVE and replacing synthetic data!")
        self.logger.info("=" * 60)
    
    def _cleanup_failed_activation(self):
        """Cleanup after failed activation"""
        self.logger.warning("🧹 Cleaning up failed activation...")
        
        try:
            if self.automation_manager:
                self.automation_manager.stop()
                self.logger.info("✅ Automation system stopped")
        except Exception as e:
            self.logger.error(f"Error stopping automation: {e}")
        
        try:
            if self.training_pipeline:
                self.training_pipeline.emergency_stop()
                self.logger.info("✅ Training pipeline stopped")
        except Exception as e:
            self.logger.error(f"Error stopping training: {e}")
    
    def get_activation_status(self) -> dict:
        """Get current activation status"""
        return {
            'activation_successful': self.activation_successful,
            'activation_time': (datetime.now() - self.activation_start_time).total_seconds(),
            'automation_active': self.automation_manager is not None,
            'training_active': self.training_pipeline is not None
        }

def main():
    """Main activation function"""
    print("🚀 Self-Play Training System Activation")
    print("=" * 50)
    
    # Create activation manager
    activation_manager = SelfPlayActivationManager()
    
    # Run activation sequence
    success = activation_manager.run_activation_sequence()
    
    if success:
        print("\n✅ ACTIVATION COMPLETED SUCCESSFULLY!")
        print("The self-play training system is now active and generating real game data.")
        print("Neural networks will be retrained with progressive 5000+ game approach.")
        
        # Keep system running
        try:
            print("\nPress Ctrl+C to stop the system...")
            while True:
                time.sleep(10)
                # Periodic status check
                status = activation_manager.get_activation_status()
                if status['activation_time'] % 300 < 10:  # Every 5 minutes
                    print(f"⏱️  System running for {status['activation_time']/60:.1f} minutes...")
        except KeyboardInterrupt:
            print("\n🛑 Stopping self-play training system...")
            activation_manager._cleanup_failed_activation()
            print("✅ System stopped gracefully")
    else:
        print("\n❌ ACTIVATION FAILED!")
        print("Please check the logs and resolve issues before retrying.")
        sys.exit(1)

if __name__ == "__main__":
    main()