"""
Battlesnake Self-Play Automation System

This module orchestrates multiple Battlesnake servers and automates game execution
using the Battlesnake CLI. Handles server lifecycle, game scheduling, and process
management for high-throughput data collection.

Architecture compliance: 100+ games/hour, multi-server concurrency, robust error handling
"""

import os
import sys
import time
import subprocess
import threading
import queue
import logging
import signal
import psutil
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

from config.self_play_config import get_config, SystemConfig, ServerConfig, GameConfig

@dataclass
class ServerProcess:
    """Represents a running Battlesnake server process"""
    config: ServerConfig
    process: subprocess.Popen
    start_time: datetime
    status: str  # "starting", "ready", "busy", "error", "stopped"
    last_health_check: Optional[datetime] = None
    games_served: int = 0
    error_count: int = 0
    
    @property
    def uptime_seconds(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def is_healthy(self) -> bool:
        if self.process is None:
            return False
        try:
            return self.process.poll() is None and self.status in ["ready", "busy"]
        except:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'port': self.config.port,
            'status': self.status,
            'uptime_seconds': self.uptime_seconds,
            'games_served': self.games_served,
            'error_count': self.error_count,
            'memory_mb': self._get_memory_usage(),
            'cpu_percent': self._get_cpu_usage()
        }
    
    def _get_memory_usage(self) -> float:
        try:
            if self.process and self.process.pid:
                proc = psutil.Process(self.process.pid)
                return proc.memory_info().rss / 1024 / 1024  # MB
        except:
            pass
        return 0.0
    
    def _get_cpu_usage(self) -> float:
        try:
            if self.process and self.process.pid:
                proc = psutil.Process(self.process.pid)
                return proc.cpu_percent()
        except:
            pass
        return 0.0

@dataclass
class GameExecution:
    """Represents a single game execution"""
    game_id: str
    server_ports: List[int]
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "scheduled"  # scheduled, running, completed, failed, timeout
    game_length_turns: Optional[int] = None
    winner: Optional[str] = None
    error_message: Optional[str] = None
    log_file: Optional[str] = None
    data_extracted: bool = False
    
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ServerPool:
    """Manages a pool of Battlesnake server processes"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.servers: Dict[int, ServerProcess] = {}
        self.logger = logging.getLogger(__name__)
        self._shutdown_event = threading.Event()
        self._health_check_thread: Optional[threading.Thread] = None
        
        # Create workspace directory
        self.workspace_dir = Path.cwd()
        self.cargo_build_done = False
    
    def start_all_servers(self) -> bool:
        """Start all configured servers"""
        self.logger.info(f"Starting {len(self.config.servers)} Battlesnake servers...")
        
        # Ensure cargo build is done first
        if not self._ensure_cargo_build():
            return False
        
        success_count = 0
        for server_config in self.config.servers:
            if self._start_server(server_config):
                success_count += 1
            else:
                self.logger.error(f"Failed to start server on port {server_config.port}")
        
        if success_count == len(self.config.servers):
            self.logger.info("All servers started successfully")
            self._start_health_monitoring()
            return True
        else:
            self.logger.error(f"Only {success_count}/{len(self.config.servers)} servers started")
            return False
    
    def _ensure_cargo_build(self) -> bool:
        """Ensure the Rust project is built before starting servers"""
        if self.cargo_build_done:
            return True
            
        self.logger.info("Building Rust project...")
        try:
            result = subprocess.run(
                ["cargo", "build", "--release"],
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes max
            )
            
            if result.returncode == 0:
                self.logger.info("Cargo build completed successfully")
                self.cargo_build_done = True
                return True
            else:
                self.logger.error(f"Cargo build failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Cargo build timed out")
            return False
        except Exception as e:
            self.logger.error(f"Cargo build error: {e}")
            return False
    
    def _start_server(self, server_config: ServerConfig) -> bool:
        """Start a single server process"""
        if server_config.port in self.servers:
            self.logger.warning(f"Server on port {server_config.port} already exists")
            return True
        
        # Prepare environment with PORT set (not ROCKET_PORT)
        env = os.environ.copy()
        env.update(server_config.get_env_vars())
        
        self.logger.info(f"Starting server {server_config.name} on port {server_config.port}")
        
        try:
            # Start the Rust server process
            process = subprocess.Popen(
                ["cargo", "run", "--release"],
                cwd=self.workspace_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            server_process = ServerProcess(
                config=server_config,
                process=process,
                start_time=datetime.now(),
                status="starting"
            )
            
            self.servers[server_config.port] = server_process
            
            # Wait for server to be ready
            if self._wait_for_server_ready(server_process):
                server_process.status = "ready"
                self.logger.info(f"Server on port {server_config.port} is ready")
                return True
            else:
                self.logger.error(f"Server on port {server_config.port} failed to become ready")
                self._stop_server(server_config.port)
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start server on port {server_config.port}: {e}")
            return False
    
    def _wait_for_server_ready(self, server_process: ServerProcess, timeout_seconds: int = 30) -> bool:
        """Wait for server to be ready to accept requests"""
        import requests
        
        start_time = time.time()
        url = f"http://localhost:{server_process.config.port}/"
        
        while time.time() - start_time < timeout_seconds:
            if server_process.process.poll() is not None:
                # Process died
                return False
            
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    return True
            except:
                pass
            
            time.sleep(1)
        
        return False
    
    def _start_health_monitoring(self):
        """Start health monitoring thread"""
        if self._health_check_thread is None:
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True
            )
            self._health_check_thread.start()
            self.logger.info("Health monitoring started")
    
    def _health_check_loop(self):
        """Continuous health checking for all servers"""
        while not self._shutdown_event.is_set():
            try:
                for port, server in list(self.servers.items()):
                    self._check_server_health(server)
                
                time.sleep(self.config.health_check_interval_s)
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                time.sleep(5)
    
    def _check_server_health(self, server: ServerProcess):
        """Check health of a single server"""
        if not server.is_healthy:
            self.logger.warning(f"Server on port {server.config.port} is unhealthy")
            server.status = "error"
            server.error_count += 1
            
            # Restart if too many errors
            if server.error_count >= 3:
                self.logger.info(f"Restarting server on port {server.config.port}")
                self._restart_server(server.config.port)
        else:
            server.last_health_check = datetime.now()
            if server.status == "error":
                server.status = "ready"
    
    def _restart_server(self, port: int) -> bool:
        """Restart a specific server"""
        if port in self.servers:
            server_config = self.servers[port].config
            self._stop_server(port)
            time.sleep(2)  # Brief pause
            return self._start_server(server_config)
        return False
    
    def _stop_server(self, port: int):
        """Stop a specific server"""
        if port in self.servers:
            server = self.servers[port]
            server.status = "stopped"
            
            if server.process:
                try:
                    # Graceful shutdown first
                    server.process.terminate()
                    try:
                        server.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # Force kill if needed
                        server.process.kill()
                        server.process.wait()
                except Exception as e:
                    self.logger.error(f"Error stopping server on port {port}: {e}")
            
            del self.servers[port]
            self.logger.info(f"Server on port {port} stopped")
    
    def stop_all_servers(self):
        """Stop all servers and cleanup"""
        self.logger.info("Stopping all servers...")
        
        self._shutdown_event.set()
        
        # Stop health monitoring
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5)
        
        # Stop all servers
        for port in list(self.servers.keys()):
            self._stop_server(port)
        
        self.logger.info("All servers stopped")
    
    def get_available_server(self) -> Optional[ServerProcess]:
        """Get an available server for game execution"""
        for server in self.servers.values():
            if server.status == "ready":
                return server
        return None
    
    def mark_server_busy(self, port: int):
        """Mark server as busy during game execution"""
        if port in self.servers:
            self.servers[port].status = "busy"
    
    def mark_server_ready(self, port: int):
        """Mark server as ready after game completion"""
        if port in self.servers:
            self.servers[port].status = "ready"
            self.servers[port].games_served += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats for all servers"""
        stats = {
            'total_servers': len(self.servers),
            'healthy_servers': sum(1 for s in self.servers.values() if s.is_healthy),
            'total_games_served': sum(s.games_served for s in self.servers.values()),
            'servers': [s.get_stats() for s in self.servers.values()]
        }
        return stats

class GameOrchestrator:
    """Orchestrates automated game execution using Battlesnake CLI"""
    
    def __init__(self, config: SystemConfig, server_pool: ServerPool):
        self.config = config
        self.server_pool = server_pool
        self.logger = logging.getLogger(__name__)
        
        self.game_queue = queue.Queue()
        self.completed_games: List[GameExecution] = []
        self._executor = ThreadPoolExecutor(max_workers=config.data_collection.concurrent_servers)
        self._shutdown_event = threading.Event()
        
        # Ensure battlesnake CLI is available
        self._verify_battlesnake_cli()
    
    def _verify_battlesnake_cli(self):
        """Verify that battlesnake CLI is installed and available"""
        try:
            result = subprocess.run(
                ["battlesnake", "version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self.logger.info(f"Battlesnake CLI available: {result.stdout.strip()}")
            else:
                raise Exception("Battlesnake CLI not responding correctly")
        except Exception as e:
            self.logger.error(f"Battlesnake CLI not available: {e}")
            raise RuntimeError("Battlesnake CLI is required but not available")
    
    def schedule_games(self, num_games: int):
        """Schedule a batch of games for execution"""
        self.logger.info(f"Scheduling {num_games} games")
        
        for i in range(num_games):
            game_id = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:04d}"
            
            # For now, use single-server solo games
            # In future versions, we can add multi-server multi-player games
            server = self.server_pool.get_available_server()
            if server:
                game = GameExecution(
                    game_id=game_id,
                    server_ports=[server.config.port],
                    start_time=datetime.now(),
                    status="scheduled"
                )
                self.game_queue.put(game)
            else:
                self.logger.warning("No available server for game scheduling")
                break
    
    def start_game_execution(self) -> threading.Thread:
        """Start the game execution loop in a separate thread"""
        execution_thread = threading.Thread(
            target=self._game_execution_loop,
            daemon=True
        )
        execution_thread.start()
        self.logger.info("Game execution started")
        return execution_thread
    
    def _game_execution_loop(self):
        """Main game execution loop"""
        futures = []
        
        while not self._shutdown_event.is_set():
            # Submit new games to thread pool
            while len(futures) < self.config.data_collection.concurrent_servers:
                try:
                    game = self.game_queue.get_nowait()
                    future = self._executor.submit(self._execute_game, game)
                    futures.append(future)
                except queue.Empty:
                    break
            
            # Check for completed games
            completed_futures = []
            for future in futures:
                if future.done():
                    try:
                        completed_game = future.result()
                        self.completed_games.append(completed_game)
                        self.logger.info(f"Game {completed_game.game_id} completed: {completed_game.status}")
                    except Exception as e:
                        self.logger.error(f"Game execution error: {e}")
                    completed_futures.append(future)
            
            # Remove completed futures
            for future in completed_futures:
                futures.remove(future)
            
            time.sleep(0.5)
    
    def _execute_game(self, game: GameExecution) -> GameExecution:
        """Execute a single game"""
        game.status = "running"
        game.start_time = datetime.now()
        
        # Mark server as busy
        for port in game.server_ports:
            self.server_pool.mark_server_busy(port)
        
        try:
            # Create log file for this game
            log_dir = Path("logs/games")
            log_dir.mkdir(parents=True, exist_ok=True)
            game.log_file = str(log_dir / f"{game.game_id}.log")
            
            # Build battlesnake CLI command
            cmd = self._build_game_command(game)
            
            self.logger.info(f"Starting game {game.game_id}: {' '.join(cmd)}")
            
            # Execute the game
            with open(game.log_file, 'w') as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                # Wait for game completion with timeout
                try:
                    return_code = process.wait(timeout=300)  # 5 minute timeout
                    
                    if return_code == 0:
                        game.status = "completed"
                        self._extract_game_results(game)
                    else:
                        game.status = "failed"
                        game.error_message = f"Game process returned {return_code}"
                        
                except subprocess.TimeoutExpired:
                    process.kill()
                    game.status = "timeout"
                    game.error_message = "Game execution timed out"
                    
        except Exception as e:
            game.status = "failed"
            game.error_message = str(e)
            self.logger.error(f"Game {game.game_id} execution error: {e}")
        
        finally:
            game.end_time = datetime.now()
            # Mark servers as ready
            for port in game.server_ports:
                self.server_pool.mark_server_ready(port)
        
        return game
    
    def _build_game_command(self, game: GameExecution) -> List[str]:
        """Build the battlesnake CLI command for a game"""
        cmd = ["battlesnake", "play"]
        
        # Add game configuration arguments
        cmd.extend(self.config.game.get_cli_args())
        
        # Add snake URLs
        for port in game.server_ports:
            cmd.extend(["--name", f"snake-{port}", "--url", f"http://localhost:{port}"])
        
        # Add output options for data extraction
        cmd.extend(["--output", "json", "--quiet"])
        
        return cmd
    
    def _extract_game_results(self, game: GameExecution):
        """Extract game results from log file"""
        if not game.log_file or not Path(game.log_file).exists():
            return
        
        try:
            with open(game.log_file, 'r') as f:
                content = f.read()
                
            # Basic extraction - in future versions this will be more sophisticated
            lines = content.split('\n')
            for line in lines:
                if "turns" in line.lower():
                    # Try to extract turn count
                    try:
                        import re
                        match = re.search(r'turn[s]?\s*[:=]?\s*(\d+)', line.lower())
                        if match:
                            game.game_length_turns = int(match.group(1))
                    except:
                        pass
                
                if "winner" in line.lower() or "won" in line.lower():
                    # Try to extract winner information
                    game.winner = "extracted"  # Placeholder
                    
        except Exception as e:
            self.logger.error(f"Error extracting game results for {game.game_id}: {e}")
    
    def stop(self):
        """Stop game execution"""
        self.logger.info("Stopping game orchestrator...")
        self._shutdown_event.set()
        self._executor.shutdown(wait=True)
        self.logger.info("Game orchestrator stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        total_games = len(self.completed_games)
        if total_games == 0:
            return {'total_games': 0, 'status_breakdown': {}}
        
        status_counts = {}
        total_turns = 0
        valid_turn_games = 0
        
        for game in self.completed_games:
            status_counts[game.status] = status_counts.get(game.status, 0) + 1
            if game.game_length_turns:
                total_turns += game.game_length_turns
                valid_turn_games += 1
        
        avg_turns = total_turns / valid_turn_games if valid_turn_games > 0 else 0
        
        return {
            'total_games': total_games,
            'status_breakdown': status_counts,
            'average_game_length_turns': avg_turns,
            'games_in_queue': self.game_queue.qsize()
        }

class SelfPlayAutomationManager:
    """Main automation manager that coordinates servers and games"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        self.server_pool = ServerPool(self.config)
        self.orchestrator = GameOrchestrator(self.config, self.server_pool)
        
        self.start_time = datetime.now()
        self.target_games_per_hour = self.config.data_collection.target_games_per_hour
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._running = False
        self._execution_thread: Optional[threading.Thread] = None
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def start(self) -> bool:
        """Start the automation system"""
        self.logger.info("Starting self-play automation system...")
        
        # Start server pool
        if not self.server_pool.start_all_servers():
            self.logger.error("Failed to start server pool")
            return False
        
        # Start game orchestrator
        self._execution_thread = self.orchestrator.start_game_execution()
        
        self._running = True
        self.logger.info("Self-play automation system started successfully")
        return True
    
    def run_batch(self, num_games: int) -> bool:
        """Run a batch of games"""
        if not self._running:
            self.logger.error("System not started")
            return False
        
        self.logger.info(f"Running batch of {num_games} games...")
        self.orchestrator.schedule_games(num_games)
        
        # Wait for all games to complete
        start_time = time.time()
        timeout_seconds = (num_games / self.target_games_per_hour) * 3600 * 2  # 2x expected time
        
        while self.orchestrator.game_queue.qsize() > 0 or self._games_in_progress():
            if time.time() - start_time > timeout_seconds:
                self.logger.warning("Batch execution timed out")
                return False
            time.sleep(5)
        
        self.logger.info(f"Batch of {num_games} games completed")
        return True
    
    def _games_in_progress(self) -> bool:
        """Check if any games are currently in progress"""
        for server in self.server_pool.servers.values():
            if server.status == "busy":
                return True
        return False
    
    def run_continuous(self, duration_hours: Optional[float] = None):
        """Run continuous game generation"""
        if not self._running:
            self.logger.error("System not started")
            return
        
        self.logger.info(f"Starting continuous operation{f' for {duration_hours} hours' if duration_hours else ''}")
        
        start_time = time.time()
        last_schedule_time = 0
        games_per_batch = self.config.data_collection.max_games_per_batch
        batch_interval = (games_per_batch / self.target_games_per_hour) * 3600  # seconds between batches
        
        try:
            while True:
                current_time = time.time()
                
                # Check if we should stop based on duration
                if duration_hours and (current_time - start_time) / 3600 >= duration_hours:
                    break
                
                # Schedule new batch if needed
                if current_time - last_schedule_time >= batch_interval:
                    if self.orchestrator.game_queue.qsize() < games_per_batch:
                        self.orchestrator.schedule_games(games_per_batch)
                        last_schedule_time = current_time
                
                # Log stats periodically
                if int(current_time) % 300 == 0:  # Every 5 minutes
                    self._log_system_stats()
                
                time.sleep(10)  # Check every 10 seconds
                
        except KeyboardInterrupt:
            self.logger.info("Continuous operation interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in continuous operation: {e}")
    
    def stop(self):
        """Stop the automation system"""
        if not self._running:
            return
        
        self.logger.info("Stopping automation system...")
        self._running = False
        
        # Stop orchestrator
        self.orchestrator.stop()
        
        # Stop server pool
        self.server_pool.stop_all_servers()
        
        self.logger.info("Automation system stopped")
    
    def _log_system_stats(self):
        """Log comprehensive system statistics"""
        server_stats = self.server_pool.get_stats()
        orchestrator_stats = self.orchestrator.get_stats()
        
        uptime = (datetime.now() - self.start_time).total_seconds() / 3600  # hours
        games_per_hour = orchestrator_stats['total_games'] / uptime if uptime > 0 else 0
        
        self.logger.info(f"=== System Stats (Uptime: {uptime:.1f}h) ===")
        self.logger.info(f"Servers: {server_stats['healthy_servers']}/{server_stats['total_servers']} healthy")
        self.logger.info(f"Games completed: {orchestrator_stats['total_games']}")
        self.logger.info(f"Current rate: {games_per_hour:.1f} games/hour (target: {self.target_games_per_hour})")
        self.logger.info(f"Games in queue: {orchestrator_stats['games_in_queue']}")
        
        if 'status_breakdown' in orchestrator_stats:
            status_str = ", ".join(f"{status}: {count}" for status, count in orchestrator_stats['status_breakdown'].items())
            self.logger.info(f"Game status: {status_str}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        server_stats = self.server_pool.get_stats()
        orchestrator_stats = self.orchestrator.get_stats()
        
        uptime = (datetime.now() - self.start_time).total_seconds() / 3600
        games_per_hour = orchestrator_stats['total_games'] / uptime if uptime > 0 else 0
        
        return {
            'system': {
                'uptime_hours': uptime,
                'target_games_per_hour': self.target_games_per_hour,
                'actual_games_per_hour': games_per_hour,
                'efficiency_percent': (games_per_hour / self.target_games_per_hour) * 100 if self.target_games_per_hour > 0 else 0
            },
            'servers': server_stats,
            'orchestrator': orchestrator_stats,
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Battlesnake Self-Play Automation")
    parser.add_argument("--batch", type=int, help="Run a batch of N games")
    parser.add_argument("--continuous", type=float, help="Run continuously for N hours")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start automation manager
    manager = SelfPlayAutomationManager(args.config)
    
    try:
        if not manager.start():
            sys.exit(1)
        
        if args.batch:
            success = manager.run_batch(args.batch)
            sys.exit(0 if success else 1)
        elif args.continuous is not None:
            manager.run_continuous(args.continuous if args.continuous > 0 else None)
        else:
            # Default: run continuous
            print("Starting continuous operation. Press Ctrl+C to stop.")
            manager.run_continuous()
    
    finally:
        manager.stop()