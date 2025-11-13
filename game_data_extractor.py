"""
Game Data Extraction System

This module extracts training data from Battlesnake CLI games and Rust server logs.
Parses log output in real-time to capture game states, heuristic scores, and outcomes
for neural network training data generation.

Architecture compliance: Extract sophisticated heuristic scores (Safety: 8-50pts, Territory: 3-9pts, etc.)
"""

import re
import json
import logging
import threading
import queue
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, NamedTuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import subprocess
import requests

from config.self_play_config import get_config

@dataclass
class Coordinate:
    """Represents a coordinate on the Battlesnake board"""
    x: int
    y: int
    
    def to_dict(self) -> Dict[str, int]:
        return {'x': self.x, 'y': self.y}
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'Coordinate':
        return cls(x=data['x'], y=data['y'])

@dataclass
class SnakeState:
    """Represents the state of a snake at a specific turn"""
    id: str
    name: str
    health: int
    body: List[Coordinate]
    head: Coordinate
    length: int
    shout: str = ""
    
    @property
    def neck(self) -> Optional[Coordinate]:
        return self.body[1] if len(self.body) >= 2 else None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'health': self.health,
            'body': [coord.to_dict() for coord in self.body],
            'head': self.head.to_dict(),
            'length': self.length,
            'shout': self.shout
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SnakeState':
        return cls(
            id=data['id'],
            name=data['name'],
            health=data['health'],
            body=[Coordinate.from_dict(coord) for coord in data['body']],
            head=Coordinate.from_dict(data['head']),
            length=data['length'],
            shout=data.get('shout', '')
        )

@dataclass
class BoardState:
    """Represents the complete board state at a specific turn"""
    width: int
    height: int
    food: List[Coordinate]
    hazards: List[Coordinate]
    snakes: List[SnakeState]
    turn: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'width': self.width,
            'height': self.height,
            'food': [coord.to_dict() for coord in self.food],
            'hazards': [coord.to_dict() for coord in self.hazards],
            'snakes': [snake.to_dict() for snake in self.snakes],
            'turn': self.turn
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoardState':
        return cls(
            width=data['width'],
            height=data['height'],
            food=[Coordinate.from_dict(coord) for coord in data['food']],
            hazards=[Coordinate.from_dict(coord) for coord in data.get('hazards', [])],
            snakes=[SnakeState.from_dict(snake) for snake in data['snakes']],
            turn=data['turn']
        )

@dataclass
class HeuristicScores:
    """Heuristic component scores from the Rust decision system"""
    safety: float = 0.0          # 8-50 points from collision detection
    territory: float = 0.0       # 3-9 points from Voronoi BFS
    opponent: float = 0.0        # 0-6 points from opponent modeling  
    food: float = 0.0           # 0-11 points from A* pathfinding
    exploration: float = 0.0     # ~30 points from exploration
    
    # Additional scores that might be extracted
    cutting: float = 0.0         # Opponent cutting positions
    survival: float = 0.0        # Long-term survival assessment
    aggression: float = 0.0      # Aggressive move scoring
    
    @property
    def total_score(self) -> float:
        return (self.safety + self.territory + self.opponent + 
                self.food + self.exploration + self.cutting + 
                self.survival + self.aggression)
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'HeuristicScores':
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

@dataclass
class MoveDecision:
    """Represents a move decision with context and scoring"""
    snake_id: str
    turn: int
    move: str  # "up", "down", "left", "right"
    board_state: BoardState
    heuristic_scores: HeuristicScores
    decision_time_ms: float
    timestamp: datetime
    server_port: int
    
    # Move evaluation scores for all possible moves
    move_evaluations: Dict[str, float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['board_state'] = self.board_state.to_dict()
        data['heuristic_scores'] = self.heuristic_scores.to_dict()
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MoveDecision':
        return cls(
            snake_id=data['snake_id'],
            turn=data['turn'],
            move=data['move'],
            board_state=BoardState.from_dict(data['board_state']),
            heuristic_scores=HeuristicScores.from_dict(data['heuristic_scores']),
            decision_time_ms=data['decision_time_ms'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            server_port=data['server_port'],
            move_evaluations=data.get('move_evaluations')
        )

@dataclass
class GameData:
    """Complete data for a single game"""
    game_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    board_width: int = 11
    board_height: int = 11
    initial_snakes: List[str] = None
    moves: List[MoveDecision] = None
    final_state: Optional[BoardState] = None
    winner: Optional[str] = None
    game_length_turns: int = 0
    termination_reason: str = ""
    
    def __post_init__(self):
        if self.moves is None:
            self.moves = []
        if self.initial_snakes is None:
            self.initial_snakes = []
    
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        data['moves'] = [move.to_dict() for move in self.moves]
        if self.final_state:
            data['final_state'] = self.final_state.to_dict()
        return data

class LogPatterns:
    """Regular expressions for parsing various log formats"""
    
    # Rust server log patterns
    MOVE_DECISION = re.compile(r'MOVE (\d+):\s*(.+)')
    HEURISTIC_SAFETY = re.compile(r'Safety.*?score[:\s]*([+-]?\d+\.?\d*)', re.IGNORECASE)
    HEURISTIC_TERRITORY = re.compile(r'Territory.*?score[:\s]*([+-]?\d+\.?\d*)', re.IGNORECASE) 
    HEURISTIC_FOOD = re.compile(r'Food.*?score[:\s]*([+-]?\d+\.?\d*)', re.IGNORECASE)
    HEURISTIC_OPPONENT = re.compile(r'Opponent.*?score[:\s]*([+-]?\d+\.?\d*)', re.IGNORECASE)
    HEURISTIC_EXPLORATION = re.compile(r'Exploration.*?score[:\s]*([+-]?\d+\.?\d*)', re.IGNORECASE)
    
    # Enhanced patterns for more detailed extraction
    REQUEST_ID = re.compile(r'Request\s+ID[:\s]*([a-f0-9-]+)', re.IGNORECASE)
    DECISION_TIME = re.compile(r'Decision\s+time[:\s]*(\d+\.?\d*)\s*ms', re.IGNORECASE)
    TOTAL_SCORE = re.compile(r'Total.*?score[:\s]*([+-]?\d+\.?\d*)', re.IGNORECASE)
    
    # Battlesnake CLI output patterns  
    GAME_START = re.compile(r'Game\s+started.*?ID[:\s]*([a-f0-9-]+)', re.IGNORECASE)
    GAME_END = re.compile(r'Game\s+(?:ended|completed|finished)', re.IGNORECASE)
    TURN_START = re.compile(r'Turn\s+(\d+)', re.IGNORECASE)
    WINNER = re.compile(r'Winner[:\s]*(.+)', re.IGNORECASE)
    GAME_RESULT = re.compile(r'Result[:\s]*(.+)', re.IGNORECASE)
    
    # JSON data patterns
    JSON_BOARD_STATE = re.compile(r'\{.*?"board".*?\}', re.DOTALL)
    JSON_GAME_STATE = re.compile(r'\{.*?"game".*?\}', re.DOTALL)

class ServerLogParser:
    """Parses Rust server logs to extract heuristic scores"""
    
    def __init__(self, server_port: int):
        self.server_port = server_port
        self.logger = logging.getLogger(f"{__name__}.server_{server_port}")
        self.current_request_context = {}
        
    def parse_log_line(self, line: str, timestamp: datetime = None) -> Optional[Dict[str, Any]]:
        """Parse a single log line and extract relevant data"""
        if timestamp is None:
            timestamp = datetime.now()
        
        line = line.strip()
        if not line:
            return None
        
        extracted_data = {}
        
        # Extract request ID for context correlation
        request_match = LogPatterns.REQUEST_ID.search(line)
        if request_match:
            extracted_data['request_id'] = request_match.group(1)
        
        # Extract move decision
        move_match = LogPatterns.MOVE_DECISION.search(line)
        if move_match:
            extracted_data['type'] = 'move_decision'
            extracted_data['turn'] = int(move_match.group(1))
            extracted_data['move'] = move_match.group(2).strip()
        
        # Extract heuristic scores
        heuristic_scores = {}
        
        safety_match = LogPatterns.HEURISTIC_SAFETY.search(line)
        if safety_match:
            heuristic_scores['safety'] = float(safety_match.group(1))
        
        territory_match = LogPatterns.HEURISTIC_TERRITORY.search(line)
        if territory_match:
            heuristic_scores['territory'] = float(territory_match.group(1))
        
        food_match = LogPatterns.HEURISTIC_FOOD.search(line)
        if food_match:
            heuristic_scores['food'] = float(food_match.group(1))
            
        opponent_match = LogPatterns.HEURISTIC_OPPONENT.search(line)
        if opponent_match:
            heuristic_scores['opponent'] = float(opponent_match.group(1))
            
        exploration_match = LogPatterns.HEURISTIC_EXPLORATION.search(line)
        if exploration_match:
            heuristic_scores['exploration'] = float(exploration_match.group(1))
        
        if heuristic_scores:
            extracted_data['heuristic_scores'] = heuristic_scores
        
        # Extract decision time
        time_match = LogPatterns.DECISION_TIME.search(line)
        if time_match:
            extracted_data['decision_time_ms'] = float(time_match.group(1))
        
        # Extract total score
        total_match = LogPatterns.TOTAL_SCORE.search(line)
        if total_match:
            extracted_data['total_score'] = float(total_match.group(1))
        
        if extracted_data:
            extracted_data['server_port'] = self.server_port
            extracted_data['timestamp'] = timestamp
            extracted_data['raw_line'] = line
            return extracted_data
        
        return None

class GameLogParser:
    """Parses Battlesnake CLI game output to extract game states"""
    
    def __init__(self, game_id: str):
        self.game_id = game_id
        self.logger = logging.getLogger(f"{__name__}.game_{game_id}")
        
    def parse_game_log(self, log_content: str) -> GameData:
        """Parse complete game log content"""
        game_data = GameData(
            game_id=self.game_id,
            start_time=datetime.now()
        )
        
        lines = log_content.split('\n')
        current_turn = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to extract JSON data first
            json_data = self._extract_json_from_line(line)
            if json_data:
                self._process_json_data(json_data, game_data)
                continue
            
            # Parse text-based patterns
            turn_match = LogPatterns.TURN_START.search(line)
            if turn_match:
                current_turn = int(turn_match.group(1))
                continue
            
            winner_match = LogPatterns.WINNER.search(line)
            if winner_match:
                game_data.winner = winner_match.group(1).strip()
                continue
            
            if LogPatterns.GAME_END.search(line):
                game_data.end_time = datetime.now()
                game_data.game_length_turns = current_turn
                continue
        
        return game_data
    
    def _extract_json_from_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Extract JSON data from a log line"""
        # Look for JSON patterns
        json_match = LogPatterns.JSON_BOARD_STATE.search(line)
        if not json_match:
            json_match = LogPatterns.JSON_GAME_STATE.search(line)
        
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try to parse the entire line as JSON
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None
    
    def _process_json_data(self, json_data: Dict[str, Any], game_data: GameData):
        """Process JSON data and update game data"""
        # Check if this is a board state
        if 'board' in json_data:
            board_data = json_data['board']
            try:
                board_state = BoardState.from_dict(board_data)
                # This could be part of a move decision - store for correlation
            except Exception as e:
                self.logger.warning(f"Failed to parse board state: {e}")
        
        # Check if this is game metadata
        if 'game' in json_data:
            game_info = json_data['game']
            if 'id' in game_info:
                game_data.game_id = game_info['id']

class RealTimeGameExtractor:
    """Real-time extractor that monitors active games and server logs"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.active_games: Dict[str, GameData] = {}
        self.server_parsers: Dict[int, ServerLogParser] = {}
        self.completed_games: List[GameData] = []
        
        # Real-time processing
        self.log_queue = queue.Queue()
        self.processing_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Server log monitoring
        self.server_processes: Dict[int, subprocess.Popen] = {}
        
    def start_monitoring(self):
        """Start real-time log monitoring"""
        self.logger.info("Starting real-time game data extraction")
        
        # Initialize server parsers
        for server_config in self.config.servers:
            self.server_parsers[server_config.port] = ServerLogParser(server_config.port)
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
        self.logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.logger.info("Stopping real-time monitoring...")
        self.shutdown_event.set()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        self.logger.info("Real-time monitoring stopped")
    
    def monitor_game(self, game_id: str, log_file: str):
        """Monitor a specific game's log file"""
        if game_id in self.active_games:
            self.logger.warning(f"Game {game_id} is already being monitored")
            return
        
        game_data = GameData(
            game_id=game_id,
            start_time=datetime.now()
        )
        self.active_games[game_id] = game_data
        
        # Start monitoring the log file
        threading.Thread(
            target=self._monitor_log_file,
            args=(game_id, log_file),
            daemon=True
        ).start()
        
        self.logger.info(f"Started monitoring game {game_id}")
    
    def _monitor_log_file(self, game_id: str, log_file: str):
        """Monitor a log file for a specific game"""
        log_path = Path(log_file)
        
        # Wait for log file to be created
        max_wait_time = 30  # seconds
        wait_time = 0
        while not log_path.exists() and wait_time < max_wait_time:
            time.sleep(0.5)
            wait_time += 0.5
        
        if not log_path.exists():
            self.logger.error(f"Log file {log_file} not found for game {game_id}")
            return
        
        self.logger.info(f"Monitoring log file: {log_file}")
        
        # Tail the log file
        try:
            with open(log_path, 'r') as f:
                # Go to end of file initially
                f.seek(0, 2)
                
                while not self.shutdown_event.is_set():
                    line = f.readline()
                    if line:
                        self.log_queue.put({
                            'game_id': game_id,
                            'line': line.strip(),
                            'timestamp': datetime.now()
                        })
                    else:
                        time.sleep(0.1)  # No new data, wait briefly
        
        except Exception as e:
            self.logger.error(f"Error monitoring log file {log_file}: {e}")
        
        finally:
            # Mark game as completed when log monitoring ends
            if game_id in self.active_games:
                self._finalize_game(game_id)
    
    def _processing_loop(self):
        """Main processing loop for log data"""
        while not self.shutdown_event.is_set():
            try:
                # Process log entries with timeout
                try:
                    log_entry = self.log_queue.get(timeout=1.0)
                    self._process_log_entry(log_entry)
                except queue.Empty:
                    continue
            
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(1)
    
    def _process_log_entry(self, log_entry: Dict[str, Any]):
        """Process a single log entry"""
        game_id = log_entry['game_id']
        line = log_entry['line']
        timestamp = log_entry['timestamp']
        
        if game_id not in self.active_games:
            return
        
        game_data = self.active_games[game_id]
        
        # Parse the line using the game log parser
        game_parser = GameLogParser(game_id)
        
        # Try to extract heuristic data by correlating with server logs
        # This is a simplified approach - in practice, we'd need more sophisticated correlation
        extracted_data = self._correlate_with_server_logs(line, timestamp)
        
        if extracted_data:
            self._update_game_data(game_data, extracted_data)
    
    def _correlate_with_server_logs(self, line: str, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Correlate game log line with server logs to extract heuristic scores"""
        # This is a placeholder for more sophisticated correlation logic
        # In a full implementation, this would:
        # 1. Parse the game log line for move decisions
        # 2. Query recent server log entries around the same timestamp
        # 3. Extract heuristic scores from matching server entries
        # 4. Return combined data
        
        # For now, return basic extracted data
        return None
    
    def _update_game_data(self, game_data: GameData, extracted_data: Dict[str, Any]):
        """Update game data with extracted information"""
        if 'type' in extracted_data and extracted_data['type'] == 'move_decision':
            # Create a move decision object
            move_decision = MoveDecision(
                snake_id=extracted_data.get('snake_id', 'unknown'),
                turn=extracted_data.get('turn', 0),
                move=extracted_data.get('move', 'up'),
                board_state=None,  # Would be populated from correlated data
                heuristic_scores=HeuristicScores.from_dict(
                    extracted_data.get('heuristic_scores', {})
                ),
                decision_time_ms=extracted_data.get('decision_time_ms', 0),
                timestamp=extracted_data.get('timestamp', datetime.now()),
                server_port=extracted_data.get('server_port', 8000)
            )
            
            game_data.moves.append(move_decision)
    
    def _finalize_game(self, game_id: str):
        """Finalize a completed game"""
        if game_id in self.active_games:
            game_data = self.active_games[game_id]
            game_data.end_time = datetime.now()
            
            self.completed_games.append(game_data)
            del self.active_games[game_id]
            
            self.logger.info(f"Finalized game {game_id}: {len(game_data.moves)} moves extracted")
    
    def get_completed_games(self) -> List[GameData]:
        """Get all completed games"""
        return self.completed_games.copy()
    
    def get_active_games(self) -> Dict[str, GameData]:
        """Get currently active games"""
        return self.active_games.copy()
    
    def extract_game_from_file(self, game_id: str, log_file: str) -> Optional[GameData]:
        """Extract game data from a completed log file"""
        log_path = Path(log_file)
        if not log_path.exists():
            self.logger.error(f"Log file not found: {log_file}")
            return None
        
        try:
            with open(log_path, 'r') as f:
                content = f.read()
            
            parser = GameLogParser(game_id)
            game_data = parser.parse_game_log(content)
            
            self.logger.info(f"Extracted game {game_id}: {len(game_data.moves)} moves")
            return game_data
            
        except Exception as e:
            self.logger.error(f"Error extracting game from file {log_file}: {e}")
            return None
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get comprehensive extraction statistics"""
        total_moves = sum(len(game.moves) for game in self.completed_games)
        total_games = len(self.completed_games)
        active_games = len(self.active_games)
        
        # Calculate average heuristic scores if available
        all_heuristic_scores = []
        for game in self.completed_games:
            for move in game.moves:
                all_heuristic_scores.append(move.heuristic_scores)
        
        avg_scores = {}
        if all_heuristic_scores:
            score_fields = ['safety', 'territory', 'opponent', 'food', 'exploration']
            for field in score_fields:
                values = [getattr(scores, field) for scores in all_heuristic_scores]
                if values:
                    avg_scores[field] = sum(values) / len(values)
        
        return {
            'completed_games': total_games,
            'active_games': active_games,
            'total_moves_extracted': total_moves,
            'average_moves_per_game': total_moves / total_games if total_games > 0 else 0,
            'average_heuristic_scores': avg_scores,
            'extraction_timestamp': datetime.now().isoformat()
        }

# Enhanced server log monitoring for direct heuristic extraction
class DirectServerMonitor:
    """Direct monitoring of server stdout/stderr for heuristic extraction"""
    
    def __init__(self, server_port: int, process: subprocess.Popen):
        self.server_port = server_port
        self.process = process
        self.logger = logging.getLogger(f"{__name__}.direct_monitor_{server_port}")
        self.parser = ServerLogParser(server_port)
        
        self.extracted_data = queue.Queue()
        self.monitoring_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
    
    def start_monitoring(self):
        """Start monitoring server stdout/stderr"""
        self.monitoring_thread = threading.Thread(
            target=self._monitor_output,
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info(f"Started direct monitoring for server on port {self.server_port}")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.shutdown_event.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _monitor_output(self):
        """Monitor server output for heuristic data"""
        while not self.shutdown_event.is_set() and self.process.poll() is None:
            try:
                # Monitor stdout
                if self.process.stdout:
                    line = self.process.stdout.readline()
                    if line:
                        self._process_output_line(line.strip(), 'stdout')
                
                # Monitor stderr  
                if self.process.stderr:
                    line = self.process.stderr.readline()
                    if line:
                        self._process_output_line(line.strip(), 'stderr')
                        
            except Exception as e:
                self.logger.error(f"Error monitoring server output: {e}")
                break
    
    def _process_output_line(self, line: str, source: str):
        """Process a single output line"""
        if not line:
            return
        
        extracted = self.parser.parse_log_line(line)
        if extracted:
            extracted['source'] = source
            self.extracted_data.put(extracted)
    
    def get_recent_data(self, max_items: int = 100) -> List[Dict[str, Any]]:
        """Get recent extracted data"""
        items = []
        try:
            while len(items) < max_items and not self.extracted_data.empty():
                items.append(self.extracted_data.get_nowait())
        except queue.Empty:
            pass
        return items

if __name__ == "__main__":
    # Testing and demonstration
    logging.basicConfig(level=logging.INFO)
    
    # Test log parsing
    sample_log_lines = [
        "MOVE 15: right",
        "Safety score: 42.5",
        "Territory score: 7.2", 
        "Food score: 3.1",
        "Opponent score: 1.8",
        "Exploration score: 28.7",
        "Decision time: 45.3 ms",
        "Total score: 83.1"
    ]
    
    parser = ServerLogParser(8000)
    
    print("=== Sample Log Parsing ===")
    for line in sample_log_lines:
        result = parser.parse_log_line(line)
        if result:
            print(f"Input: {line}")
            print(f"Extracted: {result}")
            print()
    
    # Test JSON parsing
    sample_board_json = {
        "width": 11,
        "height": 11,
        "food": [{"x": 5, "y": 5}, {"x": 8, "y": 2}],
        "hazards": [],
        "snakes": [{
            "id": "snake1",
            "name": "Test Snake",
            "health": 95,
            "body": [{"x": 3, "y": 3}, {"x": 3, "y": 4}],
            "head": {"x": 3, "y": 3},
            "length": 2,
            "shout": ""
        }],
        "turn": 15
    }
    
    print("=== JSON Parsing Test ===")
    try:
        board_state = BoardState.from_dict(sample_board_json)
        print(f"Parsed board state: {board_state.width}x{board_state.height}, turn {board_state.turn}")
        print(f"Snakes: {len(board_state.snakes)}, Food: {len(board_state.food)}")
    except Exception as e:
        print(f"Error parsing board state: {e}")
    
    print("\n=== Game Data Extraction Ready ===")
    print("Use RealTimeGameExtractor for live monitoring")
    print("Use extract_game_from_file() for post-processing")