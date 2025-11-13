"""
Training Data Pipeline

This module converts extracted game data into neural network training format with
12-channel board encoding, feature vectors, and target labels. Implements the
architecture-specified data format for supervised learning from heuristic decisions.

Architecture compliance: 12-channel board encoding, heuristic supervision, 100+ games/hour processing
"""

import numpy as np
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque, defaultdict
import concurrent.futures
from enum import IntEnum

from game_data_extractor import GameData, MoveDecision, BoardState, SnakeState, HeuristicScores, Coordinate
from config.self_play_config import get_config

class Move(IntEnum):
    """Move directions as integers for neural network targets"""
    UP = 0
    DOWN = 1
    LEFT = 2  
    RIGHT = 3
    
    @classmethod
    def from_string(cls, move_str: str) -> 'Move':
        """Convert move string to enum"""
        move_str = move_str.lower().strip()
        mapping = {
            'up': cls.UP,
            'down': cls.DOWN, 
            'left': cls.LEFT,
            'right': cls.RIGHT
        }
        return mapping.get(move_str, cls.UP)
    
    def to_string(self) -> str:
        """Convert enum to string"""
        return self.name.lower()

@dataclass
class TrainingSample:
    """Complete training sample for neural network"""
    
    # Input tensors
    board_state: np.ndarray          # [11, 11, 12] - 12-channel board encoding
    snake_features: np.ndarray       # [32] - Snake-specific features
    game_context: np.ndarray         # [16] - Game state context
    
    # Target outputs
    target_move: int                 # 0-3: chosen move direction
    position_value: float           # -50 to +50: position evaluation
    move_probabilities: np.ndarray  # [4] - move preference distribution
    win_probability: float          # 0-1: predicted game outcome
    
    # Heuristic supervision
    heuristic_scores: Dict[str, float]  # Component scores from Rust system
    
    # Metadata
    game_id: str
    turn: int
    snake_id: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'board_state': self.board_state.tolist(),
            'snake_features': self.snake_features.tolist(),
            'game_context': self.game_context.tolist(),
            'target_move': self.target_move,
            'position_value': self.position_value,
            'move_probabilities': self.move_probabilities.tolist(),
            'win_probability': self.win_probability,
            'heuristic_scores': self.heuristic_scores,
            'game_id': self.game_id,
            'turn': self.turn,
            'snake_id': self.snake_id,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingSample':
        """Create from dictionary"""
        return cls(
            board_state=np.array(data['board_state']),
            snake_features=np.array(data['snake_features']),
            game_context=np.array(data['game_context']),
            target_move=data['target_move'],
            position_value=data['position_value'],
            move_probabilities=np.array(data['move_probabilities']),
            win_probability=data['win_probability'],
            heuristic_scores=data['heuristic_scores'],
            game_id=data['game_id'],
            turn=data['turn'],
            snake_id=data['snake_id'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

class BoardEncoder:
    """Encodes Battlesnake board states into 12-channel tensors"""
    
    BOARD_SIZE = 11
    NUM_CHANNELS = 12
    
    # Channel indices
    CH_OUR_SNAKE = 0           # Our snake head/body
    CH_OPPONENT_1 = 1          # Opponent snake 1 
    CH_OPPONENT_2 = 2          # Opponent snake 2
    CH_OPPONENT_3 = 3          # Opponent snake 3
    CH_FOOD = 4                # Food locations
    CH_WALLS = 5               # Wall/boundary positions
    CH_OUR_TERRITORY = 6       # Our territory control
    CH_OPPONENT_TERRITORY = 7  # Opponent territories (combined)
    CH_DANGER_ZONES = 8        # Immediate collision risks
    CH_MOVEMENT_HISTORY = 9    # Recent position trail
    CH_STRATEGIC_POSITIONS = 10 # Cutting points, tactical locations
    CH_GAME_STATE = 11         # Turn number, urgency indicators
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def encode_board(self, board_state: BoardState, our_snake_id: str, 
                     move_history: Optional[List[MoveDecision]] = None) -> np.ndarray:
        """Encode board state into 12-channel tensor"""
        
        # Initialize tensor [height, width, channels] - note order for consistency
        tensor = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE, self.NUM_CHANNELS), dtype=np.float32)
        
        # Find our snake and opponents
        our_snake = None
        opponents = []
        
        for snake in board_state.snakes:
            if snake.id == our_snake_id:
                our_snake = snake
            else:
                opponents.append(snake)
        
        if our_snake is None:
            self.logger.warning(f"Our snake {our_snake_id} not found in board state")
            return tensor
        
        # Channel 0: Our snake
        self._encode_snake(tensor, our_snake, self.CH_OUR_SNAKE, is_our_snake=True)
        
        # Channels 1-3: Opponent snakes (up to 3)
        for i, opponent in enumerate(opponents[:3]):
            self._encode_snake(tensor, opponent, self.CH_OPPONENT_1 + i, is_our_snake=False)
        
        # Channel 4: Food
        self._encode_food(tensor, board_state.food)
        
        # Channel 5: Walls/boundaries
        self._encode_walls(tensor, board_state.width, board_state.height)
        
        # Channel 6-7: Territory control
        self._encode_territory_control(tensor, board_state, our_snake_id)
        
        # Channel 8: Danger zones
        self._encode_danger_zones(tensor, board_state, our_snake)
        
        # Channel 9: Movement history
        self._encode_movement_history(tensor, move_history, our_snake_id)
        
        # Channel 10: Strategic positions
        self._encode_strategic_positions(tensor, board_state, our_snake)
        
        # Channel 11: Game state information
        self._encode_game_state(tensor, board_state, our_snake)
        
        return tensor
    
    def _encode_snake(self, tensor: np.ndarray, snake: SnakeState, channel: int, is_our_snake: bool):
        """Encode a snake into a channel"""
        if not snake.body:
            return
        
        # Head gets higher value
        head = snake.head
        if self._is_valid_coord(head):
            tensor[head.y, head.x, channel] = 1.0
        
        # Body segments get decreasing values based on distance from head
        for i, coord in enumerate(snake.body[1:], 1):  # Skip head (already encoded)
            if self._is_valid_coord(coord):
                # Body segments get lower values, with tail being lowest
                value = max(0.3, 1.0 - (i * 0.1))  # 0.9, 0.8, 0.7, ..., min 0.3
                tensor[coord.y, coord.x, channel] = value
    
    def _encode_food(self, tensor: np.ndarray, food: List[Coordinate]):
        """Encode food locations"""
        for coord in food:
            if self._is_valid_coord(coord):
                tensor[coord.y, coord.x, self.CH_FOOD] = 1.0
    
    def _encode_walls(self, tensor: np.ndarray, width: int, height: int):
        """Encode wall/boundary positions"""
        # Mark boundaries - note: using actual board size, not tensor size
        # This creates a border effect for edge detection
        
        # Top and bottom boundaries
        tensor[0, :, self.CH_WALLS] = 1.0
        tensor[self.BOARD_SIZE-1, :, self.CH_WALLS] = 1.0
        
        # Left and right boundaries  
        tensor[:, 0, self.CH_WALLS] = 1.0
        tensor[:, self.BOARD_SIZE-1, self.CH_WALLS] = 1.0
    
    def _encode_territory_control(self, tensor: np.ndarray, board_state: BoardState, our_snake_id: str):
        """Encode territory control using simplified Voronoi-like calculation"""
        our_snake = None
        opponents = []
        
        for snake in board_state.snakes:
            if snake.id == our_snake_id:
                our_snake = snake
            else:
                opponents.append(snake)
        
        if not our_snake or not our_snake.body:
            return
        
        # Simple territory calculation: distance to nearest snake head
        for y in range(self.BOARD_SIZE):
            for x in range(self.BOARD_SIZE):
                coord = Coordinate(x, y)
                
                # Skip occupied positions
                if self._is_occupied(coord, board_state):
                    continue
                
                our_distance = self._manhattan_distance(coord, our_snake.head)
                min_opponent_distance = float('inf')
                
                for opponent in opponents:
                    if opponent.body:
                        opp_distance = self._manhattan_distance(coord, opponent.head)
                        min_opponent_distance = min(min_opponent_distance, opp_distance)
                
                # Territory assignment based on closest snake
                if our_distance < min_opponent_distance:
                    # Our territory - stronger value for closer positions
                    value = max(0.1, 1.0 - (our_distance / self.BOARD_SIZE))
                    tensor[y, x, self.CH_OUR_TERRITORY] = value
                elif min_opponent_distance < float('inf'):
                    # Opponent territory
                    value = max(0.1, 1.0 - (min_opponent_distance / self.BOARD_SIZE))
                    tensor[y, x, self.CH_OPPONENT_TERRITORY] = value
    
    def _encode_danger_zones(self, tensor: np.ndarray, board_state: BoardState, our_snake: SnakeState):
        """Encode immediate collision risks"""
        if not our_snake.body:
            return
        
        head = our_snake.head
        
        # Check each direction from our head
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        
        for dx, dy in directions:
            check_x, check_y = head.x + dx, head.y + dy
            danger_level = 0.0
            
            # Boundary collision
            if not (0 <= check_x < self.BOARD_SIZE and 0 <= check_y < self.BOARD_SIZE):
                danger_level = 1.0
            else:
                check_coord = Coordinate(check_x, check_y)
                
                # Snake body collision
                for snake in board_state.snakes:
                    for body_coord in snake.body:
                        if body_coord.x == check_x and body_coord.y == check_y:
                            danger_level = max(danger_level, 0.8)
                
                # Head-to-head collision risk (opponent heads)
                for snake in board_state.snakes:
                    if snake.id != our_snake.id and snake.body:
                        # Check if opponent could move to this position
                        opp_head = snake.head
                        if abs(opp_head.x - check_x) + abs(opp_head.y - check_y) == 1:
                            # Opponent could move here - risk depends on length comparison
                            if len(snake.body) >= len(our_snake.body):
                                danger_level = max(danger_level, 0.6)  # We lose
                            else:
                                danger_level = max(danger_level, 0.3)  # We win
            
            # Mark danger zone
            if danger_level > 0 and 0 <= check_x < self.BOARD_SIZE and 0 <= check_y < self.BOARD_SIZE:
                tensor[check_y, check_x, self.CH_DANGER_ZONES] = danger_level
    
    def _encode_movement_history(self, tensor: np.ndarray, move_history: Optional[List[MoveDecision]], 
                                our_snake_id: str):
        """Encode recent movement trail"""
        if not move_history:
            return
        
        # Get recent moves for our snake (last 5 turns)
        recent_moves = []
        for move_decision in reversed(move_history[-5:]):
            if move_decision.snake_id == our_snake_id:
                recent_moves.append(move_decision)
        
        # Encode positions with decreasing intensity
        for i, move_decision in enumerate(reversed(recent_moves)):
            if move_decision.board_state and move_decision.board_state.snakes:
                our_snake = None
                for snake in move_decision.board_state.snakes:
                    if snake.id == our_snake_id:
                        our_snake = snake
                        break
                
                if our_snake and our_snake.body:
                    head = our_snake.head
                    if self._is_valid_coord(head):
                        # More recent positions get higher values
                        value = max(0.2, 1.0 - (i * 0.2))
                        tensor[head.y, head.x, self.CH_MOVEMENT_HISTORY] = value
    
    def _encode_strategic_positions(self, tensor: np.ndarray, board_state: BoardState, our_snake: SnakeState):
        """Encode strategic positions like cutting points"""
        if not our_snake.body:
            return
        
        # Simple strategic position detection
        # Look for positions that could cut off opponents or control key areas
        
        for y in range(self.BOARD_SIZE):
            for x in range(self.BOARD_SIZE):
                coord = Coordinate(x, y)
                
                if self._is_occupied(coord, board_state):
                    continue
                
                strategic_value = 0.0
                
                # Center positions are generally more strategic
                center_distance = abs(x - self.BOARD_SIZE//2) + abs(y - self.BOARD_SIZE//2)
                if center_distance <= 2:
                    strategic_value += 0.3
                
                # Positions that could cut opponent escape routes
                cutting_value = self._calculate_cutting_value(coord, board_state, our_snake)
                strategic_value += cutting_value
                
                # Positions near food but not immediately dangerous
                for food_coord in board_state.food:
                    food_distance = self._manhattan_distance(coord, food_coord)
                    if food_distance <= 2:
                        strategic_value += max(0.0, 0.2 - food_distance * 0.1)
                
                if strategic_value > 0:
                    tensor[y, x, self.CH_STRATEGIC_POSITIONS] = min(1.0, strategic_value)
    
    def _encode_game_state(self, tensor: np.ndarray, board_state: BoardState, our_snake: SnakeState):
        """Encode game state information uniformly across the board"""
        # Turn-based urgency (higher values for later turns)
        turn_urgency = min(1.0, board_state.turn / 200.0)
        
        # Health urgency (higher values for low health)
        health_urgency = max(0.0, 1.0 - our_snake.health / 100.0) if our_snake else 0.0
        
        # Food scarcity (higher values when little food available)
        food_scarcity = max(0.0, 1.0 - len(board_state.food) / 5.0)
        
        # Combine game state factors
        game_state_value = (turn_urgency + health_urgency + food_scarcity) / 3.0
        
        # Fill entire channel with this value
        tensor[:, :, self.CH_GAME_STATE] = game_state_value
    
    def _calculate_cutting_value(self, coord: Coordinate, board_state: BoardState, our_snake: SnakeState) -> float:
        """Calculate how valuable a position is for cutting off opponents"""
        cutting_value = 0.0
        
        for snake in board_state.snakes:
            if snake.id == our_snake.id or not snake.body:
                continue
            
            # Check if this position would reduce opponent's accessible area
            opponent_head = snake.head
            distance_to_opponent = self._manhattan_distance(coord, opponent_head)
            
            if 2 <= distance_to_opponent <= 4:
                # Position is in good range for cutting
                cutting_value += 0.2
        
        return min(cutting_value, 0.8)
    
    def _is_valid_coord(self, coord: Coordinate) -> bool:
        """Check if coordinate is within board bounds"""
        return 0 <= coord.x < self.BOARD_SIZE and 0 <= coord.y < self.BOARD_SIZE
    
    def _is_occupied(self, coord: Coordinate, board_state: BoardState) -> bool:
        """Check if position is occupied by any snake"""
        for snake in board_state.snakes:
            for body_coord in snake.body:
                if body_coord.x == coord.x and body_coord.y == coord.y:
                    return True
        return False
    
    def _manhattan_distance(self, coord1: Coordinate, coord2: Coordinate) -> int:
        """Calculate Manhattan distance between two coordinates"""
        return abs(coord1.x - coord2.x) + abs(coord1.y - coord2.y)

class FeatureExtractor:
    """Extracts additional feature vectors from game state"""
    
    SNAKE_FEATURES_SIZE = 32
    GAME_CONTEXT_SIZE = 16
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_snake_features(self, our_snake: SnakeState, board_state: BoardState) -> np.ndarray:
        """Extract snake-specific features [32 dimensional]"""
        features = np.zeros(self.SNAKE_FEATURES_SIZE, dtype=np.float32)
        
        if not our_snake or not our_snake.body:
            return features
        
        idx = 0
        
        # Basic snake properties [8 features]
        features[idx] = our_snake.health / 100.0  # Normalized health
        features[idx+1] = len(our_snake.body) / 20.0  # Normalized length (assume max ~20)
        features[idx+2] = our_snake.head.x / 11.0  # Normalized head X
        features[idx+3] = our_snake.head.y / 11.0  # Normalized head Y
        
        # Distance to walls [4 features]
        features[idx+4] = our_snake.head.x / 11.0  # Distance to left wall
        features[idx+5] = (10 - our_snake.head.x) / 11.0  # Distance to right wall  
        features[idx+6] = our_snake.head.y / 11.0  # Distance to top wall
        features[idx+7] = (10 - our_snake.head.y) / 11.0  # Distance to bottom wall
        idx += 8
        
        # Food-related features [4 features]
        if board_state.food:
            distances_to_food = [self._manhattan_distance(our_snake.head, food) 
                               for food in board_state.food]
            min_food_distance = min(distances_to_food)
            avg_food_distance = sum(distances_to_food) / len(distances_to_food)
            
            features[idx] = min_food_distance / 20.0  # Normalized min food distance
            features[idx+1] = avg_food_distance / 20.0  # Normalized avg food distance
            features[idx+2] = len(board_state.food) / 10.0  # Normalized food count
            features[idx+3] = 1.0 if our_snake.health <= 30 else 0.0  # Hunger indicator
        idx += 4
        
        # Opponent-related features [12 features - 4 per opponent max]
        opponents = [s for s in board_state.snakes if s.id != our_snake.id and s.body][:3]
        
        for i in range(3):  # Support up to 3 opponents
            base_idx = idx + i * 4
            if i < len(opponents):
                opponent = opponents[i]
                features[base_idx] = len(opponent.body) / 20.0  # Opponent length
                features[base_idx+1] = opponent.health / 100.0  # Opponent health
                features[base_idx+2] = self._manhattan_distance(our_snake.head, opponent.head) / 20.0  # Distance
                features[base_idx+3] = 1.0 if len(opponent.body) > len(our_snake.body) else 0.0  # Size threat
        idx += 12
        
        # Space and mobility features [8 features] 
        accessible_spaces = self._count_accessible_spaces(our_snake.head, board_state, max_distance=5)
        features[idx] = accessible_spaces / 50.0  # Normalized accessible space
        
        # Count available moves
        available_moves = self._count_safe_moves(our_snake, board_state)
        features[idx+1] = available_moves / 4.0  # Normalized (0-4 moves)
        
        # Territory control estimate
        controlled_territory = self._estimate_territory_control(our_snake, board_state)
        features[idx+2] = controlled_territory / 121.0  # Normalized (max 11x11)
        
        # Body density (how tightly coiled)
        body_density = self._calculate_body_density(our_snake)
        features[idx+3] = body_density
        
        # Remaining features for future use
        for j in range(4, 8):
            features[idx+j] = 0.0  # Reserved for future features
        
        return features
    
    def extract_game_context(self, board_state: BoardState, our_snake: SnakeState) -> np.ndarray:
        """Extract game context features [16 dimensional]"""
        features = np.zeros(self.GAME_CONTEXT_SIZE, dtype=np.float32)
        
        idx = 0
        
        # Game progress [4 features]
        features[idx] = min(1.0, board_state.turn / 200.0)  # Turn progress
        features[idx+1] = len(board_state.snakes) / 4.0  # Snake count (normalized for 4 max)
        features[idx+2] = len(board_state.food) / 10.0  # Food abundance
        features[idx+3] = len([s for s in board_state.snakes if len(s.body) >= 10]) / 4.0  # Large snakes
        idx += 4
        
        # Board state [4 features]
        total_snake_length = sum(len(s.body) for s in board_state.snakes)
        features[idx] = total_snake_length / 121.0  # Board occupancy
        features[idx+1] = self._calculate_board_congestion(board_state)  # Congestion metric
        features[idx+2] = self._calculate_food_distribution(board_state)  # Food spread
        features[idx+3] = self._calculate_danger_level(board_state, our_snake)  # Overall danger
        idx += 4
        
        # Competition state [4 features]
        if our_snake:
            our_rank = self._get_snake_rank_by_length(our_snake, board_state)
            features[idx] = our_rank / len(board_state.snakes)  # Our ranking
            
            length_advantage = self._get_length_advantage(our_snake, board_state)
            features[idx+1] = max(-1.0, min(1.0, length_advantage / 10.0))  # Length advantage
            
            health_advantage = self._get_health_advantage(our_snake, board_state)
            features[idx+2] = max(-1.0, min(1.0, health_advantage / 100.0))  # Health advantage
            
            features[idx+3] = 1.0 if our_snake.health < 30 else 0.0  # Critical health
        idx += 4
        
        # Strategic context [4 features - reserved for future use]
        for j in range(4):
            features[idx+j] = 0.0
        
        return features
    
    def _manhattan_distance(self, coord1: Coordinate, coord2: Coordinate) -> int:
        """Calculate Manhattan distance"""
        return abs(coord1.x - coord2.x) + abs(coord1.y - coord2.y)
    
    def _count_accessible_spaces(self, start: Coordinate, board_state: BoardState, max_distance: int = 5) -> int:
        """Count accessible empty spaces using BFS"""
        if not start:
            return 0
        
        visited = set()
        queue = deque([(start.x, start.y, 0)])
        visited.add((start.x, start.y))
        count = 0
        
        while queue:
            x, y, dist = queue.popleft()
            if dist > max_distance:
                continue
                
            count += 1
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < 11 and 0 <= ny < 11 and 
                    (nx, ny) not in visited and
                    not self._is_position_occupied(nx, ny, board_state)):
                    visited.add((nx, ny))
                    queue.append((nx, ny, dist + 1))
        
        return count
    
    def _count_safe_moves(self, our_snake: SnakeState, board_state: BoardState) -> int:
        """Count number of safe moves available"""
        if not our_snake.body:
            return 0
        
        head = our_snake.head
        safe_moves = 0
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = head.x + dx, head.y + dy
            if (0 <= nx < 11 and 0 <= ny < 11 and
                not self._is_position_occupied(nx, ny, board_state)):
                safe_moves += 1
        
        return safe_moves
    
    def _estimate_territory_control(self, our_snake: SnakeState, board_state: BoardState) -> float:
        """Estimate territory controlled by our snake"""
        if not our_snake.body:
            return 0.0
        
        controlled = 0
        head = our_snake.head
        
        for x in range(11):
            for y in range(11):
                if self._is_position_occupied(x, y, board_state):
                    continue
                
                our_distance = abs(x - head.x) + abs(y - head.y)
                min_opponent_distance = float('inf')
                
                for snake in board_state.snakes:
                    if snake.id != our_snake.id and snake.body:
                        opp_distance = abs(x - snake.head.x) + abs(y - snake.head.y)
                        min_opponent_distance = min(min_opponent_distance, opp_distance)
                
                if our_distance < min_opponent_distance:
                    controlled += 1
        
        return float(controlled)
    
    def _calculate_body_density(self, snake: SnakeState) -> float:
        """Calculate how densely packed the snake's body is"""
        if len(snake.body) < 3:
            return 0.0
        
        total_distance = 0
        for i in range(len(snake.body) - 1):
            total_distance += self._manhattan_distance(snake.body[i], snake.body[i+1])
        
        # Perfect density would be length-1 (each segment adjacent)
        perfect_distance = len(snake.body) - 1
        density = perfect_distance / total_distance if total_distance > 0 else 0.0
        return min(1.0, density)
    
    def _calculate_board_congestion(self, board_state: BoardState) -> float:
        """Calculate how congested the board is"""
        total_positions = 11 * 11
        occupied_positions = sum(len(snake.body) for snake in board_state.snakes)
        return occupied_positions / total_positions
    
    def _calculate_food_distribution(self, board_state: BoardState) -> float:
        """Calculate how well distributed food is across the board"""
        if len(board_state.food) < 2:
            return 1.0
        
        total_distance = 0
        count = 0
        
        for i, food1 in enumerate(board_state.food):
            for food2 in board_state.food[i+1:]:
                total_distance += self._manhattan_distance(food1, food2)
                count += 1
        
        avg_distance = total_distance / count if count > 0 else 0
        # Normalize by maximum possible distance (corner to corner)
        max_distance = 20  # Manhattan distance across 11x11 board
        return min(1.0, avg_distance / max_distance)
    
    def _calculate_danger_level(self, board_state: BoardState, our_snake: SnakeState) -> float:
        """Calculate overall danger level for our snake"""
        if not our_snake or not our_snake.body:
            return 1.0
        
        danger = 0.0
        
        # Health danger
        if our_snake.health < 30:
            danger += 0.3
        elif our_snake.health < 50:
            danger += 0.1
        
        # Space danger
        safe_moves = self._count_safe_moves(our_snake, board_state)
        if safe_moves <= 1:
            danger += 0.4
        elif safe_moves == 2:
            danger += 0.2
        
        # Opponent threat
        for snake in board_state.snakes:
            if snake.id != our_snake.id and snake.body:
                distance = self._manhattan_distance(our_snake.head, snake.head)
                if distance <= 3 and len(snake.body) >= len(our_snake.body):
                    danger += 0.1
        
        return min(1.0, danger)
    
    def _get_snake_rank_by_length(self, our_snake: SnakeState, board_state: BoardState) -> int:
        """Get our snake's ranking by length (1 = longest)"""
        lengths = sorted([len(snake.body) for snake in board_state.snakes], reverse=True)
        our_length = len(our_snake.body)
        return lengths.index(our_length) + 1
    
    def _get_length_advantage(self, our_snake: SnakeState, board_state: BoardState) -> float:
        """Get average length advantage over opponents"""
        our_length = len(our_snake.body)
        opponent_lengths = [len(snake.body) for snake in board_state.snakes 
                          if snake.id != our_snake.id and snake.body]
        
        if not opponent_lengths:
            return 0.0
        
        avg_opponent_length = sum(opponent_lengths) / len(opponent_lengths)
        return our_length - avg_opponent_length
    
    def _get_health_advantage(self, our_snake: SnakeState, board_state: BoardState) -> float:
        """Get average health advantage over opponents"""
        our_health = our_snake.health
        opponent_healths = [snake.health for snake in board_state.snakes 
                          if snake.id != our_snake.id and snake.body]
        
        if not opponent_healths:
            return 0.0
        
        avg_opponent_health = sum(opponent_healths) / len(opponent_healths)
        return our_health - avg_opponent_health
    
    def _is_position_occupied(self, x: int, y: int, board_state: BoardState) -> bool:
        """Check if position is occupied by any snake"""
        for snake in board_state.snakes:
            for body_coord in snake.body:
                if body_coord.x == x and body_coord.y == y:
                    return True
        return False

class TrainingDataProcessor:
    """Main processor that converts game data to training samples"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        self.board_encoder = BoardEncoder()
        self.feature_extractor = FeatureExtractor()
        
        # Processing statistics
        self.processed_games = 0
        self.generated_samples = 0
        self.invalid_samples = 0
        
    def process_game(self, game_data: GameData) -> List[TrainingSample]:
        """Process a complete game into training samples"""
        if not game_data.moves:
            self.logger.warning(f"Game {game_data.game_id} has no moves to process")
            return []
        
        samples = []
        move_history = []  # For encoding movement history
        
        # Calculate game outcome value for all moves
        game_outcome = self._calculate_game_outcome(game_data)
        
        for move_decision in game_data.moves:
            try:
                sample = self._process_move_decision(move_decision, move_history, game_outcome)
                if sample:
                    samples.append(sample)
                    self.generated_samples += 1
                else:
                    self.invalid_samples += 1
                
                # Add to history for subsequent moves
                move_history.append(move_decision)
                
            except Exception as e:
                self.logger.error(f"Error processing move in game {game_data.game_id}: {e}")
                self.invalid_samples += 1
        
        self.processed_games += 1
        self.logger.info(f"Game {game_data.game_id}: {len(samples)} training samples generated")
        
        return samples
    
    def _process_move_decision(self, move_decision: MoveDecision, move_history: List[MoveDecision], 
                              game_outcome: float) -> Optional[TrainingSample]:
        """Process a single move decision into a training sample"""
        
        if not move_decision.board_state or not move_decision.board_state.snakes:
            return None
        
        # Find our snake in the board state
        our_snake = None
        for snake in move_decision.board_state.snakes:
            if snake.id == move_decision.snake_id:
                our_snake = snake
                break
        
        if not our_snake:
            self.logger.warning(f"Snake {move_decision.snake_id} not found in board state")
            return None
        
        try:
            # Encode board state
            board_tensor = self.board_encoder.encode_board(
                move_decision.board_state, 
                move_decision.snake_id,
                move_history
            )
            
            # Extract features
            snake_features = self.feature_extractor.extract_snake_features(
                our_snake, 
                move_decision.board_state
            )
            
            game_context = self.feature_extractor.extract_game_context(
                move_decision.board_state, 
                our_snake
            )
            
            # Convert move to target
            target_move = Move.from_string(move_decision.move)
            
            # Calculate position value from heuristic scores
            position_value = self._calculate_position_value(move_decision.heuristic_scores)
            
            # Generate move probability distribution (softmax of heuristic preferences)
            move_probabilities = self._generate_move_probabilities(move_decision, our_snake)
            
            # Create training sample
            sample = TrainingSample(
                board_state=board_tensor,
                snake_features=snake_features,
                game_context=game_context,
                target_move=int(target_move),
                position_value=position_value,
                move_probabilities=move_probabilities,
                win_probability=game_outcome,
                heuristic_scores=move_decision.heuristic_scores.to_dict(),
                game_id=move_decision.board_state.game_id if hasattr(move_decision.board_state, 'game_id') else 'unknown',
                turn=move_decision.turn,
                snake_id=move_decision.snake_id,
                timestamp=move_decision.timestamp
            )
            
            return sample
            
        except Exception as e:
            self.logger.error(f"Error creating training sample: {e}")
            return None
    
    def _calculate_game_outcome(self, game_data: GameData) -> float:
        """Calculate game outcome value (1.0 = win, 0.0 = loss)"""
        if not game_data.moves:
            return 0.5  # No data
        
        # Simple heuristic: if we survived until the end and made many moves, we did well
        # In a real implementation, we'd parse the actual game outcome
        
        our_snake_id = game_data.moves[0].snake_id if game_data.moves else None
        if not our_snake_id:
            return 0.5
        
        # Check if we're still alive at the end
        if game_data.final_state:
            our_final_snake = None
            for snake in game_data.final_state.snakes:
                if snake.id == our_snake_id:
                    our_final_snake = snake
                    break
            
            if our_final_snake:
                # We survived - check our relative performance
                our_length = len(our_final_snake.body)
                other_lengths = [len(s.body) for s in game_data.final_state.snakes 
                               if s.id != our_snake_id]
                
                if not other_lengths:
                    return 1.0  # We're the only survivor
                
                avg_other_length = sum(other_lengths) / len(other_lengths)
                
                # Scale outcome based on relative performance
                if our_length > avg_other_length:
                    return 0.8  # We did well
                else:
                    return 0.6  # We survived but didn't dominate
            else:
                return 0.2  # We died
        
        # Fallback: estimate based on game length and final heuristic scores
        if len(game_data.moves) > 50:  # Long game suggests we did reasonably well
            return 0.6
        else:
            return 0.4
    
    def _calculate_position_value(self, heuristic_scores: HeuristicScores) -> float:
        """Convert heuristic scores to position value (-50 to +50)"""
        # Combine all heuristic components
        total_score = heuristic_scores.total_score
        
        # The heuristic system already provides good scoring
        # Safety: 8-50, Territory: 3-9, Food: 0-11, Opponent: 0-6, Exploration: ~30
        # Total range approximately -50 to +110
        
        # Normalize to -50 to +50 range
        normalized = max(-50.0, min(50.0, total_score))
        
        return float(normalized)
    
    def _generate_move_probabilities(self, move_decision: MoveDecision, our_snake: SnakeState) -> np.ndarray:
        """Generate move probability distribution from heuristic preferences"""
        
        # Initialize with small base probabilities
        probabilities = np.full(4, 0.05, dtype=np.float32)  # Small base for all moves
        
        # Get the chosen move
        chosen_move = Move.from_string(move_decision.move)
        
        # The chosen move gets the highest probability
        probabilities[int(chosen_move)] = 0.7
        
        # Distribute remaining probability based on move safety
        if move_decision.board_state:
            remaining_prob = 0.25
            safe_moves = []
            
            head = our_snake.head
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
            
            for i, (dx, dy) in enumerate(directions):
                if i == int(chosen_move):
                    continue  # Skip chosen move
                
                new_x, new_y = head.x + dx, head.y + dy
                
                # Check if move is safe
                is_safe = True
                if not (0 <= new_x < 11 and 0 <= new_y < 11):
                    is_safe = False
                else:
                    # Check collision with snakes
                    for snake in move_decision.board_state.snakes:
                        for body_coord in snake.body:
                            if body_coord.x == new_x and body_coord.y == new_y:
                                is_safe = False
                                break
                        if not is_safe:
                            break
                
                if is_safe:
                    safe_moves.append(i)
            
            # Distribute remaining probability among safe moves
            if safe_moves:
                prob_per_safe_move = remaining_prob / len(safe_moves)
                for move_idx in safe_moves:
                    probabilities[move_idx] += prob_per_safe_move
        
        # Ensure probabilities sum to 1.0
        probabilities = probabilities / probabilities.sum()
        
        return probabilities
    
    def process_games_batch(self, games: List[GameData]) -> List[TrainingSample]:
        """Process multiple games in parallel"""
        all_samples = []
        
        # Use thread pool for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_game = {executor.submit(self.process_game, game): game for game in games}
            
            for future in concurrent.futures.as_completed(future_to_game):
                game = future_to_game[future]
                try:
                    samples = future.result()
                    all_samples.extend(samples)
                except Exception as e:
                    self.logger.error(f"Error processing game {game.game_id}: {e}")
        
        return all_samples
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        success_rate = (self.generated_samples / (self.generated_samples + self.invalid_samples) 
                       if (self.generated_samples + self.invalid_samples) > 0 else 0.0)
        
        return {
            'processed_games': self.processed_games,
            'generated_samples': self.generated_samples,
            'invalid_samples': self.invalid_samples,
            'success_rate': success_rate,
            'avg_samples_per_game': (self.generated_samples / self.processed_games 
                                   if self.processed_games > 0 else 0.0)
        }
    
    def validate_sample(self, sample: TrainingSample) -> bool:
        """Validate a training sample for quality"""
        # Check tensor shapes
        if sample.board_state.shape != (11, 11, 12):
            return False
        
        if sample.snake_features.shape != (32,):
            return False
        
        if sample.game_context.shape != (16,):
            return False
        
        if sample.move_probabilities.shape != (4,):
            return False
        
        # Check value ranges
        if not (0 <= sample.target_move <= 3):
            return False
        
        if not (-50.0 <= sample.position_value <= 50.0):
            return False
        
        if not (0.0 <= sample.win_probability <= 1.0):
            return False
        
        # Check probability distribution
        if not np.isclose(sample.move_probabilities.sum(), 1.0, atol=1e-3):
            return False
        
        # Check for NaN or infinite values
        if (np.any(np.isnan(sample.board_state)) or 
            np.any(np.isnan(sample.snake_features)) or
            np.any(np.isnan(sample.game_context)) or
            np.any(np.isnan(sample.move_probabilities))):
            return False
        
        return True

if __name__ == "__main__":
    # Testing and demonstration
    logging.basicConfig(level=logging.INFO)
    
    # Create test data
    test_coord = Coordinate(5, 5)
    test_snake = SnakeState(
        id="test_snake",
        name="Test Snake",
        health=85,
        body=[Coordinate(5, 5), Coordinate(5, 6), Coordinate(5, 7)],
        head=Coordinate(5, 5),
        length=3
    )
    
    test_board = BoardState(
        width=11,
        height=11,
        food=[Coordinate(8, 3), Coordinate(2, 9)],
        hazards=[],
        snakes=[test_snake],
        turn=25
    )
    
    test_heuristics = HeuristicScores(
        safety=35.0,
        territory=6.5,
        food=4.2,
        opponent=2.1,
        exploration=28.3
    )
    
    print("=== Testing Board Encoding ===")
    encoder = BoardEncoder()
    board_tensor = encoder.encode_board(test_board, "test_snake")
    print(f"Board tensor shape: {board_tensor.shape}")
    print(f"Non-zero channels: {[i for i in range(12) if np.any(board_tensor[:,:,i] > 0)]}")
    
    print("\n=== Testing Feature Extraction ===")
    extractor = FeatureExtractor()
    snake_features = extractor.extract_snake_features(test_snake, test_board)
    game_context = extractor.extract_game_context(test_board, test_snake)
    
    print(f"Snake features shape: {snake_features.shape}")
    print(f"Game context shape: {game_context.shape}")
    print(f"Sample snake features: {snake_features[:8]}")  # First 8 features
    
    print("\n=== Testing Training Data Processor ===")
    processor = TrainingDataProcessor()
    
    # Create a test move decision
    test_move = MoveDecision(
        snake_id="test_snake",
        turn=25,
        move="right",
        board_state=test_board,
        heuristic_scores=test_heuristics,
        decision_time_ms=42.5,
        timestamp=datetime.now(),
        server_port=8000
    )
    
    sample = processor._process_move_decision(test_move, [], 0.8)
    if sample:
        print(f"Generated training sample:")
        print(f"  Board tensor: {sample.board_state.shape}")
        print(f"  Target move: {sample.target_move} ({Move(sample.target_move).to_string()})")
        print(f"  Position value: {sample.position_value:.2f}")
        print(f"  Move probabilities: {sample.move_probabilities}")
        print(f"  Win probability: {sample.win_probability}")
        
        # Validate sample
        is_valid = processor.validate_sample(sample)
        print(f"  Sample valid: {is_valid}")
    
    print("\n=== Training Data Pipeline Ready ===")
    print("Ready to process game data into neural network training format")