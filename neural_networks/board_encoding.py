"""
Board State Encoding for Neural Networks

This module provides functions to encode Battlesnake board states into formats suitable
for neural network input (CNN-based architectures).
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import IntEnum

class CellType(IntEnum):
    """Cell type encoding for board state representation"""
    EMPTY = 0
    OWN_HEAD = 1
    OWN_BODY = 2
    OPPONENT_HEAD = 3
    OPPONENT_BODY = 4
    FOOD = 5
    WALL = 6

@dataclass
class BoardState:
    """Encoded board state representation"""
    board_width: int
    board_height: int
    
    # Multi-channel representation for CNN input
    # Shape: (channels, height, width)
    grid_channels: np.ndarray
    
    # Additional features
    our_health: int
    max_health: int
    our_length: int
    our_head_pos: Tuple[int, int]
    
    # Game metadata
    turn_number: int
    total_snake_count: int
    game_state: str  # 'ongoing', 'won', 'lost'

class BoardStateEncoder:
    """Encodes Battlesnake board states into neural network input format"""
    
    def __init__(self, max_board_size: int = 20):
        self.max_board_size = max_board_size
        self.num_channels = 7  # 7 different cell types
        
    def encode_board_state(self, 
                          board_state: Dict[str, Any],
                          you: Dict[str, Any]) -> BoardState:
        """
        Encode a battlesnake board state into CNN input format
        
        Args:
            board_state: Raw battlesnake board state
            you: Your snake information
            
        Returns:
            Encoded BoardState ready for neural network input
        """
        width = board_state['width']
        height = board_state['height']
        snakes = board_state['snakes']
        
        # Initialize multi-channel grid
        grid = np.zeros((self.num_channels, height, width), dtype=np.float32)
        
        # Encode food
        for food in board_state['food']:
            if 0 <= food['y'] < height and 0 <= food['x'] < width:
                grid[CellType.FOOD, food['y'], food['x']] = 1.0
                
        # Encode snakes
        for snake in snakes:
            is_our_snake = snake['id'] == you['id']
            
            for i, segment in enumerate(snake['body']):
                if 0 <= segment['y'] < height and 0 <= segment['x'] < width:
                    if i == 0:  # Head
                        if is_our_snake:
                            grid[CellType.OWN_HEAD, segment['y'], segment['x']] = 1.0
                        else:
                            grid[CellType.OPPONENT_HEAD, segment['y'], segment['x']] = 1.0
                    else:  # Body
                        if is_our_snake:
                            grid[CellType.OWN_BODY, segment['y'], segment['x']] = 1.0
                        else:
                            grid[CellType.OPPONENT_BODY, segment['y'], segment['x']] = 1.0
        
        # Encode walls (board boundaries)
        grid[CellType.WALL, 0, :] = 1.0  # Top boundary
        grid[CellType.WALL, height-1, :] = 1.0  # Bottom boundary
        grid[CellType.WALL, :, 0] = 1.0  # Left boundary
        grid[CellType.WALL, :, width-1] = 1.0  # Right boundary
        
        # Pad to max board size if necessary
        if height < self.max_board_size or width < self.max_board_size:
            grid = self._pad_to_max_size(grid, height, width)
            
        # Get our snake info
        our_snake = next((s for s in snakes if s['id'] == you['id']), you)
        
        return BoardState(
            board_width=width,
            board_height=height,
            grid_channels=grid,
            our_health=our_snake['health'],
            max_health=100,
            our_length=len(our_snake['body']),
            our_head_pos=(our_snake['body'][0]['y'], our_snake['body'][0]['x']),
            turn_number=board_state.get('turn', 0),
            total_snake_count=len(snakes),
            game_state='ongoing'
        )
    
    def _pad_to_max_size(self, grid: np.ndarray, height: int, width: int) -> np.ndarray:
        """Pad grid to max board size with zeros"""
        padded_grid = np.zeros((self.num_channels, self.max_board_size, self.max_board_size), dtype=np.float32)
        padded_grid[:, :height, :width] = grid
        return padded_grid
    
    def normalize_input(self, board_state: BoardState) -> np.ndarray:
        """
        Normalize board state for neural network input
        
        Args:
            board_state: Encoded board state
            
        Returns:
            Normalized input tensor
        """
        # Normalize grid channels to [0, 1]
        normalized_grid = board_state.grid_channels.copy()
        
        # Normalize health and other features to [0, 1]
        health_ratio = board_state.our_health / board_state.max_health
        length_ratio = board_state.our_length / 20.0  # Normalize by typical max length
        turn_ratio = min(board_state.turn_number / 1000.0, 1.0)  # Cap turn normalization
        
        # Add feature channels
        features = np.array([
            health_ratio,
            length_ratio, 
            turn_ratio,
            board_state.total_snake_count / 8.0,  # Normalize by max snakes
            board_state.our_head_pos[0] / board_state.board_height,  # Head Y normalized
            board_state.our_head_pos[1] / board_state.board_width   # Head X normalized
        ], dtype=np.float32)
        
        return normalized_grid, features
    
    def decode_move_probabilities(self, prediction: np.ndarray) -> Dict[str, float]:
        """Convert neural network output to move probabilities"""
        moves = ['up', 'down', 'left', 'right']
        return {move: float(pred) for move, pred in zip(moves, prediction)}
    
    def decode_position_score(self, prediction: np.ndarray) -> float:
        """Convert position evaluation output to score"""
        return float(prediction[0])
    
    def decode_win_probability(self, prediction: np.ndarray) -> float:
        """Convert game outcome output to win probability"""
        return float(prediction[0])

# Data collection and storage
@dataclass
class TrainingSample:
    """Single training sample for neural network"""
    board_state: BoardState
    target_move: str
    move_probabilities: Dict[str, float] = None
    position_score: float = None
    win_probability: float = None
    game_outcome: str = None  # 'win', 'loss', 'draw'
    
    def to_tensor_input(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to neural network input tensor"""
        encoder = BoardStateEncoder()
        grid, features = encoder.normalize_input(self.board_state)
        return grid, features
    
    def to_training_label(self, task: str) -> np.ndarray:
        """Convert to training label based on task"""
        if task == 'move_prediction':
            # Convert move to one-hot encoding
            moves = ['up', 'down', 'left', 'right']
            label = np.zeros(4)
            if self.target_move in moves:
                label[moves.index(self.target_move)] = 1.0
            return label
        elif task == 'position_evaluation':
            return np.array([self.position_score])
        elif task == 'game_outcome':
            return np.array([self.win_probability])
        else:
            raise ValueError(f"Unknown training task: {task}")

def test_board_encoding():
    """Test board state encoding functionality"""
    # Example board state
    test_board = {
        'width': 11,
        'height': 11,
        'food': [{'x': 5, 'y': 5}],
        'snakes': [
            {
                'id': 'our_snake',
                'health': 90,
                'body': [{'x': 5, 'y': 6}, {'x': 5, 'y': 7}]
            },
            {
                'id': 'opponent',
                'health': 80,
                'body': [{'x': 3, 'y': 4}, {'x': 3, 'y': 5}]
            }
        ]
    }
    
    test_you = {
        'id': 'our_snake',
        'health': 90,
        'body': [{'x': 5, 'y': 6}, {'x': 5, 'y': 7}]
    }
    
    encoder = BoardStateEncoder()
    encoded = encoder.encode_board_state(test_board, test_you)
    
    print(f"Encoded board shape: {encoded.grid_channels.shape}")
    print(f"Grid channels: {encoded.grid_channels.shape[0]}")
    print(f"Health: {encoded.our_health}")
    print(f"Head position: {encoded.our_head_pos}")
    
    # Test normalization
    grid, features = encoder.normalize_input(encoded)
    print(f"Normalized grid shape: {grid.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Features: {features}")

if __name__ == "__main__":
    test_board_encoding()