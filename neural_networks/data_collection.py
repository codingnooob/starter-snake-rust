"""
Data Collection Framework for Battlesnake Training

This module handles collecting training data from Battlesnake games,
preprocessing game states, and creating training datasets.
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import os
import logging
from pathlib import Path
from datetime import datetime
import pickle

from board_encoding import BoardState, BoardStateEncoder, TrainingSample

@dataclass
class GameRecord:
    """Record of a single game for training data collection"""
    game_id: str
    board_state: Dict[str, Any]
    our_snake_id: str
    moves_made: List[str]
    game_outcome: str  # 'win', 'loss', 'draw'
    final_score: int
    game_length: int
    timestamp: str
    metadata: Dict[str, Any]

class GameDataCollector:
    """Collects and manages training data from Battlesnake games"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize encoder
        self.encoder = BoardStateEncoder()
        
        # Data storage
        self.game_records: List[GameRecord] = []
        self.training_samples: List[TrainingSample] = []
        
        # Statistics
        self.stats = {
            'games_collected': 0,
            'samples_created': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'avg_game_length': 0
        }
    
    def record_game(self, game_data: Dict[str, Any], our_snake_id: str, moves: List[str], 
                   outcome: str, score: int) -> GameRecord:
        """
        Record a complete game for training data collection
        
        Args:
            game_data: Complete game state data
            our_snake_id: ID of our snake
            moves: List of moves made during the game
            outcome: Game outcome ('win', 'loss', 'draw')
            score: Final score
            
        Returns:
            GameRecord object
        """
        game_record = GameRecord(
            game_id=game_data.get('game', {}).get('id', f"game_{datetime.now().timestamp()}"),
            board_state=game_data.get('board', {}),
            our_snake_id=our_snake_id,
            moves_made=moves,
            game_outcome=outcome,
            final_score=score,
            game_length=len(moves),
            timestamp=datetime.now().isoformat(),
            metadata={
                'ruleset': game_data.get('game', {}).get('ruleset', {}),
                'timeout': game_data.get('game', {}).get('timeout', 500),
                'source': game_data.get('game', {}).get('source', 'custom')
            }
        )
        
        self.game_records.append(game_record)
        self.stats['games_collected'] += 1
        
        if outcome == 'win':
            self.stats['wins'] += 1
        elif outcome == 'loss':
            self.stats['losses'] += 1
        else:
            self.stats['draws'] += 1
            
        return game_record
    
    def create_training_sample(self, game_record: GameRecord, turn: int) -> Optional[TrainingSample]:
        """
        Create training sample from a specific turn in a game
        
        Args:
            game_record: Game record to extract sample from
            turn: Turn number (0-indexed)
            
        Returns:
            TrainingSample or None if invalid
        """
        if turn >= len(game_record.moves_made):
            return None
            
        # Find our snake in the board state
        board_state = game_record.board_state
        you_snake = None
        
        for snake in board_state.get('snakes', []):
            if snake['id'] == game_record.our_snake_id:
                you_snake = snake
                break
        
        if not you_snake:
            return None
        
        # Encode board state
        try:
            encoded_state = self.encoder.encode_board_state(board_state, you_snake)
            
            # Create training sample
            sample = TrainingSample(
                board_state=encoded_state,
                target_move=game_record.moves_made[turn],
                game_outcome=game_record.game_outcome
            )
            
            # Add derived labels
            self._add_derived_labels(sample, game_record, turn)
            
            return sample
            
        except Exception as e:
            self.logger.warning(f"Failed to encode state at turn {turn}: {e}")
            return None
    
    def _add_derived_labels(self, sample: TrainingSample, game_record: GameRecord, turn: int):
        """Add derived training labels to sample"""
        
        # Move probabilities (assuming uniform for now, could be improved with MCTS analysis)
        moves = ['up', 'down', 'left', 'right']
        sample.move_probabilities = {move: 0.25 for move in moves}
        sample.move_probabilities[sample.target_move] = 0.6  # Boost target move probability
        remaining_prob = 0.4 / 3  # Distribute remaining probability
        for move in moves:
            if move != sample.target_move:
                sample.move_probabilities[move] = remaining_prob
        
        # Position score based on game outcome and position in game
        if game_record.game_outcome == 'win':
            if turn < game_record.game_length * 0.5:
                sample.position_score = 0.8  # Strong early position in winning game
            else:
                sample.position_score = 0.9  # Strong late position in winning game
        elif game_record.game_outcome == 'loss':
            if turn < game_record.game_length * 0.5:
                sample.position_score = 0.3  # Poor early position in losing game
            else:
                sample.position_score = 0.1  # Poor late position in losing game
        else:  # draw
            sample.position_score = 0.5  # Neutral position in drawn game
        
        # Win probability based on remaining turns and current outcome
        remaining_turns = game_record.game_length - turn
        if game_record.game_outcome == 'win':
            sample.win_probability = 0.9 - (remaining_turns * 0.01)  # Decreases as game progresses
        elif game_record.game_outcome == 'loss':
            sample.win_probability = 0.1 + (remaining_turns * 0.01)  # Increases as game progresses (uncertainty)
        else:  # draw
            sample.win_probability = 0.5
    
    def create_training_samples(self) -> List[TrainingSample]:
        """
        Create training samples from all recorded games
        
        Returns:
            List of TrainingSample objects
        """
        samples = []
        
        for game_record in self.game_records:
            # Create samples for each turn in the game
            for turn in range(len(game_record.moves_made)):
                sample = self.create_training_sample(game_record, turn)
                if sample:
                    samples.append(sample)
        
        self.training_samples = samples
        self.stats['samples_created'] = len(samples)
        
        # Update average game length
        if self.stats['games_collected'] > 0:
            total_length = sum(record.game_length for record in self.game_records)
            self.stats['avg_game_length'] = total_length / self.stats['games_collected']
        
        self.logger.info(f"Created {len(samples)} training samples from {len(self.game_records)} games")
        
        return samples
    
    def save_data(self, filename: Optional[str] = None):
        """Save collected data to disk"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{timestamp}.pkl"
        
        filepath = self.data_dir / filename
        
        data = {
            'game_records': self.game_records,
            'training_samples': self.training_samples,
            'stats': self.stats,
            'created_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Saved {len(self.training_samples)} samples to {filepath}")
    
    def load_data(self, filename: str):
        """Load previously saved data"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.game_records = data['game_records']
        self.training_samples = data['training_samples']
        self.stats = data.get('stats', {})
        
        self.logger.info(f"Loaded {len(self.training_samples)} samples from {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return self.stats.copy()
    
    def print_statistics(self):
        """Print collection statistics"""
        stats = self.get_statistics()
        print("\n=== Data Collection Statistics ===")
        print(f"Games Collected: {stats['games_collected']}")
        print(f"Samples Created: {stats['samples_created']}")
        print(f"Wins: {stats['wins']} ({stats['wins']/max(stats['games_collected'],1)*100:.1f}%)")
        print(f"Losses: {stats['losses']} ({stats['losses']/max(stats['games_collected'],1)*100:.1f}%)")
        print(f"Draws: {stats['draws']} ({stats['draws']/max(stats['games_collected'],1)*100:.1f}%)")
        print(f"Average Game Length: {stats['avg_game_length']:.1f} turns")

class GameDataProcessor:
    """Processes raw game data into training datasets"""
    
    def __init__(self, data_dir: str = "data", processed_dir: str = "processed"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(exist_ok=True)
        
        self.encoder = BoardStateEncoder()
    
    def process_game_file(self, file_path: str, output_name: str):
        """
        Process a single game data file into training samples
        
        Args:
            file_path: Path to game data file (JSON or pickle)
            output_name: Name for processed output file
        """
        # Load raw data
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                raw_data = json.load(f)
        elif file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                raw_data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        collector = GameDataCollector()
        
        # Process individual games if raw_data is a list
        if isinstance(raw_data, list):
            for game_data in raw_data:
                self._process_single_game(game_data, collector)
        else:
            self._process_single_game(raw_data, collector)
        
        # Create training samples
        samples = collector.create_training_samples()
        
        # Save processed data
        output_path = self.processed_dir / f"{output_name}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump({
                'samples': samples,
                'stats': collector.get_statistics(),
                'processed_at': datetime.now().isoformat()
            }, f)
        
        print(f"Processed {len(samples)} samples -> {output_path}")
        collector.print_statistics()
    
    def _process_single_game(self, game_data: Dict[str, Any], collector: GameDataCollector):
        """Process a single game's data"""
        # Extract game information
        game_id = game_data.get('game', {}).get('id', 'unknown')
        our_snake_id = game_data.get('you', {}).get('id', 'unknown')
        
        # Extract moves (this would need to be adapted based on actual data format)
        moves = []
        turn_data = game_data.get('turns', [])
        for turn in turn_data:
            if 'move' in turn:
                moves.append(turn['move'])
        
        # Determine outcome
        outcome = 'draw'  # Default
        if game_data.get('you', {}).get('health', 0) > 0:
            # Check if we survived to the end
            other_snakes = [s for s in game_data.get('board', {}).get('snakes', []) 
                          if s['id'] != our_snake_id]
            if all(s.get('health', 0) <= 0 for s in other_snakes):
                outcome = 'win'
        else:
            outcome = 'loss'
        
        score = len(game_data.get('you', {}).get('body', []))
        
        # Record the game
        collector.record_game(game_data, our_snake_id, moves, outcome, score)

def simulate_game_collection():
    """Simulate collecting training data for testing"""
    print("Simulating game data collection...")
    
    collector = GameDataCollector()
    
    # Simulate a few example games
    for game_idx in range(5):
        # Create mock game data
        mock_game = {
            'game': {'id': f'game_{game_idx}'},
            'board': {
                'width': 11,
                'height': 11,
                'food': [{'x': 5, 'y': 5}],
                'snakes': [
                    {
                        'id': 'our_snake',
                        'health': 90 - game_idx * 10,
                        'body': [{'x': 5, 'y': 6}, {'x': 5, 'y': 7}]
                    },
                    {
                        'id': 'opponent_1',
                        'health': 80,
                        'body': [{'x': 3, 'y': 4}, {'x': 3, 'y': 5}]
                    }
                ]
            },
            'you': {
                'id': 'our_snake',
                'health': 90 - game_idx * 10,
                'body': [{'x': 5, 'y': 6}, {'x': 5, 'y': 7}]
            }
        }
        
        # Simulate game outcome
        if game_idx < 3:
            outcome, score = 'win', 8 + game_idx
        elif game_idx == 3:
            outcome, score = 'loss', 4
        else:
            outcome, score = 'draw', 6
        
        # Simulate moves made
        moves = ['up', 'right', 'down', 'left', 'up'] * 20  # Simulate long game
        
        collector.record_game(mock_game, 'our_snake', moves, outcome, score)
    
    # Create training samples
    samples = collector.create_training_samples()
    
    # Print statistics
    collector.print_statistics()
    
    # Save data
    collector.save_data("test_collection.pkl")
    
    return collector, samples

if __name__ == "__main__":
    # Test data collection
    collector, samples = simulate_game_collection()
    
    print(f"\nFirst sample:")
    first_sample = samples[0]
    print(f"Target move: {first_sample.target_move}")
    print(f"Position score: {first_sample.position_score}")
    print(f"Win probability: {first_sample.win_probability}")
    print(f"Move probabilities: {first_sample.move_probabilities}")