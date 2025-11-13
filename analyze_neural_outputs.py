#!/usr/bin/env python3
"""
Neural Network Output Analysis Tool

Analyzes the current ONNX models' output distributions and prediction patterns
to inform the new confidence calculation system design.
"""

import numpy as np
import onnx
import onnxruntime as ort
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuralOutputAnalyzer:
    """Analyzes neural network outputs to understand prediction patterns."""
    
    def __init__(self):
        self.models = {}
        self.model_paths = {
            'position_evaluation': 'models/position_evaluation.onnx',
            'move_prediction': 'models/move_prediction.onnx', 
            'game_outcome': 'models/game_outcome.onnx'
        }
        self.results = {}
        
    def load_models(self) -> bool:
        """Load all ONNX models for analysis."""
        logger.info("Loading ONNX models for analysis...")
        
        for name, path in self.model_paths.items():
            try:
                if Path(path).exists():
                    session = ort.InferenceSession(path)
                    self.models[name] = session
                    logger.info(f"Loaded {name} model from {path}")
                    
                    # Print model input/output info
                    inputs = [i.name for i in session.get_inputs()]
                    outputs = [o.name for o in session.get_outputs()]
                    logger.info(f"  Inputs: {inputs}")
                    logger.info(f"  Outputs: {outputs}")
                else:
                    logger.warning(f"Model not found: {path}")
                    
            except Exception as e:
                logger.error(f"Failed to load {name} model: {e}")
                
        return len(self.models) > 0
    
    def generate_sample_board_states(self, num_samples: int = 1000) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate diverse sample board states for analysis."""
        logger.info(f"Generating {num_samples} sample board states...")
        
        samples = []
        
        for i in range(num_samples):
            # Create a 20x20 board with 7 channels (as per existing system)
            grid = np.zeros((1, 7, 20, 20), dtype=np.float32)
            
            # Random board size (typical Battlesnake boards are 11x11)
            board_size = random.choice([11, 15, 19])
            
            # Channel layout: [EMPTY, OWN_HEAD, OWN_BODY, OPPONENT_HEAD, OPPONENT_BODY, FOOD, WALL]
            
            # 1. Add walls around the active board area
            grid[0, 6, :, :] = 1.0  # Fill with walls
            grid[0, 6, :board_size, :board_size] = 0.0  # Clear active area
            grid[0, 0, :board_size, :board_size] = 1.0  # Mark as empty
            
            # 2. Add our snake
            snake_length = random.randint(3, 8)
            head_x, head_y = random.randint(1, board_size-2), random.randint(1, board_size-2)
            
            # Place head
            grid[0, 0, head_y, head_x] = 0.0  # Remove empty
            grid[0, 1, head_y, head_x] = 1.0  # Add head
            
            # Place body segments
            body_positions = [(head_x, head_y)]
            for j in range(1, min(snake_length, 6)):
                # Simple body placement (could be more sophisticated)
                possible_positions = []
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nx, ny = body_positions[-1][0] + dx, body_positions[-1][1] + dy
                    if 0 <= nx < board_size and 0 <= ny < board_size:
                        if (nx, ny) not in body_positions:
                            possible_positions.append((nx, ny))
                            
                if possible_positions:
                    bx, by = random.choice(possible_positions)
                    body_positions.append((bx, by))
                    grid[0, 0, by, bx] = 0.0  # Remove empty
                    grid[0, 2, by, bx] = 1.0  # Add body
            
            # 3. Add opponent snakes
            num_opponents = random.randint(1, 3)
            for _ in range(num_opponents):
                opp_length = random.randint(3, 6)
                
                # Find empty space for opponent head
                empty_positions = []
                for y in range(board_size):
                    for x in range(board_size):
                        if grid[0, 0, y, x] == 1.0:  # Empty space
                            empty_positions.append((x, y))
                
                if empty_positions:
                    opp_head_x, opp_head_y = random.choice(empty_positions)
                    grid[0, 0, opp_head_y, opp_head_x] = 0.0  # Remove empty
                    grid[0, 3, opp_head_y, opp_head_x] = 1.0  # Add opponent head
                    
                    # Add opponent body
                    opp_positions = [(opp_head_x, opp_head_y)]
                    for j in range(1, min(opp_length, 4)):
                        possible_positions = []
                        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                            nx, ny = opp_positions[-1][0] + dx, opp_positions[-1][1] + dy
                            if 0 <= nx < board_size and 0 <= ny < board_size:
                                if grid[0, 0, ny, nx] == 1.0:  # Still empty
                                    possible_positions.append((nx, ny))
                                    
                        if possible_positions:
                            bx, by = random.choice(possible_positions)
                            opp_positions.append((bx, by))
                            grid[0, 0, by, bx] = 0.0  # Remove empty
                            grid[0, 4, by, bx] = 1.0  # Add opponent body
            
            # 4. Add food
            num_food = random.randint(1, 5)
            for _ in range(num_food):
                empty_positions = []
                for y in range(board_size):
                    for x in range(board_size):
                        if grid[0, 0, y, x] == 1.0:  # Empty space
                            empty_positions.append((x, y))
                
                if empty_positions:
                    food_x, food_y = random.choice(empty_positions)
                    grid[0, 0, food_y, food_x] = 0.0  # Remove empty
                    grid[0, 5, food_y, food_x] = 1.0  # Add food
            
            # 5. Generate features vector
            our_health = random.randint(20, 100) / 100.0
            our_length = len(body_positions)
            opponent_length = random.randint(3, 8)
            
            features = np.array([[
                our_health,  # health_ratio
                our_length / max(opponent_length, 1),  # length_ratio
                random.randint(1, 50) / 100.0,  # turn_ratio
                1.0 / (num_opponents + 1),  # snake_count_ratio
                head_x / board_size,  # head_x normalized
                head_y / board_size   # head_y normalized
            ]], dtype=np.float32)
            
            samples.append((grid, features))
            
            if (i + 1) % 200 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} samples")
        
        logger.info(f"Generated {len(samples)} sample board states")
        return samples
    
    def analyze_model_outputs(self, samples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """Analyze outputs from all models to understand prediction patterns."""
        logger.info("Analyzing model outputs...")
        
        analysis_results = {}
        
        for model_name, session in self.models.items():
            logger.info(f"Analyzing {model_name} model...")
            
            outputs = []
            input_names = [i.name for i in session.get_inputs()]
            output_names = [o.name for o in session.get_outputs()]
            
            for i, (grid, features) in enumerate(samples):
                try:
                    # Prepare inputs based on model requirements
                    inputs = {}
                    if 'grid' in input_names:
                        inputs['grid'] = grid
                    if 'features' in input_names:
                        inputs['features'] = features
                    
                    # Run inference
                    result = session.run(output_names, inputs)
                    outputs.append(result[0])  # First (usually only) output
                    
                    if (i + 1) % 200 == 0:
                        logger.info(f"  Processed {i + 1}/{len(samples)} samples")
                        
                except Exception as e:
                    logger.warning(f"Inference failed for sample {i}: {e}")
                    continue
            
            if outputs:
                outputs_array = np.array(outputs)
                analysis = self._analyze_output_distribution(model_name, outputs_array)
                analysis_results[model_name] = analysis
            else:
                logger.warning(f"No valid outputs for {model_name}")
        
        return analysis_results
    
    def _analyze_output_distribution(self, model_name: str, outputs: np.ndarray) -> Dict:
        """Analyze the distribution of outputs from a specific model."""
        analysis = {
            'model_name': model_name,
            'num_samples': len(outputs),
            'output_shape': outputs.shape,
            'statistics': {}
        }
        
        # Flatten outputs for basic statistics
        if outputs.ndim > 2:
            flat_outputs = outputs.reshape(len(outputs), -1)
        else:
            flat_outputs = outputs
        
        # Basic statistics
        analysis['statistics'] = {
            'mean': float(np.mean(flat_outputs)),
            'std': float(np.std(flat_outputs)),
            'min': float(np.min(flat_outputs)),
            'max': float(np.max(flat_outputs)),
            'median': float(np.median(flat_outputs))
        }
        
        # Model-specific analysis
        if model_name == 'move_prediction':
            # Analyze move probability distributions
            analysis['move_analysis'] = self._analyze_move_predictions(outputs)
        elif model_name == 'position_evaluation':
            # Analyze position evaluation scores
            analysis['position_analysis'] = self._analyze_position_scores(outputs)
        elif model_name == 'game_outcome':
            # Analyze outcome probabilities
            analysis['outcome_analysis'] = self._analyze_outcome_probabilities(outputs)
        
        # Calculate prediction entropy for confidence analysis
        analysis['entropy_analysis'] = self._calculate_entropy_stats(outputs)
        
        return analysis
    
    def _analyze_move_predictions(self, outputs: np.ndarray) -> Dict:
        """Analyze move prediction outputs (4-dimensional softmax probabilities)."""
        # Assuming outputs shape is (num_samples, 4) for [up, down, left, right]
        if outputs.shape[1] != 4:
            return {'error': f'Expected 4 move probabilities, got {outputs.shape[1]}'}
        
        move_names = ['up', 'down', 'left', 'right']
        
        analysis = {
            'move_statistics': {},
            'prediction_certainty': {},
            'move_preferences': {}
        }
        
        # Per-move statistics
        for i, move in enumerate(move_names):
            move_probs = outputs[:, i]
            analysis['move_statistics'][move] = {
                'mean_probability': float(np.mean(move_probs)),
                'std_probability': float(np.std(move_probs)),
                'min_probability': float(np.min(move_probs)),
                'max_probability': float(np.max(move_probs))
            }
        
        # Prediction certainty analysis
        max_probs = np.max(outputs, axis=1)
        analysis['prediction_certainty'] = {
            'mean_max_probability': float(np.mean(max_probs)),
            'std_max_probability': float(np.std(max_probs)),
            'high_certainty_samples': int(np.sum(max_probs > 0.8)),  # >80% confidence
            'medium_certainty_samples': int(np.sum((max_probs > 0.5) & (max_probs <= 0.8))),
            'low_certainty_samples': int(np.sum(max_probs <= 0.5))
        }
        
        # Move preferences (which moves are predicted most often)
        preferred_moves = np.argmax(outputs, axis=1)
        for i, move in enumerate(move_names):
            analysis['move_preferences'][move] = int(np.sum(preferred_moves == i))
        
        return analysis
    
    def _analyze_position_scores(self, outputs: np.ndarray) -> Dict:
        """Analyze position evaluation scores."""
        scores = outputs.flatten()
        
        return {
            'score_distribution': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores)),
                'q25': float(np.percentile(scores, 25)),
                'q75': float(np.percentile(scores, 75))
            },
            'score_categories': {
                'very_positive': int(np.sum(scores > 0.5)),
                'positive': int(np.sum((scores > 0.1) & (scores <= 0.5))),
                'neutral': int(np.sum((scores >= -0.1) & (scores <= 0.1))),
                'negative': int(np.sum((scores < -0.1) & (scores >= -0.5))),
                'very_negative': int(np.sum(scores < -0.5))
            }
        }
    
    def _analyze_outcome_probabilities(self, outputs: np.ndarray) -> Dict:
        """Analyze game outcome probabilities."""
        probs = outputs.flatten()
        
        return {
            'probability_distribution': {
                'mean': float(np.mean(probs)),
                'std': float(np.std(probs)),
                'min': float(np.min(probs)),
                'max': float(np.max(probs)),
                'median': float(np.median(probs))
            },
            'confidence_levels': {
                'very_confident_win': int(np.sum(probs > 0.8)),
                'confident_win': int(np.sum((probs > 0.6) & (probs <= 0.8))),
                'uncertain': int(np.sum((probs >= 0.4) & (probs <= 0.6))),
                'confident_loss': int(np.sum((probs < 0.4) & (probs >= 0.2))),
                'very_confident_loss': int(np.sum(probs < 0.2))
            }
        }
    
    def _calculate_entropy_stats(self, outputs: np.ndarray) -> Dict:
        """Calculate entropy statistics for confidence analysis."""
        entropies = []
        
        for output in outputs:
            if len(output.shape) == 1 and len(output) > 1:
                # Multi-dimensional output (like move probabilities)
                # Normalize to probabilities if needed
                probs = output
                if np.sum(probs) > 0:
                    probs = probs / np.sum(probs)
                
                # Calculate entropy
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                entropies.append(entropy)
        
        if entropies:
            max_entropy = np.log(len(outputs[0]))  # Maximum possible entropy
            normalized_entropies = np.array(entropies) / max_entropy
            
            return {
                'mean_entropy': float(np.mean(entropies)),
                'std_entropy': float(np.std(entropies)),
                'mean_normalized_entropy': float(np.mean(normalized_entropies)),
                'low_entropy_samples': int(np.sum(normalized_entropies < 0.3)),  # High confidence
                'medium_entropy_samples': int(np.sum((normalized_entropies >= 0.3) & (normalized_entropies < 0.7))),
                'high_entropy_samples': int(np.sum(normalized_entropies >= 0.7))  # Low confidence
            }
        
        return {'error': 'Could not calculate entropy for this model'}
    
    def generate_confidence_insights(self, analysis_results: Dict) -> Dict:
        """Generate insights for confidence calculation system design."""
        logger.info("Generating confidence insights...")
        
        insights = {
            'summary': {},
            'confidence_patterns': {},
            'recommendations': {}
        }
        
        # Summary of model behaviors
        for model_name, analysis in analysis_results.items():
            insights['summary'][model_name] = {
                'output_range': f"{analysis['statistics']['min']:.3f} to {analysis['statistics']['max']:.3f}",
                'mean_output': f"{analysis['statistics']['mean']:.3f}",
                'output_variability': f"{analysis['statistics']['std']:.3f}"
            }
            
            if 'entropy_analysis' in analysis:
                entropy = analysis['entropy_analysis']
                if 'mean_normalized_entropy' in entropy:
                    insights['summary'][model_name]['mean_entropy'] = f"{entropy['mean_normalized_entropy']:.3f}"
        
        # Confidence calculation recommendations
        insights['recommendations'] = {
            'entropy_thresholds': {
                'high_confidence': 'entropy < 0.3 (low entropy = high confidence)',
                'medium_confidence': '0.3 <= entropy < 0.7',
                'low_confidence': 'entropy >= 0.7 (high entropy = low confidence)'
            },
            'probability_thresholds': {
                'description': 'Based on move prediction analysis',
                'high_confidence': 'max_probability > 0.8',
                'medium_confidence': '0.5 < max_probability <= 0.8',
                'low_confidence': 'max_probability <= 0.5'
            },
            'consistency_checks': {
                'position_outcome_alignment': 'High position scores should correlate with high win probabilities',
                'move_position_alignment': 'Strong move preferences should align with positive position evaluations'
            }
        }
        
        return insights
    
    def save_results(self, analysis_results: Dict, insights: Dict, filename: str = 'neural_output_analysis.json'):
        """Save analysis results to JSON file."""
        logger.info(f"Saving results to {filename}")
        
        output_data = {
            'analysis_timestamp': str(np.datetime64('now')),
            'models_analyzed': list(self.models.keys()),
            'analysis_results': analysis_results,
            'confidence_insights': insights
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Analysis results saved to {filename}")
    
    def run_analysis(self, num_samples: int = 1000) -> None:
        """Run the complete neural output analysis."""
        logger.info("Starting neural network output analysis...")
        
        # Load models
        if not self.load_models():
            logger.error("No models loaded. Cannot proceed with analysis.")
            return
        
        # Generate sample data
        samples = self.generate_sample_board_states(num_samples)
        
        if not samples:
            logger.error("No sample data generated. Cannot proceed with analysis.")
            return
        
        # Analyze model outputs
        analysis_results = self.analyze_model_outputs(samples)
        
        if not analysis_results:
            logger.error("No analysis results generated.")
            return
        
        # Generate insights
        insights = self.generate_confidence_insights(analysis_results)
        
        # Save results
        self.save_results(analysis_results, insights)
        
        # Print summary
        logger.info("Analysis Summary:")
        logger.info(f"Models analyzed: {list(analysis_results.keys())}")
        logger.info(f"Samples processed: {num_samples}")
        
        for model_name in analysis_results.keys():
            logger.info(f"\n{model_name.upper()} MODEL:")
            stats = analysis_results[model_name]['statistics']
            logger.info(f"  Output range: {stats['min']:.3f} to {stats['max']:.3f}")
            logger.info(f"  Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
            
            if 'entropy_analysis' in analysis_results[model_name]:
                entropy = analysis_results[model_name]['entropy_analysis']
                if 'mean_normalized_entropy' in entropy:
                    logger.info(f"  Mean normalized entropy: {entropy['mean_normalized_entropy']:.3f}")
        
        logger.info("\nAnalysis complete! Check neural_output_analysis.json for detailed results.")

def main():
    """Main function to run the neural output analysis."""
    analyzer = NeuralOutputAnalyzer()
    analyzer.run_analysis(num_samples=1000)

if __name__ == '__main__':
    main()