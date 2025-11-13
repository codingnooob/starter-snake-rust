#!/usr/bin/env python3
"""
Simplified Neural Network Output Analysis Tool

Analyzes the current ONNX models using the known input format: (batch_size, 8, 11, 11)
"""

import numpy as np
import onnxruntime as ort
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
        
    def load_models(self) -> bool:
        """Load all ONNX models for analysis."""
        logger.info("Loading ONNX models for analysis...")
        
        for name, path in self.model_paths.items():
            try:
                if Path(path).exists():
                    session = ort.InferenceSession(path)
                    self.models[name] = session
                    logger.info(f"Loaded {name} model from {path}")
                else:
                    logger.warning(f"Model not found: {path}")
                    
            except Exception as e:
                logger.error(f"Failed to load {name} model: {e}")
                
        return len(self.models) > 0
    
    def generate_sample_board_states(self, num_samples: int = 1000) -> List[np.ndarray]:
        """Generate sample board states with known format: (1, 8, 11, 11)"""
        logger.info(f"Generating {num_samples} sample board states...")
        
        samples = []
        
        for i in range(num_samples):
            # Create board state with shape (1, 8, 11, 11)
            # Channels: [EMPTY, OWN_HEAD, OWN_BODY, OPPONENT_HEAD, OPPONENT_BODY, FOOD, WALL, OTHER]
            board_state = np.random.rand(1, 8, 11, 11).astype(np.float32)
            
            # Make it more realistic by ensuring only one channel is active per cell
            for batch in range(1):
                for h in range(11):
                    for w in range(11):
                        # Select random channel to be active
                        active_channel = np.random.randint(0, 8)
                        board_state[batch, :, h, w] = 0.0
                        board_state[batch, active_channel, h, w] = 1.0
            
            samples.append(board_state)
            
            if (i + 1) % 200 == 0:
                logger.info(f"Generated {i + 1}/{num_samples} samples")
        
        logger.info(f"Generated {len(samples)} sample board states")
        return samples
    
    def analyze_model_outputs(self, samples: List[np.ndarray]) -> Dict:
        """Analyze outputs from all models to understand prediction patterns."""
        logger.info("Analyzing model outputs...")
        
        analysis_results = {}
        
        for model_name, session in self.models.items():
            logger.info(f"Analyzing {model_name} model...")
            
            outputs = []
            
            for i, board_state in enumerate(samples):
                try:
                    # Create input for the model
                    inputs = {'board_state': board_state}
                    
                    # Run inference
                    result = session.run(['prediction'], inputs)
                    if result:
                        outputs.append(result[0])  # First (usually only) output
                    
                    if (i + 1) % 200 == 0:
                        logger.info(f"  Processed {i + 1}/{len(samples)} samples")
                        
                except Exception as e:
                    logger.warning(f"Inference failed for sample {i} on {model_name}: {e}")
                    continue
            
            if outputs:
                outputs_array = np.array(outputs)
                logger.info(f"  Collected {len(outputs)} valid outputs with shape {outputs_array.shape}")
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
        
        # Basic statistics
        flat_outputs = outputs.reshape(len(outputs), -1)
        analysis['statistics'] = {
            'mean': float(np.mean(flat_outputs)),
            'std': float(np.std(flat_outputs)),
            'min': float(np.min(flat_outputs)),
            'max': float(np.max(flat_outputs)),
            'median': float(np.median(flat_outputs))
        }
        
        # Model-specific analysis
        if 'move_prediction' in model_name:
            analysis['move_analysis'] = self._analyze_move_predictions(outputs)
        elif 'position_evaluation' in model_name:
            analysis['position_analysis'] = self._analyze_position_scores(outputs)
        elif 'game_outcome' in model_name:
            analysis['outcome_analysis'] = self._analyze_outcome_probabilities(outputs)
        
        # Calculate prediction entropy/confidence metrics
        analysis['confidence_analysis'] = self._analyze_confidence_patterns(outputs)
        
        return analysis
    
    def _analyze_move_predictions(self, outputs: np.ndarray) -> Dict:
        """Analyze move prediction outputs (expected shape: [N, 1, 4])"""
        # Handle potential extra dimension
        if len(outputs.shape) == 3 and outputs.shape[1] == 1:
            outputs = outputs.squeeze(1)  # Remove middle dimension
        
        if outputs.shape[1] == 4:
            move_names = ['up', 'down', 'left', 'right']
            
            analysis = {
                'move_statistics': {},
                'prediction_certainty': {},
                'move_preferences': {},
                'softmax_analysis': {}
            }
            
            # Apply softmax to get probabilities
            exp_outputs = np.exp(outputs)
            softmax_probs = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
            
            # Per-move statistics (raw outputs)
            for i, move in enumerate(move_names):
                move_vals = outputs[:, i]
                analysis['move_statistics'][move] = {
                    'mean_raw_output': float(np.mean(move_vals)),
                    'std_raw_output': float(np.std(move_vals)),
                    'min_raw_output': float(np.min(move_vals)),
                    'max_raw_output': float(np.max(move_vals))
                }
            
            # Softmax probability analysis
            for i, move in enumerate(move_names):
                move_probs = softmax_probs[:, i]
                analysis['softmax_analysis'][move] = {
                    'mean_probability': float(np.mean(move_probs)),
                    'std_probability': float(np.std(move_probs)),
                    'min_probability': float(np.min(move_probs)),
                    'max_probability': float(np.max(move_probs))
                }
            
            # Prediction certainty analysis
            max_probs = np.max(softmax_probs, axis=1)
            analysis['prediction_certainty'] = {
                'mean_max_probability': float(np.mean(max_probs)),
                'std_max_probability': float(np.std(max_probs)),
                'high_certainty_samples': int(np.sum(max_probs > 0.8)),
                'medium_certainty_samples': int(np.sum((max_probs > 0.5) & (max_probs <= 0.8))),
                'low_certainty_samples': int(np.sum(max_probs <= 0.5)),
                'uniform_baseline': 0.25  # Random baseline for 4 moves
            }
            
            # Move preferences
            preferred_moves = np.argmax(softmax_probs, axis=1)
            for i, move in enumerate(move_names):
                analysis['move_preferences'][move] = int(np.sum(preferred_moves == i))
            
            return analysis
        else:
            return {'error': f'Unexpected output shape for move predictions: {outputs.shape}'}
    
    def _analyze_position_scores(self, outputs: np.ndarray) -> Dict:
        """Analyze position evaluation scores (expected shape: [N, 1, 1])"""
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
                'very_positive': int(np.sum(scores > 0.7)),
                'positive': int(np.sum((scores > 0.3) & (scores <= 0.7))),
                'neutral': int(np.sum((scores >= -0.3) & (scores <= 0.3))),
                'negative': int(np.sum((scores < -0.3) & (scores >= -0.7))),
                'very_negative': int(np.sum(scores < -0.7))
            },
            'confidence_proxy': {
                'mean_abs_score': float(np.mean(np.abs(scores))),
                'extreme_scores': int(np.sum(np.abs(scores) > 0.5)),
                'neutral_scores': int(np.sum(np.abs(scores) < 0.1))
            }
        }
    
    def _analyze_outcome_probabilities(self, outputs: np.ndarray) -> Dict:
        """Analyze game outcome probabilities (expected shape: [N, 1, 1])"""
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
            },
            'confidence_proxy': {
                'mean_distance_from_neutral': float(np.mean(np.abs(probs - 0.5))),
                'extreme_predictions': int(np.sum((probs < 0.2) | (probs > 0.8))),
                'neutral_predictions': int(np.sum((probs >= 0.45) & (probs <= 0.55)))
            }
        }
    
    def _analyze_confidence_patterns(self, outputs: np.ndarray) -> Dict:
        """Analyze patterns that could inform confidence calculation."""
        try:
            if len(outputs.shape) >= 2 and outputs.shape[-1] > 1:
                # Multi-dimensional output - calculate entropy
                if len(outputs.shape) == 3 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)  # Remove middle dimension
                
                # Apply softmax to get probabilities
                exp_outputs = np.exp(outputs)
                probs = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
                
                # Calculate entropy
                entropies = -np.sum(probs * np.log(probs + 1e-8), axis=1)
                max_entropy = np.log(outputs.shape[1])  # Max possible entropy
                normalized_entropies = entropies / max_entropy
                
                return {
                    'entropy_statistics': {
                        'mean_entropy': float(np.mean(entropies)),
                        'std_entropy': float(np.std(entropies)),
                        'mean_normalized_entropy': float(np.mean(normalized_entropies)),
                        'min_entropy': float(np.min(entropies)),
                        'max_entropy': float(np.max(entropies))
                    },
                    'confidence_distribution': {
                        'high_confidence': int(np.sum(normalized_entropies < 0.3)),  # Low entropy
                        'medium_confidence': int(np.sum((normalized_entropies >= 0.3) & (normalized_entropies < 0.7))),
                        'low_confidence': int(np.sum(normalized_entropies >= 0.7))   # High entropy
                    },
                    'max_probability_stats': {
                        'mean_max_prob': float(np.mean(np.max(probs, axis=1))),
                        'std_max_prob': float(np.std(np.max(probs, axis=1)))
                    }
                }
            else:
                # Single-value outputs - use absolute deviation from neutral
                values = outputs.flatten()
                
                # Assume outputs are in [0,1] range or [-1,1] range
                if np.min(values) >= 0:
                    # [0,1] range - neutral is 0.5
                    deviations = np.abs(values - 0.5)
                    neutral_point = 0.5
                else:
                    # [-1,1] range - neutral is 0.0
                    deviations = np.abs(values)
                    neutral_point = 0.0
                
                return {
                    'deviation_statistics': {
                        'mean_deviation': float(np.mean(deviations)),
                        'std_deviation': float(np.std(deviations)),
                        'max_deviation': float(np.max(deviations)),
                        'neutral_point': neutral_point
                    },
                    'confidence_distribution': {
                        'high_confidence': int(np.sum(deviations > 0.4)),     # Far from neutral
                        'medium_confidence': int(np.sum((deviations >= 0.2) & (deviations <= 0.4))),
                        'low_confidence': int(np.sum(deviations < 0.2))      # Close to neutral
                    }
                }
        except Exception as e:
            return {'error': f'Confidence analysis failed: {e}'}
    
    def generate_confidence_insights(self, analysis_results: Dict) -> Dict:
        """Generate insights for the new confidence calculation system."""
        insights = {
            'summary': {},
            'confidence_recommendations': {},
            'threshold_suggestions': {}
        }
        
        # Summarize each model
        for model_name, analysis in analysis_results.items():
            model_summary = {
                'samples_analyzed': analysis['num_samples'],
                'output_shape': str(analysis['output_shape']),
                'value_range': f"{analysis['statistics']['min']:.3f} to {analysis['statistics']['max']:.3f}",
                'mean_output': f"{analysis['statistics']['mean']:.3f} ± {analysis['statistics']['std']:.3f}"
            }
            
            # Add model-specific insights
            if 'confidence_analysis' in analysis:
                conf_analysis = analysis['confidence_analysis']
                if 'entropy_statistics' in conf_analysis:
                    model_summary['entropy_method'] = 'Available'
                    model_summary['mean_normalized_entropy'] = f"{conf_analysis['entropy_statistics']['mean_normalized_entropy']:.3f}"
                elif 'deviation_statistics' in conf_analysis:
                    model_summary['confidence_method'] = 'Deviation from neutral'
                    model_summary['mean_deviation'] = f"{conf_analysis['deviation_statistics']['mean_deviation']:.3f}"
            
            insights['summary'][model_name] = model_summary
        
        # Generate recommendations for confidence calculation
        insights['confidence_recommendations'] = {
            'entropy_based': {
                'description': 'For multi-output models (move prediction)',
                'formula': 'confidence = 1.0 - (entropy / max_entropy)',
                'high_threshold': 'entropy < 0.3 * max_entropy',
                'medium_threshold': '0.3 * max_entropy <= entropy < 0.7 * max_entropy',
                'low_threshold': 'entropy >= 0.7 * max_entropy'
            },
            'deviation_based': {
                'description': 'For single-output models (position, outcome)',
                'formula': 'confidence = abs(value - neutral_point) / max_possible_deviation',
                'high_threshold': 'deviation > 0.4',
                'medium_threshold': '0.2 <= deviation <= 0.4',
                'low_threshold': 'deviation < 0.2'
            },
            'max_probability_based': {
                'description': 'Alternative for discrete prediction models',
                'formula': 'confidence = max(probabilities)',
                'high_threshold': 'max_prob > 0.8',
                'medium_threshold': '0.5 < max_prob <= 0.8',
                'low_threshold': 'max_prob <= 0.5'
            }
        }
        
        # Suggest specific thresholds based on analysis
        if 'move_prediction' in analysis_results:
            move_analysis = analysis_results['move_prediction']
            if 'prediction_certainty' in move_analysis:
                certainty = move_analysis['prediction_certainty']
                mean_max_prob = certainty['mean_max_probability']
                
                insights['threshold_suggestions']['move_prediction'] = {
                    'empirical_high_threshold': f"max_probability > {mean_max_prob + 0.1:.2f}",
                    'empirical_medium_threshold': f"{mean_max_prob - 0.1:.2f} <= max_probability <= {mean_max_prob + 0.1:.2f}",
                    'empirical_low_threshold': f"max_probability < {mean_max_prob - 0.1:.2f}",
                    'baseline_random': certainty['uniform_baseline']
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
        logger.info("Starting simplified neural network output analysis...")
        
        # Load models
        if not self.load_models():
            logger.error("No models loaded. Cannot proceed with analysis.")
            return
        
        # Generate sample data
        samples = self.generate_sample_board_states(num_samples)
        
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
        logger.info("\n" + "="*60)
        logger.info("NEURAL NETWORK OUTPUT ANALYSIS SUMMARY")
        logger.info("="*60)
        
        for model_name, results in analysis_results.items():
            logger.info(f"\n{model_name.upper()} MODEL:")
            logger.info(f"  Samples processed: {results['num_samples']}")
            logger.info(f"  Output shape: {results['output_shape']}")
            stats = results['statistics']
            logger.info(f"  Value range: {stats['min']:.3f} to {stats['max']:.3f}")
            logger.info(f"  Mean ± Std: {stats['mean']:.3f} ± {stats['std']:.3f}")
            
            # Model-specific insights
            if 'move_analysis' in results and 'prediction_certainty' in results['move_analysis']:
                cert = results['move_analysis']['prediction_certainty']
                logger.info(f"  Average max probability: {cert['mean_max_probability']:.3f}")
                logger.info(f"  High certainty samples: {cert['high_certainty_samples']} ({100*cert['high_certainty_samples']/results['num_samples']:.1f}%)")
            
            if 'confidence_analysis' in results:
                conf = results['confidence_analysis']
                if 'entropy_statistics' in conf:
                    logger.info(f"  Mean normalized entropy: {conf['entropy_statistics']['mean_normalized_entropy']:.3f}")
                elif 'deviation_statistics' in conf:
                    logger.info(f"  Mean deviation from neutral: {conf['deviation_statistics']['mean_deviation']:.3f}")
        
        logger.info(f"\nDetailed analysis saved to: neural_output_analysis.json")
        logger.info("="*60)

def main():
    """Main function to run the neural output analysis."""
    analyzer = NeuralOutputAnalyzer()
    analyzer.run_analysis(num_samples=1000)

if __name__ == '__main__':
    main()