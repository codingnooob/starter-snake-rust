#!/usr/bin/env python3
"""
Fixed Neural Network Output Analysis Tool

Analyzes the current ONNX models' output distributions using the correct input format
that matches what the models actually expect.
"""

import numpy as np
import onnx
import onnxruntime as ort
import json
import logging
from pathlib import Path
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
                    input_shapes = [i.shape for i in session.get_inputs()]
                    output_shapes = [o.shape for o in session.get_outputs()]
                    logger.info(f"  Inputs: {inputs} with shapes: {input_shapes}")
                    logger.info(f"  Outputs: {outputs} with shapes: {output_shapes}")
                else:
                    logger.warning(f"Model not found: {path}")
                    
            except Exception as e:
                logger.error(f"Failed to load {name} model: {e}")
                
        return len(self.models) > 0
    
    def generate_sample_board_states(self, num_samples: int = 1000) -> List[np.ndarray]:
        """Generate sample board states in the format the models expect."""
        logger.info(f"Generating {num_samples} sample board states...")
        
        samples = []
        
        # First, let's inspect what the models actually expect
        if self.models:
            sample_model = list(self.models.values())[0]
            input_info = sample_model.get_inputs()[0]
            logger.info(f"Model input info: name='{input_info.name}', shape={input_info.shape}, type={input_info.type}")
            
            # Try to create the right input format
            if input_info.shape:
                # Use the actual expected shape from the model
                expected_shape = [dim if dim > 0 else 1 for dim in input_info.shape]
                logger.info(f"Using input shape: {expected_shape}")
                
                for i in range(num_samples):
                    # Create a board state matching expected input shape
                    if len(expected_shape) == 4:  # (batch, channels, height, width)
                        batch, channels, height, width = expected_shape
                        board_state = np.random.rand(batch, channels, height, width).astype(np.float32)
                    elif len(expected_shape) == 3:  # (channels, height, width) 
                        channels, height, width = expected_shape
                        board_state = np.random.rand(1, channels, height, width).astype(np.float32)
                    elif len(expected_shape) == 2:  # (batch, features)
                        batch, features = expected_shape
                        board_state = np.random.rand(batch, features).astype(np.float32)
                    else:
                        # Default fallback
                        board_state = np.random.rand(1, 400).astype(np.float32)
                    
                    samples.append(board_state)
                    
                    if (i + 1) % 200 == 0:
                        logger.info(f"Generated {i + 1}/{num_samples} samples")
            else:
                logger.warning("Could not determine input shape from model")
                # Create default samples
                for i in range(num_samples):
                    board_state = np.random.rand(1, 400).astype(np.float32)
                    samples.append(board_state)
        
        logger.info(f"Generated {len(samples)} sample board states")
        return samples
    
    def analyze_model_outputs(self, samples: List[np.ndarray]) -> Dict:
        """Analyze outputs from all models to understand prediction patterns."""
        logger.info("Analyzing model outputs...")
        
        analysis_results = {}
        
        for model_name, session in self.models.items():
            logger.info(f"Analyzing {model_name} model...")
            
            outputs = []
            input_names = [i.name for i in session.get_inputs()]
            output_names = [o.name for o in session.get_outputs()]
            
            logger.info(f"  Model expects inputs: {input_names}")
            logger.info(f"  Model provides outputs: {output_names}")
            
            for i, board_state in enumerate(samples):
                try:
                    # Create inputs matching what this specific model expects
                    inputs = {}
                    if len(input_names) == 1:
                        # Single input model
                        inputs[input_names[0]] = board_state
                    else:
                        # Multiple input model (shouldn't happen with current models)
                        logger.warning(f"Unexpected multiple inputs for {model_name}: {input_names}")
                        inputs[input_names[0]] = board_state
                    
                    # Run inference
                    result = session.run(output_names, inputs)
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
        logger.info(f"Analyzing output distribution for {model_name}")
        logger.info(f"Output array shape: {outputs.shape}")
        
        analysis = {
            'model_name': model_name,
            'num_samples': len(outputs),
            'output_shape': outputs.shape,
            'statistics': {}
        }
        
        # Flatten outputs for basic statistics if multidimensional
        if len(outputs.shape) > 1:
            flat_outputs = outputs.reshape(len(outputs), -1)
        else:
            flat_outputs = outputs.reshape(-1, 1)
        
        # Basic statistics
        analysis['statistics'] = {
            'mean': float(np.mean(flat_outputs)),
            'std': float(np.std(flat_outputs)),
            'min': float(np.min(flat_outputs)),
            'max': float(np.max(flat_outputs)),
            'median': float(np.median(flat_outputs)),
            'shape_info': f"Raw: {outputs.shape}, Flattened: {flat_outputs.shape}"
        }
        
        # Model-specific analysis
        if 'move_prediction' in model_name:
            analysis['move_analysis'] = self._analyze_move_predictions(outputs)
        elif 'position_evaluation' in model_name:
            analysis['position_analysis'] = self._analyze_position_scores(outputs)
        elif 'game_outcome' in model_name:
            analysis['outcome_analysis'] = self._analyze_outcome_probabilities(outputs)
        
        # Calculate prediction entropy for confidence analysis
        analysis['entropy_analysis'] = self._calculate_entropy_stats(outputs)
        
        return analysis
    
    def _analyze_move_predictions(self, outputs: np.ndarray) -> Dict:
        """Analyze move prediction outputs."""
        logger.info(f"Analyzing move predictions with shape: {outputs.shape}")
        
        # Handle different possible output shapes
        if len(outputs.shape) == 2 and outputs.shape[1] == 4:
            # Standard (num_samples, 4) shape for move probabilities
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
            
            # Move preferences
            preferred_moves = np.argmax(outputs, axis=1)
            for i, move in enumerate(move_names):
                analysis['move_preferences'][move] = int(np.sum(preferred_moves == i))
            
            return analysis
        else:
            return {'error': f'Unexpected output shape for move predictions: {outputs.shape}'}
    
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
                'very_positive': int(np.sum(scores > 0.7)),
                'positive': int(np.sum((scores > 0.3) & (scores <= 0.7))),
                'neutral': int(np.sum((scores >= -0.3) & (scores <= 0.3))),
                'negative': int(np.sum((scores < -0.3) & (scores >= -0.7))),
                'very_negative': int(np.sum(scores < -0.7))
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
        
        try:
            if len(outputs.shape) == 2 and outputs.shape[1] > 1:
                # Multi-dimensional output (like move probabilities)
                for output in outputs:
                    # Ensure we have a valid probability distribution
                    if np.sum(output) > 0:
                        # Normalize to probabilities
                        probs = output / np.sum(output)
                        # Calculate entropy (add small epsilon to avoid log(0))
                        entropy = -np.sum(probs * np.log(probs + 1e-8))
                        entropies.append(entropy)
            elif len(outputs.shape) == 2 and outputs.shape[1] == 1:
                # Single-value outputs - can't calculate entropy directly
                # Instead, calculate "pseudo-entropy" based on confidence
                for output in outputs:
                    # For single values, use distance from extremes as proxy for uncertainty
                    val = output[0]
                    # Map [0,1] to entropy-like measure where 0.5 = maximum "entropy"
                    pseudo_entropy = 4 * val * (1 - val)  # Maximum at 0.5, minimum at 0 or 1
                    entropies.append(pseudo_entropy)
            else:
                # 1D outputs
                for output_val in outputs.flatten():
                    # Similar pseudo-entropy for scalar outputs
                    if 0 <= output_val <= 1:
                        pseudo_entropy = 4 * output_val * (1 - output_val)
                    else:
                        # For outputs outside [0,1], use absolute distance from 0
                        pseudo_entropy = min(abs(output_val), 2.0)  # Cap at 2.0
                    entropies.append(pseudo_entropy)
        
            if entropies:
                entropies = np.array(entropies)
                max_entropy = np.log(outputs.shape[1]) if len(outputs.shape) == 2 and outputs.shape[1] > 1 else 1.0
                normalized_entropies = entropies / max_entropy
                
                return {
                    'mean_entropy': float(np.mean(entropies)),
                    'std_entropy': float(np.std(entropies)),
                    'mean_normalized_entropy': float(np.mean(normalized_entropies)),
                    'min_entropy': float(np.min(entropies)),
                    'max_entropy': float(np.max(entropies)),
                    'low_entropy_samples': int(np.sum(normalized_entropies < 0.3)),  # High confidence
                    'medium_entropy_samples': int(np.sum((normalized_entropies >= 0.3) & (normalized_entropies < 0.7))),
                    'high_entropy_samples': int(np.sum(normalized_entropies >= 0.7)),  # Low confidence
                    'entropy_calculation_method': 'actual' if len(outputs.shape) == 2 and outputs.shape[1] > 1 else 'pseudo'
                }
            else:
                return {'error': 'Could not calculate entropy statistics'}
        
        except Exception as e:
            logger.warning(f"Error calculating entropy: {e}")
            return {'error': f'Entropy calculation failed: {e}'}
    
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
                'samples_processed': analysis['num_samples'],
                'output_shape': str(analysis['output_shape']),
                'output_range': f"{analysis['statistics']['min']:.3f} to {analysis['statistics']['max']:.3f}",
                'mean_output': f"{analysis['statistics']['mean']:.3f}",
                'output_variability': f"{analysis['statistics']['std']:.3f}"
            }
            
            if 'entropy_analysis' in analysis and 'mean_normalized_entropy' in analysis['entropy_analysis']:
                entropy = analysis['entropy_analysis']
                insights['summary'][model_name]['mean_entropy'] = f"{entropy['mean_normalized_entropy']:.3f}"
                insights['summary'][model_name]['entropy_method'] = entropy.get('entropy_calculation_method', 'unknown')
        
        # Model-specific patterns
        for model_name, analysis in analysis_results.items():
            model_patterns = {}
            
            if 'move_analysis' in analysis and 'prediction_certainty' in analysis['move_analysis']:
                certainty = analysis['move_analysis']['prediction_certainty']
                model_patterns['certainty_distribution'] = {
                    'high_certainty_fraction': certainty['high_certainty_samples'] / analysis['num_samples'],
                    'medium_certainty_fraction': certainty['medium_certainty_samples'] / analysis['num_samples'],
                    'low_certainty_fraction': certainty['low_certainty_samples'] / analysis['num_samples']
                }
            
            if 'entropy_analysis' in analysis and 'low_entropy_samples' in analysis['entropy_analysis']:
                entropy = analysis['entropy_analysis']
                total_samples = analysis['num_samples']
                model_patterns['entropy_distribution'] = {
                    'low_entropy_fraction': entropy['low_entropy_samples'] / total_samples,
                    'medium_entropy_fraction': entropy['medium_entropy_samples'] / total_samples,
                    'high_entropy_fraction': entropy['high_entropy_samples'] / total_samples
                }
            
            if model_patterns:
                insights['confidence_patterns'][model_name] = model_patterns
        
        # Recommendations for new confidence system
        insights['recommendations'] = {
            'entropy_based_confidence': {
                'description': 'Use normalized entropy for uncertainty measurement',
                'high_confidence_threshold': 'normalized_entropy < 0.3',
                'medium_confidence_threshold': '0.3 <= normalized_entropy < 0.7',
                'low_confidence_threshold': 'normalized_entropy >= 0.7'
            },
            'probability_based_confidence': {
                'description': 'Use max probability for discrete predictions',
                'high_confidence_threshold': 'max_probability > 0.8',
                'medium_confidence_threshold': '0.5 < max_probability <= 0.8',
                'low_confidence_threshold': 'max_probability <= 0.5'
            },
            'model_specific_insights': {}
        }
        
        # Add model-specific recommendations
        for model_name, analysis in analysis_results.items():
            model_rec = {}
            
            if 'move_analysis' in analysis:
                move_analysis = analysis['move_analysis']
                if 'prediction_certainty' in move_analysis:
                    model_rec['suggested_thresholds'] = {
                        'high_confidence': f"max_prob > {move_analysis['prediction_certainty']['mean_max_probability']:.2f}",
                        'uses_discrete_probabilities': True
                    }
            elif 'position_analysis' in analysis:
                model_rec['value_range'] = analysis['position_analysis']['score_distribution']
                model_rec['confidence_proxy'] = 'Use absolute distance from neutral (0.0) as confidence indicator'
            elif 'outcome_analysis' in analysis:
                model_rec['probability_range'] = analysis['outcome_analysis']['probability_distribution'] 
                model_rec['confidence_proxy'] = 'Use distance from 0.5 (uncertain) as confidence indicator'
            
            if model_rec:
                insights['recommendations']['model_specific_insights'][model_name] = model_rec
        
        return insights
    
    def save_results(self, analysis_results: Dict, insights: Dict, filename: str = 'neural_output_analysis.json'):
        """Save analysis results to JSON file."""
        logger.info(f"Saving results to {filename}")
        
        output_data = {
            'analysis_timestamp': str(np.datetime64('now')),
            'models_analyzed': list(self.models.keys()),
            'analysis_results': analysis_results,
            'confidence_insights': insights,
            'model_input_output_info': {}
        }
        
        # Add model metadata
        for model_name, session in self.models.items():
            input_info = session.get_inputs()[0] if session.get_inputs() else None
            output_info = session.get_outputs()[0] if session.get_outputs() else None
            
            output_data['model_input_output_info'][model_name] = {
                'input_name': input_info.name if input_info else 'unknown',
                'input_shape': input_info.shape if input_info else 'unknown',
                'output_name': output_info.name if output_info else 'unknown',
                'output_shape': output_info.shape if output_info else 'unknown'
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
        logger.info(f"Samples processed per model: {num_samples}")
        
        for model_name, results in analysis_results.items():
            logger.info(f"\n{model_name.upper()} MODEL:")
            stats = results['statistics']
            logger.info(f"  Valid outputs: {results['num_samples']}")
            logger.info(f"  Output shape: {results['output_shape']}")
            logger.info(f"  Value range: {stats['min']:.3f} to {stats['max']:.3f}")
            logger.info(f"  Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
            
            if 'entropy_analysis' in results and 'mean_normalized_entropy' in results['entropy_analysis']:
                entropy = results['entropy_analysis']
                logger.info(f"  Mean normalized entropy: {entropy['mean_normalized_entropy']:.3f}")
                logger.info(f"  Entropy calculation: {entropy.get('entropy_calculation_method', 'unknown')}")
        
        logger.info("\nAnalysis complete! Check neural_output_analysis.json for detailed results.")

def main():
    """Main function to run the neural output analysis."""
    analyzer = NeuralOutputAnalyzer()
    analyzer.run_analysis(num_samples=1000)

if __name__ == '__main__':
    main()