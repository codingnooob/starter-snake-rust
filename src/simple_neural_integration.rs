// Simple Neural Network Integration for Battlesnake
// Provides basic neural network evaluation as part of the decision pipeline

use crate::logic::{Board, Battlesnake, Coord};
use crate::logic::{Direction, SafetyChecker, ReachabilityAnalyzer, FoodSeeker};
use log::{info, warn, debug};
use anyhow::Result;
use ndarray::Array2;

/// Simple neural network evaluator that mimics neural network behavior
/// using simplified heuristics and pattern recognition
pub struct SimpleNeuralEvaluator {
    position_weights: PositionWeights,
    move_biases: MoveBiases,
    confidence_threshold: f32,
}

#[derive(Debug, Clone)]
struct PositionWeights {
    health_weight: f32,
    space_weight: f32,
    food_weight: f32,
    safety_weight: f32,
    opponent_awareness_weight: f32,
}

#[derive(Debug, Clone)]
struct MoveBiases {
    up_bias: f32,
    down_bias: f32,
    left_bias: f32,
    right_bias: f32,
}

impl Default for PositionWeights {
    fn default() -> Self {
        Self {
            health_weight: 0.15,
            space_weight: 0.25,
            food_weight: 0.20,
            safety_weight: 0.30,
            opponent_awareness_weight: 0.10,
        }
    }
}

impl Default for MoveBiases {
    fn default() -> Self {
        Self {
            up_bias: 0.0,    // NEUTRAL: No bias toward upward movement
            down_bias: 0.0,  // NEUTRAL: No bias toward downward movement
            left_bias: 0.0,  // NEUTRAL: No bias toward left movement
            right_bias: 0.0, // NEUTRAL: No bias toward right movement
        }
    }
}

impl SimpleNeuralEvaluator {
    pub fn new() -> Self {
        Self {
            position_weights: PositionWeights::default(),
            move_biases: MoveBiases::default(),
            confidence_threshold: 0.25,
        }
    }

    /// Main neural network evaluation function
    pub fn evaluate_board_state(&self, board: &Board, you: &Battlesnake) -> Result<NeuralEvaluationResult> {
        // Encode board state (similar to neural network input preparation)
        let input_features = self.encode_input_features(board, you)?;
        
        // Apply "neural network" layers (simplified linear transformations)
        let hidden_layer = self.apply_hidden_layer(&input_features)?;
        
        // Apply output layer to get position score and move probabilities
        let (position_score, move_probabilities) = self.apply_output_layer(&hidden_layer)?;
        
        // Calculate confidence based on evaluation strength
        let confidence = self.calculate_confidence(&input_features, &move_probabilities);
        
        let result = NeuralEvaluationResult {
            position_score,
            move_probabilities,
            confidence,
            evaluation_method: "NeuralNetwork".to_string(),
        };
        
        info!("NEURAL NETWORK: Position score: {:.3}, Confidence: {:.3}", 
              result.position_score, result.confidence);
        
        Ok(result)
    }

    /// Encode board state into features (simulating neural network input encoding)
    fn encode_input_features(&self, board: &Board, you: &Battlesnake) -> Result<Vec<f32>> {
        let mut features = Vec::new();
        
        // Health and length features
        features.push((you.health as f32) / 100.0);
        features.push((you.body.len() as f32) / 20.0);
        
        // Board complexity features
        features.push((board.snakes.len() as f32) / 8.0);
        features.push((board.food.len() as f32) / 10.0);
        
        // Position features
        features.push(you.head.x as f32 / board.width as f32);
        features.push(you.head.y as f32 / board.height as f32);
        
        // Safety features
        let safe_moves = SafetyChecker::calculate_safe_moves(you, board, &board.snakes);
        features.push(safe_moves.len() as f32 / 4.0);
        
        // Space accessibility
        let reachable_space = ReachabilityAnalyzer::count_reachable_spaces(&you.head, board, &board.snakes);
        features.push(reachable_space as f32 / 100.0);
        
        // Food proximity
        let nearest_food_distance = if !board.food.is_empty() {
            board.food.iter()
                .map(|food| ((you.head.x - food.x).abs() + (you.head.y - food.y).abs()) as f32)
                .fold(f32::INFINITY, f32::min)
        } else {
            10.0
        };
        features.push(1.0 / (nearest_food_distance + 1.0));
        
        // Opponent threat assessment
        let opponent_threat = self.assess_opponent_threat(you, board);
        features.push(opponent_threat);
        
        Ok(features)
    }

    /// Apply hidden layer transformations (simulating neural network hidden layers)
    fn apply_hidden_layer(&self, input_features: &[f32]) -> Result<Vec<f32>> {
        let mut hidden_layer = Vec::new();
        
        // Hidden neuron 1: Overall position strength
        let strength = input_features[0] * 0.4 + input_features[1] * 0.3 + input_features[7] * 0.3;
        hidden_layer.push(tanh_activation(strength));
        
        // Hidden neuron 2: Safety assessment
        let safety = input_features[6] * 0.5 + input_features[7] * 0.3 + (1.0 - input_features[8]) * 0.2;
        hidden_layer.push(tanh_activation(safety));
        
        // Hidden neuron 3: Strategic positioning
        let positioning = input_features[4] * 0.3 + input_features[5] * 0.3 + input_features[2] * 0.4;
        hidden_layer.push(tanh_activation(positioning));
        
        // Hidden neuron 4: Food and growth potential
        let growth = input_features[3] * 0.4 + input_features[8] * 0.4 + input_features[1] * 0.2;
        hidden_layer.push(tanh_activation(growth));
        
        Ok(hidden_layer)
    }

    /// Apply output layer to get final scores and move probabilities
    fn apply_output_layer(&self, hidden_layer: &[f32]) -> Result<(f32, Array2<f32>)> {
        // Position score from hidden layer
        let position_score = hidden_layer[0] * 0.5 + hidden_layer[1] * 0.3 + hidden_layer[2] * 0.2;
        
        // Calculate move probabilities
        let mut move_probs = Array2::zeros((1, 4));
        
        // Up move probability
        let up_score = hidden_layer[1] * 0.4 + hidden_layer[3] * 0.3 + self.move_biases.up_bias;
        move_probs[[0, 0]] = sigmoid(up_score);
        
        // Down move probability  
        let down_score = hidden_layer[0] * 0.3 + hidden_layer[2] * 0.4 + self.move_biases.down_bias;
        move_probs[[0, 1]] = sigmoid(down_score);
        
        // Left move probability
        let left_score = hidden_layer[1] * 0.3 + hidden_layer[3] * 0.3 + self.move_biases.left_bias;
        move_probs[[0, 2]] = sigmoid(left_score);
        
        // Right move probability
        let right_score = hidden_layer[1] * 0.3 + hidden_layer[3] * 0.3 + self.move_biases.right_bias;
        move_probs[[0, 3]] = sigmoid(right_score);
        
        // Normalize probabilities
        let sum: f32 = move_probs.iter().sum();
        if sum > 0.0 {
            for prob in move_probs.iter_mut() {
                *prob /= sum;
            }
        }
        
        Ok((position_score, move_probs))
    }

    /// Calculate confidence in the neural network evaluation
    fn calculate_confidence(&self, input_features: &[f32], move_probs: &Array2<f32>) -> f32 {
        // Confidence based on input feature consistency and prediction certainty
        let feature_consistency = input_features.iter()
            .map(|&f| if f >= 0.0 && f <= 1.0 { 1.0 } else { 0.0 })
            .sum::<f32>() / input_features.len() as f32;
        
        // Prediction certainty (how clear the best move is)
        let max_prob = move_probs.iter().fold(0.0, f32::max);
        let prediction_certainty = (max_prob - 0.25) * 2.0; // 0.25 is expected random chance
        
        (feature_consistency + prediction_certainty * 0.5).clamp(0.0, 1.0)
    }

    /// Assess threat from opponents
    fn assess_opponent_threat(&self, you: &Battlesnake, board: &Board) -> f32 {
        let mut threat_score = 0.0;
        
        for opponent in &board.snakes {
            if opponent.id != you.id {
                // Check if opponent is close and might be threatening
                let distance = ((you.head.x - opponent.head.x).abs() + (you.head.y - opponent.head.y).abs()) as f32;
                let threat_proximity = if distance < 5.0 { 1.0 / distance } else { 0.0 };
                
                // Consider opponent size relative to ours
                let size_ratio = opponent.body.len() as f32 / you.body.len().max(1) as f32;
                let size_threat = if size_ratio > 1.0 { size_ratio - 1.0 } else { 0.0 };
                
                threat_score += threat_proximity * size_threat * 0.5;
            }
        }
        
        threat_score.clamp(0.0, 1.0)
    }

    /// Choose best move based on neural network evaluation
    pub fn choose_best_move(&self, board: &Board, you: &Battlesnake, available_moves: &[Direction]) -> Result<Direction> {
        let eval_result = self.evaluate_board_state(board, you)?;
        
        if eval_result.confidence >= self.confidence_threshold {
            // Use neural network recommendation
            let mut best_move = Direction::Up;
            let mut best_score = f32::NEG_INFINITY;
            
            for (i, mv) in [Direction::Up, Direction::Down, Direction::Left, Direction::Right].iter().enumerate() {
                let score = eval_result.move_probabilities[[0, i]];
                if available_moves.contains(mv) && score > best_score {
                    best_score = score;
                    best_move = *mv;
                }
            }
            
            info!("NEURAL NETWORK: Selected move {:?} with confidence {:.3}", best_move, eval_result.confidence);
            Ok(best_move)
        } else {
            // Fallback to heuristic if confidence too low
            warn!("NEURAL NETWORK: Confidence too low ({:.3}), falling back to heuristic", eval_result.confidence);
            self.fallback_move_selection(board, you, available_moves)
        }
    }

    /// Fallback move selection when neural network confidence is low
    fn fallback_move_selection(&self, board: &Board, you: &Battlesnake, available_moves: &[Direction]) -> Result<Direction> {
        let mut best_move = available_moves[0];
        let mut best_score = f32::NEG_INFINITY;
        
        for mv in available_moves {
            let next_pos = you.head.apply_direction(mv);
            
            // Safety score
            let safety_score = if SafetyChecker::is_safe_coordinate(&next_pos, board, &board.snakes) { 1.0 } else { 0.0 };
            
            // Food bonus
            let food_bonus = if !board.food.is_empty() {
                board.food.iter()
                    .map(|food| {
                        let current_distance = ((you.head.x - food.x).abs() + (you.head.y - food.y).abs()) as f32;
                        let next_distance = ((next_pos.x - food.x).abs() + (next_pos.y - food.y).abs()) as f32;
                        if next_distance < current_distance { 0.5 } else { 0.0 }
                    })
                    .sum::<f32>()
            } else { 0.0 };
            
            // Space bonus
            let space_bonus = {
                let reachable_spaces = ReachabilityAnalyzer::count_reachable_spaces(&next_pos, board, &board.snakes);
                reachable_spaces as f32 / 100.0
            };
            
            let total_score = safety_score * 0.6 + food_bonus * 0.2 + space_bonus * 0.2;
            
            if total_score > best_score {
                best_score = total_score;
                best_move = *mv;
            }
        }
        
        Ok(best_move)
    }
}

#[derive(Debug, Clone)]
pub struct NeuralEvaluationResult {
    pub position_score: f32,
    pub move_probabilities: Array2<f32>,
    pub confidence: f32,
    pub evaluation_method: String,
}

// Activation functions (simulating neural network activations)
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh_activation(x: f32) -> f32 {
    x.tanh()
}

impl Default for SimpleNeuralEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_evaluator_creation() {
        let evaluator = SimpleNeuralEvaluator::new();
        assert!(evaluator.confidence_threshold > 0.0);
    }

    #[test]
    fn test_activation_functions() {
        assert!(sigmoid(0.0) > 0.0);
        assert!(sigmoid(0.0) < 1.0);
        assert_eq!(tanh_activation(0.0), 0.0);
        assert!(tanh_activation(1.0).abs() <= 1.0);
    }

    #[test]
    fn test_confidence_calculation() {
        let evaluator = SimpleNeuralEvaluator::new();
        let dummy_features = vec![0.5, 0.6, 0.7, 0.8, 0.3, 0.4, 0.9, 0.7, 0.2, 0.1];
        let dummy_probs = Array2::from_elem((1, 4), 0.25);
        
        let confidence = evaluator.calculate_confidence(&dummy_features, &dummy_probs);
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }
}