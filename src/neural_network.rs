// Neural Network Inference Module for Rust Battlesnake
// Phase 3: Simple Neural Network Integration
//
// This module provides simplified neural network interfaces
// for position evaluation and move prediction.

use ndarray::{Array2, Array3, s};
use anyhow::{Result, Context, anyhow};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use parking_lot::RwLock;
use log::{info, warn, error};

// Import required types from existing logic
use crate::main::{Board, Battlesnake, Coord};

// Neural network inference types
#[derive(Debug, Clone)]
pub struct NeuralNetworkInput {
    pub board_state: Array3<f32>, // (channels, height, width)
    pub features: Array2<f32>,    // (batch_size, feature_dim)
}

#[derive(Debug, Clone)]
pub struct NeuralNetworkOutput {
    pub position_score: Option<f32>,
    pub move_probabilities: Option<Array2<f32>>,
    pub win_probability: Option<f32>,
}

pub enum NeuralNetworkType {
    PositionEvaluation,
    MovePrediction,
    GameOutcome,
}

impl std::fmt::Display for NeuralNetworkType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NeuralNetworkType::PositionEvaluation => write!(f, "PositionEvaluation"),
            NeuralNetworkType::MovePrediction => write!(f, "MovePrediction"),
            NeuralNetworkType::GameOutcome => write!(f, "GameOutcome"),
        }
    }
}

/// ONNX Model Loader and Inference Engine
pub struct ONNXInferenceEngine {
    models: Arc<RwLock<HashMap<NeuralNetworkType, onnxruntime::Session>>>,
    metadata: Arc<RwLock<HashMap<NeuralNetworkType, ModelMetadata>>>,
}

#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub model_name: String,
    pub input_shape: (i64, i64, i64),
    pub feature_dim: usize,
    pub output_shape: Vec<i64>,
    pub provider: String,
    pub version: String,
}

impl ONNXInferenceEngine {
    /// Create new ONNX inference engine
    pub fn new() -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Load ONNX model from file
    pub fn load_model(
        &self,
        model_path: &str,
        model_type: NeuralNetworkType,
    ) -> Result<()> {
        info!("Loading ONNX model: {} for type: {}", model_path, model_type);
        
        // Create ONNX session
        let session = onnxruntime::Session::builder()
            .with_optimization_level(onnxruntime::OptimizationLevel::Basic)
            .commit_from_file(model_path)
            .map_err(|e| anyhow!("Failed to load ONNX model: {}", e))?;

        // Store session
        {
            let mut models = self.models.write();
            models.insert(model_type.clone(), session);
        }

        // Load and store metadata
        let metadata_path = format!("{}.json", model_path.trim_end_matches(".onnx"));
        let model_metadata = if std::path::Path::new(&metadata_path).exists() {
            self.load_metadata(&metadata_path)?
        } else {
            self.create_default_metadata(&model_type, &session)?
        };

        {
            let mut metadata = self.metadata.write();
            metadata.insert(model_type.clone(), model_metadata);
        }

        info!("Successfully loaded ONNX model: {} for type: {}", model_path, model_type);
        Ok(())
    }

    /// Load model metadata from JSON file
    fn load_metadata(&self, metadata_path: &str) -> Result<ModelMetadata> {
        let content = std::fs::read_to_string(metadata_path)
            .context(format!("Failed to read metadata file: {}", metadata_path))?;
        let json_data: serde_json::Value = serde_json::from_str(&content)?;
        
        Ok(ModelMetadata {
            model_name: json_data["model_name"].as_str().unwrap_or_default().to_string(),
            input_shape: (
                json_data["input_shape"][0].as_i64().unwrap_or(7),
                json_data["input_shape"][1].as_i64().unwrap_or(20),
                json_data["input_shape"][2].as_i64().unwrap_or(20),
            ),
            feature_dim: json_data["feature_dim"].as_u64().unwrap_or(6) as usize,
            output_shape: json_data["output_shape"].as_array()
                .and_then(|arr| arr.iter().map(|v| v.as_i64()).collect::<Option<Vec<_>>>())
                .unwrap_or_else(|| vec![1]),
            provider: json_data["onnx_version"].as_str().unwrap_or("unknown").to_string(),
            version: json_data["exported_at"].as_str().unwrap_or("unknown").to_string(),
        })
    }

    /// Create default metadata when JSON file doesn't exist
    fn create_default_metadata(&self, model_type: &NeuralNetworkType, session: &onnxruntime::Session) -> Result<ModelMetadata> {
        // Extract input/output information from ONNX model
        let input_info = session.input().first()
            .ok_or_else(|| anyhow!("No input information found in ONNX model"))?;
        
        let input_shape = if let Some(input_shape) = input_info.input_info.dims() {
            if input_shape.len() >= 4 {
                (input_shape[1], input_shape[2], input_shape[3]) // Remove batch dimension
            } else {
                (7, 20, 20) // Default
            }
        } else {
            (7, 20, 20) // Default
        };

        let output_shape = if let Some(output_info) = session.output().first() {
            if let Some(output_dims) = output_info.output_info.dims() {
                output_dims.iter().copied().collect()
            } else {
                vec![1]
            }
        } else {
            vec![1]
        };

        Ok(ModelMetadata {
            model_name: format!("{}_model", model_type),
            input_shape,
            feature_dim: 6, // Based on our encoding design
            output_shape,
            provider: "onnxruntime-rs".to_string(),
            version: "1.0.0".to_string(),
        })
    }

    /// Run inference on neural network
    pub fn run_inference(
        &self,
        model_type: &NeuralNetworkType,
        input: &NeuralNetworkInput,
    ) -> Result<NeuralNetworkOutput> {
        let models = self.models.read();
        let model = models.get(model_type)
            .ok_or_else(|| anyhow!("Model not loaded for type: {}", model_type))?;

        let metadata = self.metadata.read();
        let model_metadata = metadata.get(model_type)
            .ok_or_else(|| anyhow!("Metadata not found for model type: {}", model_type))?;

        // Prepare input tensors
        let grid_tensor = self.prepare_grid_tensor(input, model_metadata)?;
        let feature_tensor = self.prepare_feature_tensor(input, model_metadata)?;

        // Create input map
        let mut input_map = std::collections::HashMap::new();
        input_map.insert("grid".to_string(), grid_tensor);
        input_map.insert("features".to_string(), feature_tensor);

        // Run inference
        let outputs = model.run(input_map)
            .map_err(|e| anyhow!("ONNX inference failed: {}", e))?;

        let output_tensor = outputs.get("output")
            .ok_or_else(|| anyhow!("No output tensor found in ONNX model"))?;

        // Parse output based on model type
        let output = self.parse_output(model_type, output_tensor, model_metadata)?;

        info!("Successfully ran inference for model type: {}", model_type);
        Ok(output)
    }

    /// Prepare grid tensor for inference
    fn prepare_grid_tensor(
        &self,
        input: &NeuralNetworkInput,
        metadata: &ModelMetadata,
    ) -> Result<onnxruntime::Tensor> {
        // Ensure correct dimensions
        let (channels, height, width) = metadata.input_shape;
        let batch_size = 1;

        // Create tensor with correct shape
        let mut grid_data = vec![0.0f32; (batch_size * channels as usize * height as usize * width as usize) as usize];

        // Copy input data (assuming input.board_state is already in correct format)
        for c in 0..channels as usize {
            for h in 0..height as usize {
                for w in 0..width as usize {
                    let src_idx = (c * height as usize + h) * width as usize + w;
                    let dst_idx = (0 * channels as usize + c) * (height as usize * width as usize) + h * width as usize + w;
                    if src_idx < input.board_state.len() && dst_idx < grid_data.len() {
                        grid_data[dst_idx] = input.board_state[[c, h, w]];
                    }
                }
            }
        }

        onnxruntime::Tensor::new(
            &onnxruntime::DataType::FLOAT32,
            onnxruntime::Shape::new(vec![batch_size, channels, height, width]),
            &grid_data,
        ).map_err(|e| anyhow!("Failed to create grid tensor: {}", e))
    }

    /// Prepare feature tensor for inference
    fn prepare_feature_tensor(
        &self,
        input: &NeuralNetworkInput,
        metadata: &ModelMetadata,
    ) -> Result<onnxruntime::Tensor> {
        let batch_size = 1;
        let feature_dim = metadata.feature_dim;

        let mut feature_data = vec![0.0f32; batch_size * feature_dim];
        
        // Copy feature data
        for i in 0..feature_dim {
            if i < input.features.len() {
                feature_data[i] = input.features[[0, i]];
            }
        }

        onnxruntime::Tensor::new(
            &onnxruntime::DataType::FLOAT32,
            onnxruntime::Shape::new(vec![batch_size, feature_dim as i64]),
            &feature_data,
        ).map_err(|e| anyhow!("Failed to create feature tensor: {}", e))
    }

    /// Parse ONNX output tensor based on model type
    fn parse_output(
        &self,
        model_type: &NeuralNetworkType,
        output_tensor: &onnxruntime::Tensor,
        metadata: &ModelMetadata,
    ) -> Result<NeuralNetworkOutput> {
        let output_data = output_tensor.data::<f32>()
            .map_err(|e| anyhow!("Failed to extract float data from tensor: {}", e))?;

        match model_type {
            NeuralNetworkType::PositionEvaluation => {
                let score = if !output_data.is_empty() {
                    output_data[0]
                } else {
                    0.0
                };
                
                Ok(NeuralNetworkOutput {
                    position_score: Some(score),
                    move_probabilities: None,
                    win_probability: None,
                })
            }
            NeuralNetworkType::MovePrediction => {
                // Output should be 4 move probabilities
                let mut probs = Array2::zeros((1, 4));
                for i in 0..4.min(output_data.len()) {
                    probs[[0, i]] = output_data[i];
                }
                
                Ok(NeuralNetworkOutput {
                    position_score: None,
                    move_probabilities: Some(probs),
                    win_probability: None,
                })
            }
            NeuralNetworkType::GameOutcome => {
                let win_prob = if !output_data.is_empty() {
                    output_data[0]
                } else {
                    0.5
                };
                
                Ok(NeuralNetworkOutput {
                    position_score: None,
                    move_probabilities: None,
                    win_probability: Some(win_prob),
                })
            }
        }
    }

    /// Check if model is loaded
    pub fn is_model_loaded(&self, model_type: &NeuralNetworkType) -> bool {
        let models = self.models.read();
        models.contains_key(model_type)
    }

    /// Get model metadata
    pub fn get_model_metadata(&self, model_type: &NeuralNetworkType) -> Option<ModelMetadata> {
        let metadata = self.metadata.read();
        metadata.get(model_type).cloned()
    }
}

/// Neural Network Board State Encoder
/// Converts Battlesnake board state to neural network input format
pub struct BoardStateEncoder {
    max_board_size: usize,
    num_channels: usize,
}

impl BoardStateEncoder {
    /// Create new board state encoder
    pub fn new(max_board_size: usize) -> Self {
        Self {
            max_board_size,
            num_channels: 7, // EMPTY, OWN_HEAD, OWN_BODY, OPPONENT_HEAD, OPPONENT_BODY, FOOD, WALL
        }
    }

    /// Encode board state for neural network input
    pub fn encode_board_state(&self, board: &Board, our_snake: &Battlesnake) -> NeuralNetworkInput {
        let height = board.height as usize;
        let width = board.width as usize;

        // Initialize grid (channels, height, width)
        let mut grid = Array3::zeros((self.num_channels, height, width));

        // Encode food
        for food in &board.food {
            if (food.y as usize) < height && (food.x as usize) < width {
                grid[[5, food.y as usize, food.x as usize]] = 1.0; // FOOD channel
            }
        }

        // Encode snakes
        for snake in &board.snakes {
            let is_our_snake = snake.id == our_snake.id;
            
            for (i, segment) in snake.body.iter().enumerate() {
                if (segment.y as usize) < height && (segment.x as usize) < width {
                    if i == 0 { // Head
                        if is_our_snake {
                            grid[[1, segment.y as usize, segment.x as usize]] = 1.0; // OWN_HEAD
                        } else {
                            grid[[3, segment.y as usize, segment.x as usize]] = 1.0; // OPPONENT_HEAD
                        }
                    } else { // Body
                        if is_our_snake {
                            grid[[2, segment.y as usize, segment.x as usize]] = 1.0; // OWN_BODY
                        } else {
                            grid[[4, segment.y as usize, segment.x as usize]] = 1.0; // OPPONENT_BODY
                        }
                    }
                }
            }
        }

        // Encode walls (board boundaries)
        for x in 0..width {
            grid[[6, 0, x]] = 1.0; // Top boundary
            grid[[6, height-1, x]] = 1.0; // Bottom boundary
        }
        for y in 0..height {
            grid[[6, y, 0]] = 1.0; // Left boundary
            grid[[6, y, width-1]] = 1.0; // Right boundary
        }

        // Pad to max board size if necessary
        let grid = self.pad_to_max_size(grid, height, width);

        // Create feature vector
        let mut features = Array2::zeros((1, 6));
        features[[0, 0]] = (our_snake.health as f32) / 100.0; // Health ratio
        features[[0, 1]] = (our_snake.body.len() as f32) / 20.0; // Length ratio
        features[[0, 2]] = board.turn as f32 / 1000.0; // Turn ratio
        features[[0, 3]] = (board.snakes.len() as f32) / 8.0; // Snake count ratio
        features[[0, 4]] = (our_snake.body[0].y as f32) / (board.height as f32); // Head Y
        features[[0, 5]] = (our_snake.body[0].x as f32) / (board.width as f32); // Head X

        NeuralNetworkInput { board_state: grid, features }
    }

    /// Pad grid to maximum board size
    fn pad_to_max_size(&self, grid: Array3<f32>, height: usize, width: usize) -> Array3<f32> {
        if height == self.max_board_size && width == self.max_board_size {
            return grid;
        }

        let mut padded_grid = Array3::zeros((self.num_channels, self.max_board_size, self.max_board_size));
        padded_grid.slice_mut(s![.., ..height, ..width]).assign(&grid);
        padded_grid
    }
}

/// Neural Network Evaluation Integrator
/// Combines neural network predictions with traditional search algorithms
pub struct NeuralNetworkEvaluator {
    inference_engine: Arc<ONNXInferenceEngine>,
    board_encoder: BoardStateEncoder,
    fallback_enabled: bool,
    confidence_threshold: f32,
}

impl NeuralNetworkEvaluator {
    /// Create new neural network evaluator
    pub fn new() -> Self {
        Self {
            inference_engine: Arc::new(ONNXInferenceEngine::new()),
            board_encoder: BoardStateEncoder::new(20),
            fallback_enabled: true,
            confidence_threshold: 0.3,
        }
    }

    /// Load ONNX models for all network types
    pub fn load_models(&self, model_dir: &str) -> Result<()> {
        let model_types = [
            (NeuralNetworkType::PositionEvaluation, "position_evaluation.onnx"),
            (NeuralNetworkType::MovePrediction, "move_prediction.onnx"),
            (NeuralNetworkType::GameOutcome, "game_outcome.onnx"),
        ];

        for (model_type, model_filename) in &model_types {
            let model_path = format!("{}/{}", model_dir, model_filename);
            if std::path::Path::new(&model_path).exists() {
                self.inference_engine.load_model(&model_path, model_type.clone())
                    .map_err(|e| warn!("Failed to load {} model: {}", model_type, e))
                    .ok();
            } else {
                warn!("ONNX model not found: {}", model_path);
            }
        }

        Ok(())
    }

    /// Evaluate board position using neural networks
    pub fn evaluate_position(&self, board: &Board, our_snake: &Battlesnake) -> Result<f32> {
        // Try neural network evaluation
        if self.inference_engine.is_model_loaded(&NeuralNetworkType::PositionEvaluation) {
            match self.evaluate_with_neural_network(board, our_snake) {
                Ok(score) => {
                    info!("Neural network position evaluation: {:.3}", score);
                    return Ok(score);
                }
                Err(e) => {
                    warn!("Neural network evaluation failed: {}", e);
                    if self.fallback_enabled {
                        info!("Falling back to heuristic evaluation");
                        return Ok(self.heuristic_position_evaluation(board, our_snake));
                    }
                }
            }
        }

        // Fallback to heuristic
        if self.fallback_enabled {
            info!("Using heuristic position evaluation");
            Ok(self.heuristic_position_evaluation(board, our_snake))
        } else {
            Err(anyhow!("Neural network evaluation failed and fallback disabled"))
        }
    }

    /// Evaluate position using neural network
    fn evaluate_with_neural_network(&self, board: &Board, our_snake: &Battlesnake) -> Result<f32> {
        let input = self.board_encoder.encode_board_state(board, our_snake);
        let output = self.inference_engine.run_inference(&NeuralNetworkType::PositionEvaluation, &input)?;
        
        output.position_score
            .ok_or_else(|| anyhow!("No position score returned from neural network"))
            .map(|score| score * 2.0 - 1.0) // Normalize from [0,1] to [-1,1]
    }

    /// Get move probabilities using neural network
    pub fn get_move_probabilities(&self, board: &Board, our_snake: &Battlesnake) -> Result<Array2<f32>> {
        if self.inference_engine.is_model_loaded(&NeuralNetworkType::MovePrediction) {
            let input = self.board_encoder.encode_board_state(board, our_snake);
            let output = self.inference_engine.run_inference(&NeuralNetworkType::MovePrediction, &input)?;
            
            if let Some(probs) = output.move_probabilities {
                info!("Neural network move probabilities: {:?}", probs);
                return Ok(probs);
            }
        }

        // Fallback to uniform distribution
        warn!("Using uniform move probabilities as fallback");
        Ok(Array2::ones((1, 4)) * 0.25)
    }

    /// Get win probability for current position
    pub fn get_win_probability(&self, board: &Board, our_snake: &Battlesnake) -> Result<f32> {
        if self.inference_engine.is_model_loaded(&NeuralNetworkType::GameOutcome) {
            let input = self.board_encoder.encode_board_state(board, our_snake);
            let output = self.inference_engine.run_inference(&NeuralNetworkType::GameOutcome, &input)?;
            
            if let Some(prob) = output.win_probability {
                info!("Neural network win probability: {:.3}", prob);
                return Ok(prob);
            }
        }

        // Fallback to heuristic
        warn!("Using heuristic win probability as fallback");
        Ok(self.heuristic_win_probability(board, our_snake))
    }

    /// Heuristic position evaluation (fallback)
    fn heuristic_position_evaluation(&self, board: &Board, our_snake: &Battlesnake) -> f32 {
        // Simple heuristic: health + space advantage - danger
        let health_score = (our_snake.health as f32) / 100.0;
        
        // Space advantage (simplified)
        let space_score = 0.5;
        
        // Danger penalty
        let danger_score = 0.2;
        
        (health_score + space_score - danger_score) * 2.0 - 1.0
    }

    /// Heuristic win probability (fallback)
    fn heuristic_win_probability(&self, board: &Board, our_snake: &Battlesnake) -> f32 {
        // Simple heuristic based on health and snake length
        let health_factor = (our_snake.health as f32) / 100.0;
        let length_factor = (our_snake.body.len() as f32) / 10.0;
        
        (health_factor + length_factor).min(1.0)
    }

    /// Enable/disable fallback to heuristics
    pub fn set_fallback_enabled(&mut self, enabled: bool) {
        self.fallback_enabled = enabled;
    }

    /// Set confidence threshold for neural network usage
    pub fn set_confidence_threshold(&mut self, threshold: f32) {
        self.confidence_threshold = threshold;
    }
}

// Global singleton for neural network inference
use lazy_static::lazy_static;
lazy_static! {
    static ref NEURAL_NETWORK_EVALUATOR: Arc<Mutex<NeuralNetworkEvaluator>> = {
        Arc::new(Mutex::new(NeuralNetworkEvaluator::new()))
    };
}

/// Get global neural network evaluator
pub fn get_neural_network_evaluator() -> Arc<Mutex<NeuralNetworkEvaluator>> {
    NEURAL_NETWORK_EVALUATOR.clone()
}

/// Initialize neural network evaluator with models
pub fn initialize_neural_networks(model_dir: &str) -> Result<()> {
    let evaluator = get_neural_network_evaluator();
    let mut evaluator = evaluator.lock().map_err(|e| anyhow!("Failed to lock evaluator: {}", e))?;
    
    evaluator.load_models(model_dir)?;
    
    info!("Neural network evaluator initialized with models from: {}", model_dir);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_state_encoder() {
        let encoder = BoardStateEncoder::new(20);
        
        // Create test board
        let board = Board {
            height: 11,
            width: 11,
            food: vec![Coord { x: 5, y: 5 }],
            snakes: vec![],
        };
        
        let snake = Battlesnake {
            id: "test".to_string(),
            name: "test".to_string(),
            health: 100,
            body: vec![Coord { x: 5, y: 6 }, Coord { x: 5, y: 7 }],
            head: Coord { x: 5, y: 6 },
            length: 2,
            latency: "0".to_string(),
            shout: None,
        };
        
        let input = encoder.encode_board_state(&board, &snake);
        
        assert_eq!(input.board_state.dim(), (7, 20, 20));
        assert_eq!(input.features.dim(), (1, 6));
    }

    #[test]
    fn test_neural_network_output_parsing() {
        let engine = ONNXInferenceEngine::new();
        
        // This would test output parsing, but we need actual ONNX model
        // In real tests, we would mock the ONNX session
    }
}