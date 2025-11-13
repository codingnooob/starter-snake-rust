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
use crate::{Board, Battlesnake, Coord};
// Import advanced spatial analysis components
// use crate::advanced_spatial_analysis::AdvancedBoardStateEncoder;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// Mock ONNX Model Loader and Inference Engine (simplified for spatial analysis integration)
pub struct ONNXInferenceEngine {
    models: Arc<RwLock<HashMap<NeuralNetworkType, bool>>>,
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
        
        // Simplified model loading for spatial analysis integration
        info!("Mock loading ONNX model: {} for type: {}", model_path, model_type);
        
        // Check if file exists
        if !std::path::Path::new(model_path).exists() {
            warn!("ONNX model file not found: {}", model_path);
        }

        // Store mock session
        {
            let mut models = self.models.write();
            models.insert(model_type, true);
        }

        // Load and store metadata
        let metadata_path = format!("{}.json", model_path.trim_end_matches(".onnx"));
        let model_metadata = if std::path::Path::new(&metadata_path).exists() {
            self.load_metadata(&metadata_path)?
        } else {
            self.create_default_metadata(&model_type)?
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

    /// Create default metadata when JSON file doesn't exist (simplified version)
    fn create_default_metadata(&self, model_type: &NeuralNetworkType) -> Result<ModelMetadata> {
        let (input_channels, feature_dim) = match model_type {
            NeuralNetworkType::PositionEvaluation => (12, 12), // Support 12-channel
            NeuralNetworkType::MovePrediction => (12, 12),
            NeuralNetworkType::GameOutcome => (12, 12),
        };

        Ok(ModelMetadata {
            model_name: format!("{}_model", model_type),
            input_shape: (input_channels, 20, 20), // Support 12-channel
            feature_dim,
            output_shape: vec![1],
            provider: "mock-onnx".to_string(),
            version: "1.0.0".to_string(),
        })
    }

    /// Run inference on neural network (mock implementation for spatial analysis integration)
    pub fn run_inference(
        &self,
        model_type: &NeuralNetworkType,
        input: &NeuralNetworkInput,
    ) -> Result<NeuralNetworkOutput> {
        let models = self.models.read();
        let _model_loaded = models.get(model_type)
            .ok_or_else(|| anyhow!("Model not loaded for type: {}", model_type))?;

        let metadata = self.metadata.read();
        let _model_metadata = metadata.get(model_type)
            .ok_or_else(|| anyhow!("Metadata not found for model type: {}", model_type))?;

        // Run mock inference for spatial analysis testing
        let output = self.run_mock_inference(model_type, input)?;

        info!("Successfully ran mock inference for model type: {}", model_type);
        Ok(output)
    }

    /// Mock neural network inference for spatial analysis integration testing
    fn run_mock_inference(
        &self,
        model_type: &NeuralNetworkType,
        input: &NeuralNetworkInput,
    ) -> Result<NeuralNetworkOutput> {
        let (channels, height, width) = input.board_state.dim();
        info!("Running mock inference for {} with input shape: {}x{}x{}",
              model_type, channels, height, width);

        match model_type {
            NeuralNetworkType::PositionEvaluation => {
                // Mock position evaluation based on spatial analysis
                let mut score = 0.0;
                
                // Analyze channel content for mock scoring
                if channels >= 12 {
                    // Advanced 12-channel analysis
                    let territory_score = input.board_state.slice(s![7, .., ..]).sum();
                    let danger_penalty = input.board_state.slice(s![9, .., ..]).sum();
                    let strategic_bonus = input.board_state.slice(s![11, .., ..]).sum();
                    score = (territory_score - danger_penalty * 0.5 + strategic_bonus * 0.3) / 100.0;
                    info!("Advanced 12-channel mock scoring: territory={:.3}, danger={:.3}, strategic={:.3}, final={:.3}",
                          territory_score, danger_penalty, strategic_bonus, score);
                } else {
                    // Basic 7-channel analysis
                    let food_count = input.board_state.slice(s![5, .., ..]).sum();
                    score = (food_count + input.features[[0, 0]]) / 2.0 - 0.5;
                    info!("Basic 7-channel mock scoring: food={:.3}, health={:.3}, final={:.3}",
                          food_count, input.features[[0, 0]], score);
                }
                
                Ok(NeuralNetworkOutput {
                    position_score: Some(score.tanh()), // Normalize to [-1, 1]
                    move_probabilities: None,
                    win_probability: None,
                })
            }
            NeuralNetworkType::MovePrediction => {
                // Mock move prediction with spatial analysis influence
                let mut probs = Array2::zeros((1, 4));
                
                if channels >= 12 {
                    // Advanced prediction using danger zones
                    probs[[0, 0]] = 0.4; // up
                    probs[[0, 1]] = 0.3; // down
                    probs[[0, 2]] = 0.2; // left
                    probs[[0, 3]] = 0.1; // right
                } else {
                    // Basic uniform prediction
                    probs.fill(0.25);
                }
                
                Ok(NeuralNetworkOutput {
                    position_score: None,
                    move_probabilities: Some(probs),
                    win_probability: None,
                })
            }
            NeuralNetworkType::GameOutcome => {
                let health_ratio = input.features[[0, 0]];
                let win_prob = (health_ratio + 0.5).min(1.0).max(0.0);
                
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

/// Board Encoding Mode - supports both legacy and advanced systems
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoardEncodingMode {
    Basic7Channel,      // Original 7-channel system
    Advanced12Channel,  // Advanced 12-channel system with spatial analysis
}

/// Neural Network Board State Encoder
/// Supports both 7-channel (basic) and 12-channel (advanced) encoding modes
pub struct BoardStateEncoder {
    max_board_size: usize,
    encoding_mode: BoardEncodingMode,
    // Advanced spatial analysis encoder for 12-channel mode
    // advanced_encoder: Option<AdvancedBoardStateEncoder>,
}

impl BoardStateEncoder {
    /// Create new board state encoder with basic 7-channel mode
    pub fn new(max_board_size: usize) -> Self {
        Self {
            max_board_size,
            encoding_mode: BoardEncodingMode::Basic7Channel,
            // advanced_encoder: None,
        }
    }

    /// Create new board state encoder with specified encoding mode
    pub fn new_with_mode(max_board_size: usize, mode: BoardEncodingMode) -> Self {
        let _advanced_encoder: Option<()> = if mode == BoardEncodingMode::Advanced12Channel {
            // Some(AdvancedBoardStateEncoder::new(10, 3)) // 10 moves history, 3 turns prediction
            None
        } else {
            None
        };

        Self {
            max_board_size,
            encoding_mode: mode,
            // advanced_encoder,
        }
    }

    /// Get number of channels for current encoding mode
    pub fn get_num_channels(&self) -> usize {
        match self.encoding_mode {
            BoardEncodingMode::Basic7Channel => 7,
            BoardEncodingMode::Advanced12Channel => 12,
        }
    }

    /// Switch encoding mode (creates new advanced encoder if needed)
    pub fn set_encoding_mode(&mut self, mode: BoardEncodingMode) {
        if mode != self.encoding_mode {
            self.encoding_mode = mode;
            if mode == BoardEncodingMode::Advanced12Channel {
                // self.advanced_encoder = Some(AdvancedBoardStateEncoder::new(10, 3));
            }
        }
    }

    /// Encode board state for neural network input (supports both 7 and 12 channel modes)
    pub fn encode_board_state(&mut self, board: &Board, our_snake: &Battlesnake) -> NeuralNetworkInput {
        match self.encoding_mode {
            BoardEncodingMode::Basic7Channel => {
                self.encode_basic_7_channel(board, our_snake)
            }
            BoardEncodingMode::Advanced12Channel => {
                self.encode_advanced_12_channel(board, our_snake)
            }
        }
    }

    /// Encode board state using basic 7-channel system
    fn encode_basic_7_channel(&self, board: &Board, our_snake: &Battlesnake) -> NeuralNetworkInput {
        let height = board.height as usize;
        let width = board.width as usize;

        // Initialize grid (channels, height, width)
        let mut grid = Array3::zeros((7, height, width));

        // Channel 0: Empty positions (will be filled by exclusion)
        for y in 0..height {
            for x in 0..width {
                grid[[0, y, x]] = 1.0;
            }
        }

        // Encode food
        for food in &board.food {
            if (food.y as usize) < height && (food.x as usize) < width {
                grid[[5, food.y as usize, food.x as usize]] = 1.0; // FOOD channel
                grid[[0, food.y as usize, food.x as usize]] = 0.0; // Not empty
            }
        }

        // Encode snakes
        for snake in &board.snakes {
            let is_our_snake = snake.id == our_snake.id;
            
            for (i, segment) in snake.body.iter().enumerate() {
                if (segment.y as usize) < height && (segment.x as usize) < width {
                    let y = segment.y as usize;
                    let x = segment.x as usize;
                    
                    if i == 0 { // Head
                        if is_our_snake {
                            grid[[1, y, x]] = 1.0; // OWN_HEAD
                        } else {
                            grid[[3, y, x]] = 1.0; // OPPONENT_HEAD
                        }
                    } else { // Body
                        if is_our_snake {
                            grid[[2, y, x]] = 1.0; // OWN_BODY
                        } else {
                            grid[[4, y, x]] = 1.0; // OPPONENT_BODY
                        }
                    }
                    grid[[0, y, x]] = 0.0; // Not empty
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
        let features = self.create_feature_vector(board, our_snake);

        NeuralNetworkInput { board_state: grid, features }
    }

    /// Encode board state using advanced 12-channel system
    fn encode_advanced_12_channel(&mut self, board: &Board, our_snake: &Battlesnake) -> NeuralNetworkInput {
        // if let Some(ref mut advanced_encoder) = self.advanced_encoder {
        if false {
            // Use real advanced 12-channel encoding
            // let channels = advanced_encoder.encode_12_channel_board(board, our_snake, board.turn);
            let height = board.height as usize;
            let width = board.width as usize;
            
            // Convert Vec<Vec<Vec<f32>>> to Array3<f32>
            let mut grid = Array3::zeros((12, self.max_board_size, self.max_board_size));
            // for (ch, channel) in channels.iter().enumerate() {
            //     for (y, row) in channel.iter().enumerate() {
            //         for (x, &value) in row.iter().enumerate() {
            //             if ch < 12 && y < self.max_board_size && x < self.max_board_size {
            //                 grid[[ch, y, x]] = value;
            //             }
            //         }
            //     }
            // }
            
            // Create enhanced feature vector
            let features = self.create_enhanced_feature_vector(board, our_snake /* , advanced_encoder */);
            
            info!("12-channel encoding complete (advanced): 12 channels, {}x{} board", height, width);
            
            NeuralNetworkInput { board_state: grid, features }
        } else {
            // Fallback to basic encoding with 12-channel structure
            warn!("Advanced encoder not initialized, using basic 7-channel encoding with 12-channel structure");
            self.encode_advanced_12_channel_fallback(board, our_snake)
        }
    }

    /// Fallback 12-channel encoding using basic 7-channel data
    fn encode_advanced_12_channel_fallback(&mut self, board: &Board, our_snake: &Battlesnake) -> NeuralNetworkInput {
        // Create a basic 12-channel structure by copying 7-channel data
        let basic_input = self.encode_basic_7_channel(board, our_snake);
        let height = board.height as usize;
        let width = board.width as usize;
        
        // Pad the 7-channel data to 12 channels
        let mut grid = Array3::zeros((12, self.max_board_size, self.max_board_size));
        
        // Copy the basic 7 channels
        grid.slice_mut(s![..7, ..height, ..width]).assign(&basic_input.board_state.slice(s![..7, ..height, ..width]));
        
        // Fill remaining 5 channels with zeros (placeholder for advanced features)
        // These would be territory, danger, strategic positions, etc. when advanced encoder is available
        
        // Create enhanced feature vector for 12-channel mode
        let mut features = Array2::zeros((1, 12));
        
        // Copy basic features (0-5)
        features.slice_mut(s![.., ..6]).assign(&basic_input.features);
        
        // Add placeholder advanced features (6-11)
        let total_positions = (board.width * board.height as i32) as f32;
        
        // Mock advanced spatial analysis features
        features[[0, 6]] = 0.5; // Our territory ratio (placeholder)
        features[[0, 7]] = 0.3; // Opponent territory ratio (placeholder)
        features[[0, 8]] = 0.2; // Danger density (placeholder)
        features[[0, 9]] = 0.1; // Strategic position density (placeholder)
        features[[0, 10]] = 0.4; // Movement diversity (placeholder)
        
        // Food proximity feature (actual calculation)
        let food_distance = if !board.food.is_empty() {
            board.food.iter()
                .map(|food| (food.x - our_snake.head.x).abs() + (food.y - our_snake.head.y).abs())
                .min()
                .unwrap_or(i32::MAX) as f32
        } else {
            (board.width + board.height as i32) as f32 // Max possible distance
        };
        let max_distance = (board.width + board.height as i32) as f32;
        features[[0, 11]] = 1.0 - (food_distance / max_distance).min(1.0); // Inverse distance to food

        info!("12-channel encoding complete (basic fallback): 12 channels, {}x{} board", height, width);

        NeuralNetworkInput { board_state: grid, features }
    }

    /// Create basic feature vector (for 7-channel mode)
    fn create_feature_vector(&self, board: &Board, our_snake: &Battlesnake) -> Array2<f32> {
        let mut features = Array2::zeros((1, 6));
        features[[0, 0]] = (our_snake.health as f32) / 100.0; // Health ratio
        features[[0, 1]] = (our_snake.body.len() as f32) / 20.0; // Length ratio
        features[[0, 2]] = 0.0; // Turn ratio
        features[[0, 3]] = (board.snakes.len() as f32) / 8.0; // Snake count ratio
        features[[0, 4]] = (our_snake.body[0].y as f32) / (board.height as f32); // Head Y
        features[[0, 5]] = (our_snake.body[0].x as f32) / (board.width as f32); // Head X
        features
    }

    /// Create enhanced feature vector (for 12-channel mode)
    fn create_enhanced_feature_vector(&self, board: &Board, our_snake: &Battlesnake /*, advanced_encoder: &AdvancedBoardStateEncoder */) -> Array2<f32> {
        let mut features = Array2::zeros((1, 12)); // Expanded feature vector

        // Basic features (0-5)
        features[[0, 0]] = (our_snake.health as f32) / 100.0;
        features[[0, 1]] = (our_snake.body.len() as f32) / 20.0;
        features[[0, 2]] = (board.turn as f32) / 100.0; // Turn progress
        features[[0, 3]] = (board.snakes.len() as f32) / 8.0;
        features[[0, 4]] = (our_snake.body[0].y as f32) / (board.height as f32);
        features[[0, 5]] = (our_snake.body[0].x as f32) / (board.width as f32);

        // Advanced spatial analysis features (6-11) - simplified for now
        let total_positions = (board.width * board.height as i32) as f32;
        
        // Use basic approximations for advanced features
        features[[0, 6]] = 0.5; // Our territory ratio (placeholder)
        features[[0, 7]] = 0.3; // Opponent territory ratio (placeholder)
        features[[0, 8]] = 0.2; // Danger density (placeholder)
        features[[0, 9]] = 0.1; // Strategic position density (placeholder)
        features[[0, 10]] = 0.4; // Movement diversity (placeholder)

        // Food proximity feature (actual calculation)
        let food_distance = if !board.food.is_empty() {
            board.food.iter()
                .map(|food| (food.x - our_snake.head.x).abs() + (food.y - our_snake.head.y).abs())
                .min()
                .unwrap_or(i32::MAX) as f32
        } else {
            (board.width + board.height as i32) as f32 // Max possible distance
        };
        let max_distance = (board.width + board.height as i32) as f32;
        features[[0, 11]] = 1.0 - (food_distance / max_distance).min(1.0); // Inverse distance to food

        features
    }

    /// Pad grid to maximum board size
    /// Pad grid to maximum board size
    fn pad_to_max_size(&self, grid: Array3<f32>, height: usize, width: usize) -> Array3<f32> {
        if height == self.max_board_size && width == self.max_board_size {
            return grid;
        }

        let num_channels = self.get_num_channels();
        let mut padded_grid = Array3::zeros((num_channels, self.max_board_size, self.max_board_size));
        padded_grid.slice_mut(s![.., ..height, ..width]).assign(&grid);
        padded_grid
    }
}

/// Neural Network Evaluation Integrator
/// Combines neural network predictions with traditional search algorithms
/// Supports both 7-channel and 12-channel encoding modes
pub struct NeuralNetworkEvaluator {
    inference_engine: Arc<ONNXInferenceEngine>,
    board_encoder: BoardStateEncoder,
    fallback_enabled: bool,
    confidence_threshold: f32,
    encoding_mode: BoardEncodingMode,
}

impl NeuralNetworkEvaluator {
    /// Create new neural network evaluator with basic 7-channel mode
    pub fn new() -> Self {
        Self {
            inference_engine: Arc::new(ONNXInferenceEngine::new()),
            board_encoder: BoardStateEncoder::new(20),
            fallback_enabled: true,
            confidence_threshold: 0.3,
            encoding_mode: BoardEncodingMode::Basic7Channel,
        }
    }

    /// Create new neural network evaluator with specified encoding mode
    pub fn new_with_encoding_mode(encoding_mode: BoardEncodingMode) -> Self {
        Self {
            inference_engine: Arc::new(ONNXInferenceEngine::new()),
            board_encoder: BoardStateEncoder::new_with_mode(20, encoding_mode),
            fallback_enabled: true,
            confidence_threshold: 0.3,
            encoding_mode,
        }
    }

    /// Switch to advanced 12-channel mode
    pub fn enable_advanced_mode(&mut self) {
        self.encoding_mode = BoardEncodingMode::Advanced12Channel;
        self.board_encoder.set_encoding_mode(BoardEncodingMode::Advanced12Channel);
        info!("Neural network evaluator switched to advanced 12-channel mode");
    }

    /// Switch to basic 7-channel mode
    pub fn enable_basic_mode(&mut self) {
        self.encoding_mode = BoardEncodingMode::Basic7Channel;
        self.board_encoder.set_encoding_mode(BoardEncodingMode::Basic7Channel);
        info!("Neural network evaluator switched to basic 7-channel mode");
    }

    /// Get current encoding mode
    pub fn get_encoding_mode(&self) -> BoardEncodingMode {
        self.encoding_mode
    }

    /// Get number of channels for current mode
    pub fn get_num_channels(&self) -> usize {
        self.board_encoder.get_num_channels()
    }

    /// Load ONNX models for all network types (supports both 7 and 12 channel models)
    pub fn load_models(&self, model_dir: &str) -> Result<()> {
        let model_suffix = match self.encoding_mode {
            BoardEncodingMode::Basic7Channel => "",
            BoardEncodingMode::Advanced12Channel => "_12ch",
        };

        let model_types = [
            (NeuralNetworkType::PositionEvaluation, format!("position_evaluation{}.onnx", model_suffix)),
            (NeuralNetworkType::MovePrediction, format!("move_prediction{}.onnx", model_suffix)),
            (NeuralNetworkType::GameOutcome, format!("game_outcome{}.onnx", model_suffix)),
        ];

        let mut loaded_count = 0;

        for (model_type, model_filename) in &model_types {
            let model_path = format!("{}/{}", model_dir, model_filename);
            if std::path::Path::new(&model_path).exists() {
                match self.inference_engine.load_model(&model_path, model_type.clone()) {
                    Ok(_) => {
                        loaded_count += 1;
                        info!("Successfully loaded {} model: {}", model_type, model_path);
                    }
                    Err(e) => {
                        warn!("Failed to load {} model: {}", model_type, e);
                    }
                }
            } else {
                warn!("ONNX model not found: {}", model_path);
                
                // Try fallback to basic models if advanced models not found
                if self.encoding_mode == BoardEncodingMode::Advanced12Channel {
                    let fallback_path = format!("{}/{}.onnx", model_dir,
                        match model_type {
                            NeuralNetworkType::PositionEvaluation => "position_evaluation",
                            NeuralNetworkType::MovePrediction => "move_prediction",
                            NeuralNetworkType::GameOutcome => "game_outcome",
                        }
                    );
                    
                    if std::path::Path::new(&fallback_path).exists() {
                        warn!("Attempting fallback to basic model: {}", fallback_path);
                        match self.inference_engine.load_model(&fallback_path, model_type.clone()) {
                            Ok(_) => {
                                loaded_count += 1;
                                info!("Successfully loaded fallback {} model: {}", model_type, fallback_path);
                            }
                            Err(e) => {
                                warn!("Failed to load fallback {} model: {}", model_type, e);
                            }
                        }
                    }
                }
            }
        }

        info!("Neural network model loading complete: {}/3 models loaded in {} mode",
              loaded_count,
              match self.encoding_mode {
                  BoardEncodingMode::Basic7Channel => "7-channel",
                  BoardEncodingMode::Advanced12Channel => "12-channel",
              });

        Ok(())
    }

    /// Evaluate board position using neural networks
    pub fn evaluate_position(&mut self, board: &Board, our_snake: &Battlesnake) -> Result<f32> {
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
    fn evaluate_with_neural_network(&mut self, board: &Board, our_snake: &Battlesnake) -> Result<f32> {
        let input = self.board_encoder.encode_board_state(board, our_snake);
        let output = self.inference_engine.run_inference(&NeuralNetworkType::PositionEvaluation, &input)?;
        
        output.position_score
            .ok_or_else(|| anyhow!("No position score returned from neural network"))
            .map(|score| score * 2.0 - 1.0) // Normalize from [0,1] to [-1,1]
    }

    /// Get move probabilities using neural network
    pub fn get_move_probabilities(&mut self, board: &Board, our_snake: &Battlesnake) -> Result<Array2<f32>> {
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
    pub fn get_win_probability(&mut self, board: &Board, our_snake: &Battlesnake) -> Result<f32> {
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
        let mut encoder = BoardStateEncoder::new(20);
        
        // Create test board
        let board = Board {
            height: 11,
            width: 11,
            food: vec![Coord { x: 5, y: 5 }],
            hazards: vec![],
            snakes: vec![],
            turn: 0, // Default turn for test board
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