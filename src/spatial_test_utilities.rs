//! Comprehensive Test Utilities for 12-Channel Spatial Analysis System
//! 
//! This module provides robust testing infrastructure for all spatial analysis components:
//! - Mock game state generation with configurable scenarios
//! - Board configuration builders with edge cases
//! - Snake position generators for complex multi-snake scenarios  
//! - Fuzzing helpers for random board generation
//! - Performance measurement utilities
//! - Property-based test helpers and validation functions

use crate::{Board, Battlesnake, Coord, Game, GameState};
use crate::advanced_spatial_analysis::{
    VoronoiTerritoryAnalyzer, DangerZonePredictor, MovementHistoryTracker, 
    StrategicPositionAnalyzer, AdvancedBoardStateEncoder
};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Instant;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

// ============================================================================
// MOCK GAME STATE BUILDERS
// ============================================================================

/// Builder for creating test game states with configurable parameters
#[derive(Debug, Clone)]
pub struct GameStateBuilder {
    width: i32,
    height: u32,
    our_snake: Option<Battlesnake>,
    opponent_snakes: Vec<Battlesnake>,
    food_positions: Vec<Coord>,
    hazards: Vec<Coord>,
    turn: i32,
    game_id: String,
}

impl GameStateBuilder {
    pub fn new() -> Self {
        Self {
            width: 11,
            height: 11,
            our_snake: None,
            opponent_snakes: Vec::new(),
            food_positions: Vec::new(),
            hazards: Vec::new(),
            turn: 0,
            game_id: "test-game".to_string(),
        }
    }

    pub fn with_board_size(mut self, width: i32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn with_our_snake(mut self, snake: Battlesnake) -> Self {
        self.our_snake = Some(snake);
        self
    }

    pub fn with_opponent(mut self, snake: Battlesnake) -> Self {
        self.opponent_snakes.push(snake);
        self
    }

    pub fn with_food(mut self, positions: Vec<Coord>) -> Self {
        self.food_positions = positions;
        self
    }

    pub fn with_turn(mut self, turn: i32) -> Self {
        self.turn = turn;
        self
    }

    pub fn with_game_id(mut self, id: String) -> Self {
        self.game_id = id;
        self
    }

    pub fn build(self) -> GameState {
        let our_snake = self.our_snake.unwrap_or_else(|| {
            create_test_snake("our-snake", Coord { x: 5, y: 5 }, 3, 100)
        });

        let mut all_snakes = vec![our_snake.clone()];
        all_snakes.extend(self.opponent_snakes);

        let board = Board {
            width: self.width,
            height: self.height,
            food: self.food_positions,
            snakes: all_snakes,
            hazards: self.hazards,
        };

        let game = Game {
            id: self.game_id,
            ruleset: HashMap::new(),
            timeout: 500,
        };

        GameState {
            game,
            turn: self.turn,
            board,
            you: our_snake,
        }
    }
}

// ============================================================================
// SNAKE BUILDERS AND GENERATORS
// ============================================================================

/// Create a test snake with specified parameters
pub fn create_test_snake(id: &str, head: Coord, length: i32, health: i32) -> Battlesnake {
    let mut body = vec![head];
    
    // Generate body segments behind head
    for i in 1..length {
        let segment = Coord {
            x: head.x,
            y: head.y - i, // Place segments below head
        };
        body.push(segment);
    }

    Battlesnake {
        id: id.to_string(),
        name: format!("Test Snake {}", id),
        health,
        body,
        head,
        length,
        latency: "0".to_string(),
        shout: None,
    }
}

/// Create a snake positioned along a specific path
pub fn create_snake_with_path(id: &str, path: Vec<Coord>, health: i32) -> Battlesnake {
    if path.is_empty() {
        panic!("Snake path cannot be empty");
    }

    let head = path[0];
    let length = path.len() as i32;

    Battlesnake {
        id: id.to_string(),
        name: format!("Path Snake {}", id),
        health,
        body: path,
        head,
        length,
        latency: "0".to_string(),
        shout: None,
    }
}

/// Generate multiple opponent snakes for testing scenarios
pub fn create_multi_opponent_scenario(board_size: (i32, u32), num_opponents: usize) -> Vec<Battlesnake> {
    let mut opponents = Vec::new();
    let spacing = board_size.0 / (num_opponents + 1) as i32;
    
    for i in 0..num_opponents {
        let x = spacing * (i + 1) as i32;
        let y = board_size.1 as i32 / 2;
        let head = Coord { x, y };
        
        let opponent = create_test_snake(&format!("opponent-{}", i), head, 3, 100);
        opponents.push(opponent);
    }
    
    opponents
}

// ============================================================================
// BOARD SCENARIO GENERATORS
// ============================================================================

/// Generate various board scenarios for testing
pub struct BoardScenarioGenerator;

impl BoardScenarioGenerator {
    /// Create an empty board with just our snake
    pub fn empty_board() -> GameState {
        GameStateBuilder::new()
            .with_board_size(11, 11)
            .with_our_snake(create_test_snake("us", Coord { x: 5, y: 5 }, 3, 100))
            .build()
    }

    /// Create a crowded board with multiple opponents
    pub fn crowded_board() -> GameState {
        let opponents = create_multi_opponent_scenario((11, 11), 3);
        let mut builder = GameStateBuilder::new()
            .with_board_size(11, 11)
            .with_our_snake(create_test_snake("us", Coord { x: 5, y: 5 }, 5, 100));

        for opponent in opponents {
            builder = builder.with_opponent(opponent);
        }

        builder.build()
    }

    /// Create a corner trap scenario
    pub fn corner_trap_scenario() -> GameState {
        GameStateBuilder::new()
            .with_board_size(11, 11)
            .with_our_snake(create_test_snake("us", Coord { x: 0, y: 0 }, 3, 50))
            .with_opponent(create_test_snake("opponent", Coord { x: 2, y: 2 }, 10, 100))
            .build()
    }

    /// Create a food competition scenario
    pub fn food_competition_scenario() -> GameState {
        let food = vec![
            Coord { x: 5, y: 5 },
            Coord { x: 7, y: 3 },
            Coord { x: 3, y: 7 },
        ];

        GameStateBuilder::new()
            .with_board_size(11, 11)
            .with_our_snake(create_test_snake("us", Coord { x: 4, y: 4 }, 3, 30))
            .with_opponent(create_test_snake("competitor", Coord { x: 6, y: 6 }, 4, 40))
            .with_food(food)
            .build()
    }

    /// Create a head-to-head collision scenario
    pub fn head_collision_scenario() -> GameState {
        GameStateBuilder::new()
            .with_board_size(11, 11)
            .with_our_snake(create_test_snake("us", Coord { x: 5, y: 5 }, 4, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 7, y: 5 }, 4, 100))
            .build()
    }

    /// Create a large board scenario for performance testing
    pub fn large_board_scenario() -> GameState {
        let opponents = create_multi_opponent_scenario((19, 19), 7);
        let mut builder = GameStateBuilder::new()
            .with_board_size(19, 19)
            .with_our_snake(create_test_snake("us", Coord { x: 9, y: 9 }, 8, 100));

        for opponent in opponents {
            builder = builder.with_opponent(opponent);
        }

        // Add scattered food
        let food = vec![
            Coord { x: 3, y: 3 }, Coord { x: 15, y: 3 },
            Coord { x: 3, y: 15 }, Coord { x: 15, y: 15 },
            Coord { x: 9, y: 2 }, Coord { x: 2, y: 9 },
        ];

        builder.with_food(food).build()
    }
}

// ============================================================================
// FUZZING AND RANDOM GENERATION
// ============================================================================

/// Random board generator for property-based testing
pub struct RandomBoardGenerator {
    rng: StdRng,
    min_width: i32,
    max_width: i32,
    min_height: u32,
    max_height: u32,
}

impl RandomBoardGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            min_width: 7,
            max_width: 19,
            min_height: 7,
            max_height: 19,
        }
    }

    pub fn with_size_bounds(mut self, min_width: i32, max_width: i32, min_height: u32, max_height: u32) -> Self {
        self.min_width = min_width;
        self.max_width = max_width;
        self.min_height = min_height;
        self.max_height = max_height;
        self
    }

    /// Generate a random game state
    pub fn generate_random_game_state(&mut self) -> GameState {
        let width = self.rng.gen_range(self.min_width..=self.max_width);
        let height = self.rng.gen_range(self.min_height..=self.max_height);
        
        let num_opponents = self.rng.gen_range(0..=4);
        let num_food = self.rng.gen_range(1..=8);

        // Generate our snake
        let our_head = self.generate_random_coord(width, height as i32);
        let our_length = self.rng.gen_range(3..=8);
        let our_snake = self.generate_random_snake("us", our_head, our_length, width, height as i32);

        // Generate opponents
        let mut opponents = Vec::new();
        for i in 0..num_opponents {
            let opp_head = self.generate_non_colliding_coord(width, height as i32, &[our_snake.clone()], &opponents);
            let opp_length = self.rng.gen_range(3..=10);
            let opponent = self.generate_random_snake(&format!("opp-{}", i), opp_head, opp_length, width, height as i32);
            opponents.push(opponent);
        }

        // Generate food
        let mut all_snakes = vec![our_snake.clone()];
        all_snakes.extend(opponents.clone());
        let food = self.generate_random_food(width, height as i32, num_food, &all_snakes);

        let mut builder = GameStateBuilder::new()
            .with_board_size(width, height)
            .with_our_snake(our_snake)
            .with_food(food)
            .with_turn(self.rng.gen_range(0..=100));

        for opponent in opponents {
            builder = builder.with_opponent(opponent);
        }

        builder.build()
    }

    fn generate_random_coord(&mut self, width: i32, height: i32) -> Coord {
        Coord {
            x: self.rng.gen_range(0..width),
            y: self.rng.gen_range(0..height),
        }
    }

    fn generate_non_colliding_coord(&mut self, width: i32, height: i32, our_snake: &[Battlesnake], opponents: &[Battlesnake]) -> Coord {
        for _ in 0..100 { // Max attempts
            let coord = self.generate_random_coord(width, height);
            let mut is_safe = true;

            // Check collision with existing snakes
            for snake in our_snake.iter().chain(opponents.iter()) {
                for segment in &snake.body {
                    if *segment == coord || (segment.x - coord.x).abs() + (segment.y - coord.y).abs() <= 2 {
                        is_safe = false;
                        break;
                    }
                }
                if !is_safe { break; }
            }

            if is_safe {
                return coord;
            }
        }

        // Fallback to center if can't find safe position
        Coord { x: width / 2, y: height / 2 }
    }

    fn generate_random_snake(&mut self, id: &str, head: Coord, length: i32, width: i32, height: i32) -> Battlesnake {
        let mut body = vec![head];
        let mut current = head;

        // Generate body by random walk, avoiding boundaries
        for _ in 1..length {
            let directions = [(-1, 0), (1, 0), (0, -1), (0, 1)];
            let mut valid_moves = Vec::new();

            for (dx, dy) in directions.iter() {
                let next = Coord {
                    x: current.x + dx,
                    y: current.y + dy,
                };

                if next.x >= 0 && next.x < width && next.y >= 0 && next.y < height && !body.contains(&next) {
                    valid_moves.push(next);
                }
            }

            if let Some(&next_pos) = valid_moves.get(self.rng.gen_range(0..valid_moves.len().max(1))) {
                current = next_pos;
                body.push(current);
            } else {
                // Can't extend further, stop here
                break;
            }
        }

        Battlesnake {
            id: id.to_string(),
            name: format!("Random Snake {}", id),
            health: self.rng.gen_range(1..=100),
            body,
            head,
            length: body.len() as i32,
            latency: "0".to_string(),
            shout: None,
        }
    }

    fn generate_random_food(&mut self, width: i32, height: i32, count: usize, snakes: &[Battlesnake]) -> Vec<Coord> {
        let mut food = Vec::new();

        for _ in 0..count {
            for _ in 0..50 { // Max attempts per food
                let coord = self.generate_random_coord(width, height);
                let mut is_safe = true;

                // Ensure food doesn't spawn on snakes
                for snake in snakes {
                    if snake.body.contains(&coord) {
                        is_safe = false;
                        break;
                    }
                }

                if is_safe && !food.contains(&coord) {
                    food.push(coord);
                    break;
                }
            }
        }

        food
    }
}

// ============================================================================
// PERFORMANCE MEASUREMENT UTILITIES
// ============================================================================

/// Performance measurement utilities for benchmarking spatial analysis components
pub struct PerformanceMeasurement {
    pub component_times: HashMap<String, f32>,
    pub total_time: f32,
}

impl PerformanceMeasurement {
    pub fn new() -> Self {
        Self {
            component_times: HashMap::new(),
            total_time: 0.0,
        }
    }

    /// Measure the performance of a spatial analysis operation
    pub fn measure<F, R>(mut self, component_name: &str, operation: F) -> (R, Self)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let elapsed = start.elapsed().as_secs_f32() * 1000.0;
        
        self.component_times.insert(component_name.to_string(), elapsed);
        self.total_time += elapsed;
        
        (result, self)
    }

    /// Check if total time is within budget
    pub fn is_within_budget(&self, budget_ms: f32) -> bool {
        self.total_time <= budget_ms
    }

    /// Get the slowest component
    pub fn get_slowest_component(&self) -> Option<(&String, &f32)> {
        self.component_times.iter().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    }
}

// ============================================================================
// VALIDATION AND ASSERTION HELPERS
// ============================================================================

/// Validation helpers for testing spatial analysis outputs
pub struct SpatialAnalysisValidator;

impl SpatialAnalysisValidator {
    /// Validate that channel values are in expected range [0.0, 1.0]
    pub fn validate_channel_range(channel: &Vec<Vec<f32>>, channel_name: &str) -> Result<(), String> {
        for (y, row) in channel.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                if value < 0.0 || value > 1.0 {
                    return Err(format!("Channel {} at ({}, {}) has invalid value: {} (expected [0.0, 1.0])", 
                                     channel_name, x, y, value));
                }
            }
        }
        Ok(())
    }

    /// Validate channel dimensions match board size
    pub fn validate_channel_dimensions(channel: &Vec<Vec<f32>>, board: &Board, channel_name: &str) -> Result<(), String> {
        let expected_height = board.height as usize;
        let expected_width = board.width as usize;

        if channel.len() != expected_height {
            return Err(format!("Channel {} height mismatch: got {}, expected {}", 
                             channel_name, channel.len(), expected_height));
        }

        for (y, row) in channel.iter().enumerate() {
            if row.len() != expected_width {
                return Err(format!("Channel {} width at row {} mismatch: got {}, expected {}", 
                                 channel_name, y, row.len(), expected_width));
            }
        }

        Ok(())
    }

    /// Validate that territory analysis results are consistent
    pub fn validate_territory_consistency(our_territory: &Vec<Vec<f32>>, opponent_territory: &Vec<Vec<f32>>) -> Result<(), String> {
        if our_territory.len() != opponent_territory.len() {
            return Err("Territory channel dimensions mismatch".to_string());
        }

        for (y, (our_row, opp_row)) in our_territory.iter().zip(opponent_territory.iter()).enumerate() {
            if our_row.len() != opp_row.len() {
                return Err(format!("Territory row {} length mismatch", y));
            }

            for (x, (&our_val, &opp_val)) in our_row.iter().zip(opp_row.iter()).enumerate() {
                // Territory values should not be excessive when combined
                if our_val + opp_val > 1.5 {
                    return Err(format!("Combined territory values at ({}, {}) are excessive: {} + {} = {}", 
                                     x, y, our_val, opp_val, our_val + opp_val));
                }
            }
        }

        Ok(())
    }

    /// Validate that danger zone values make sense
    pub fn validate_danger_zones(danger_channel: &Vec<Vec<f32>>, board: &Board) -> Result<(), String> {
        for snake in &board.snakes {
            for segment in &snake.body {
                if segment.x >= 0 && segment.x < board.width && 
                   segment.y >= 0 && segment.y < board.height as i32 {
                    let x = segment.x as usize;
                    let y = segment.y as usize;
                    
                    // Snake body positions should have high danger values (> 0.7)
                    if danger_channel[y][x] < 0.7 {
                        return Err(format!("Snake body at ({}, {}) should have high danger, got: {}", 
                                         x, y, danger_channel[y][x]));
                    }
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// PROPERTY-BASED TEST HELPERS
// ============================================================================

/// Generate valid board configurations for property testing
pub fn arbitrary_valid_board_config(width_range: (i32, i32), height_range: (u32, u32)) -> impl Iterator<Item = (i32, u32)> {
    let mut configs = Vec::new();
    for width in width_range.0..=width_range.1 {
        for height in height_range.0..=height_range.1 {
            configs.push((width, height));
        }
    }
    configs.into_iter()
}

/// Generate edge case coordinates for boundary testing
pub fn generate_boundary_coordinates(width: i32, height: u32) -> Vec<Coord> {
    vec![
        Coord { x: 0, y: 0 },                                    // Top-left corner
        Coord { x: width - 1, y: 0 },                           // Top-right corner
        Coord { x: 0, y: height as i32 - 1 },                   // Bottom-left corner
        Coord { x: width - 1, y: height as i32 - 1 },           // Bottom-right corner
        Coord { x: width / 2, y: height as i32 / 2 },           // Center
        Coord { x: 0, y: height as i32 / 2 },                   // Left edge center
        Coord { x: width - 1, y: height as i32 / 2 },           // Right edge center
        Coord { x: width / 2, y: 0 },                           // Top edge center
        Coord { x: width / 2, y: height as i32 - 1 },           // Bottom edge center
    ]
}

// ============================================================================
// TEST ASSERTION MACROS
// ============================================================================

/// Assert that performance is within acceptable limits
#[macro_export]
macro_rules! assert_performance_budget {
    ($measurement:expr, $budget_ms:expr) => {
        assert!(
            $measurement.is_within_budget($budget_ms),
            "Performance budget exceeded: {:.2}ms > {:.2}ms. Slowest component: {:?}",
            $measurement.total_time,
            $budget_ms,
            $measurement.get_slowest_component()
        );
    };
}

/// Assert that channel values are valid
#[macro_export]
macro_rules! assert_valid_channel {
    ($channel:expr, $board:expr, $name:expr) => {
        SpatialAnalysisValidator::validate_channel_range($channel, $name)
            .expect(&format!("Channel {} range validation failed", $name));
        SpatialAnalysisValidator::validate_channel_dimensions($channel, $board, $name)
            .expect(&format!("Channel {} dimension validation failed", $name));
    };
}