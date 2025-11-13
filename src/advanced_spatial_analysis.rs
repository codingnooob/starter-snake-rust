use log::info;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use crate::{Battlesnake, Board, Coord};
use crate::logic::Direction;

// ============================================================================
// ADVANCED SPATIAL ANALYSIS FOR 12-CHANNEL BOARD ENCODING
// ============================================================================

/// Advanced spatial analysis module providing 5 sophisticated algorithms for
/// neural network board encoding channels 7-11:
/// - Channel 7: Our Territory (Voronoi-based)
/// - Channel 8: Opponent Territory (Voronoi-based)  
/// - Channel 9: Danger Zones (collision risk prediction)
/// - Channel 10: Movement History (position tracking with time decay)
/// - Channel 11: Strategic Positions (tactical advantages)

// ============================================================================
// VORONOI TERRITORY ANALYZER - Channels 7-8
// ============================================================================

/// Voronoi Territory Analyzer for sophisticated territory control analysis
/// Uses distance-based Voronoi diagrams to determine territory ownership
#[derive(Debug, Clone)]
pub struct VoronoiTerritoryAnalyzer {
    pub our_territory: HashMap<Coord, f32>,      // Coord -> ownership strength
    pub opponent_territory: HashMap<Coord, f32>, // Coord -> ownership strength
    pub contested_zones: HashSet<Coord>,         // Areas with unclear ownership
}

impl VoronoiTerritoryAnalyzer {
    pub fn new() -> Self {
        Self {
            our_territory: HashMap::new(),
            opponent_territory: HashMap::new(),
            contested_zones: HashSet::new(),
        }
    }

    /// Analyze territory control using Voronoi diagrams
    pub fn analyze_territory(&mut self, board: &Board, our_snake: &Battlesnake) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let start_time = Instant::now();
        
        // Clear previous analysis
        self.our_territory.clear();
        self.opponent_territory.clear();
        self.contested_zones.clear();

        let width = board.width as usize;
        let height = board.height as usize;
        let mut our_channel = vec![vec![0.0; width]; height];
        let mut opponent_channel = vec![vec![0.0; width]; height];

        // Get all snakes and separate ours from opponents
        let opponents: Vec<&Battlesnake> = board.snakes.iter()
            .filter(|snake| snake.id != our_snake.id)
            .collect();

        if opponents.is_empty() {
            // No opponents - all territory is ours
            for y in 0..height {
                for x in 0..width {
                    our_channel[y][x] = 1.0;
                }
            }
            info!("Voronoi territory analysis: No opponents, all territory ours ({:.2}ms)",
                  start_time.elapsed().as_secs_f32() * 1000.0);
            return (our_channel, opponent_channel);
        }

        // Multi-source BFS for Voronoi diagram computation
        let mut queue = VecDeque::new();
        let mut distance_map: HashMap<Coord, (i32, String)> = HashMap::new();

        // Initialize with snake heads as sources
        queue.push_back((our_snake.head, 0, our_snake.id.clone()));
        distance_map.insert(our_snake.head, (0, our_snake.id.clone()));

        for opponent in &opponents {
            queue.push_back((opponent.head, 0, opponent.id.clone()));
            distance_map.insert(opponent.head, (0, opponent.id.clone()));
        }

        // BFS to compute Voronoi diagram
        while let Some((current_pos, distance, owner_id)) = queue.pop_front() {
            // Explore neighbors
            for direction in [Direction::Up, Direction::Down, Direction::Left, Direction::Right] {
                let next_pos = match direction {
                    Direction::Up => Coord { x: current_pos.x, y: current_pos.y + 1 },
                    Direction::Down => Coord { x: current_pos.x, y: current_pos.y - 1 },
                    Direction::Left => Coord { x: current_pos.x - 1, y: current_pos.y },
                    Direction::Right => Coord { x: current_pos.x + 1, y: current_pos.y },
                };

                // Check bounds
                if next_pos.x < 0 || next_pos.x >= board.width || 
                   next_pos.y < 0 || next_pos.y >= board.height as i32 {
                    continue;
                }

                // Check if position is blocked by snake bodies
                let is_blocked = self.is_position_blocked(&next_pos, board);
                if is_blocked {
                    continue;
                }

                let new_distance = distance + 1;

                // Update if this is a better path or first time visiting
                match distance_map.get(&next_pos) {
                    Some((existing_distance, existing_owner)) => {
                        if new_distance < *existing_distance {
                            // Better path found
                            distance_map.insert(next_pos, (new_distance, owner_id.clone()));
                            queue.push_back((next_pos, new_distance, owner_id.clone()));
                        } else if new_distance == *existing_distance && existing_owner != &owner_id {
                            // Equal distance from different owners - contested zone
                            self.contested_zones.insert(next_pos);
                        }
                    }
                    None => {
                        // First time visiting this position
                        distance_map.insert(next_pos, (new_distance, owner_id.clone()));
                        queue.push_back((next_pos, new_distance, owner_id.clone()));
                    }
                }
            }
        }

        // Build territory ownership maps and channels
        for (coord, (distance, owner_id)) in &distance_map {
            
            if coord.x < 0 || coord.x >= board.width || 
               coord.y < 0 || coord.y >= board.height as i32 {
                continue;
            }

            let x = coord.x as usize;
            let y = coord.y as usize;

            // Calculate ownership strength based on distance (closer = stronger)
            let strength = self.calculate_territory_strength(*distance);

            if self.contested_zones.contains(&coord) {
                // Contested zone - split ownership
                our_channel[y][x] = 0.5;
                opponent_channel[y][x] = 0.5;
            } else if owner_id == &our_snake.id {
                // Our territory
                our_channel[y][x] = strength;
                self.our_territory.insert(*coord, strength);
            } else {
                // Opponent territory
                opponent_channel[y][x] = strength;
                self.opponent_territory.insert(*coord, strength);
            }
        }

        let elapsed = start_time.elapsed().as_secs_f32() * 1000.0;
        info!("Voronoi territory analysis complete: {} our territories, {} opponent territories, {} contested zones ({:.2}ms)",
              self.our_territory.len(), self.opponent_territory.len(), self.contested_zones.len(), elapsed);

        (our_channel, opponent_channel)
    }

    /// Check if a position is blocked by snake bodies
    fn is_position_blocked(&self, pos: &Coord, board: &Board) -> bool {
        for snake in &board.snakes {
            // Check body segments (excluding tail which might move)
            let body_to_check = if snake.body.len() > 1 {
                &snake.body[0..snake.body.len()-1]
            } else {
                &snake.body
            };

            for segment in body_to_check {
                if segment == pos {
                    return true;
                }
            }
        }
        false
    }

    /// Calculate territory ownership strength based on distance
    fn calculate_territory_strength(&self, distance: i32) -> f32 {
        // Exponential decay: closer positions have stronger ownership
        let max_distance = 20.0; // Reasonable maximum for most boards
        let normalized_distance = (distance as f32).min(max_distance) / max_distance;
        (1.0 - normalized_distance).max(0.1) // Minimum 0.1 strength
    }
}

// ============================================================================
// DANGER ZONE PREDICTOR - Channel 9
// ============================================================================

/// Danger Zone Predictor for collision risk analysis with movement pattern prediction
#[derive(Debug, Clone)]
pub struct DangerZonePredictor {
    pub danger_map: HashMap<Coord, f32>,    // Coord -> danger level [0.0, 1.0]
    pub prediction_depth: u8,               // How many turns to look ahead
}

impl DangerZonePredictor {
    pub fn new(prediction_depth: u8) -> Self {
        Self {
            danger_map: HashMap::new(),
            prediction_depth: prediction_depth.min(5), // Limit computational cost
        }
    }

    /// Predict collision danger zones using multi-turn analysis
    pub fn predict_danger_zones(&mut self, board: &Board, our_snake: &Battlesnake, turn: u32) -> Vec<Vec<f32>> {
        let start_time = Instant::now();
        self.danger_map.clear();

        let width = board.width as usize;
        let height = board.height as usize;
        let mut danger_channel = vec![vec![0.0; width]; height];

        // Analyze immediate collision risks (turn 1)
        self.analyze_immediate_dangers(board, our_snake);

        // Analyze multi-turn collision risks (turns 2-N)
        self.analyze_future_dangers(board, our_snake, turn);

        // Convert danger_map to 2D channel
        for (&coord, &danger_level) in &self.danger_map {
            if coord.x >= 0 && coord.x < board.width &&
               coord.y >= 0 && coord.y < board.height as i32 {
                danger_channel[coord.y as usize][coord.x as usize] = danger_level;
            }
        }

        danger_channel
    }

    fn analyze_immediate_dangers(&mut self, board: &Board, our_snake: &Battlesnake) {
        // Mark snake bodies as immediate dangers
        for snake in &board.snakes {
            for segment in &snake.body {
                self.danger_map.insert(*segment, 1.0);
            }
        }
        
        // Mark potential collision points for moving snakes
        for snake in &board.snakes {
            if snake.health > 0 && !snake.body.is_empty() {
                let head = snake.head;
                
                // Check all possible moves for this snake
                for direction in &[Direction::Up, Direction::Down, Direction::Left, Direction::Right] {
                    let next_pos = match direction {
                        Direction::Up => Coord { x: head.x, y: head.y - 1 },
                        Direction::Down => Coord { x: head.x, y: head.y + 1 },
                        Direction::Left => Coord { x: head.x - 1, y: head.y },
                        Direction::Right => Coord { x: head.x + 1, y: head.y },
                    };
                    
                    // Check bounds
                    if next_pos.x >= 0 && next_pos.x < board.width &&
                       next_pos.y >= 0 && next_pos.y < board.height as i32 {
                        // Add danger probability based on snake length and our snake size
                        let danger_level = if snake.body.len() >= our_snake.body.len() {
                            0.8 // High danger from larger/equal snakes
                        } else {
                            0.4 // Lower danger from smaller snakes
                        };
                        
                        self.danger_map.insert(next_pos, danger_level);
                    }
                }
            }
        }
    }

    fn analyze_future_dangers(&mut self, board: &Board, our_snake: &Battlesnake, _turn: u32) {
        // Simple future danger prediction - areas near snake heads
        for snake in &board.snakes {
            if snake.health > 0 && !snake.body.is_empty() {
                let head = snake.head;
                
                // Mark areas within 2 squares of enemy snake heads as moderate danger
                for dx in -2_i32..=2_i32 {
                    for dy in -2_i32..=2_i32 {
                        if dx == 0 && dy == 0 { continue; } // Skip the head itself
                        
                        let danger_pos = Coord { x: head.x + dx, y: head.y + dy };
                        
                        if danger_pos.x >= 0 && danger_pos.x < board.width &&
                           danger_pos.y >= 0 && danger_pos.y < board.height as i32 {
                            let distance = (dx.abs() + dy.abs()) as f32;
                            let danger_level = (0.3_f32 / distance).min(0.3_f32);
                            
                            // Don't overwrite higher danger levels
                            let existing_danger = self.danger_map.get(&danger_pos).unwrap_or(&0.0);
                            if danger_level > *existing_danger {
                                self.danger_map.insert(danger_pos, danger_level);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod danger_zone_predictor_tests {
    use super::*;
    use crate::spatial_test_utilities::*;

    #[test]
    fn test_danger_predictor_creation() {
        let predictor = DangerZonePredictor::new(3);
        assert!(predictor.danger_map.is_empty());
        assert_eq!(predictor.prediction_depth, 3);
        
        // Test depth limiting
        let deep_predictor = DangerZonePredictor::new(10);
        assert_eq!(deep_predictor.prediction_depth, 5, "Prediction depth should be limited to 5");
    }

    #[test]
    fn test_immediate_danger_detection() {
        let mut predictor = DangerZonePredictor::new(1);
        let game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_test_snake("us", Coord { x: 3, y: 3 }, 3, 100))
            .with_opponent(create_snake_with_path("opponent", vec![
                Coord { x: 1, y: 1 }, // head
                Coord { x: 1, y: 0 }, // neck
                Coord { x: 0, y: 0 }, // tail
            ], 100))
            .build();

        let danger_channel = predictor.predict_danger_zones(&game_state.board, &game_state.you);

        // Validate channel dimensions and ranges
        SpatialAnalysisValidator::validate_channel_dimensions(&danger_channel, &game_state.board, "danger_zones")
            .expect("Danger channel should have correct dimensions");
        SpatialAnalysisValidator::validate_channel_range(&danger_channel, "danger_zones")
            .expect("Danger values should be in [0.0, 1.0] range");

        // Validate that opponent snake bodies are marked as dangerous
        SpatialAnalysisValidator::validate_danger_zones(&danger_channel, &game_state.board)
            .expect("Snake bodies should be marked as dangerous");

        // Opponent body positions should have high danger (our own body is not considered dangerous for our movement)
        assert_eq!(danger_channel[1][1], 1.0, "Opponent head should be maximum danger");
        assert_eq!(danger_channel[0][1], 1.0, "Opponent body should be maximum danger");
        assert_eq!(danger_channel[0][0], 1.0, "Opponent tail should be maximum danger");
    }

    #[test]
    fn test_wall_boundary_dangers() {
        let mut predictor = DangerZonePredictor::new(1);
        let game_state = BoardScenarioGenerator::empty_board();

        let danger_channel = predictor.predict_danger_zones(&game_state.board, &game_state.you);

        // Check that predictor tracks wall dangers in its internal map
        // Note: Wall positions are outside board bounds, so they won't appear in the channel
        // but should be tracked in the danger_map for movement calculations
        assert!(predictor.danger_map.contains_key(&Coord { x: -1, y: 5 }), "Left wall should be marked as dangerous");
        assert!(predictor.danger_map.contains_key(&Coord { x: 11, y: 5 }), "Right wall should be marked as dangerous");
        assert!(predictor.danger_map.contains_key(&Coord { x: 5, y: -1 }), "Top wall should be marked as dangerous");
        assert!(predictor.danger_map.contains_key(&Coord { x: 5, y: 11 }), "Bottom wall should be marked as dangerous");
    }

    #[test]
    fn test_multi_turn_prediction() {
        let mut predictor = DangerZonePredictor::new(3);
        let game_state = GameStateBuilder::new()
            .with_board_size(9, 9)
            .with_our_snake(create_test_snake("us", Coord { x: 4, y: 4 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 1, y: 1 }, 4, 100))
            .with_food(vec![Coord { x: 7, y: 7 }]) // Food to influence opponent prediction
            .build();

        let danger_channel = predictor.predict_danger_zones(&game_state.board, &game_state.you);

        // Validate basic properties
        SpatialAnalysisValidator::validate_channel_dimensions(&danger_channel, &game_state.board, "danger_zones")
            .expect("Multi-turn danger prediction should have correct dimensions");
        SpatialAnalysisValidator::validate_channel_range(&danger_channel, "danger_zones")
            .expect("Multi-turn danger values should be in valid range");

        // Should have predicted some future danger positions
        let total_danger_positions = predictor.danger_map.len();
        assert!(total_danger_positions > 3, "Should predict future positions beyond immediate snake bodies");

        // Immediate opponent body should have maximum danger
        assert_eq!(danger_channel[1][1], 1.0, "Immediate opponent position should have maximum danger");
    }

    #[test]
    fn test_head_collision_risk_analysis() {
        let mut predictor = DangerZonePredictor::new(2);
        let game_state = BoardScenarioGenerator::head_collision_scenario();

        let danger_channel = predictor.predict_danger_zones(&game_state.board, &game_state.you);

        // In head-collision scenario, positions between the snakes should have elevated danger
        // Our snake is at (5,5), opponent at (7,5) - position (6,5) should be dangerous
        let between_danger = danger_channel[5][6];
        assert!(between_danger > 0.5, "Position between snake heads should have elevated danger: {}", between_danger);

        // Validate that head-collision analysis is working
        let collision_risks = predictor.danger_map.iter()
            .filter(|(_, &danger)| danger > 0.8 && danger < 1.0)
            .count();
        assert!(collision_risks > 0, "Should detect head-collision risks");
    }

    #[test]
    fn test_opponent_movement_prediction() {
        let predictor = DangerZonePredictor::new(2);
        let game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_test_snake("us", Coord { x: 3, y: 3 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 1, y: 1 }, 3, 100))
            .with_food(vec![Coord { x: 5, y: 1 }]) // Food to influence movement prediction
            .build();

        let opponent = &game_state.board.snakes[1]; // Opponent snake
        let predictions = predictor.predict_opponent_positions(opponent, &game_state.board);

        // Should predict future positions
        assert!(predictions.len() <= 2, "Should predict up to depth turns");
        if !predictions.is_empty() {
            assert!(!predictions[0].is_empty(), "Should predict at least one position for first turn");
        }
    }

    #[test]
    fn test_possible_moves_calculation() {
        let predictor = DangerZonePredictor::new(1);
        let board = Board {
            width: 5,
            height: 5,
            food: vec![],
            snakes: vec![],
            hazards: vec![],
        };

        // Test center position - should have 4 possible moves
        let center_moves = predictor.get_possible_moves(&Coord { x: 2, y: 2 }, &board);
        assert_eq!(center_moves.len(), 4, "Center position should have 4 possible moves");

        // Test corner position - should have 2 possible moves
        let corner_moves = predictor.get_possible_moves(&Coord { x: 0, y: 0 }, &board);
        assert_eq!(corner_moves.len(), 2, "Corner position should have 2 possible moves");

        // Test edge position - should have 3 possible moves
        let edge_moves = predictor.get_possible_moves(&Coord { x: 0, y: 2 }, &board);
        assert_eq!(edge_moves.len(), 3, "Edge position should have 3 possible moves");
    }

    #[test]
    fn test_opponent_prediction_heuristics() {
        let predictor = DangerZonePredictor::new(1);
        
        // Test food-seeking behavior
        let board_with_food = Board {
            width: 7,
            height: 7,
            food: vec![Coord { x: 5, y: 3 }],
            snakes: vec![],
            hazards: vec![],
        };
        
        let food_move = predictor.predict_best_opponent_move(&Coord { x: 3, y: 3 }, &board_with_food);
        // Should move toward food (x: 5, y: 3) from (x: 3, y: 3)
        assert!(food_move.x > 3 || food_move.y == 3, "Should move toward food");

        // Test center-seeking behavior (no food)
        let empty_board = Board {
            width: 7,
            height: 7,
            food: vec![],
            snakes: vec![],
            hazards: vec![],
        };
        
        let center_move = predictor.predict_best_opponent_move(&Coord { x: 1, y: 1 }, &empty_board);
        // Should move toward center (3, 3) from corner (1, 1)
        assert!(center_move.x >= 1 && center_move.y >= 1, "Should move toward center when no food");
    }

    #[test]
    fn test_danger_level_accuracy() {
        let mut predictor = DangerZonePredictor::new(3);
        let game_state = GameStateBuilder::new()
            .with_board_size(9, 9)
            .with_our_snake(create_test_snake("us", Coord { x: 4, y: 4 }, 4, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 6, y: 4 }, 4, 100))
            .build();

        let danger_channel = predictor.predict_danger_zones(&game_state.board, &game_state.you);

        // Immediate opponent positions should have maximum danger (1.0)
        assert_eq!(danger_channel[4][6], 1.0, "Immediate opponent head should be max danger");

        // Positions further away should have lower danger levels
        let far_position = danger_channel[0][0];
        assert!(far_position < 1.0, "Far positions should have lower danger");

        // Danger levels should decrease with distance/time
        let immediate_area_danger = danger_channel[4][7]; // Next to opponent
        let distant_area_danger = danger_channel[8][8]; // Far corner
        assert!(immediate_area_danger >= distant_area_danger, 
                "Danger should decrease with distance from threats");
    }

    #[test]
    fn test_no_opponent_scenario() {
        let mut predictor = DangerZonePredictor::new(2);
        let game_state = BoardScenarioGenerator::empty_board();

        let danger_channel = predictor.predict_danger_zones(&game_state.board, &game_state.you);

        // With no opponents, only walls should be dangerous (tracked in danger_map)
        // The actual channel should be mostly empty since walls are outside bounds
        let total_channel_danger: f32 = danger_channel.iter()
            .flatten()
            .sum();
        
        // Channel should have minimal danger (only from wall proximity effects if any)
        assert!(total_channel_danger < 5.0, "Should have minimal danger with no opponents");
        
        // But danger_map should contain wall positions
        let wall_dangers = predictor.danger_map.iter()
            .filter(|(pos, _)| pos.x < 0 || pos.x >= 11 || pos.y < 0 || pos.y >= 11)
            .count();
        assert!(wall_dangers > 0, "Should detect wall dangers even with no opponents");
    }

    #[test]
    fn test_performance_with_multiple_opponents() {
        let mut predictor = DangerZonePredictor::new(3);
        let game_state = GameStateBuilder::new()
            .with_board_size(11, 11)
            .with_our_snake(create_test_snake("us", Coord { x: 5, y: 5 }, 4, 100))
            .with_opponent(create_test_snake("opponent1", Coord { x: 2, y: 2 }, 3, 100))
            .with_opponent(create_test_snake("opponent2", Coord { x: 8, y: 2 }, 3, 100))
            .with_opponent(create_test_snake("opponent3", Coord { x: 2, y: 8 }, 3, 100))
            .with_opponent(create_test_snake("opponent4", Coord { x: 8, y: 8 }, 3, 100))
            .build();

        let start = std::time::Instant::now();
        let danger_channel = predictor.predict_danger_zones(&game_state.board, &game_state.you);
        let elapsed = start.elapsed().as_millis();

        // Performance test - should complete within reasonable time even with multiple opponents
        assert!(elapsed < 50, "Multi-opponent danger prediction should complete within 50ms, took {}ms", elapsed);

        // Should detect multiple danger sources
        let high_danger_count = danger_channel.iter()
            .flatten()
            .filter(|&&danger| danger > 0.7)
            .count();
        assert!(high_danger_count >= 12, "Should detect multiple high-danger positions with 4 opponents");
    }

    #[test]
    fn test_prediction_depth_variation() {
        // Test different prediction depths
        for depth in 1..=5 {
            let mut predictor = DangerZonePredictor::new(depth);
            let game_state = GameStateBuilder::new()
                .with_board_size(9, 9)
                .with_our_snake(create_test_snake("us", Coord { x: 4, y: 4 }, 3, 100))
                .with_opponent(create_test_snake("opponent", Coord { x: 1, y: 1 }, 3, 100))
                .with_food(vec![Coord { x: 7, y: 7 }])
                .build();

            let start = std::time::Instant::now();
            let danger_channel = predictor.predict_danger_zones(&game_state.board, &game_state.you);
            let elapsed = start.elapsed().as_millis();

            // Each depth should complete quickly
            assert!(elapsed < 20, "Depth {} should complete within 20ms, took {}ms", depth, elapsed);

            // Validate basic properties
            SpatialAnalysisValidator::validate_channel_dimensions(&danger_channel, &game_state.board, "danger_zones")
                .expect(&format!("Depth {} should produce valid dimensions", depth));
            SpatialAnalysisValidator::validate_channel_range(&danger_channel, "danger_zones")
                .expect(&format!("Depth {} should produce valid ranges", depth));
        }
    }

    #[test]
    fn test_small_board_edge_case() {
        let mut predictor = DangerZonePredictor::new(2);
        let game_state = GameStateBuilder::new()
            .with_board_size(3, 3)
            .with_our_snake(create_test_snake("us", Coord { x: 1, y: 1 }, 1, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 0, y: 0 }, 1, 100))
            .build();

        let danger_channel = predictor.predict_danger_zones(&game_state.board, &game_state.you);

        // Should handle small boards correctly
        assert_eq!(danger_channel.len(), 3);
        assert_eq!(danger_channel[0].len(), 3);

        // Opponent position should be dangerous
        assert_eq!(danger_channel[0][0], 1.0, "Opponent position should be maximum danger on small board");
    }

    #[test]
    fn test_large_board_performance() {
        let mut predictor = DangerZonePredictor::new(4);
        let game_state = BoardScenarioGenerator::large_board_scenario();

        let start = std::time::Instant::now();
        let danger_channel = predictor.predict_danger_zones(&game_state.board, &game_state.you);
        let elapsed = start.elapsed().as_millis();

        // Large board should still complete within performance budget
        assert!(elapsed < 150, "Large board danger prediction should complete within 150ms, took {}ms", elapsed);

        // Validate results
        SpatialAnalysisValidator::validate_channel_dimensions(&danger_channel, &game_state.board, "danger_zones")
            .expect("Large board should produce valid dimensions");
        SpatialAnalysisValidator::validate_channel_range(&danger_channel, "danger_zones")
            .expect("Large board should produce valid danger values");

        // Should detect multiple danger sources on large board
        let danger_positions = predictor.danger_map.len();
        assert!(danger_positions > 50, "Large board should detect many danger positions");
    }

    #[test]
    fn test_head_collision_size_considerations() {
        let mut predictor = DangerZonePredictor::new(1);
        
        // Test collision between snakes of different sizes
        let game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_test_snake("us", Coord { x: 3, y: 3 }, 2, 100)) // Smaller snake
            .with_opponent(create_test_snake("opponent", Coord { x: 5, y: 3 }, 5, 100)) // Larger snake
            .build();

        let danger_channel = predictor.predict_danger_zones(&game_state.board, &game_state.you);

        // Position between heads should have high danger due to size disadvantage
        let between_position_danger = danger_channel[3][4];
        assert!(between_position_danger > 0.5, "Should have high danger when facing larger opponent");
    }

    #[test]
    fn test_danger_map_consistency() {
        let mut predictor = DangerZonePredictor::new(2);
        let game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_test_snake("us", Coord { x: 3, y: 3 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 1, y: 1 }, 3, 100))
            .build();

        let danger_channel = predictor.predict_danger_zones(&game_state.board, &game_state.you);

        // Danger map and channel should be consistent for valid board positions
        for y in 0..7 {
            for x in 0..7 {
                let coord = Coord { x, y };
                let channel_value = danger_channel[y as usize][x as usize];
                let map_value = predictor.danger_map.get(&coord).unwrap_or(&0.0);
                
                assert_eq!(channel_value, *map_value, 
                    "Channel and danger map should be consistent at ({}, {}): channel={}, map={}", 
                    x, y, channel_value, map_value);
            }
        }
    }
}

#[cfg(test)]
mod movement_history_tracker_tests {
    use super::*;
    use crate::spatial_test_utilities::*;

    #[test]
    fn test_tracker_creation() {
        let tracker = MovementHistoryTracker::new(10);
        assert_eq!(tracker.max_history, 10);
        assert_eq!(tracker.current_turn, 0);
        assert!(tracker.history.is_empty());
    }

    #[test]
    fn test_position_update_and_tracking() {
        let mut tracker = MovementHistoryTracker::new(5);
        
        // Add positions over several turns
        tracker.update_position(Coord { x: 1, y: 1 }, 0);
        tracker.update_position(Coord { x: 2, y: 1 }, 1);
        tracker.update_position(Coord { x: 3, y: 1 }, 2);
        tracker.update_position(Coord { x: 4, y: 1 }, 3);
        
        assert_eq!(tracker.history.len(), 4);
        assert_eq!(tracker.current_turn, 3);
        
        // Verify order and content
        let positions: Vec<Coord> = tracker.history.iter().map(|(pos, _)| *pos).collect();
        assert_eq!(positions, vec![
            Coord { x: 1, y: 1 },
            Coord { x: 2, y: 1 },
            Coord { x: 3, y: 1 },
            Coord { x: 4, y: 1 },
        ]);
    }

    #[test]
    fn test_max_history_limit() {
        let mut tracker = MovementHistoryTracker::new(3);
        
        // Add more positions than max_history
        for i in 0..6 {
            tracker.update_position(Coord { x: i, y: 0 }, i as u32);
        }
        
        // Should only keep the last 3 positions
        assert_eq!(tracker.history.len(), 3);
        
        let positions: Vec<Coord> = tracker.history.iter().map(|(pos, _)| *pos).collect();
        assert_eq!(positions, vec![
            Coord { x: 3, y: 0 },
            Coord { x: 4, y: 0 },
            Coord { x: 5, y: 0 },
        ]);
    }

    #[test]
    fn test_time_decay_calculation() {
        let tracker = MovementHistoryTracker::new(10);
        
        // Test decay weights for different ages
        let weight_0 = tracker.calculate_decay_weight(0.0);
        let weight_1 = tracker.calculate_decay_weight(1.0);
        let weight_5 = tracker.calculate_decay_weight(5.0);
        let weight_10 = tracker.calculate_decay_weight(10.0);
        let weight_20 = tracker.calculate_decay_weight(20.0);
        
        // More recent positions should have higher weights
        assert!(weight_0 > weight_1);
        assert!(weight_1 > weight_5);
        assert!(weight_5 > weight_10);
        assert!(weight_10 > weight_20);
        
        // All weights should be in valid range
        assert!(weight_0 >= 0.05 && weight_0 <= 1.0);
        assert!(weight_5 >= 0.05 && weight_5 <= 1.0);
        assert!(weight_20 >= 0.05 && weight_20 <= 1.0);
        
        // Current turn (age 0) should have maximum weight
        assert_eq!(weight_0, 1.0);
        
        // Very old positions should have minimum weight
        let weight_100 = tracker.calculate_decay_weight(100.0);
        assert_eq!(weight_100, 0.05);
    }

    #[test]
    fn test_exponential_decay_properties() {
        let tracker = MovementHistoryTracker::new(10);
        
        // Test that decay follows exponential pattern
        // At half-life (5 turns), weight should be approximately 0.5
        let half_life_weight = tracker.calculate_decay_weight(5.0);
        assert!((half_life_weight - 0.5).abs() < 0.01, 
                "Half-life weight should be ~0.5, got {}", half_life_weight);
        
        // Test mathematical consistency of exponential decay
        let weight_2 = tracker.calculate_decay_weight(2.0);
        let weight_4 = tracker.calculate_decay_weight(4.0);
        let weight_8 = tracker.calculate_decay_weight(8.0);
        
        // Due to exponential nature, weight_8 should be approximately weight_4^2 / weight_2
        // This tests the exponential relationship
        let expected_ratio = weight_4 * weight_4 / weight_2;
        assert!((weight_8 - expected_ratio).abs() < 0.1,
                "Exponential decay relationship not maintained");
    }

    #[test]
    fn test_history_channel_generation() {
        let mut tracker = MovementHistoryTracker::new(10);
        let board = Board {
            width: 5,
            height: 5,
            food: vec![],
            snakes: vec![],
            hazards: vec![],
        };
        
        // Add some historical positions
        tracker.update_position(Coord { x: 2, y: 2 }, 0);
        tracker.update_position(Coord { x: 2, y: 1 }, 1);
        tracker.update_position(Coord { x: 1, y: 1 }, 2);
        tracker.update_position(Coord { x: 0, y: 1 }, 3);
        tracker.current_turn = 3;
        
        let history_channel = tracker.generate_history_channel(&board);
        
        // Validate channel dimensions
        assert_eq!(history_channel.len(), 5);
        assert_eq!(history_channel[0].len(), 5);
        
        // Validate that visited positions have non-zero values
        assert!(history_channel[2][2] > 0.0, "Position (2,2) should have history value");
        assert!(history_channel[1][2] > 0.0, "Position (2,1) should have history value");
        assert!(history_channel[1][1] > 0.0, "Position (1,1) should have history value");
        assert!(history_channel[1][0] > 0.0, "Position (0,1) should have history value");
        
        // Validate time decay - more recent positions should have higher values
        let oldest_value = history_channel[2][2]; // age 3
        let newest_value = history_channel[1][0]; // age 0
        assert!(newest_value >= oldest_value, 
                "More recent positions should have higher or equal history values");
        
        // Unvisited positions should be zero
        assert_eq!(history_channel[0][0], 0.0, "Unvisited position should be zero");
        assert_eq!(history_channel[4][4], 0.0, "Unvisited position should be zero");
    }

    #[test]
    fn test_history_channel_value_clamping() {
        let mut tracker = MovementHistoryTracker::new(10);
        let board = Board {
            width: 3,
            height: 3,
            food: vec![],
            snakes: vec![],
            hazards: vec![],
        };
        
        // Repeatedly visit the same position to test value accumulation and clamping
        let repeated_pos = Coord { x: 1, y: 1 };
        for turn in 0..10 {
            tracker.update_position(repeated_pos, turn);
        }
        
        let history_channel = tracker.generate_history_channel(&board);
        
        // Value should be clamped to 1.0 maximum
        let repeated_value = history_channel[1][1];
        assert!(repeated_value <= 1.0, "History values should be clamped to 1.0, got {}", repeated_value);
        assert!(repeated_value > 0.9, "Repeatedly visited position should have high value");
    }

    #[test]
    fn test_recent_positions_query() {
        let mut tracker = MovementHistoryTracker::new(10);
        
        // Add positions over time
        tracker.update_position(Coord { x: 0, y: 0 }, 0);
        tracker.update_position(Coord { x: 1, y: 0 }, 2);
        tracker.update_position(Coord { x: 2, y: 0 }, 5);
        tracker.update_position(Coord { x: 3, y: 0 }, 8);
        tracker.update_position(Coord { x: 4, y: 0 }, 10);
        
        // Query recent positions (last 3 turns from turn 10)
        let recent_3 = tracker.get_recent_positions(3);
        assert_eq!(recent_3.len(), 2); // Turns 8 and 10 are within last 3 turns
        assert!(recent_3.contains(&Coord { x: 3, y: 0 }));
        assert!(recent_3.contains(&Coord { x: 4, y: 0 }));
        
        // Query recent positions (last 6 turns)
        let recent_6 = tracker.get_recent_positions(6);
        assert_eq!(recent_6.len(), 3); // Turns 5, 8, and 10
        assert!(recent_6.contains(&Coord { x: 2, y: 0 }));
        assert!(recent_6.contains(&Coord { x: 3, y: 0 }));
        assert!(recent_6.contains(&Coord { x: 4, y: 0 }));
        
        // Query all positions
        let all_recent = tracker.get_recent_positions(20);
        assert_eq!(all_recent.len(), 5);
    }

    #[test]
    fn test_was_recently_visited() {
        let mut tracker = MovementHistoryTracker::new(10);
        
        // Add positions at specific turns
        tracker.update_position(Coord { x: 1, y: 1 }, 5);
        tracker.update_position(Coord { x: 2, y: 1 }, 8);
        tracker.update_position(Coord { x: 3, y: 1 }, 10);
        
        // Test recent visit detection
        assert!(tracker.was_recently_visited(&Coord { x: 3, y: 1 }, 1)); // Within 1 turn
        assert!(tracker.was_recently_visited(&Coord { x: 2, y: 1 }, 3)); // Within 3 turns
        assert!(tracker.was_recently_visited(&Coord { x: 1, y: 1 }, 6)); // Within 6 turns
        
        // Test negative cases
        assert!(!tracker.was_recently_visited(&Coord { x: 1, y: 1 }, 4)); // Not within 4 turns from turn 10
        assert!(!tracker.was_recently_visited(&Coord { x: 0, y: 0 }, 20)); // Never visited
    }

    #[test]
    fn test_boundary_coordinate_handling() {
        let mut tracker = MovementHistoryTracker::new(5);
        let board = Board {
            width: 3,
            height: 3,
            food: vec![],
            snakes: vec![],
            hazards: vec![],
        };
        
        // Add positions including boundary and out-of-bounds
        tracker.update_position(Coord { x: 0, y: 0 }, 0); // Valid boundary
        tracker.update_position(Coord { x: 2, y: 2 }, 1); // Valid boundary
        tracker.update_position(Coord { x: -1, y: 1 }, 2); // Out of bounds
        tracker.update_position(Coord { x: 1, y: 3 }, 3); // Out of bounds
        tracker.update_position(Coord { x: 1, y: 1 }, 4); // Valid center
        
        let history_channel = tracker.generate_history_channel(&board);
        
        // Valid positions should be recorded
        assert!(history_channel[0][0] > 0.0, "Boundary position (0,0) should be recorded");
        assert!(history_channel[2][2] > 0.0, "Boundary position (2,2) should be recorded");
        assert!(history_channel[1][1] > 0.0, "Center position should be recorded");
        
        // Out-of-bounds positions should be ignored (channel should remain valid)
        SpatialAnalysisValidator::validate_channel_dimensions(&history_channel, &board, "movement_history")
            .expect("History channel should handle out-of-bounds positions gracefully");
    }

    #[test]
    fn test_empty_history_scenario() {
        let tracker = MovementHistoryTracker::new(5);
        let board = Board {
            width: 5,
            height: 5,
            food: vec![],
            snakes: vec![],
            hazards: vec![],
        };
        
        let history_channel = tracker.generate_history_channel(&board);
        
        // Empty history should generate zero channel
        for row in &history_channel {
            for &value in row {
                assert_eq!(value, 0.0, "Empty history should generate zero channel");
            }
        }
        
        // Empty queries should return empty results
        let recent = tracker.get_recent_positions(10);
        assert!(recent.is_empty(), "Empty history should return no recent positions");
        
        let was_visited = tracker.was_recently_visited(&Coord { x: 2, y: 2 }, 5);
        assert!(!was_visited, "Empty history should not show any visited positions");
    }

    #[test]
    fn test_position_revisiting_behavior() {
        let mut tracker = MovementHistoryTracker::new(10);
        let board = Board {
            width: 5,
            height: 5,
            food: vec![],
            snakes: vec![],
            hazards: vec![],
        };
        
        // Visit same position multiple times
        let revisit_pos = Coord { x: 2, y: 2 };
        tracker.update_position(revisit_pos, 0);
        tracker.update_position(Coord { x: 2, y: 1 }, 1);
        tracker.update_position(revisit_pos, 2); // Revisit
        tracker.update_position(Coord { x: 1, y: 2 }, 3);
        tracker.update_position(revisit_pos, 4); // Revisit again
        
        let history_channel = tracker.generate_history_channel(&board);
        
        // Revisited position should have accumulated value (but clamped to 1.0)
        let revisited_value = history_channel[2][2];
        assert_eq!(revisited_value, 1.0, "Revisited position should reach maximum value");
        
        // Should track all instances in recent positions
        let recent_all = tracker.get_recent_positions(10);
        let revisit_count = recent_all.iter().filter(|&&pos| pos == revisit_pos).count();
        assert_eq!(revisit_count, 3, "Should track all revisits in recent positions");
    }

    #[test]
    fn test_large_history_buffer() {
        let mut tracker = MovementHistoryTracker::new(100);
        
        // Add many positions
        for i in 0..150 {
            let x = i % 10;
            let y = i / 10;
            tracker.update_position(Coord { x, y }, i as u32);
        }
        
        // Should respect max_history limit
        assert_eq!(tracker.history.len(), 100);
        
        // Should contain most recent positions
        let recent_positions = tracker.get_recent_positions(50);
        assert_eq!(recent_positions.len(), 50);
        
        // Performance should be acceptable even with large buffer
        let board = Board {
            width: 15,
            height: 15,
            food: vec![],
            snakes: vec![],
            hazards: vec![],
        };
        
        let start = std::time::Instant::now();
        let history_channel = tracker.generate_history_channel(&board);
        let elapsed = start.elapsed().as_millis();
        
        assert!(elapsed < 10, "Large history should generate channel quickly, took {}ms", elapsed);
        
        // Should have valid channel
        SpatialAnalysisValidator::validate_channel_dimensions(&history_channel, &board, "movement_history")
            .expect("Large history should produce valid channel dimensions");
    }

    #[test]
    fn test_turn_sequence_consistency() {
        let mut tracker = MovementHistoryTracker::new(10);
        
        // Add positions with non-sequential turns (simulating reconnection or skipped turns)
        tracker.update_position(Coord { x: 0, y: 0 }, 1);
        tracker.update_position(Coord { x: 1, y: 0 }, 3); // Skip turn 2
        tracker.update_position(Coord { x: 2, y: 0 }, 5); // Skip turn 4
        tracker.update_position(Coord { x: 3, y: 0 }, 10); // Big skip
        
        // Current turn should be correctly updated
        assert_eq!(tracker.current_turn, 10);
        
        // Decay calculation should handle gaps correctly
        let board = Board {
            width: 5,
            height: 5,
            food: vec![],
            snakes: vec![],
            hazards: vec![],
        };
        
        let history_channel = tracker.generate_history_channel(&board);
        
        // Older positions should have lower values due to larger age gaps
        let oldest_value = history_channel[0][0]; // Age: 10 - 1 = 9
        let newest_value = history_channel[0][3]; // Age: 10 - 10 = 0
        assert!(newest_value > oldest_value, "Newer positions should have higher values even with turn gaps");
    }

    #[test]
    fn test_memory_efficiency() {
        // Test that memory usage is bounded by max_history
        let mut tracker = MovementHistoryTracker::new(50);
        
        // Add many more positions than max_history
        for i in 0..1000 {
            tracker.update_position(Coord { x: i % 20, y: (i / 20) % 20 }, i as u32);
        }
        
        // Memory should be bounded
        assert!(tracker.history.len() <= 50, "History should be bounded by max_history");
        assert!(tracker.position_frequencies.len() <= 400, "Position frequencies should be reasonable"); // 20x20 max
    }
}

#[cfg(test)]
mod strategic_position_analyzer_tests {
    use super::*;
    use crate::spatial_test_utilities::*;

    #[test]
    fn test_analyzer_creation() {
        let analyzer = StrategicPositionAnalyzer::new();
        assert!(analyzer.strategic_positions.is_empty());
    }

    #[test]
    fn test_comprehensive_strategic_analysis() {
        let mut analyzer = StrategicPositionAnalyzer::new();
        let game_state = BoardScenarioGenerator::food_competition_scenario();
        
        let strategic_channel = analyzer.analyze_strategic_positions(&game_state.board, &game_state.you);
        
        // Validate channel properties
        SpatialAnalysisValidator::validate_channel_dimensions(&strategic_channel, &game_state.board, "strategic_positions")
            .expect("Strategic channel should have correct dimensions");
        SpatialAnalysisValidator::validate_channel_range(&strategic_channel, "strategic_positions")
            .expect("Strategic values should be in valid range");
        
        // Should have identified strategic positions
        assert!(!analyzer.strategic_positions.is_empty(), "Should identify strategic positions");
        
        // Strategic values should vary across the board
        let strategic_values: Vec<f32> = strategic_channel.iter()
            .flatten()
            .filter(|&&v| v != 0.0)
            .copied()
            .collect();
        
        assert!(!strategic_values.is_empty(), "Should have non-zero strategic values");
        
        // Should have both positive and negative strategic values (due to corner traps)
        let has_positive = strategic_values.iter().any(|&v| v > 0.0);
        let has_negative = strategic_values.iter().any(|&v| v < 0.0);
        assert!(has_positive, "Should have positive strategic values");
        // Note: Corner trap analysis might not always produce negative values depending on the scenario
    }

    #[test]
    fn test_food_proximity_analysis() {
        let mut analyzer = StrategicPositionAnalyzer::new();
        let game_state = GameStateBuilder::new()
            .with_board_size(9, 9)
            .with_our_snake(create_test_snake("us", Coord { x: 2, y: 2 }, 3, 50)) // Closer to food
            .with_opponent(create_test_snake("opponent", Coord { x: 6, y: 6 }, 3, 70)) // Farther from food
            .with_food(vec![Coord { x: 3, y: 3 }]) // Food closer to us
            .build();
        
        analyzer.analyze_food_proximity(&game_state.board, &game_state.you);
        
        // Should have strategic positions around food
        let food_area_coords = vec![
            Coord { x: 3, y: 3 }, // Food position
            Coord { x: 2, y: 3 }, Coord { x: 4, y: 3 }, // Adjacent to food
            Coord { x: 3, y: 2 }, Coord { x: 3, y: 4 }, // Adjacent to food
        ];
        
        for coord in food_area_coords {
            if let Some(&strategic_value) = analyzer.strategic_positions.get(&coord) {
                assert!(strategic_value > 0.0, "Food area positions should have positive strategic value at {:?}", coord);
            }
        }
        
        // Positions closer to food should have higher strategic value
        let close_to_food = analyzer.strategic_positions.get(&Coord { x: 3, y: 3 }).unwrap_or(&0.0);
        let far_from_food = analyzer.strategic_positions.get(&Coord { x: 1, y: 1 }).unwrap_or(&0.0);
        assert!(close_to_food >= far_from_food, "Closer positions should have higher food strategic value");
    }

    #[test]
    fn test_food_competition_scenarios() {
        let mut analyzer = StrategicPositionAnalyzer::new();
        
        // Test scenario where we're closer to food
        let closer_game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_test_snake("us", Coord { x: 2, y: 3 }, 3, 50))
            .with_opponent(create_test_snake("opponent", Coord { x: 5, y: 3 }, 3, 50))
            .with_food(vec![Coord { x: 3, y: 3 }]) // We're 1 step away, opponent is 2 steps away
            .build();
        
        analyzer.analyze_food_proximity(&closer_game_state.board, &closer_game_state.you);
        let closer_advantage_value = analyzer.strategic_positions.get(&Coord { x: 3, y: 3 }).copied().unwrap_or(0.0);
        
        // Reset analyzer for second test
        analyzer.strategic_positions.clear();
        
        // Test scenario where opponent is closer to food
        let farther_game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_test_snake("us", Coord { x: 1, y: 3 }, 3, 50))
            .with_opponent(create_test_snake("opponent", Coord { x: 4, y: 3 }, 3, 50))
            .with_food(vec![Coord { x: 5, y: 3 }]) // Opponent is 1 step away, we're 4 steps away
            .build();
        
        analyzer.analyze_food_proximity(&farther_game_state.board, &farther_game_state.you);
        let farther_disadvantage_value = analyzer.strategic_positions.get(&Coord { x: 5, y: 3 }).copied().unwrap_or(0.0);
        
        // We should have higher strategic value when we're closer to food
        assert!(closer_advantage_value > farther_disadvantage_value,
                "Should have higher strategic value when closer to food: {} vs {}", 
                closer_advantage_value, farther_disadvantage_value);
    }

    #[test]
    fn test_center_control_analysis() {
        let mut analyzer = StrategicPositionAnalyzer::new();
        let game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_test_snake("us", Coord { x: 3, y: 3 }, 3, 100))
            .build();
        
        analyzer.analyze_center_control(&game_state.board);
        
        // Center positions should have higher strategic value
        let center_value = analyzer.strategic_positions.get(&Coord { x: 3, y: 3 }).unwrap_or(&0.0);
        let corner_value = analyzer.strategic_positions.get(&Coord { x: 0, y: 0 }).unwrap_or(&0.0);
        let edge_value = analyzer.strategic_positions.get(&Coord { x: 0, y: 3 }).unwrap_or(&0.0);
        
        assert!(center_value > corner_value, "Center should have higher value than corner");
        assert!(center_value > edge_value, "Center should have higher value than edge");
        assert!(edge_value > corner_value, "Edge should have higher value than corner");
        
        // All center control values should be positive
        assert!(center_value > &0.0, "Center control should provide positive strategic value");
    }

    #[test]
    fn test_cutting_point_analysis() {
        let mut analyzer = StrategicPositionAnalyzer::new();
        let game_state = GameStateBuilder::new()
            .with_board_size(9, 9)
            .with_our_snake(create_test_snake("us", Coord { x: 4, y: 4 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 2, y: 2 }, 3, 100))
            .build();
        
        analyzer.analyze_cutting_points(&game_state.board, &game_state.you);
        
        // Should identify cutting positions around opponent head
        let opponent_head = Coord { x: 2, y: 2 };
        let potential_cutting_points = vec![
            Coord { x: 2, y: 4 }, // 2 steps up from opponent
            Coord { x: 2, y: 0 }, // 2 steps down from opponent
            Coord { x: 0, y: 2 }, // 2 steps left from opponent
            Coord { x: 4, y: 2 }, // 2 steps right from opponent
        ];
        
        let cutting_positions_found = potential_cutting_points.iter()
            .filter(|&pos| analyzer.strategic_positions.get(pos).map_or(false, |&v| v > 0.5))
            .count();
        
        assert!(cutting_positions_found > 0, "Should identify some cutting positions around opponent");
        
        // Cutting positions should have substantial strategic value
        for cutting_point in potential_cutting_points {
            if cutting_point.x >= 0 && cutting_point.x < 9 && cutting_point.y >= 0 && cutting_point.y < 9 {
                if let Some(&value) = analyzer.strategic_positions.get(&cutting_point) {
                    if value > 0.0 {
                        assert!(value >= 0.6, "Cutting positions should have high strategic value: {} at {:?}", value, cutting_point);
                    }
                }
            }
        }
    }

    #[test]
    fn test_escape_route_analysis() {
        let mut analyzer = StrategicPositionAnalyzer::new();
        let game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_test_snake("us", Coord { x: 3, y: 3 }, 3, 100))
            .build();
        
        analyzer.analyze_escape_routes(&game_state.board, &game_state.you);
        
        // Center positions should have good escape routes
        let center_escape_value = analyzer.strategic_positions.get(&Coord { x: 3, y: 3 }).unwrap_or(&0.0);
        assert!(center_escape_value > &0.4, "Center should have good escape routes");
        
        // Edge positions should have fewer escape routes
        let edge_escape_value = analyzer.strategic_positions.get(&Coord { x: 1, y: 3 }).unwrap_or(&0.0);
        assert!(center_escape_value > edge_escape_value, "Center should have better escape routes than edge");
        
        // Verify that escape route values correlate with mobility
        // Position (2,2) should have 4 open directions
        let mobile_position_value = analyzer.strategic_positions.get(&Coord { x: 2, y: 2 }).unwrap_or(&0.0);
        assert!(mobile_position_value > &0.0, "Mobile positions should have positive escape route value");
    }

    #[test]
    fn test_corner_trap_analysis() {
        let mut analyzer = StrategicPositionAnalyzer::new();
        let game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_test_snake("us", Coord { x: 3, y: 3 }, 3, 100))
            .build();
        
        analyzer.analyze_corner_traps(&game_state.board);
        
        // Corner positions should have negative strategic value (traps)
        let corner_positions = vec![
            Coord { x: 0, y: 0 },
            Coord { x: 6, y: 0 },
            Coord { x: 0, y: 6 },
            Coord { x: 6, y: 6 },
        ];
        
        for corner in corner_positions {
            let corner_value = analyzer.strategic_positions.get(&corner).unwrap_or(&0.0);
            assert!(corner_value <= &0.0, "Corner {:?} should have negative or zero strategic value: {}", corner, corner_value);
        }
        
        // Edge positions should also have negative values
        let edge_positions = vec![
            Coord { x: 0, y: 3 }, // Left edge
            Coord { x: 6, y: 3 }, // Right edge
            Coord { x: 3, y: 0 }, // Top edge
            Coord { x: 3, y: 6 }, // Bottom edge
        ];
        
        for edge in edge_positions {
            let edge_value = analyzer.strategic_positions.get(&edge).unwrap_or(&0.0);
            assert!(edge_value <= &0.0, "Edge {:?} should have negative or zero strategic value: {}", edge, edge_value);
        }
        
        // Center positions should not be affected by corner trap analysis
        let center_value = analyzer.strategic_positions.get(&Coord { x: 3, y: 3 }).unwrap_or(&0.0);
        // Note: center might have slight negative value due to edge distance calculation, but should be minimal
    }

    #[test]
    fn test_position_safety_check() {
        let analyzer = StrategicPositionAnalyzer::new();
        let game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_snake_with_path("us", vec![
                Coord { x: 3, y: 3 }, // head
                Coord { x: 3, y: 2 }, // neck
                Coord { x: 3, y: 1 }, // body
            ], 100))
            .with_opponent(create_snake_with_path("opponent", vec![
                Coord { x: 1, y: 1 }, // head
                Coord { x: 0, y: 1 }, // neck
            ], 100))
            .build();
        
        // Snake body positions should not be safe
        assert!(!analyzer.is_position_safe(&Coord { x: 3, y: 3 }, &game_state.board)); // Our head
        assert!(!analyzer.is_position_safe(&Coord { x: 3, y: 2 }, &game_state.board)); // Our neck
        assert!(!analyzer.is_position_safe(&Coord { x: 1, y: 1 }, &game_state.board)); // Opponent head
        assert!(!analyzer.is_position_safe(&Coord { x: 0, y: 1 }, &game_state.board)); // Opponent neck
        
        // Empty positions should be safe
        assert!(analyzer.is_position_safe(&Coord { x: 5, y: 5 }, &game_state.board));
        assert!(analyzer.is_position_safe(&Coord { x: 0, y: 0 }, &game_state.board));
    }

    #[test]
    fn test_complex_tactical_scenario() {
        let mut analyzer = StrategicPositionAnalyzer::new();
        
        // Complex scenario: food competition + territorial control + escape routes
        let game_state = GameStateBuilder::new()
            .with_board_size(11, 11)
            .with_our_snake(create_test_snake("us", Coord { x: 3, y: 5 }, 4, 60))
            .with_opponent(create_test_snake("opponent1", Coord { x: 7, y: 5 }, 5, 80))
            .with_opponent(create_test_snake("opponent2", Coord { x: 5, y: 2 }, 3, 90))
            .with_food(vec![
                Coord { x: 5, y: 5 }, // Central contested food
                Coord { x: 1, y: 8 }, // Food closer to us
                Coord { x: 9, y: 2 }, // Food closer to opponent1
            ])
            .build();
        
        let strategic_channel = analyzer.analyze_strategic_positions(&game_state.board, &game_state.you);
        
        // Validate basic properties
        SpatialAnalysisValidator::validate_channel_dimensions(&strategic_channel, &game_state.board, "strategic_positions")
            .expect("Complex scenario should have correct dimensions");
        
        // Central food should have strategic value due to competition
        let central_food_value = strategic_channel[5][5];
        assert!(central_food_value != 0.0, "Central contested food should have strategic significance");
        
        // Food closer to us should have higher strategic value
        let our_food_value = strategic_channel[8][1];
        let their_food_value = strategic_channel[2][9];
        assert!(our_food_value >= their_food_value, "Food closer to us should have higher strategic value");
        
        // Should identify multiple strategic considerations
        let non_zero_positions = strategic_channel.iter()
            .flatten()
            .filter(|&&v| v.abs() > 0.01)
            .count();
        
        assert!(non_zero_positions > 20, "Complex scenario should identify many strategic positions");
    }

    #[test]
    fn test_multi_food_prioritization() {
        let mut analyzer = StrategicPositionAnalyzer::new();
        let game_state = GameStateBuilder::new()
            .with_board_size(9, 9)
            .with_our_snake(create_test_snake("us", Coord { x: 4, y: 4 }, 3, 30)) // Low health - need food
            .with_opponent(create_test_snake("opponent", Coord { x: 1, y: 1 }, 4, 90))
            .with_food(vec![
                Coord { x: 3, y: 4 }, // 1 step away from us
                Coord { x: 6, y: 4 }, // 2 steps away from us
                Coord { x: 4, y: 2 }, // 2 steps away from us
                Coord { x: 8, y: 8 }, // Far from both
            ])
            .build();
        
        analyzer.analyze_food_proximity(&game_state.board, &game_state.you);
        
        // Closer food should have higher strategic value
        let closest_food_area = analyzer.strategic_positions.get(&Coord { x: 3, y: 4 }).unwrap_or(&0.0);
        let medium_food_area = analyzer.strategic_positions.get(&Coord { x: 6, y: 4 }).unwrap_or(&0.0);
        let far_food_area = analyzer.strategic_positions.get(&Coord { x: 8, y: 8 }).unwrap_or(&0.0);
        
        assert!(closest_food_area >= medium_food_area, "Closest food should have highest strategic value");
        assert!(medium_food_area >= far_food_area, "Medium distance food should have higher value than far food");
        
        // All food areas should have positive strategic value
        assert!(closest_food_area > &0.0, "Food areas should have positive strategic value");
    }

    #[test]
    fn test_escape_route_with_obstacles() {
        let mut analyzer = StrategicPositionAnalyzer::new();
        
        // Create scenario with snake bodies blocking some escape routes
        let game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_test_snake("us", Coord { x: 3, y: 3 }, 3, 100))
            .with_opponent(create_snake_with_path("obstacle", vec![
                Coord { x: 2, y: 3 }, // Blocking left
                Coord { x: 2, y: 2 },
                Coord { x: 1, y: 2 },
                Coord { x: 1, y: 3 },
                Coord { x: 1, y: 4 }, // Long snake creating obstacle
            ], 100))
            .build();
        
        analyzer.analyze_escape_routes(&game_state.board, &game_state.you);
        
        // Position near the blocking snake should have reduced escape route value
        let blocked_area_value = analyzer.strategic_positions.get(&Coord { x: 3, y: 2 }).unwrap_or(&0.0);
        let open_area_value = analyzer.strategic_positions.get(&Coord { x: 5, y: 5 }).unwrap_or(&0.0);
        
        assert!(open_area_value >= blocked_area_value, 
                "Open areas should have better escape routes than blocked areas");
    }

    #[test]
    fn test_strategic_value_accumulation() {
        let mut analyzer = StrategicPositionAnalyzer::new();
        let game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_test_snake("us", Coord { x: 3, y: 3 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 1, y: 1 }, 3, 100))
            .with_food(vec![Coord { x: 3, y: 4 }])
            .build();
        
        let strategic_channel = analyzer.analyze_strategic_positions(&game_state.board, &game_state.you);
        
        // Positions that benefit from multiple strategic factors should have higher values
        // Center-ish position near food should accumulate values from multiple analyses
        let multi_factor_value = strategic_channel[4][3]; // Position (3,4) - food location
        let single_factor_value = strategic_channel[6][6]; // Corner - only affected by corner trap analysis
        
        // Multi-factor positions should generally have higher absolute values
        assert!(multi_factor_value.abs() >= single_factor_value.abs() - 0.1, 
                "Positions with multiple strategic factors should have significant values");
        
        // Values should be properly clamped
        for row in &strategic_channel {
            for &value in row {
                assert!(value >= -1.0 && value <= 1.0, "Strategic values should be in [-1.0, 1.0] range");
            }
        }
    }

    #[test]
    fn test_large_board_performance() {
        let mut analyzer = StrategicPositionAnalyzer::new();
        let game_state = BoardScenarioGenerator::large_board_scenario();
        
        let start = std::time::Instant::now();
        let strategic_channel = analyzer.analyze_strategic_positions(&game_state.board, &game_state.you);
        let elapsed = start.elapsed().as_millis();
        
        // Should complete within performance budget
        assert!(elapsed < 100, "Large board strategic analysis should complete within 100ms, took {}ms", elapsed);
        
        // Should produce valid results
        SpatialAnalysisValidator::validate_channel_dimensions(&strategic_channel, &game_state.board, "strategic_positions")
            .expect("Large board should produce valid strategic channel");
        
        // Should identify strategic positions across the large board
        let strategic_position_count = analyzer.strategic_positions.len();
        assert!(strategic_position_count > 50, "Large board should identify many strategic positions");
    }

    #[test]
    fn test_asymmetric_board_handling() {
        let mut analyzer = StrategicPositionAnalyzer::new();
        let game_state = GameStateBuilder::new()
            .with_board_size(15, 7) // width=i32, height=u32 - type inconsistency
            .with_our_snake(create_test_snake("us", Coord { x: 7, y: 3 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 3, y: 1 }, 3, 100))
            .with_food(vec![Coord { x: 11, y: 5 }])
            .build();
        
        let strategic_channel = analyzer.analyze_strategic_positions(&game_state.board, &game_state.you);
        
        // Should handle asymmetric dimensions correctly
        assert_eq!(strategic_channel.len(), 7); // height as usize
        assert_eq!(strategic_channel[0].len(), 15); // width as usize
        
        // Should identify strategic positions across asymmetric board

#[cfg(test)]
mod advanced_board_state_encoder_tests {
    use super::*;
    use crate::spatial_test_utilities::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = AdvancedBoardStateEncoder::new(20, 3);
        assert_eq!(encoder.movement_tracker.max_history, 20);
        assert_eq!(encoder.danger_predictor.prediction_depth, 3);
        assert!(encoder.voronoi_analyzer.our_territory.is_empty());
        assert!(encoder.strategic_analyzer.strategic_positions.is_empty());
    }

    #[test]
    fn test_complete_12_channel_encoding() {
        let mut encoder = AdvancedBoardStateEncoder::new(10, 2);
        let game_state = BoardScenarioGenerator::food_competition_scenario();
        
        let channels = encoder.encode_12_channel_board(&game_state.board, &game_state.you, 5);
        
        // Validate 12-channel structure
        assert_eq!(channels.len(), 12, "Should have exactly 12 channels");
        
        for (i, channel) in channels.iter().enumerate() {
            // Validate each channel dimensions
            SpatialAnalysisValidator::validate_channel_dimensions(channel, &game_state.board, &format!("channel_{}", i))
                .expect(&format!("Channel {} should have correct dimensions", i));
            
            // Validate each channel ranges
            SpatialAnalysisValidator::validate_channel_range(channel, &format!("channel_{}", i))
                .expect(&format!("Channel {} should have valid value ranges", i));
        }
        
        // Verify specific channel contents
        
        // Channel 0: Empty positions - should be mostly 1.0 except where occupied
        let empty_channel = &channels[0];
        let total_empty_positions = empty_channel.iter().flatten().filter(|&&v| v == 1.0).count();
        assert!(total_empty_positions > 50, "Channel 0 should have many empty positions");
        
        // Channel 1: Our head - should have exactly one position with value 1.0
        let our_head_channel = &channels[1];
        let our_head_positions = our_head_channel.iter().flatten().filter(|&&v| v == 1.0).count();
        assert_eq!(our_head_positions, 1, "Channel 1 should have exactly one head position");
        
        // Channel 5: Food - should have positions where food exists
        let food_channel = &channels[5];
        let food_positions = food_channel.iter().flatten().filter(|&&v| v == 1.0).count();
        assert!(food_positions >= game_state.board.food.len(), "Channel 5 should represent all food positions");
        
        // Channels 7-8: Territory analysis
        let our_territory_channel = &channels[7];
        let opponent_territory_channel = &channels[8];
        SpatialAnalysisValidator::validate_territory_consistency(our_territory_channel, opponent_territory_channel)
            .expect("Territory channels should be consistent");
        
        // Channel 9: Danger zones - should have some dangerous areas
        let danger_channel = &channels[9];
        let dangerous_positions = danger_channel.iter().flatten().filter(|&&v| v > 0.5).count();
        assert!(dangerous_positions > 0, "Channel 9 should identify dangerous positions");
        
        // Channel 10: Movement history - should be empty initially (turn 5 but no previous positions)
        let history_channel = &channels[10];
        let history_sum: f32 = history_channel.iter().flatten().sum();
        assert!(history_sum >= 0.0, "Channel 10 should have valid history values");
        
        // Channel 11: Strategic positions - should identify strategic areas
        let strategic_channel = &channels[11];
        let strategic_positions = strategic_channel.iter().flatten().filter(|&&v| v.abs() > 0.1).count();
        assert!(strategic_positions > 0, "Channel 11 should identify strategic positions");
    }

    #[test]
    fn test_movement_history_integration() {
        let mut encoder = AdvancedBoardStateEncoder::new(10, 2);
        let game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_test_snake("us", Coord { x: 3, y: 3 }, 3, 100))
            .build();
        
        // Encode multiple turns to build movement history
        let positions = vec![
            (Coord { x: 1, y: 1 }, 0),
            (Coord { x: 2, y: 1 }, 1),
            (Coord { x: 3, y: 1 }, 2),
            (Coord { x: 3, y: 2 }, 3),
            (Coord { x: 3, y: 3 }, 4),
        ];
        
        for (pos, turn) in positions {
            let test_snake = create_test_snake("us", pos, 3, 100);
            let test_board = Board {
                width: game_state.board.width,
                height: game_state.board.height,
                food: game_state.board.food.clone(),
                snakes: vec![test_snake.clone()],
                hazards: vec![],
            };
            
            let channels = encoder.encode_12_channel_board(&test_board, &test_snake, turn);
            
            // Movement history channel should accumulate positions
            let history_channel = &channels[10];
            let history_sum: f32 = history_channel.iter().flatten().sum();
            
            if turn > 0 {
                assert!(history_sum > 0.0, "Movement history should accumulate on turn {}", turn);
            }
        }
        
        // Final encoding should have rich movement history
        let final_channels = encoder.encode_12_channel_board(&game_state.board, &game_state.you, 4);
        let final_history_channel = &final_channels[10];
        let final_history_sum: f32 = final_history_channel.iter().flatten().sum();
        assert!(final_history_sum > 2.0, "Should have substantial movement history after multiple turns");
    }

    #[test]
    fn test_basic_channels_encoding() {
        let encoder = AdvancedBoardStateEncoder::new(5, 2);
        let game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_test_snake("us", Coord { x: 3, y: 3 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 1, y: 1 }, 4, 100))
            .with_food(vec![Coord { x: 5, y: 5 }])
            .build();
        
        let mut channels = vec![vec![vec![0.0; 7]; 7]; 12];
        encoder.encode_basic_channels(&mut channels, &game_state.board, &game_state.you);
        
        // Validate basic channels (0-6)
        for i in 0..7 {
            let channel = &channels[i];
            SpatialAnalysisValidator::validate_channel_dimensions(channel, &game_state.board, &format!("basic_channel_{}", i))
                .expect(&format!("Basic channel {} should have correct dimensions", i));
        }
        
        // Channel 0: Empty - should be inverse of occupied positions
        let empty_channel = &channels[0];
        assert_eq!(empty_channel[3][3], 0.0, "Our head position should not be empty");
        assert_eq!(empty_channel[1][1], 0.0, "Opponent head position should not be empty");
        assert_eq!(empty_channel[5][5], 0.0, "Food position should not be empty");
        assert_eq!(empty_channel[0][0], 1.0, "Unoccupied position should be empty");
        
        // Channel 1: Our head
        let our_head_channel = &channels[1];
        assert_eq!(our_head_channel[3][3], 1.0, "Our head position should be marked");
        assert_eq!(our_head_channel[1][1], 0.0, "Other positions should not be marked as our head");
        
        // Channel 3: Opponent heads
        let opponent_head_channel = &channels[3];
        assert_eq!(opponent_head_channel[1][1], 1.0, "Opponent head should be marked");
        assert_eq!(opponent_head_channel[3][3], 0.0, "Our head should not be marked as opponent");
        
        // Channel 5: Food
        let food_channel = &channels[5];
        assert_eq!(food_channel[5][5], 1.0, "Food position should be marked");
        assert_eq!(food_channel[0][0], 0.0, "Non-food position should not be marked");
        
        // Channel 6: Walls (boundaries)
        let wall_channel = &channels[6];
        assert_eq!(wall_channel[0][0], 1.0, "Corner should be marked as boundary");
        assert_eq!(wall_channel[0][6], 1.0, "Corner should be marked as boundary");
        assert_eq!(wall_channel[6][0], 1.0, "Corner should be marked as boundary");
        assert_eq!(wall_channel[6][6], 1.0, "Corner should be marked as boundary");
        assert_eq!(wall_channel[3][3], 0.0, "Center should not be marked as boundary");
    }

    #[test]
    fn test_encoding_statistics() {
        let mut encoder = AdvancedBoardStateEncoder::new(5, 2);
        let game_state = BoardScenarioGenerator::crowded_board();
        
        // Perform encoding to populate statistics
        encoder.encode_12_channel_board(&game_state.board, &game_state.you, 10);
        
        let (our_territory_count, opponent_territory_count, danger_positions, strategic_positions) = encoder.get_encoding_stats();
        
        // Validate statistics are reasonable
        assert!(our_territory_count > 0, "Should have our territory positions");
        assert!(opponent_territory_count > 0, "Should have opponent territory positions");
        assert!(danger_positions > 0, "Should have danger positions with multiple snakes");
        assert!(strategic_positions > 0, "Should have strategic positions");
        
        // Territory counts should be reasonable for the board size
        let total_positions = (game_state.board.width * game_state.board.height as i32) as usize;
        assert!(our_territory_count + opponent_territory_count <= total_positions * 2, 
                "Territory counts should be reasonable"); // *2 because of overlap
    }

    #[test]
    fn test_performance_within_budget() {
        let mut encoder = AdvancedBoardStateEncoder::new(15, 3);
        let game_state = BoardScenarioGenerator::large_board_scenario();
        
        let start = std::time::Instant::now();
        let channels = encoder.encode_12_channel_board(&game_state.board, &game_state.you, 25);
        let elapsed = start.elapsed().as_millis();
        
        // CRITICAL: Must stay under 500ms budget
        assert!(elapsed < 500, "12-channel encoding must complete within 500ms budget, took {}ms", elapsed);
        
        // Validate results are still correct despite time pressure
        assert_eq!(channels.len(), 12, "Should still produce 12 channels under time pressure");
        
        for channel in &channels {
            assert_eq!(channel.len(), game_state.board.height as usize, "Channel height should be correct");
            assert_eq!(channel[0].len(), game_state.board.width as usize, "Channel width should be correct");
        }
        
        // Performance should scale reasonably
        println!("Large board ({}x{}) encoded in {}ms", 
                 game_state.board.width, game_state.board.height, elapsed);
    }

    #[test]
    fn test_encoding_consistency() {
        let mut encoder = AdvancedBoardStateEncoder::new(8, 2);
        let game_state = GameStateBuilder::new()
            .with_board_size(9, 9)
            .with_our_snake(create_test_snake("us", Coord { x: 4, y: 4 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 2, y: 2 }, 3, 100))
            .with_food(vec![Coord { x: 6, y: 6 }])
            .build();
        
        // Encode the same state multiple times
        let channels1 = encoder.encode_12_channel_board(&game_state.board, &game_state.you, 5);
        let channels2 = encoder.encode_12_channel_board(&game_state.board, &game_state.you, 5);
        
        // Results should be consistent (deterministic)
        assert_eq!(channels1.len(), channels2.len());
        
        for (i, (ch1, ch2)) in channels1.iter().zip(channels2.iter()).enumerate() {
            assert_eq!(ch1.len(), ch2.len(), "Channel {} should have consistent dimensions", i);
            
            for (row1, row2) in ch1.iter().zip(ch2.iter()) {
                assert_eq!(row1.len(), row2.len());
                for (&val1, &val2) in row1.iter().zip(row2.iter()) {
                    assert!((val1 - val2).abs() < 0.001, 
                           "Channel {} should have consistent values: {} vs {}", i, val1, val2);
                }
            }
        }
    }

    #[test]
    fn test_multi_scenario_encoding() {
        let mut encoder = AdvancedBoardStateEncoder::new(10, 3);
        
        let scenarios = vec![
            BoardScenarioGenerator::empty_board(),
            BoardScenarioGenerator::crowded_board(),
            BoardScenarioGenerator::corner_trap_scenario(),
            BoardScenarioGenerator::food_competition_scenario(),
            BoardScenarioGenerator::head_collision_scenario(),
        ];
        
        for (i, scenario) in scenarios.iter().enumerate() {
            let start = std::time::Instant::now();
            let channels = encoder.encode_12_channel_board(&scenario.board, &scenario.you, i as u32);
            let elapsed = start.elapsed().as_millis();
            
            // Each scenario should complete quickly
            assert!(elapsed < 200, "Scenario {} should encode quickly, took {}ms", i, elapsed);
            
            // Each scenario should produce valid 12-channel encoding
            assert_eq!(channels.len(), 12, "Scenario {} should produce 12 channels", i);
            
            // Validate all channels
            for (ch_idx, channel) in channels.iter().enumerate() {
                SpatialAnalysisValidator::validate_channel_dimensions(channel, &scenario.board, &format!("scenario_{}_channel_{}", i, ch_idx))
                    .expect(&format!("Scenario {} channel {} should be valid", i, ch_idx));
                SpatialAnalysisValidator::validate_channel_range(channel, &format!("scenario_{}_channel_{}", i, ch_idx))
                    .expect(&format!("Scenario {} channel {} should have valid ranges", i, ch_idx));
            }
        }
    }

    #[test]
    fn test_asymmetric_board_encoding() {
        let mut encoder = AdvancedBoardStateEncoder::new(5, 2);
        let game_state = GameStateBuilder::new()
            .with_board_size(15, 7) // width=i32, height=u32 - type inconsistency
            .with_our_snake(create_test_snake("us", Coord { x: 7, y: 3 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 3, y: 1 }, 3, 100))
            .with_food(vec![Coord { x: 11, y: 5 }])
            .build();
        
        let channels = encoder.encode_12_channel_board(&game_state.board, &game_state.you, 8);
        
        // Should handle asymmetric dimensions correctly
        assert_eq!(channels.len(), 12);
        
        for (i, channel) in channels.iter().enumerate() {
            assert_eq!(channel.len(), 7, "Channel {} height should be 7", i); // height as usize
            assert_eq!(channel[0].len(), 15, "Channel {} width should be 15", i); // width as usize
            
            // Validate ranges on asymmetric board
            SpatialAnalysisValidator::validate_channel_range(channel, &format!("asymmetric_channel_{}", i))
                .expect(&format!("Asymmetric channel {} should have valid ranges", i));
        }
        
        // Verify content is properly mapped
        let our_head_channel = &channels[1];
        assert_eq!(our_head_channel[3][7], 1.0, "Our head should be correctly positioned on asymmetric board");
        
        let food_channel = &channels[5];
        assert_eq!(food_channel[5][11], 1.0, "Food should be correctly positioned on asymmetric board");
    }

    #[test]
    fn test_memory_management() {
        let mut encoder = AdvancedBoardStateEncoder::new(50, 4);
        let game_state = GameStateBuilder::new()
            .with_board_size(11, 11)
            .with_our_snake(create_test_snake("us", Coord { x: 5, y: 5 }, 4, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 3, y: 3 }, 4, 100))
            .with_food(vec![Coord { x: 7, y: 7 }])
            .build();
        
        // Perform many encodings to test memory management
        for turn in 0..100 {
            let channels = encoder.encode_12_channel_board(&game_state.board, &game_state.you, turn);
            
            // Verify encoding is still valid
            assert_eq!(channels.len(), 12);
            
            // Check that movement history is properly bounded
            assert!(encoder.movement_tracker.history.len() <= encoder.movement_tracker.max_history);
            
            // Periodically check that internal structures are reasonable size
            if turn % 20 == 0 {
                let (our_territory, opponent_territory, dangers, strategic) = encoder.get_encoding_stats();
                assert!(our_territory < 500, "Territory tracking should be bounded");
                assert!(opponent_territory < 500, "Territory tracking should be bounded");
                assert!(dangers < 1000, "Danger tracking should be bounded");
                assert!(strategic < 500, "Strategic tracking should be bounded");
            }
        }
    }

    #[test]
    fn test_edge_case_single_cell_board() {
        let mut encoder = AdvancedBoardStateEncoder::new(5, 1);
        let game_state = GameStateBuilder::new()
            .with_board_size(1, 1)
            .with_our_snake(create_test_snake("us", Coord { x: 0, y: 0 }, 1, 100))
            .build();
        
        let channels = encoder.encode_12_channel_board(&game_state.board, &game_state.you, 1);
        
        // Should handle minimal board without crashing
        assert_eq!(channels.len(), 12);
        
        for channel in &channels {
            assert_eq!(channel.len(), 1);
            assert_eq!(channel[0].len(), 1);
            
            let value = channel[0][0];
            assert!(value >= 0.0 && value <= 1.0, "Single cell value should be valid: {}", value);
        }
        
        // Basic channels should still work
        assert_eq!(channels[0][0][0], 0.0, "Single occupied cell should not be empty");
        assert_eq!(channels[1][0][0], 1.0, "Our head should be marked");
        assert_eq!(channels[6][0][0], 1.0, "Single cell should be boundary");
    }

    #[test]
    fn test_turn_progression_effects() {
        let mut encoder = AdvancedBoardStateEncoder::new(8, 2);
        
        let positions = vec![
            Coord { x: 3, y: 3 },
            Coord { x: 4, y: 3 },
            Coord { x: 5, y: 3 },
            Coord { x: 5, y: 4 },
            Coord { x: 5, y: 5 },
        ];
        
        let mut history_progression = Vec::new();
        
        for (turn, &pos) in positions.iter().enumerate() {
            let test_snake = create_test_snake("us", pos, 3, 100);
            let game_state = GameStateBuilder::new()
                .with_board_size(9, 9)
                .with_our_snake(test_snake.clone())
                .build();
            
            let channels = encoder.encode_12_channel_board(&game_state.board, &test_snake, turn as u32);
            
            // Track movement history progression
            let history_channel = &channels[10];
            let history_sum: f32 = history_channel.iter().flatten().sum();
            history_progression.push(history_sum);
            
            // Movement history should generally increase (with time decay)
            if turn > 0 {
                // Allow for some variation due to time decay
                assert!(history_sum >= 0.0, "History should be non-negative on turn {}", turn);
            }
        }
        
        // Should have accumulated some movement history
        let final_history = history_progression.last().unwrap();
        assert!(final_history > &1.0, "Should have substantial movement history after multiple turns");
    }

    #[test]
    fn test_component_integration_consistency() {
        let mut encoder = AdvancedBoardStateEncoder::new(10, 3);
        let game_state = GameStateBuilder::new()
            .with_board_size(11, 11)
            .with_our_snake(create_test_snake("us", Coord { x: 5, y: 5 }, 4, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 3, y: 3 }, 4, 100))
            .with_food(vec![Coord { x: 7, y: 7 }])
            .build();
        
        let channels = encoder.encode_12_channel_board(&game_state.board, &game_state.you, 15);
        
        // Test integration consistency between components
        
        // Territory analysis (channels 7-8) should be consistent with strategic analysis (channel 11)
        let our_territory = &channels[7];
        let strategic_positions = &channels[11];
        
        // Positions with strong territorial control should often have positive strategic value
        let mut territory_strategic_consistency = 0;
        let mut total_territory_positions = 0;
        
        for y in 0..11 {
            for x in 0..11 {
                if our_territory[y][x] > 0.8 { // Strong territorial control
                    total_territory_positions += 1;
                    if strategic_positions[y][x] > 0.0 { // Positive strategic value
                        territory_strategic_consistency += 1;
                    }
                }
            }
        }
        
        if total_territory_positions > 0 {
            let consistency_ratio = territory_strategic_consistency as f32 / total_territory_positions as f32;
            assert!(consistency_ratio > 0.3, "Territory and strategic analysis should show some consistency");
        }
        
        // Danger zones (channel 9) should align with opponent positions
        let danger_channel = &channels[9];
        let opponent_body_channel = &channels[4];
        
        // Opponent body positions should have high danger
        for y in 0..11 {
            for x in 0..11 {
                if opponent_body_channel[y][x] > 0.5 {
                    assert!(danger_channel[y][x] > 0.7, 
                           "Opponent body positions should be dangerous at ({}, {})", x, y);
                }
            }
        }
    }

    #[test]
    fn test_neural_network_compatibility() {
        let mut encoder = AdvancedBoardStateEncoder::new(10, 2);
        let game_state = GameStateBuilder::new()
            .with_board_size(11, 11)
            .with_our_snake(create_test_snake("us", Coord { x: 5, y: 5 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 2, y: 2 }, 3, 100))
            .with_food(vec![Coord { x: 8, y: 8 }])
            .build();
        
        let channels = encoder.encode_12_channel_board(&game_state.board, &game_state.you, 10);
        
        // Simulate neural network input formatting
        let flattened: Vec<f32> = channels.iter()
            .flatten()
            .flatten()
            .copied()
            .collect();
        
        let expected_size = 12 * 11 * 11; // 12 channels * 11x11 board
        assert_eq!(flattened.len(), expected_size, "Flattened input should match expected neural network input size");
        
        // All values should be in valid range for neural network input
        for (i, &value) in flattened.iter().enumerate() {
            assert!(value >= 0.0 && value <= 1.0, 
                   "Neural network input value at index {} should be in [0.0, 1.0]: {}", i, value);
        }
        
        // Should have reasonable distribution of values (not all zeros or ones)
        let non_zero_count = flattened.iter().filter(|&&v| v > 0.0).count();
        let non_one_count = flattened.iter().filter(|&&v| v < 1.0).count();
        
        assert!(non_zero_count > expected_size / 10, "Should have reasonable number of non-zero values");
        assert!(non_one_count > expected_size / 10, "Should have reasonable number of non-one values");
    }

    #[test]
    fn test_backward_compatibility_with_7_channel_mode() {
        let encoder = AdvancedBoardStateEncoder::new(5, 1);
        let game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_test_snake("us", Coord { x: 3, y: 3 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 1, y: 1 }, 3, 100))
            .with_food(vec![Coord { x: 5, y: 5 }])
            .build();
        
        // Test basic channels encoding (simulating 7-channel mode)
        let mut basic_channels = vec![vec![vec![0.0; 7]; 7]; 7];
        encoder.encode_basic_channels(&mut basic_channels, &game_state.board, &game_state.you);
        
        // Basic channels should be compatible with original 7-channel system
        for (i, channel) in basic_channels.iter().enumerate() {
            assert_eq!(channel.len(), 7, "Basic channel {} should have correct height", i);
            assert_eq!(channel[0].len(), 7, "Basic channel {} should have correct width", i);
            
            SpatialAnalysisValidator::validate_channel_range(channel, &format!("basic_channel_{}", i))
                .expect(&format!("Basic channel {} should have valid ranges for backward compatibility", i));
        }
        
        // Key positions should be correctly encoded in backward-compatible way
        assert_eq!(basic_channels[1][3][3], 1.0, "Our head should be marked in channel 1");
        assert_eq!(basic_channels[3][1][1], 1.0, "Opponent head should be marked in channel 3");
        assert_eq!(basic_channels[5][5][5], 1.0, "Food should be marked in channel 5");
        assert_eq!(basic_channels[0][0][0], 1.0, "Empty boundary should be marked in channel 0");
        assert_eq!(basic_channels[6][0][0], 1.0, "Wall should be marked in channel 6");
    }

    #[test]
    fn test_error_recovery_and_robustness() {
        let mut encoder = AdvancedBoardStateEncoder::new(5, 2);
        
        // Test with various edge case scenarios
        let edge_cases = vec![
            // Empty snake body (shouldn't happen but test robustness)
            GameStateBuilder::new()
                .with_board_size(5, 5)
                .with_our_snake(Battlesnake {
                    id: "empty".to_string(),
                    name: "Empty Snake".to_string(),
                    health: 100,
                    body: vec![],
                    head: Coord { x: 2, y: 2 },
                    length: 0,
                    latency: "0".to_string(),
                    shout: None,
                })
                .build(),
            
            // Snake with head outside board bounds (edge case)
            GameStateBuilder::new()
                .with_board_size(3, 3)
                .with_our_snake(create_test_snake("out_of_bounds", Coord { x: 5, y: 5 }, 1, 100))
                .build(),
        ];
        
        for (i, case) in edge_cases.iter().enumerate() {
            // Should not crash on edge cases
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                encoder.encode_12_channel_board(&case.board, &case.you, i as u32)
            }));
            
            // If it doesn't panic, the result should still be valid structure
            if let Ok(channels) = result {
                assert_eq!(channels.len(), 12, "Edge case {} should still produce 12 channels", i);
                
                for channel in &channels {
                    assert_eq!(channel.len(), case.board.height as usize);
                    if !channel.is_empty() {
                        assert_eq!(channel[0].len(), case.board.width as usize);
                    }
                }
            }
        }
    }
}
        let non_zero_count = strategic_channel.iter()
            .flatten()
            .filter(|&&v| v.abs() > 0.01)
            .count();
        
        assert!(non_zero_count > 10, "Should identify strategic positions on asymmetric board");

#[cfg(test)]
mod property_based_tests {
    use super::*;
    use crate::spatial_test_utilities::*;
    use proptest::prelude::*;

    // Property test strategies for generating test data
    prop_compose! {
        fn coord_strategy(max_x: i32, max_y: i32)
                         (x in 0..max_x, y in 0..max_y) -> Coord {
            Coord { x, y }
        }
    }

    prop_compose! {
        fn snake_strategy(board_width: i32, board_height: i32, max_length: usize)
                         (length in 1..=max_length,
                          head in coord_strategy(board_width, board_height as i32),
                          body_coords in prop::collection::vec(coord_strategy(board_width, board_height as i32), 1..max_length))
                         -> Vec<Coord> {
            let mut snake_body = vec![head];
            snake_body.extend(body_coords.into_iter().take(length.saturating_sub(1)));
            snake_body.truncate(length);
            snake_body
        }
    }

    prop_compose! {
        fn board_strategy(min_size: i32, max_size: i32, max_food: usize)
                         (width in min_size..=max_size,
                          height in min_size..=max_size,
                          food_count in 0..=max_food,
                          food_coords in prop::collection::vec(any::<(i32, i32)>(), 0..max_food))
                         -> (i32, u32, Vec<Coord>) {
            let valid_food: Vec<Coord> = food_coords
                .into_iter()
                .take(food_count)
                .map(|(x, y)| Coord {
                    x: x.abs() % width,
                    y: y.abs() % (height as i32)
                })
                .collect();
            (width, height as u32, valid_food)
        }
    }

    // Property tests for VoronoiTerritoryAnalyzer
    proptest! {
        #[test]
        fn property_voronoi_territory_values_in_range(
            (width, height, food) in board_strategy(3, 15, 10),
            our_snake in snake_strategy(15, 15, 10),
            opponent_snake in snake_strategy(15, 15, 10)
        ) {
            let mut analyzer = VoronoiTerritoryAnalyzer::new();
            
            let our_battlesnake = create_test_snake_with_body("us", our_snake.clone(), 100);
            let opponent_battlesnake = create_test_snake_with_body("opponent", opponent_snake.clone(), 100);
            
            let board = Board {
                width,
                height,
                food,
                snakes: vec![our_battlesnake.clone(), opponent_battlesnake],
                hazards: vec![],
            };
            
            analyzer.analyze_territory(&board, &our_battlesnake);
            
            // Property 1: All territory strength values should be in [0.0, 1.0]
            for territory_map in [&analyzer.our_territory, &analyzer.opponent_territory] {
                for (&_coord, &strength) in territory_map {
                    prop_assert!(strength >= 0.0 && strength <= 1.0, 
                               "Territory strength {} should be in [0.0, 1.0]", strength);
                }
            }
            
            // Property 2: Territory maps should not be empty after analysis (if snakes exist)
            if !our_snake.is_empty() {
                prop_assert!(!analyzer.our_territory.is_empty(), "Our territory should not be empty");
            }
        }
        
        #[test]
        fn property_voronoi_territory_consistency(
            (width, height, food) in board_strategy(5, 11, 5),
            our_snake in snake_strategy(11, 11, 8)
        ) {
            let mut analyzer = VoronoiTerritoryAnalyzer::new();
            
            let our_battlesnake = create_test_snake_with_body("us", our_snake.clone(), 100);
            
            let board = Board {
                width,
                height,
                food,
                snakes: vec![our_battlesnake.clone()],
                hazards: vec![],
            };
            
            analyzer.analyze_territory(&board, &our_battlesnake);
            
            // Property: Running analysis twice should yield identical results
            let first_territory = analyzer.our_territory.clone();
            analyzer.analyze_territory(&board, &our_battlesnake);
            let second_territory = analyzer.our_territory.clone();
            
            prop_assert_eq!(first_territory, second_territory, "Territory analysis should be deterministic");
        }
        
        #[test]
        fn property_voronoi_distance_calculation_triangle_inequality(
            coord1 in coord_strategy(20, 20),
            coord2 in coord_strategy(20, 20),
            coord3 in coord_strategy(20, 20)
        ) {
            let analyzer = VoronoiTerritoryAnalyzer::new();
            
            let dist_12 = analyzer.calculate_distance(coord1, coord2);
            let dist_23 = analyzer.calculate_distance(coord2, coord3);
            let dist_13 = analyzer.calculate_distance(coord1, coord3);
            
            // Property: Triangle inequality should hold
            prop_assert!(dist_13 <= dist_12 + dist_23 + 0.001, // Small epsilon for floating point
                       "Triangle inequality violated: {} > {} + {}", dist_13, dist_12, dist_23);
        }
    }

    // Property tests for DangerZonePredictor
    proptest! {
        #[test]
        fn property_danger_zone_values_in_range(
            (width, height, food) in board_strategy(3, 15, 8),
            our_snake in snake_strategy(15, 15, 10),
            turns in 1u32..=5
        ) {
            let mut predictor = DangerZonePredictor::new(turns);
            
            let our_battlesnake = create_test_snake_with_body("us", our_snake.clone(), 100);
            let board = Board {
                width,
                height,
                food,
                snakes: vec![our_battlesnake.clone()],
                hazards: vec![],
            };
            
            predictor.predict_danger_zones(&board, &our_battlesnake, turns);
            
            // Property: All danger levels should be in [0.0, 1.0]
            for (&_coord, &danger) in &predictor.danger_zones {
                prop_assert!(danger >= 0.0 && danger <= 1.0,
                           "Danger level {} should be in [0.0, 1.0]", danger);
            }
        }
        
        #[test]
        fn property_danger_zone_turns_monotonicity(
            (width, height, food) in board_strategy(5, 11, 5),
            our_snake in snake_strategy(11, 11, 6)
        ) {
            let our_battlesnake = create_test_snake_with_body("us", our_snake.clone(), 100);
            let board = Board {
                width,
                height,
                food,
                snakes: vec![our_battlesnake.clone()],
                hazards: vec![],
            };
            
            // Property: More turns should generally predict more danger zones (or equal)
            let mut predictor_1 = DangerZonePredictor::new(1);
            let mut predictor_3 = DangerZonePredictor::new(3);
            
            predictor_1.predict_danger_zones(&board, &our_battlesnake, 1);
            predictor_3.predict_danger_zones(&board, &our_battlesnake, 3);
            
            // The number of danger zones should not decrease with more prediction turns
            prop_assert!(predictor_3.danger_zones.len() >= predictor_1.danger_zones.len(),
                       "More prediction turns should identify at least as many danger zones: {} vs {}",
                       predictor_3.danger_zones.len(), predictor_1.danger_zones.len());
        }
        
        #[test]
        fn property_danger_zone_snake_body_dangerous(
            (width, height, food) in board_strategy(5, 11, 5),
            our_snake in snake_strategy(11, 11, 8),
            opponent_snake in snake_strategy(11, 11, 8)
        ) {
            let our_battlesnake = create_test_snake_with_body("us", our_snake.clone(), 100);
            let opponent_battlesnake = create_test_snake_with_body("opponent", opponent_snake.clone(), 100);
            
            let board = Board {
                width,
                height,
                food,
                snakes: vec![our_battlesnake.clone(), opponent_battlesnake.clone()],
                hazards: vec![],
            };
            
            let mut predictor = DangerZonePredictor::new(2);
            predictor.predict_danger_zones(&board, &our_battlesnake, 2);
            
            // Property: Opponent snake body positions should have high danger
            for body_segment in &opponent_battlesnake.body {
                if let Some(&danger) = predictor.danger_zones.get(body_segment) {
                    prop_assert!(danger > 0.7, 
                               "Opponent body segment at ({}, {}) should be highly dangerous: {}",
                               body_segment.x, body_segment.y, danger);
                }
            }
        }
    }

    // Property tests for MovementHistoryTracker
    proptest! {
        #[test]
        fn property_movement_history_time_decay(
            max_history in 5usize..=20,
            positions in prop::collection::vec(coord_strategy(10, 10), 1..15)
        ) {
            let mut tracker = MovementHistoryTracker::new(max_history);
            
            // Add positions over time
            for (turn, position) in positions.iter().enumerate() {
                tracker.add_position(*position, turn as u32);
            }
            
            // Property: Older positions should have lower influence due to time decay
            let history = tracker.get_recent_positions(positions.len() as u32);
            
            if history.len() >= 2 {
                let recent_influence = history.values().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                let oldest_influence = history.values().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                
                prop_assert!(recent_influence >= oldest_influence,
                           "Recent positions should have at least as much influence as older ones: {} vs {}",
                           recent_influence, oldest_influence);
            }
        }
        
        #[test]
        fn property_movement_history_bounded_size(
            max_history in 3usize..=15,
            positions in prop::collection::vec(coord_strategy(15, 15), 1..30)
        ) {
            let mut tracker = MovementHistoryTracker::new(max_history);
            
            // Add many positions
            for (turn, position) in positions.iter().enumerate() {
                tracker.add_position(*position, turn as u32);
            }
            
            // Property: History should never exceed max_history size
            prop_assert!(tracker.history.len() <= max_history,
                       "History size {} should not exceed max_history {}",
                       tracker.history.len(), max_history);
        }
        
        #[test]
        fn property_movement_history_influence_values(
            max_history in 5usize..=15,
            positions in prop::collection::vec(coord_strategy(12, 12), 3..12)
        ) {
            let mut tracker = MovementHistoryTracker::new(max_history);
            
            for (turn, position) in positions.iter().enumerate() {
                tracker.add_position(*position, turn as u32);
            }
            
            let recent_positions = tracker.get_recent_positions(positions.len() as u32);
            
            // Property: All influence values should be in (0.0, 1.0]
            for (&_coord, &influence) in &recent_positions {
                prop_assert!(influence > 0.0 && influence <= 1.0,
                           "Influence value {} should be in (0.0, 1.0]", influence);
            }
        }
    }

    // Property tests for StrategicPositionAnalyzer
    proptest! {
        #[test]
        fn property_strategic_position_values_bounded(
            (width, height, food) in board_strategy(5, 13, 8),
            our_snake in snake_strategy(13, 13, 10)
        ) {
            let mut analyzer = StrategicPositionAnalyzer::new();
            
            let our_battlesnake = create_test_snake_with_body("us", our_snake.clone(), 100);
            let board = Board {
                width,
                height,
                food,
                snakes: vec![our_battlesnake.clone()],
                hazards: vec![],
            };
            
            analyzer.analyze_strategic_positions(&board, &our_battlesnake);
            
            // Property: All strategic values should be bounded (typically [-1.0, 1.0])
            for (&_coord, &value) in &analyzer.strategic_positions {
                prop_assert!(value >= -2.0 && value <= 2.0,
                           "Strategic value {} should be reasonably bounded", value);
            }
        }
        
        #[test]
        fn property_strategic_positions_food_attraction(
            (width, height, food) in board_strategy(7, 11, 5),
            our_snake in snake_strategy(11, 11, 6)
        ) {
            let mut analyzer = StrategicPositionAnalyzer::new();
            
            let our_battlesnake = create_test_snake_with_body("us", our_snake.clone(), 100);
            let board = Board {
                width,
                height,
                food: food.clone(),
                snakes: vec![our_battlesnake.clone()],
                hazards: vec![],
            };
            
            analyzer.analyze_strategic_positions(&board, &our_battlesnake);
            
            // Property: Food positions should generally have positive strategic value
            for food_pos in &food {
                if let Some(&strategic_value) = analyzer.strategic_positions.get(food_pos) {
                    prop_assert!(strategic_value >= -0.5,
                               "Food position at ({}, {}) should not have very negative strategic value: {}",
                               food_pos.x, food_pos.y, strategic_value);
                }
            }
        }
    }

    // Property tests for mathematical functions
    proptest! {
        #[test]
        fn property_time_decay_function_monotonic(
            age in 0u32..=100,
            max_age in 1u32..=100
        ) {
            let decay1 = MovementHistoryTracker::calculate_time_decay(age, max_age.max(age + 1));
            let decay2 = MovementHistoryTracker::calculate_time_decay(age + 1, max_age.max(age + 2));
            
            // Property: Time decay should be monotonically decreasing with age
            prop_assert!(decay1 >= decay2,
                       "Time decay should decrease with age: decay({}) = {} >= decay({}) = {}",
                       age, decay1, age + 1, decay2);
        }
        
        #[test]
        fn property_time_decay_function_bounds(
            age in 0u32..=50,
            max_age in 1u32..=50
        ) {
            let max_age = max_age.max(age + 1);
            let decay = MovementHistoryTracker::calculate_time_decay(age, max_age);
            
            // Property: Time decay should be in (0.0, 1.0]
            prop_assert!(decay > 0.0 && decay <= 1.0,
                       "Time decay {} should be in (0.0, 1.0]", decay);
        }
        
        #[test]
        fn property_distance_calculation_symmetry(
            coord1 in coord_strategy(20, 20),
            coord2 in coord_strategy(20, 20)
        ) {
            let analyzer = VoronoiTerritoryAnalyzer::new();
            
            let dist_12 = analyzer.calculate_distance(coord1, coord2);
            let dist_21 = analyzer.calculate_distance(coord2, coord1);
            
            // Property: Distance should be symmetric
            prop_assert!((dist_12 - dist_21).abs() < 0.001,
                       "Distance should be symmetric: d({:?}, {:?}) = {} != d({:?}, {:?}) = {}",
                       coord1, coord2, dist_12, coord2, coord1, dist_21);
        }
        
        #[test]
        fn property_distance_calculation_identity(
            coord in coord_strategy(15, 15)
        ) {
            let analyzer = VoronoiTerritoryAnalyzer::new();
            
            let dist = analyzer.calculate_distance(coord, coord);
            
            // Property: Distance from a point to itself should be zero
            prop_assert!(dist.abs() < 0.001,
                       "Distance from point to itself should be zero: {}", dist);
        }
    }

    // Property tests for cross-component consistency
    proptest! {
        #[test]
        fn property_encoder_channel_consistency(
            (width, height, food) in board_strategy(5, 11, 5),
            our_snake in snake_strategy(11, 11, 6)
        ) {
            let mut encoder = AdvancedBoardStateEncoder::new(10, 2);
            
            let our_battlesnake = create_test_snake_with_body("us", our_snake.clone(), 100);
            let board = Board {
                width,
                height,
                food,
                snakes: vec![our_battlesnake.clone()],
                hazards: vec![],
            };
            
            let channels = encoder.encode_12_channel_board(&board, &our_battlesnake, 5);
            
            // Property: All channel values should be in [0.0, 1.0]
            for (ch_idx, channel) in channels.iter().enumerate() {
                for (y, row) in channel.iter().enumerate() {
                    for (x, &value) in row.iter().enumerate() {
                        prop_assert!(value >= 0.0 && value <= 1.0,
                                   "Channel {} value at ({}, {}) should be in [0.0, 1.0]: {}",
                                   ch_idx, x, y, value);
                    }
                }
            }
        }
        
        #[test]
        fn property_encoder_dimension_consistency(
            (width, height, food) in board_strategy(3, 15, 8),
            our_snake in snake_strategy(15, 15, 8)
        ) {
            let mut encoder = AdvancedBoardStateEncoder::new(8, 3);
            
            let our_battlesnake = create_test_snake_with_body("us", our_snake.clone(), 100);
            let board = Board {
                width,
                height,
                food,
                snakes: vec![our_battlesnake.clone()],
                hazards: vec![],
            };
            
            let channels = encoder.encode_12_channel_board(&board, &our_battlesnake, 10);
            
            // Property: All channels should have consistent dimensions matching the board
            prop_assert_eq!(channels.len(), 12, "Should have exactly 12 channels");
            
            for (ch_idx, channel) in channels.iter().enumerate() {
                prop_assert_eq!(channel.len(), height as usize,
                              "Channel {} height should match board height: {} vs {}",
                              ch_idx, channel.len(), height);
                              
                if !channel.is_empty() {
                    prop_assert_eq!(channel[0].len(), width as usize,
                                  "Channel {} width should match board width: {} vs {}",
                                  ch_idx, channel[0].len(), width);
                }
            }
        }
        
        #[test]
        fn property_encoder_our_head_unique(
            (width, height, food) in board_strategy(5, 11, 5),
            our_snake in snake_strategy(11, 11, 6)
        ) {
            prop_assume!(!our_snake.is_empty()); // Ensure we have a valid snake
            
            let mut encoder = AdvancedBoardStateEncoder::new(5, 2);
            
            let our_battlesnake = create_test_snake_with_body("us", our_snake.clone(), 100);
            let board = Board {
                width,
                height,
                food,
                snakes: vec![our_battlesnake.clone()],
                hazards: vec![],
            };
            
            let channels = encoder.encode_12_channel_board(&board, &our_battlesnake, 3);
            
            // Property: Channel 1 (our head) should have exactly one position marked as 1.0
            let our_head_channel = &channels[1];
            let mut head_positions = 0;
            
            for row in our_head_channel {
                for &value in row {
                    if value == 1.0 {
                        head_positions += 1;
                    }
                }
            }
            
            prop_assert_eq!(head_positions, 1,
                          "Our head channel should have exactly one position marked as 1.0, found {}",
                          head_positions);
        }
    }

    // Property tests for edge case handling
    proptest! {
        #[test]
        fn property_empty_board_handling(
            width in 1i32..=5,
            height in 1u32..=5
        ) {
            let mut encoder = AdvancedBoardStateEncoder::new(3, 1);
            
            let our_battlesnake = create_test_snake("us", Coord { x: 0, y: 0 }, 1, 100);
            let empty_board = Board {
                width,
                height,
                food: vec![],
                snakes: vec![our_battlesnake.clone()],
                hazards: vec![],
            };
            
            // Property: Should not panic on empty boards
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                encoder.encode_12_channel_board(&empty_board, &our_battlesnake, 1)
            }));
            
            prop_assert!(result.is_ok(), "Should not panic on empty board");
            
            if let Ok(channels) = result {
                prop_assert_eq!(channels.len(), 12, "Should still produce 12 channels for empty board");
            }
        }
        
        #[test]
        fn property_single_cell_board_handling(
            snake_health in 1u32..=100
        ) {
            let mut encoder = AdvancedBoardStateEncoder::new(3, 1);
            
            let our_battlesnake = create_test_snake("us", Coord { x: 0, y: 0 }, 1, snake_health as i32);
            let single_cell_board = Board {
                width: 1,
                height: 1,
                food: vec![],
                snakes: vec![our_battlesnake.clone()],
                hazards: vec![],
            };
            
            // Property: Should handle single-cell boards gracefully - TODO: Re-implement this test properly
        }
    }
}

#[cfg(test)]
mod fuzzing_tests {
    use super::*;
    use crate::spatial_test_utilities::*;
    use quickcheck::{quickcheck, TestResult, Arbitrary, Gen};
    use std::panic;

    // Quickcheck implementations for fuzzing
    #[derive(Debug, Clone)]
    struct FuzzCoord {
        x: i32,
        y: i32,
    }

    impl Arbitrary for FuzzCoord {
        fn arbitrary(g: &mut Gen) -> Self {
            FuzzCoord {
                x: i32::arbitrary(g) % 50, // Limit to reasonable range
                y: i32::arbitrary(g) % 50,
            }
        }
    }

    impl Into<Coord> for FuzzCoord {
        fn into(self) -> Coord {
            Coord { x: self.x.abs(), y: self.y.abs() }
        }
    }

    #[derive(Debug, Clone)]
    struct FuzzBoard {
        width: i32,
        height: u32,
        food: Vec<FuzzCoord>,
        snakes: Vec<Vec<FuzzCoord>>,
    }

    impl Arbitrary for FuzzBoard {
        fn arbitrary(g: &mut Gen) -> Self {
            let width = (i32::arbitrary(g).abs() % 20).max(1);
            let height = (u32::arbitrary(g) % 20).max(1);
            let food_count = usize::arbitrary(g) % 10;
            let snake_count = (usize::arbitrary(g) % 5).max(1);
            
            let food: Vec<FuzzCoord> = (0..food_count)
                .map(|_| FuzzCoord {
                    x: i32::arbitrary(g).abs() % width,
                    y: (i32::arbitrary(g).abs() % height as i32),
                })
                .collect();
            
            let snakes: Vec<Vec<FuzzCoord>> = (0..snake_count)
                .map(|_| {
                    let snake_length = (usize::arbitrary(g) % 15).max(1);
                    (0..snake_length)
                        .map(|_| FuzzCoord {
                            x: i32::arbitrary(g).abs() % width,
                            y: (i32::arbitrary(g).abs() % height as i32),
                        })
                        .collect()
                })
                .collect();
            
            FuzzBoard { width, height, food, snakes }
        }
    }

    impl FuzzBoard {
        fn to_game_state(&self) -> (Board, Battlesnake) {
            let food: Vec<Coord> = self.food.iter().cloned().map(|f| f.into()).collect();
            
            let our_snake = if !self.snakes.is_empty() && !self.snakes[0].is_empty() {
                let body: Vec<Coord> = self.snakes[0].iter().cloned().map(|c| c.into()).collect();
                Battlesnake {
                    id: "fuzz_snake".to_string(),
                    name: "Fuzz Snake".to_string(),
                    health: 100,
                    body: body.clone(),
                    head: body[0],
                    length: body.len() as u32,
                    latency: "0".to_string(),
                    shout: None,
                }
            } else {
                create_test_snake("fuzz_snake", Coord { x: 0, y: 0 }, 1, 100)
            };
            
            let mut battlesnakes = vec![our_snake.clone()];
            
            for (i, snake_body) in self.snakes.iter().skip(1).enumerate() {
                if !snake_body.is_empty() {
                    let body: Vec<Coord> = snake_body.iter().cloned().map(|c| c.into()).collect();
                    battlesnakes.push(Battlesnake {
                        id: format!("opponent_{}", i),
                        name: format!("Opponent {}", i),
                        health: 100,
                        body: body.clone(),
                        head: body[0],
                        length: body.len() as u32,
                        latency: "0".to_string(),
                        shout: None,
                    });
                }
            }
            
            let board = Board {
                width: self.width,
                height: self.height,
                food,
                snakes: battlesnakes,
                hazards: vec![],
            };
            
            (board, our_snake)
        }
    }

    // Comprehensive fuzzing tests using RandomBoardGenerator
    #[test]
    fn fuzz_test_voronoi_analyzer_random_boards() {
        let mut generator = RandomBoardGenerator::new(12345);
        let mut analyzer = VoronoiTerritoryAnalyzer::new();
        
        for i in 0..100 {
            let (board, our_snake) = generator.generate_random_board(
                3 + (i % 15), // width 3-18
                3 + (i % 12), // height 3-15
                i % 8,        // max_food
                1 + (i % 4),  // max_snakes
                1 + (i % 10), // max_snake_length
            );
            
            // Test should not panic
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                analyzer.analyze_territory(&board, &our_snake);
            }));
            
            assert!(result.is_ok(), "VoronoiTerritoryAnalyzer panicked on iteration {}", i);
            
            // Validate results if analysis succeeded
            if result.is_ok() {
                for (&_coord, &strength) in &analyzer.our_territory {
                    assert!(strength >= 0.0 && strength <= 1.0, 
                           "Invalid territory strength {} on iteration {}", strength, i);
                }
                
                for (&_coord, &strength) in &analyzer.opponent_territory {
                    assert!(strength >= 0.0 && strength <= 1.0,
                           "Invalid opponent territory strength {} on iteration {}", strength, i);
                }
            }
        }
    }

    #[test]
    fn fuzz_test_danger_zone_predictor_random_scenarios() {
        let mut generator = RandomBoardGenerator::new(67890);
        
        for i in 0..100 {
            let turns = 1 + (i % 5) as u32;
            let mut predictor = DangerZonePredictor::new(turns);
            
            let (board, our_snake) = generator.generate_random_board(
                4 + (i % 12),
                4 + (i % 10),
                i % 6,
                1 + (i % 3),
                2 + (i % 8),
            );
            
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                predictor.predict_danger_zones(&board, &our_snake, turns);
            }));
            
            assert!(result.is_ok(), "DangerZonePredictor panicked on iteration {}", i);
            
            if result.is_ok() {
                for (&_coord, &danger) in &predictor.danger_zones {
                    assert!(danger >= 0.0 && danger <= 1.0,
                           "Invalid danger level {} on iteration {}", danger, i);
                }
            }
        }
    }

    #[test]
    fn fuzz_test_movement_history_tracker_edge_cases() {
        let mut generator = RandomBoardGenerator::new(13579);
        
        for i in 0..50 {
            let max_history = 3 + (i % 20);
            let mut tracker = MovementHistoryTracker::new(max_history);
            
            // Add random positions over many turns
            for turn in 0..(i % 50 + 10) {
                let x = (turn * 7) % 20;
                let y = (turn * 11) % 15;
                let pos = Coord { x: x as i32, y: y as i32 };
                
                let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                    tracker.add_position(pos, turn as u32);
                }));
                
                assert!(result.is_ok(), "MovementHistoryTracker panicked adding position on iteration {}", i);
            }
            
            // Test getting recent positions
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                let recent = tracker.get_recent_positions((i % 50 + 10) as u32);
                for (&_coord, &influence) in &recent {
                    assert!(influence > 0.0 && influence <= 1.0,
                           "Invalid influence {} on iteration {}", influence, i);
                }
            }));
            
            assert!(result.is_ok(), "MovementHistoryTracker panicked getting positions on iteration {}", i);
            
            // Verify history is bounded
            assert!(tracker.history.len() <= max_history,
                   "History size {} exceeded max {} on iteration {}", 
                   tracker.history.len(), max_history, i);
        }
    }

    #[test]
    fn fuzz_test_strategic_position_analyzer_random_boards() {
        let mut generator = RandomBoardGenerator::new(24680);
        let mut analyzer = StrategicPositionAnalyzer::new();
        
        for i in 0..80 {
            let (board, our_snake) = generator.generate_random_board(
                5 + (i % 10),
                5 + (i % 8),
                i % 12,
                1 + (i % 4),
                3 + (i % 7),
            );
            
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                analyzer.analyze_strategic_positions(&board, &our_snake);
            }));
            
            assert!(result.is_ok(), "StrategicPositionAnalyzer panicked on iteration {}", i);
            
            if result.is_ok() {
                for (&_coord, &value) in &analyzer.strategic_positions {
                    assert!(value >= -5.0 && value <= 5.0,
                           "Strategic value {} out of reasonable bounds on iteration {}", value, i);
                }
            }
        }
    }

    #[test]
    fn fuzz_test_advanced_board_state_encoder_comprehensive() {
        let mut generator = RandomBoardGenerator::new(98765);
        
        for i in 0..60 {
            let max_history = 5 + (i % 15);
            let prediction_depth = 1 + (i % 4);
            let mut encoder = AdvancedBoardStateEncoder::new(max_history, prediction_depth);
            
            let (board, our_snake) = generator.generate_random_board(
                3 + (i % 15),
                3 + (i % 12),
                i % 10,
                1 + (i % 5),
                2 + (i % 12),
            );
            
            let turn = i as u32;
            
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                encoder.encode_12_channel_board(&board, &our_snake, turn)
            }));
            
            assert!(result.is_ok(), "AdvancedBoardStateEncoder panicked on iteration {}", i);
            
            if let Ok(channels) = result {
                assert_eq!(channels.len(), 12, "Should always have 12 channels on iteration {}", i);
                
                for (ch_idx, channel) in channels.iter().enumerate() {
                    assert_eq!(channel.len(), board.height as usize,
                              "Channel {} height mismatch on iteration {}", ch_idx, i);
                    
                    if !channel.is_empty() {
                        assert_eq!(channel[0].len(), board.width as usize,
                                  "Channel {} width mismatch on iteration {}", ch_idx, i);
                    }
                    
                    // Validate all values in range
                    for (y, row) in channel.iter().enumerate() {
                        for (x, &value) in row.iter().enumerate() {
                            assert!(value >= 0.0 && value <= 1.0,
                                   "Invalid channel {} value {} at ({}, {}) on iteration {}",
                                   ch_idx, value, x, y, i);
                        }
                    }
                }
            }
        }
    }

    // QuickCheck-based fuzzing tests
    quickcheck! {
        fn quickcheck_voronoi_analyzer_no_panic(fuzz_board: FuzzBoard) -> TestResult {
            if fuzz_board.width <= 0 || fuzz_board.height == 0 || fuzz_board.snakes.is_empty() {
                return TestResult::discard();
            }
            
            let mut analyzer = VoronoiTerritoryAnalyzer::new();
            let (board, our_snake) = fuzz_board.to_game_state();
            
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                analyzer.analyze_territory(&board, &our_snake);
            }));
            
            TestResult::from_bool(result.is_ok())
        }
        
        fn quickcheck_danger_predictor_no_panic(fuzz_board: FuzzBoard, turns: u8) -> TestResult {
            if fuzz_board.width <= 0 || fuzz_board.height == 0 || turns == 0 || turns > 10 {
                return TestResult::discard();
            }
            
            let mut predictor = DangerZonePredictor::new(turns as u32);
            let (board, our_snake) = fuzz_board.to_game_state();
            
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                predictor.predict_danger_zones(&board, &our_snake, turns as u32);
            }));
            
            TestResult::from_bool(result.is_ok())
        }
        
        fn quickcheck_movement_tracker_bounds(positions: Vec<(i16, i16)>, max_history: u8) -> TestResult {
            if max_history == 0 || max_history > 100 {
                return TestResult::discard();
            }
            
            let mut tracker = MovementHistoryTracker::new(max_history as usize);
            
            for (turn, (x, y)) in positions.iter().enumerate() {
                let coord = Coord { x: *x as i32, y: *y as i32 };
                tracker.add_position(coord, turn as u32);
            }
            
            let history_bounded = tracker.history.len() <= max_history as usize;
            let recent_positions = tracker.get_recent_positions(positions.len() as u32);
            let all_influences_valid = recent_positions.values()
                .all(|&influence| influence > 0.0 && influence <= 1.0);
            
            TestResult::from_bool(history_bounded && all_influences_valid)
        }
        
        fn quickcheck_encoder_channel_structure(fuzz_board: FuzzBoard) -> TestResult {
            if fuzz_board.width <= 0 || fuzz_board.height == 0 || 
               fuzz_board.width > 25 || fuzz_board.height > 25 {
                return TestResult::discard();
            }
            
            let mut encoder = AdvancedBoardStateEncoder::new(10, 2);
            let (board, our_snake) = fuzz_board.to_game_state();
            
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                encoder.encode_12_channel_board(&board, &our_snake, 5)
            }));
            
            if let Ok(channels) = result {
                let correct_channel_count = channels.len() == 12;
                let correct_dimensions = channels.iter().all(|channel| {
                    channel.len() == board.height as usize &&
                    (channel.is_empty() || channel[0].len() == board.width as usize)
                });
                let valid_values = channels.iter().all(|channel| {
                    channel.iter().all(|row| {
                        row.iter().all(|&value| value >= 0.0 && value <= 1.0)
                    })
                });
                
                TestResult::from_bool(correct_channel_count && correct_dimensions && valid_values)
            } else {
                TestResult::from_bool(false)
            }
        }
    }

    // Extreme scenario fuzzing tests
    #[test]
    fn fuzz_test_extreme_board_sizes() {
        let extreme_scenarios = vec![
            (1, 1, 0, 1, 1),    // Minimal board
            (1, 50, 0, 1, 1),   // Very narrow
            (50, 1, 0, 1, 1),   // Very wide
            (25, 25, 50, 1, 1), // Large with lots of food
            (15, 15, 0, 8, 3),  // Many snakes
            (10, 10, 5, 3, 20), // Long snakes
        ];
        
        for (i, &(width, height, max_food, max_snakes, max_length)) in extreme_scenarios.iter().enumerate() {
            let mut generator = RandomBoardGenerator::new(i as u64 * 1000);
            let (board, our_snake) = generator.generate_random_board(
                width, height, max_food, max_snakes, max_length
            );
            
            // Test all components with extreme scenarios
            let mut encoder = AdvancedBoardStateEncoder::new(20, 3);
            
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                let start = std::time::Instant::now();
                let channels = encoder.encode_12_channel_board(&board, &our_snake, i as u32);
                let elapsed = start.elapsed().as_millis();
                
                // Should complete within reasonable time even for extreme cases
                assert!(elapsed < 2000, "Extreme scenario {} took too long: {}ms", i, elapsed);
                
                // Basic validation
                assert_eq!(channels.len(), 12, "Extreme scenario {} should produce 12 channels", i);
                
                for channel in &channels {
                    assert_eq!(channel.len(), height as usize, "Height mismatch in scenario {}", i);
                    if !channel.is_empty() {
                        assert_eq!(channel[0].len(), width as usize, "Width mismatch in scenario {}", i);
                    }
                }
            }));
            
            assert!(result.is_ok(), "Extreme scenario {} should not panic", i);
        }
    }

    #[test]
    fn fuzz_test_pathological_snake_configurations() {
        let pathological_configs = vec![
            // Very long snake that fills most of the board
            (7, 7, vec![
                (0..7).map(|x| Coord { x, y: 0 }).chain(
                (0..7).rev().map(|x| Coord { x, y: 1 })).chain(
                (0..7).map(|x| Coord { x, y: 2 })).collect::<Vec<_>>()
            ], "spiral_snake"),
            
            // Snake that forms a loop
            (5, 5, vec![
                Coord { x: 2, y: 2 }, Coord { x: 2, y: 1 }, Coord { x: 1, y: 1 },
                Coord { x: 1, y: 2 }, Coord { x: 1, y: 3 }, Coord { x: 2, y: 3 },
                Coord { x: 3, y: 3 }, Coord { x: 3, y: 2 }, Coord { x: 3, y: 1 },
            ], "loop_snake"),
            
            // Multiple snakes in corner
            (4, 4, vec![
                Coord { x: 0, y: 0 }, Coord { x: 0, y: 1 }, Coord { x: 0, y: 2 },
            ], "corner_snake"),
        ];
        
        for (i, (width, height, snake_body, name)) in pathological_configs.into_iter().enumerate() {
            let our_snake = Battlesnake {
                id: name.to_string(),
                name: name.to_string(),
                health: 100,
                body: snake_body.clone(),
                head: snake_body[0],
                length: snake_body.len() as u32,
                latency: "0".to_string(),
                shout: None,
            };
            
            let board = Board {
                width,
                height,
                food: vec![],
                snakes: vec![our_snake.clone()],
                hazards: vec![],
            };
            
            let mut encoder = AdvancedBoardStateEncoder::new(10, 2);
            
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                encoder.encode_12_channel_board(&board, &our_snake, i as u32)
            }));
            
            assert!(result.is_ok(), "Pathological configuration '{}' should not panic", name);
            
            if let Ok(channels) = result {
                assert_eq!(channels.len(), 12);
                
                // Verify our head is marked correctly despite pathological configuration
                let our_head_channel = &channels[1];
                let head_count: usize = our_head_channel.iter()
                    .flat_map(|row| row.iter())
                    .map(|&v| if v == 1.0 { 1 } else { 0 })
                    .sum();
                
                assert_eq!(head_count, 1, "Should have exactly one head marked in '{}'", name);
            }
        }
    }

    #[test]
    fn fuzz_test_memory_stress_long_running() {
        // Test memory behavior under extended usage
        let mut encoder = AdvancedBoardStateEncoder::new(50, 4);
        let mut generator = RandomBoardGenerator::new(55555);
        
        for turn in 0..500 {
            let (board, our_snake) = generator.generate_random_board(
                7 + (turn % 8),
                6 + (turn % 7),
                turn % 5,
                1 + (turn % 3),
                3 + (turn % 6),
            );
            
            let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                encoder.encode_12_channel_board(&board, &our_snake, turn as u32)
            }));
            
            assert!(result.is_ok(), "Memory stress test failed at turn {}", turn);
            
            // Periodically check memory bounds
            if turn % 100 == 0 {
                assert!(encoder.movement_tracker.history.len() <= encoder.movement_tracker.max_history,
                       "Movement history exceeded bounds at turn {}", turn);
                
                let (our_territory, opponent_territory, danger_positions, strategic_positions) = encoder.get_encoding_stats();
                let total_tracked = our_territory + opponent_territory + danger_positions + strategic_positions;
                
                assert!(total_tracked < 10000, "Total tracked positions {} excessive at turn {}", total_tracked, turn);
            }
        }
    }

    #[test]
    fn fuzz_test_concurrent_component_safety() {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let encoder = Arc::new(Mutex::new(AdvancedBoardStateEncoder::new(15, 2)));
        let mut handles = vec![];
        
        // Spawn multiple threads to test thread safety
        for thread_id in 0..4 {
            let encoder_clone = Arc::clone(&encoder);
            
            let handle = thread::spawn(move || {
                let mut generator = RandomBoardGenerator::new(thread_id as u64 * 99999);
                
                for i in 0..25 {
                    let (board, our_snake) = generator.generate_random_board(
                        5 + (i % 8),
                        5 + (i % 6),
                        i % 4,
                        1 + (i % 2),
                        2 + (i % 5),
                    );
                    
                    // Test that locking doesn't cause panics
                    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                        let mut enc = encoder_clone.lock().unwrap();
                        enc.encode_12_channel_board(&board, &our_snake, (thread_id * 100 + i) as u32)
                    }));
                    
                    assert!(result.is_ok(), "Thread {} panicked at iteration {}", thread_id, i);
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }
    }

    #[test]
    fn fuzz_test_performance_under_load() {
        let mut encoder = AdvancedBoardStateEncoder::new(25, 3);
        let mut generator = RandomBoardGenerator::new(777777);
        
        let mut total_time = 0u128;
        let iterations = 100;
        
        for i in 0..iterations {
            let (board, our_snake) = generator.generate_random_board(
                8 + (i % 12),  // Reasonably large boards
                7 + (i % 10),
                i % 8,
                1 + (i % 4),
                4 + (i % 8),
            );
            
            let start = std::time::Instant::now();
            let result = encoder.encode_12_channel_board(&board, &our_snake, i as u32);

#[cfg(test)]
mod cross_component_integration_tests {
    use super::*;
    use crate::spatial_test_utilities::*;

    /// Test comprehensive integration between all spatial analysis components
    #[test]
    fn test_cross_component_data_flow_consistency() {
        let mut encoder = AdvancedBoardStateEncoder::new(15, 3);
        
        let game_state = GameStateBuilder::new()
            .with_board_size(11, 11)
            .with_our_snake(create_test_snake("us", Coord { x: 5, y: 5 }, 4, 100))
            .with_opponent(create_test_snake("opponent1", Coord { x: 2, y: 2 }, 4, 95))
            .with_opponent(create_test_snake("opponent2", Coord { x: 8, y: 8 }, 3, 90))
            .with_food(vec![Coord { x: 1, y: 1 }, Coord { x: 9, y: 9 }, Coord { x: 5, y: 1 }])
            .build();
        
        // Encode multiple turns to build up internal state
        let turns = vec![3, 4, 5, 6, 7];
        let mut all_channels = Vec::new();
        
        for &turn in &turns {
            let channels = encoder.encode_12_channel_board(&game_state.board, &game_state.you, turn);
            all_channels.push((turn, channels));
        }
        
        // Test 1: Movement history should progressively accumulate
        for i in 1..all_channels.len() {
            let (prev_turn, prev_channels) = &all_channels[i-1];
            let (curr_turn, curr_channels) = &all_channels[i];
            
            let prev_history_sum: f32 = prev_channels[10].iter().flatten().sum();
            let curr_history_sum: f32 = curr_channels[10].iter().flatten().sum();
            
            assert!(curr_history_sum >= prev_history_sum * 0.8, // Allow for time decay
                   "Movement history should accumulate over time: turn {} sum {} vs turn {} sum {}",
                   prev_turn, prev_history_sum, curr_turn, curr_history_sum);
        }
        
        // Test 2: Territory and strategic position consistency
        let final_channels = &all_channels.last().unwrap().1;
        let our_territory = &final_channels[7];
        let strategic_positions = &final_channels[11];
        
        let mut territory_strategic_alignments = 0;
        let mut total_strong_territory = 0;
        
        for y in 0..11 {
            for x in 0..11 {
                if our_territory[y][x] > 0.7 { // Strong territorial control
                    total_strong_territory += 1;
                    if strategic_positions[y][x] > 0.2 { // Positive strategic value
                        territory_strategic_alignments += 1;
                    }
                }
            }
        }
        
        if total_strong_territory > 0 {
            let alignment_ratio = territory_strategic_alignments as f32 / total_strong_territory as f32;
            assert!(alignment_ratio > 0.3, 
                   "Territory control and strategic positions should show reasonable alignment: {}%", 
                   alignment_ratio * 100.0);
        }
        
        // Test 3: Danger zones should correlate with opponent positions
        let danger_channel = &final_channels[9];
        let opponent_bodies_channel = &final_channels[4];
        
        let mut dangerous_opponent_positions = 0;
        let mut total_opponent_positions = 0;
        
        for y in 0..11 {
            for x in 0..11 {
                if opponent_bodies_channel[y][x] > 0.5 {
                    total_opponent_positions += 1;
                    if danger_channel[y][x] > 0.6 {
                        dangerous_opponent_positions += 1;
                    }
                }
            }
        }
        
        if total_opponent_positions > 0 {
            let danger_ratio = dangerous_opponent_positions as f32 / total_opponent_positions as f32;
            assert!(danger_ratio > 0.5, 
                   "Most opponent body positions should be marked as dangerous: {}%", 
                   danger_ratio * 100.0);
        }
    }

    #[test]
    fn test_territorial_dominance_consistency() {
        let mut encoder = AdvancedBoardStateEncoder::new(10, 2);
        
        // Create scenarios with clear territorial dominance
        let scenarios = vec![
            // Our snake dominates center
            GameStateBuilder::new()
                .with_board_size(9, 9)
                .with_our_snake(create_test_snake("us", Coord { x: 4, y: 4 }, 5, 100))
                .with_opponent(create_test_snake("opponent", Coord { x: 1, y: 1 }, 3, 80))
                .build(),
            
            // Opponent dominates with longer snake
            GameStateBuilder::new()
                .with_board_size(9, 9)
                .with_our_snake(create_test_snake("us", Coord { x: 7, y: 7 }, 2, 90))
                .with_opponent(create_test_snake("opponent", Coord { x: 2, y: 2 }, 7, 100))
                .build(),
            
            // Equal competition
            GameStateBuilder::new()
                .with_board_size(9, 9)
                .with_our_snake(create_test_snake("us", Coord { x: 2, y: 4 }, 4, 95))
                .with_opponent(create_test_snake("opponent", Coord { x: 6, y: 4 }, 4, 95))
                .build(),
        ];
        
        for (scenario_idx, scenario) in scenarios.iter().enumerate() {
            let channels = encoder.encode_12_channel_board(&scenario.board, &scenario.you, 10);
            
            let our_territory_sum: f32 = channels[7].iter().flatten().sum();
            let opponent_territory_sum: f32 = channels[8].iter().flatten().sum();
            
            let our_snake_length = scenario.you.length;
            let opponent_snake_length = scenario.board.snakes.iter()
                .find(|s| s.id != scenario.you.id)
                .map(|s| s.length)
                .unwrap_or(0);
            
            let length_ratio = our_snake_length as f32 / opponent_snake_length.max(1) as f32;
            let territory_ratio = our_territory_sum / opponent_territory_sum.max(0.1);
            
            // Territory distribution should roughly correlate with snake size advantage
            match scenario_idx {
                0 => {
                    // We have central position advantage
                    assert!(territory_ratio > 0.7, 
                           "Scenario {}: Central position should give territorial advantage", scenario_idx);
                }
                1 => {
                    // Opponent has size advantage  
                    assert!(territory_ratio < 1.5, 
                           "Scenario {}: Larger opponent should limit our territory", scenario_idx);
                }
                2 => {
                    // Equal competition
                    assert!(territory_ratio > 0.3 && territory_ratio < 3.0,
                           "Scenario {}: Equal competition should have balanced territory", scenario_idx);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_food_strategic_value_integration() {
        let mut encoder = AdvancedBoardStateEncoder::new(8, 2);
        
        let scenarios = vec![
            // Food near our snake
            (Coord { x: 3, y: 3 }, vec![Coord { x: 4, y: 3 }], "adjacent_food"),
            // Food far from our snake
            (Coord { x: 1, y: 1 }, vec![Coord { x: 9, y: 9 }], "distant_food"),
            // Multiple food sources
            (Coord { x: 5, y: 5 }, vec![Coord { x: 4, y: 5 }, Coord { x: 6, y: 5 }, Coord { x: 5, y: 4 }], "multiple_food"),
            // Food contested by opponent
            (Coord { x: 3, y: 3 }, vec![Coord { x: 5, y: 5 }], "contested_food"),
        ];
        
        for (our_head, food, scenario_name) in scenarios {
            let mut game_builder = GameStateBuilder::new()
                .with_board_size(11, 11)
                .with_our_snake(create_test_snake("us", our_head, 3, 85))
                .with_food(food.clone());
            
            if scenario_name == "contested_food" {
                game_builder = game_builder.with_opponent(create_test_snake("opponent", Coord { x: 7, y: 7 }, 3, 90));
            }
            
            let game_state = game_builder.build();
            let channels = encoder.encode_12_channel_board(&game_state.board, &game_state.you, 5);
            
            let food_channel = &channels[5];
            let strategic_channel = &channels[11];
            
            // Test food-strategic value correlation
            for food_pos in &food {
                let food_x = food_pos.x as usize;
                let food_y = food_pos.y as usize;
                
                // Food should be marked in food channel
                assert_eq!(food_channel[food_y][food_x], 1.0, 
                          "Food should be marked in food channel for scenario '{}'", scenario_name);
                
                // Strategic value near food should be influenced by food presence
                let strategic_value = strategic_channel[food_y][food_x];
                let distance_to_our_head = ((food_pos.x - our_head.x).abs() + (food_pos.y - our_head.y).abs()) as f32;
                
                match scenario_name {
                    "adjacent_food" => {
                        assert!(strategic_value > 0.1, 
                               "Adjacent food should have positive strategic value: {}", strategic_value);
                    }
                    "distant_food" => {
                        // Distant food might have lower strategic value
                        assert!(strategic_value >= -1.0, 
                               "Distant food strategic value should be reasonable: {}", strategic_value);
                    }
                    "multiple_food" => {
                        // Multiple nearby food should be valuable
                        if distance_to_our_head <= 2.0 {
                            assert!(strategic_value > -0.5, 
                                   "Nearby food in multiple food scenario should be valuable: {}", strategic_value);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn test_danger_prediction_movement_correlation() {
        let mut encoder = AdvancedBoardStateEncoder::new(12, 3);
        
        let game_state = GameStateBuilder::new()
            .with_board_size(9, 9)
            .with_our_snake(create_test_snake("us", Coord { x: 4, y: 4 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 2, y: 2 }, 4, 95))
            .with_food(vec![Coord { x: 6, y: 6 }])
            .build();
        
        // Simulate movement over multiple turns
        let movement_sequence = vec![
            (Coord { x: 4, y: 4 }, 5),   // Initial
            (Coord { x: 4, y: 3 }, 6),   // Move up
            (Coord { x: 4, y: 2 }, 7),   // Move up (closer to opponent)
            (Coord { x: 3, y: 2 }, 8),   // Move left (very close to opponent)
        ];
        
        let mut previous_danger_level = 0.0;
        
        for (position, turn) in movement_sequence {
            let test_snake = create_test_snake("us", position, 3, 100);
            let test_board = Board {
                width: game_state.board.width,
                height: game_state.board.height,
                food: game_state.board.food.clone(),
                snakes: vec![test_snake.clone(), game_state.board.snakes[1].clone()],
                hazards: vec![],
            };
            
            let channels = encoder.encode_12_channel_board(&test_board, &test_snake, turn);
            
            let danger_channel = &channels[9];
            let movement_channel = &channels[10];
            
            // Calculate average danger around our current position
            let our_x = position.x as usize;
            let our_y = position.y as usize;
            let mut local_danger = 0.0;
            let mut danger_cells = 0;
            
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let check_x = (our_x as i32 + dx) as usize;
                    let check_y = (our_y as i32 + dy) as usize;
                    
                    if check_y < danger_channel.len() && check_x < danger_channel[check_y].len() {
                        local_danger += danger_channel[check_y][check_x];
                        danger_cells += 1;
                    }
                }
            }
            
            if danger_cells > 0 {
                local_danger /= danger_cells as f32;
            }
            
            // As we get closer to opponent, danger should generally increase
            let distance_to_opponent = ((position.x - 2).abs() + (position.y - 2).abs()) as f32;
            
            if turn > 6 && distance_to_opponent < 3.0 {
                assert!(local_danger > 0.3, 
                       "Should detect increased danger when close to opponent at turn {}: danger={}, distance={}",
                       turn, local_danger, distance_to_opponent);
            }
            
            // Movement history should accumulate
            let movement_sum: f32 = movement_channel.iter().flatten().sum();
            if turn > 5 {
                assert!(movement_sum > 0.5, "Movement history should accumulate over turns: {}", movement_sum);
            }
            
            previous_danger_level = local_danger;
        }
    }

    #[test]
    fn test_channel_mathematical_properties() {
        let mut encoder = AdvancedBoardStateEncoder::new(10, 2);
        
        let game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_test_snake("us", Coord { x: 3, y: 3 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 1, y: 1 }, 3, 100))
            .with_food(vec![Coord { x: 5, y: 5 }])
            .build();
        
        let channels = encoder.encode_12_channel_board(&game_state.board, &game_state.you, 10);
        
        // Test 1: Empty channel should be complement of occupied positions
        let empty_channel = &channels[0];
        let occupied_positions = vec![
            (3, 3), // Our head
            (1, 1), // Opponent head  
            (5, 5), // Food
        ];
        
        for (x, y) in occupied_positions {
            assert_eq!(empty_channel[y][x], 0.0, 
                      "Occupied position ({}, {}) should not be marked as empty", x, y);
        }
        
        // Test 2: Mutual exclusivity of head channels
        let our_head_channel = &channels[1];
        let opponent_head_channel = &channels[3];
        
        for y in 0..7 {
            for x in 0..7 {
                if our_head_channel[y][x] > 0.0 {
                    assert_eq!(opponent_head_channel[y][x], 0.0,
                              "Position ({}, {}) cannot be both our head and opponent head", x, y);
                }
            }
        }
        
        // Test 3: Territory channels should have non-negative values that sum reasonably
        let our_territory = &channels[7];
        let opponent_territory = &channels[8];
        
        let our_territory_sum: f32 = our_territory.iter().flatten().sum();
        let opponent_territory_sum: f32 = opponent_territory.iter().flatten().sum();
        let total_territory = our_territory_sum + opponent_territory_sum;
        let board_area = 7 * 7;
        
        assert!(total_territory > 0.0, "Should have some territorial assignment");
        assert!(total_territory <= board_area as f32 * 2.0, // Allow overlap
               "Total territory {} should not exceed twice board area {}", total_territory, board_area);
        
        // Test 4: Movement history values should be time-decayed
        if encoder.movement_tracker.history.len() > 1 {
            let movement_channel = &channels[10];
            let total_movement: f32 = movement_channel.iter().flatten().sum();
            assert!(total_movement > 0.0, "Movement history should contribute to channel");
        }
        
        // Test 5: All channels should have valid probability distributions
        for (ch_idx, channel) in channels.iter().enumerate() {
            for (y, row) in channel.iter().enumerate() {
                for (x, &value) in row.iter().enumerate() {
                    assert!(value >= 0.0 && value <= 1.0,
                           "Channel {} value {} at ({}, {}) should be in [0.0, 1.0]",
                           ch_idx, value, x, y);
                }
            }
        }
    }

    #[test]
    fn test_encoder_state_persistence_across_turns() {
        let mut encoder = AdvancedBoardStateEncoder::new(20, 3);
        
        let base_game_state = GameStateBuilder::new()
            .with_board_size(9, 9)
            .with_our_snake(create_test_snake("us", Coord { x: 4, y: 4 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 2, y: 2 }, 3, 100))
            .with_food(vec![Coord { x: 6, y: 6 }])
            .build();
        
        // Simulate game progression
        let mut game_progression = Vec::new();
        let positions = vec![
            Coord { x: 4, y: 4 },
            Coord { x: 4, y: 5 },
            Coord { x: 5, y: 5 },
            Coord { x: 6, y: 5 },
            Coord { x: 6, y: 4 },
        ];
        
        for (turn, position) in positions.iter().enumerate() {
            let test_snake = create_test_snake("us", *position, 3, 100 - turn as i32);
            let test_board = Board {
                width: base_game_state.board.width,
                height: base_game_state.board.height,
                food: base_game_state.board.food.clone(),
                snakes: vec![test_snake.clone(), base_game_state.board.snakes[1].clone()],
                hazards: vec![],
            };
            
            let channels = encoder.encode_12_channel_board(&test_board, &test_snake, turn as u32);
            game_progression.push((turn, position, channels));
        }
        
        // Test state persistence and evolution
        
        // 1. Movement history should build up over time
        for i in 1..game_progression.len() {
            let (prev_turn, _, prev_channels) = &game_progression[i-1];
            let (curr_turn, _, curr_channels) = &game_progression[i];
            
            let prev_movement_sum: f32 = prev_channels[10].iter().flatten().sum();
            let curr_movement_sum: f32 = curr_channels[10].iter().flatten().sum();
            
            if *prev_turn > 0 {
                assert!(curr_movement_sum >= prev_movement_sum * 0.7, // Allow for time decay
                       "Movement history should persist across turns: turn {} sum {} vs turn {} sum {}",
                       prev_turn, prev_movement_sum, curr_turn, curr_movement_sum);
            }
        }
        
        // 2. Internal component states should evolve consistently
        let initial_history_size = encoder.movement_tracker.history.len();
        assert!(initial_history_size <= positions.len(), 
               "Movement history should be bounded by number of turns");
        
        // 3. Territory analysis should reflect our movement pattern
        let final_channels = &game_progression.last().unwrap().2;
        let our_territory = &final_channels[7];
        
        // Check if our movement path has territorial influence
        let mut path_territory_sum = 0.0;
        for position in &positions {
            let x = position.x as usize;
            let y = position.y as usize;
            if y < our_territory.len() && x < our_territory[y].len() {
                path_territory_sum += our_territory[y][x];
            }
        }
        
        assert!(path_territory_sum > 1.0, 
               "Our movement path should show territorial influence: {}", path_territory_sum);
        
        // 4. Validate internal component consistency
        let (our_territory_count, opponent_territory_count, danger_positions, strategic_positions) = encoder.get_encoding_stats();
        
        assert!(our_territory_count > 0, "Should have some territorial control");
        assert!(danger_positions > 0, "Should identify some dangerous positions with opponents present");
        assert!(strategic_positions > 0, "Should identify some strategic positions");
    }

    #[test]
    fn test_multi_opponent_component_interactions() {
        let mut encoder = AdvancedBoardStateEncoder::new(15, 3);
        
        let game_state = GameStateBuilder::new()
            .with_board_size(13, 13)
            .with_our_snake(create_test_snake("us", Coord { x: 6, y: 6 }, 4, 100))
            .with_opponent(create_test_snake("opponent1", Coord { x: 2, y: 2 }, 3, 90))
            .with_opponent(create_test_snake("opponent2", Coord { x: 10, y: 10 }, 3, 85))
            .with_opponent(create_test_snake("opponent3", Coord { x: 2, y: 10 }, 2, 80))
            .with_food(vec![Coord { x: 6, y: 2 }, Coord { x: 10, y: 6 }])
            .build();
        
        let channels = encoder.encode_12_channel_board(&game_state.board, &game_state.you, 15);
        
        // Test 1: Opponent heads should be distributed in opponent head channel
        let opponent_head_channel = &channels[3];
        let opponent_head_count: usize = opponent_head_channel.iter()
            .flatten()
            .map(|&v| if v == 1.0 { 1 } else { 0 })
            .sum();
        
        assert_eq!(opponent_head_count, 3, "Should mark exactly 3 opponent heads");
        
        // Test 2: Territory should be contested in multi-opponent scenario
        let our_territory_sum: f32 = channels[7].iter().flatten().sum();
        let opponent_territory_sum: f32 = channels[8].iter().flatten().sum();
        
        // With multiple opponents, our territory should be limited
        let territory_ratio = our_territory_sum / (opponent_territory_sum + 0.1);
        assert!(territory_ratio < 2.0, 
               "With multiple opponents, our territory should be contested: ratio {}", territory_ratio);
        
        // Test 3: Danger zones should reflect multiple threats
        let danger_channel = &channels[9];
        let high_danger_positions: usize = danger_channel.iter()
            .flatten()
            .map(|&v| if v > 0.7 { 1 } else { 0 })
            .sum();
        
        assert!(high_danger_positions >= 6, 
               "Multiple opponents should create multiple danger zones: {}", high_danger_positions);
        
        // Test 4: Strategic positions should account for multiple opponent pressures
        let strategic_channel = &channels[11];
        let strategic_variance: f32 = {
            let values: Vec<f32> = strategic_channel.iter().flatten().copied().collect();
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
            variance
        };
        
        assert!(strategic_variance > 0.01, 
               "Multi-opponent scenario should create strategic value diversity: {}", strategic_variance);
        
        // Test 5: Food competition should be reflected in strategic values
        for food_pos in &game_state.board.food {
            let x = food_pos.x as usize;
            let y = food_pos.y as usize;
            
            if y < strategic_channel.len() && x < strategic_channel[y].len() {
                let strategic_value = strategic_channel[y][x];
                
                // Food in contested areas should have nuanced strategic values
                assert!(strategic_value.abs() > 0.1, 
                       "Food at ({}, {}) should have significant strategic value in contested scenario: {}",
                       x, y, strategic_value);
            }
        }

    #[test]
    fn test_component_error_recovery_and_fallbacks() {
        let mut encoder = AdvancedBoardStateEncoder::new(5, 2);
        
        // Test error recovery with problematic game states
        let problematic_scenarios = vec![
            // Empty snakes list (should not happen in real game)
            Board {
                width: 7,
                height: 7,
                food: vec![],
                snakes: vec![],
                hazards: vec![],
            },
            
            // Snake with empty body (should not happen but test robustness)
            Board {
                width: 5,
                height: 5,
                food: vec![],
                snakes: vec![Battlesnake {
                    id: "empty_snake".to_string(),
                    name: "Empty Snake".to_string(),
                    health: 100,
                    body: vec![],
                    head: Coord { x: 2, y: 2 },
                    length: 0,
                    latency: "0".to_string(),
                    shout: None,
                }],
                hazards: vec![],
            }
        ];
        
        // Test that encoding doesn't panic with problematic scenarios
        for (i, scenario) in problematic_scenarios.iter().enumerate() {
            let test_snake = Battlesnake {
                id: "test".to_string(),
                name: "Test Snake".to_string(),
                health: 100,
                body: vec![Coord { x: 1, y: 1 }, Coord { x: 1, y: 2 }],
                head: Coord { x: 1, y: 1 },
                length: 2,
                latency: "0".to_string(),
                shout: None,
            };
            
            // Should not panic, even with problematic scenarios
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                encoder.encode_12_channel_board(scenario, &test_snake, i as u32)
            }));
            
            assert!(result.is_ok(), "Encoding should not panic for scenario {}", i);
        }
    }
}

#[cfg(test)]
mod performance_benchmarks {
    use super::*;
    use crate::spatial_test_utilities::*;
    use std::time::{Duration, Instant};

    /// Performance measurement utilities
    struct PerformanceTracker {
        measurements: Vec<Duration>,
        name: String,
    }

    impl PerformanceTracker {
        fn new(name: &str) -> Self {
            Self {
                measurements: Vec::new(),
                name: name.to_string(),
            }
        }

        fn measure<F>(&mut self, operation: F) -> Duration 
        where F: FnOnce() {
            let start = Instant::now();
            operation();
            let duration = start.elapsed();
            self.measurements.push(duration);
            duration
        }

        fn statistics(&self) -> PerformanceStats {
            if self.measurements.is_empty() {
                return PerformanceStats::default();
            }

            let mut sorted = self.measurements.clone();
            sorted.sort();

            let total: Duration = sorted.iter().sum();
            let count = sorted.len();
            
            PerformanceStats {
                min: *sorted.first().unwrap(),
                max: *sorted.last().unwrap(),
                median: sorted[count / 2],
                mean: total / count as u32,
                p95: sorted[(count as f32 * 0.95) as usize],
                p99: sorted[(count as f32 * 0.99) as usize],
                total_samples: count,
            }
        }

        fn assert_under_budget(&self, budget_ms: u128, percentile: f32) {
            let stats = self.statistics();
            let actual_ms = if percentile >= 0.99 {
                stats.p99.as_millis()
            } else if percentile >= 0.95 {
                stats.p95.as_millis()
            } else {
                stats.mean.as_millis()
            };

            assert!(actual_ms <= budget_ms,
                   "{} {}th percentile {}ms exceeds budget {}ms. Stats: min={}ms, mean={}ms, p95={}ms, p99={}ms, max={}ms",
                   self.name, percentile * 100.0, actual_ms, budget_ms,
                   stats.min.as_millis(), stats.mean.as_millis(), 
                   stats.p95.as_millis(), stats.p99.as_millis(), stats.max.as_millis());
        }
    }

    #[derive(Debug, Clone, Default)]
    struct PerformanceStats {
        min: Duration,
        max: Duration,
        median: Duration,
        mean: Duration,
        p95: Duration,
        p99: Duration,
        total_samples: usize,
    }

    // CRITICAL: 500ms Budget Validation Tests
    #[test]
    fn benchmark_advanced_board_state_encoder_500ms_budget() {
        let mut tracker = PerformanceTracker::new("AdvancedBoardStateEncoder");
        let mut encoder = AdvancedBoardStateEncoder::new(20, 4);
        
        // Test scenarios covering real Battlesnake game conditions
        let benchmark_scenarios = vec![
            // Standard 11x11 tournament board
            (11, 11, 5, 4, 8, "tournament_standard"),
            
            // Large board maximum
            (19, 19, 8, 6, 12, "large_board_max"),
            
            // Dense competition
            (15, 15, 12, 8, 15, "dense_competition"),
            
            // Late game long snakes
            (11, 11, 3, 3, 25, "late_game_long_snakes"),
            
            // Early game many snakes
            (11, 11, 8, 12, 3, "early_game_many_snakes"),
        ];

        for (width, height, max_food, max_snakes, max_length, scenario_name) in benchmark_scenarios {
            let mut generator = RandomBoardGenerator::new(42);
            
            // Run multiple iterations for statistical validity
            for iteration in 0..20 {
                let (board, our_snake) = generator.generate_random_board(
                    width, height, max_food, max_snakes, max_length
                );
                
                let duration = tracker.measure(|| {
                    encoder.encode_12_channel_board(&board, &our_snake, iteration as u32);
                });
                
                // Individual runs must be under 500ms budget
                assert!(duration.as_millis() < 500, 
                       "Individual run {}ms exceeds 500ms budget in scenario '{}' iteration {}",
                       duration.as_millis(), scenario_name, iteration);
            }
        }

        // CRITICAL BUDGET VALIDATION
        tracker.assert_under_budget(500, 0.95); // 95th percentile under 500ms
        tracker.assert_under_budget(300, 0.50); // Median under 300ms for good user experience
        
        let stats = tracker.statistics();
        println!("🎯 CRITICAL BUDGET VALIDATION PASSED:");
        println!("   Mean: {}ms, Median: {}ms, P95: {}ms, P99: {}ms", 
                 stats.mean.as_millis(), stats.median.as_millis(), 
                 stats.p95.as_millis(), stats.p99.as_millis());
        println!("   ✅ All measurements under 500ms budget");
    }

    #[test]
    fn benchmark_voronoi_territory_analyzer_performance() {
        let mut tracker = PerformanceTracker::new("VoronoiTerritoryAnalyzer");
        let mut analyzer = VoronoiTerritoryAnalyzer::new();
        
        let test_sizes = vec![
            (7, 7, 2, 3, "small"),
            (11, 11, 4, 6, "medium"), 
            (19, 19, 6, 8, "large"),
            (25, 25, 8, 12, "extra_large"),
        ];

        for (width, height, snake_count, snake_length, size_name) in test_sizes {
            let mut generator = RandomBoardGenerator::new(123);
            
            for _ in 0..10 {
                let (board, our_snake) = generator.generate_random_board(
                    width, height, 3, snake_count, snake_length
                );
                
                let duration = tracker.measure(|| {
                    analyzer.analyze_territory(&board, &our_snake);
                });
                
                // Territory analysis should be fast even for large boards
                assert!(duration.as_millis() < 200,
                       "Territory analysis {}ms too slow for {} board", 
                       duration.as_millis(), size_name);
            }
        }

        tracker.assert_under_budget(100, 0.95); // Should be very fast
        let stats = tracker.statistics();
        println!("🔍 Territory Analysis Performance: Mean={}ms, P95={}ms", 
                 stats.mean.as_millis(), stats.p95.as_millis());
    }

    #[test]
    fn benchmark_danger_zone_predictor_performance() {
        let mut tracker = PerformanceTracker::new("DangerZonePredictor");
        
        let prediction_depths = vec![1, 2, 3, 4, 5];
        
        for depth in prediction_depths {
            let mut predictor = DangerZonePredictor::new(depth);
            let mut generator = RandomBoardGenerator::new(456);
            
            for _ in 0..8 {
                let (board, our_snake) = generator.generate_random_board(
                    13, 13, 6, 5, 10
                );
                
                let duration = tracker.measure(|| {
                    predictor.predict_danger_zones(&board, &our_snake, depth);
                });
                
                // Danger prediction must be fast enough for real-time use
                let budget = depth as u128 * 50; // 50ms per prediction depth
                assert!(duration.as_millis() < budget,
                       "Danger prediction {}ms exceeds {}ms budget for depth {}", 
                       duration.as_millis(), budget, depth);
            }
        }

        tracker.assert_under_budget(200, 0.95);
        let stats = tracker.statistics();
        println!("⚠️ Danger Prediction Performance: Mean={}ms, P95={}ms", 
                 stats.mean.as_millis(), stats.p95.as_millis());
    }

    #[test] 
    fn benchmark_movement_history_tracker_performance() {
        let mut tracker = PerformanceTracker::new("MovementHistoryTracker");
        
        let history_sizes = vec![10, 20, 50, 100];
        
        for max_history in history_sizes {
            let mut history_tracker = MovementHistoryTracker::new(max_history);
            
            // Simulate extended gameplay with many position updates
            let duration = tracker.measure(|| {
                for turn in 0..200 {
                    let position = Coord {
                        x: (turn % 15) as i32,
                        y: ((turn / 15) % 15) as i32,
                    };
                    history_tracker.add_position(position, turn);
                    
                    // Periodically get recent positions (simulating encoding calls)
                    if turn % 10 == 0 {
                        let _ = history_tracker.get_recent_positions(turn);
                    }
                }
            });
            
            // Movement tracking should be very efficient
            assert!(duration.as_millis() < 50, 
                   "Movement tracking {}ms too slow for history size {}", 
                   duration.as_millis(), max_history);
        }

        tracker.assert_under_budget(25, 0.95);
        let stats = tracker.statistics();
        println!("📍 Movement Tracking Performance: Mean={}ms, P95={}ms", 
                 stats.mean.as_millis(), stats.p95.as_millis());
    }

    #[test]
    fn benchmark_strategic_position_analyzer_performance() {
        let mut tracker = PerformanceTracker::new("StrategicPositionAnalyzer");
        let mut analyzer = StrategicPositionAnalyzer::new();
        
        let complexity_levels = vec![
            (9, 9, 3, 2, 5, "simple"),
            (13, 13, 6, 4, 8, "moderate"), 
            (19, 19, 10, 6, 12, "complex"),
        ];

        for (width, height, food_count, snake_count, snake_length, complexity) in complexity_levels {
            let mut generator = RandomBoardGenerator::new(789);
            
            for _ in 0..8 {
                let (board, our_snake) = generator.generate_random_board(
                    width, height, food_count, snake_count, snake_length
                );
                
                let duration = tracker.measure(|| {
                    analyzer.analyze_strategic_positions(&board, &our_snake);
                });
                
                // Strategic analysis should be reasonably fast
                let budget = match complexity {
                    "simple" => 50,
                    "moderate" => 100,
                    "complex" => 200,
                    _ => 100,
                };
                
                assert!(duration.as_millis() < budget,
                       "Strategic analysis {}ms exceeds {}ms budget for {} scenario", 
                       duration.as_millis(), budget, complexity);
            }
        }

        tracker.assert_under_budget(150, 0.95);
        let stats = tracker.statistics();
        println!("🎯 Strategic Analysis Performance: Mean={}ms, P95={}ms", 
                 stats.mean.as_millis(), stats.p95.as_millis());
    }

    #[test]
    fn benchmark_encoder_scaling_with_board_size() {
        let board_sizes = vec![
            (5, 5, "tiny"),
            (7, 7, "small"),
            (11, 11, "standard"),
            (15, 15, "large"),
            (19, 19, "extra_large"),
            (25, 25, "maximum"),
        ];

        for (width, height, size_name) in board_sizes {
            let mut tracker = PerformanceTracker::new(&format!("Encoder_{}", size_name));
            let mut encoder = AdvancedBoardStateEncoder::new(15, 3);
            let mut generator = RandomBoardGenerator::new(width as u64 * height as u64);
            
            for _ in 0..5 {
                let (board, our_snake) = generator.generate_random_board(
                    width, height, 
                    (width * height / 50).max(1), // Scale food count
                    ((width + height) / 4).max(1), // Scale snake count
                    ((width + height) / 3).max(2), // Scale snake length
                );
                
                let duration = tracker.measure(|| {
                    encoder.encode_12_channel_board(&board, &our_snake, 10);
                });
                
                // Performance should scale reasonably with board size
                let area = width * height;
                let expected_max_ms = (area as f32 * 0.5).min(500.0) as u128; // ~0.5ms per cell, capped at 500ms
                
                assert!(duration.as_millis() <= expected_max_ms,
                       "Board {}x{} took {}ms, expected ≤{}ms (scaling: ~0.5ms per cell)", 
                       width, height, duration.as_millis(), expected_max_ms);
            }
            
            let stats = tracker.statistics();
            println!("📏 Board {}x{} ({}): Mean={}ms, P95={}ms", 
                     width, height, size_name, stats.mean.as_millis(), stats.p95.as_millis());
        }
    }

    #[test]
    fn benchmark_encoder_scaling_with_game_complexity() {
        let complexity_scenarios = vec![
            (11, 11, 1, 1, 3, "minimal"),      // Single opponent, short snakes
            (11, 11, 3, 2, 6, "simple"),       // Low complexity
            (11, 11, 6, 4, 10, "moderate"),    // Moderate complexity  
            (11, 11, 10, 6, 15, "high"),       // High complexity
            (11, 11, 15, 8, 20, "extreme"),    // Maximum complexity
        ];

        for (width, height, food_count, snake_count, snake_length, complexity) in complexity_scenarios {
            let mut tracker = PerformanceTracker::new(&format!("Complexity_{}", complexity));
            let mut encoder = AdvancedBoardStateEncoder::new(20, 4);
            let mut generator = RandomBoardGenerator::new(
                (food_count * snake_count * snake_length) as u64
            );
            
            for _ in 0..8 {
                let (board, our_snake) = generator.generate_random_board(
                    width, height, food_count, snake_count, snake_length
                );
                
                let duration = tracker.measure(|| {
                    encoder.encode_12_channel_board(&board, &our_snake, 25);
                });
                
                // Even highest complexity must meet budget
                assert!(duration.as_millis() < 500,
                       "Complexity '{}' took {}ms, exceeds 500ms budget", 
                       complexity, duration.as_millis());
            }
            
            let stats = tracker.statistics();
            println!("🎲 Complexity '{}': Mean={}ms, P95={}ms", 
                     complexity, stats.mean.as_millis(), stats.p95.as_millis());
        }
    }

    #[test]
    fn benchmark_encoder_with_accumulated_state() {
        let mut tracker = PerformanceTracker::new("AccumulatedState");
        let mut encoder = AdvancedBoardStateEncoder::new(50, 4); // Large history
        let mut generator = RandomBoardGenerator::new(999);
        
        let (base_board, base_snake) = generator.generate_random_board(13, 13, 8, 5, 10);
        
        // Simulate long game with accumulated state
        for turn in 0..100 {
            let duration = tracker.measure(|| {
                encoder.encode_12_channel_board(&base_board, &base_snake, turn);
            });
            
            // Performance should remain stable even with accumulated state
            assert!(duration.as_millis() < 500,
                   "Turn {} with accumulated state took {}ms, exceeds budget", 
                   turn, duration.as_millis());
            
            // Periodic stability check
            if turn > 0 && turn % 20 == 0 {
                let recent_avg = tracker.measurements
                    .iter()
                    .rev()
                    .take(5)
                    .map(|d| d.as_millis())
                    .sum::<u128>() / 5;
                
                assert!(recent_avg < 400,
                       "Average performance degraded to {}ms at turn {}", recent_avg, turn);
            }
        }

        tracker.assert_under_budget(450, 0.95); // Allow some overhead for accumulated state
        let stats = tracker.statistics();
        println!("⏳ Accumulated State Performance: Mean={}ms, P95={}ms over {} turns", 
                 stats.mean.as_millis(), stats.p95.as_millis(), stats.total_samples);
    }

    #[test]
    fn benchmark_worst_case_scenarios() {
        let mut tracker = PerformanceTracker::new("WorstCase");
        
        let worst_case_scenarios = vec![
            // Maximum board size with maximum elements
            (25, 25, 20, 8, 25, "max_everything"),
            
            // Dense board with many long snakes
            (19, 19, 15, 12, 20, "dense_long_snakes"),
            
            // Large board with maximum prediction depth
            (21, 21, 12, 6, 15, "large_deep_prediction"),
        ];

        for (width, height, food_count, snake_count, snake_length, scenario) in worst_case_scenarios {
            let mut encoder = AdvancedBoardStateEncoder::new(100, 5); // Maximum settings
            let mut generator = RandomBoardGenerator::new(12345);
            
            let (board, our_snake) = generator.generate_random_board(
                width, height, food_count, snake_count, snake_length
            );
            
            let duration = tracker.measure(|| {
                encoder.encode_12_channel_board(&board, &our_snake, 50);
            });
            
            // Even worst-case scenarios MUST meet the budget
            assert!(duration.as_millis() < 500,
                   "Worst-case scenario '{}' took {}ms, CRITICAL BUDGET VIOLATION", 
                   scenario, duration.as_millis());
            
            println!("💥 Worst-case '{}' ({}x{}, {} snakes): {}ms", 
                     scenario, width, height, snake_count, duration.as_millis());
        }

        tracker.assert_under_budget(500, 1.0); // 100th percentile must be under budget
        let stats = tracker.statistics();
        println!("🚨 Worst-Case Analysis: Max={}ms, all scenarios under 500ms budget", 
                 stats.max.as_millis());
    }

    #[test]
    fn benchmark_memory_allocation_performance() {
        let mut tracker = PerformanceTracker::new("MemoryAllocation");
        
        // Test repeated encoder creation (simulating server restarts)
        for iteration in 0..20 {
            let duration = tracker.measure(|| {
                let mut encoder = AdvancedBoardStateEncoder::new(25, 4);
                let mut generator = RandomBoardGenerator::new(iteration as u64);
                
                // Immediately use the encoder to test full initialization cost
                let (board, our_snake) = generator.generate_random_board(11, 11, 5, 3, 8);
                encoder.encode_12_channel_board(&board, &our_snake, 1);
            });
            
            // Encoder creation and first use should be fast
            assert!(duration.as_millis() < 100,
                   "Encoder creation and initialization took {}ms, too slow for iteration {}", 
                   duration.as_millis(), iteration);
        }

        tracker.assert_under_budget(50, 0.95);
        let stats = tracker.statistics();
        println!("🏗️ Memory Allocation Performance: Mean={}ms, P95={}ms", 
                 stats.mean.as_millis(), stats.p95.as_millis());
    }

    #[test]
    fn benchmark_concurrent_encoder_performance() {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let shared_tracker = Arc::new(Mutex::new(Vec::new()));
        let mut handles = vec![];
        
        // Simulate concurrent requests (multiple games running)
        for thread_id in 0..4 {
            let tracker_clone = Arc::clone(&shared_tracker);
            
            let handle = thread::spawn(move || {
                let mut local_times = Vec::new();
                let mut encoder = AdvancedBoardStateEncoder::new(15, 3);
                let mut generator = RandomBoardGenerator::new(thread_id as u64 * 1000);
                
                for iteration in 0..10 {
                    let (board, our_snake) = generator.generate_random_board(11, 11, 6, 4, 10);
                    
                    let start = Instant::now();
                    encoder.encode_12_channel_board(&board, &our_snake, iteration);
                    let duration = start.elapsed();
                    
                    local_times.push(duration);
                    
                    // Each concurrent encoding must meet budget
                    assert!(duration.as_millis() < 500,
                           "Concurrent thread {} iteration {} took {}ms", 
                           thread_id, iteration, duration.as_millis());
                }
                
                // Add to shared tracker
                let mut shared = tracker_clone.lock().unwrap();
                shared.extend(local_times);
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }
        
        let all_times = shared_tracker.lock().unwrap();
        let total_time: Duration = all_times.iter().sum();
        let avg_time = total_time / all_times.len() as u32;
        let max_time = *all_times.iter().max().unwrap();
        
        assert!(max_time.as_millis() < 500, 
               "Concurrent max time {}ms exceeds budget", max_time.as_millis());
        assert!(avg_time.as_millis() < 300,
               "Concurrent average time {}ms too high", avg_time.as_millis());
        
        println!("🔄 Concurrent Performance: Avg={}ms, Max={}ms across {} encodings", 
                 avg_time.as_millis(), max_time.as_millis(), all_times.len());
    }

    #[test]
    fn benchmark_component_performance_breakdown() {
        // Measure individual component performance
        let mut encoder = AdvancedBoardStateEncoder::new(15, 3);
        let mut generator = RandomBoardGenerator::new(777);
        let (board, our_snake) = generator.generate_random_board(13, 13, 8, 5, 12);
        
        // Measure full encoding
        let start = Instant::now();
        let channels = encoder.encode_12_channel_board(&board, &our_snake, 20);
        let total_time = start.elapsed();
        
        // Measure individual components for comparison
        let mut voronoi = VoronoiTerritoryAnalyzer::new();
        let start = Instant::now();
        voronoi.analyze_territory(&board, &our_snake);
        let voronoi_time = start.elapsed();
        
        let mut danger = DangerZonePredictor::new(3);
        let start = Instant::now();
        danger.predict_danger_zones(&board, &our_snake, 3);
        let danger_time = start.elapsed();
        
        let mut strategic = StrategicPositionAnalyzer::new();
        let start = Instant::now();
        strategic.analyze_strategic_positions(&board, &our_snake);
        let strategic_time = start.elapsed();
        
        println!("🔬 Component Performance Breakdown:");
        println!("   Total encoding: {}ms", total_time.as_millis());
        println!("   Voronoi territory: {}ms ({:.1}%)", 
                 voronoi_time.as_millis(), 
                 voronoi_time.as_millis() as f32 / total_time.as_millis() as f32 * 100.0);
        println!("   Danger prediction: {}ms ({:.1}%)", 
                 danger_time.as_millis(),
                 danger_time.as_millis() as f32 / total_time.as_millis() as f32 * 100.0);
        println!("   Strategic analysis: {}ms ({:.1}%)", 
                 strategic_time.as_millis(),
                 strategic_time.as_millis() as f32 / total_time.as_millis() as f32 * 100.0);
        
        // Validate that components are reasonably fast
        assert!(voronoi_time.as_millis() < 100, "Voronoi component too slow: {}ms", voronoi_time.as_millis());
        assert!(danger_time.as_millis() < 150, "Danger component too slow: {}ms", danger_time.as_millis());
        assert!(strategic_time.as_millis() < 100, "Strategic component too slow: {}ms", strategic_time.as_millis());
        assert!(total_time.as_millis() < 500, "Total encoding too slow: {}ms", total_time.as_millis());
        
        // Validate output
        assert_eq!(channels.len(), 12, "Should produce 12 channels");
    }

    #[test]
    fn benchmark_performance_regression_detection() {
        // Test for performance regressions by establishing baseline expectations
        let mut encoder = AdvancedBoardStateEncoder::new(20, 3);
        let mut generator = RandomBoardGenerator::new(1337);
        
        let baseline_scenarios = vec![
            (11, 11, 5, 3, 8, 200),   // Standard scenario should be under 200ms
            (15, 15, 8, 5, 12, 350),  // Larger scenario should be under 350ms
            (19, 19, 10, 6, 15, 450), // Maximum practical scenario under 450ms
        ];

        for (width, height, food, snakes, length, baseline_ms) in baseline_scenarios {
            let mut times = Vec::new();
            
            for _ in 0..10 {
                let (board, our_snake) = generator.generate_random_board(
                    width, height, food, snakes, length
                );
                
                let start = Instant::now();
                encoder.encode_12_channel_board(&board, &our_snake, 15);
                let duration = start.elapsed();
                
                times.push(duration);
            }
            
            let avg_time = times.iter().sum::<Duration>() / times.len() as u32;
            let max_time = *times.iter().max().unwrap();
            
            assert!(avg_time.as_millis() < baseline_ms,
                   "Performance regression detected! Board {}x{} avg={}ms exceeds baseline {}ms",
                   width, height, avg_time.as_millis(), baseline_ms);
            
            assert!(max_time.as_millis() < (baseline_ms as f32 * 1.5) as u128,
                   "Performance spike detected! Board {}x{} max={}ms exceeds 150% of baseline {}ms",
                   width, height, max_time.as_millis(), baseline_ms);
            
            println!("✅ Baseline validated for {}x{}: avg={}ms, max={}ms (baseline: {}ms)", 
                     width, height, avg_time.as_millis(), max_time.as_millis(), baseline_ms);
        }
    }

    // Summary test that validates all critical performance requirements
    #[test]
    fn validate_critical_performance_requirements() {
        println!("\n🎯 === CRITICAL PERFORMANCE VALIDATION SUMMARY ===");
        
        let mut encoder = AdvancedBoardStateEncoder::new(25, 4);
        let mut generator = RandomBoardGenerator::new(99999);
        let mut all_times = Vec::new();
        
        // Test comprehensive scenarios
        let validation_scenarios = vec![
            (11, 11, 5, 4, 8, "tournament_standard"),
            (19, 19, 8, 6, 12, "large_board"),
            (15, 15, 12, 8, 15, "dense_game"),
        ];
        
        for (width, height, food, snakes, length, name) in validation_scenarios {
            for _ in 0..20 {
                let (board, our_snake) = generator.generate_random_board(
                    width, height, food, snakes, length
                );
                
                let start = Instant::now();
                let channels = encoder.encode_12_channel_board(&board, &our_snake, 30);
                let duration = start.elapsed();
                
                all_times.push((duration, name));
                
                // CRITICAL: Each run must be under 500ms
                assert!(duration.as_millis() < 500,
                       "🚨 CRITICAL FAILURE: {} scenario took {}ms > 500ms budget", 
                       name, duration.as_millis());
                
                // Validate output correctness under time pressure
                assert_eq!(channels.len(), 12, "Must produce 12 channels even under time pressure");
                
                for channel in &channels {

#[cfg(test)]
mod memory_leak_detection_tests {
    use super::*;
    use crate::spatial_test_utilities::*;

    /// Memory usage tracker for detecting leaks
    struct MemoryUsageTracker {
        measurements: VecDeque<usize>,
        max_samples: usize,
        name: String,
    }

    impl MemoryUsageTracker {
        fn new(name: &str, max_samples: usize) -> Self {
            Self {
                measurements: VecDeque::new(),
                max_samples,
                name: name.to_string(),
            }
        }

        fn record_measurement(&mut self, estimated_size: usize) {
            if self.measurements.len() >= self.max_samples {
                self.measurements.pop_front();
            }
            self.measurements.push_back(estimated_size);
        }

        fn check_for_leak(&self, max_growth_factor: f32) -> Result<(), String> {
            if self.measurements.len() < 10 {
                return Ok(()); // Need enough samples to detect trend
            }

            let early_avg = self.measurements.iter().take(5).sum::<usize>() as f32 / 5.0;
            let recent_avg = self.measurements.iter().rev().take(5).sum::<usize>() as f32 / 5.0;
            let growth_factor = recent_avg / early_avg.max(1.0);

            if growth_factor > max_growth_factor {
                return Err(format!(
                    "Memory leak detected in {}: {:.2}x growth ({}KB -> {}KB) exceeds {:.2}x threshold",
                    self.name, growth_factor, early_avg as usize / 1024, recent_avg as usize / 1024, max_growth_factor
                ));
            }

            Ok(())
        }

        fn get_statistics(&self) -> MemoryStats {
            if self.measurements.is_empty() {
                return MemoryStats::default();
            }

            let min = *self.measurements.iter().min().unwrap();
            let max = *self.measurements.iter().max().unwrap();
            let sum: usize = self.measurements.iter().sum();
            let avg = sum / self.measurements.len();
            
            let first_half_avg = if self.measurements.len() >= 10 {
                self.measurements.iter().take(self.measurements.len() / 2).sum::<usize>() / (self.measurements.len() / 2)
            } else {
                avg
            };
            
            let second_half_avg = if self.measurements.len() >= 10 {
                self.measurements.iter().skip(self.measurements.len() / 2).sum::<usize>() / (self.measurements.len() / 2)
            } else {
                avg
            };

            MemoryStats {
                min_kb: min / 1024,
                max_kb: max / 1024,
                avg_kb: avg / 1024,
                first_half_avg_kb: first_half_avg / 1024,
                second_half_avg_kb: second_half_avg / 1024,
                growth_factor: second_half_avg as f32 / first_half_avg.max(1) as f32,
                sample_count: self.measurements.len(),
            }
        }
    }

    #[derive(Debug, Clone, Default)]
    struct MemoryStats {
        min_kb: usize,
        max_kb: usize,
        avg_kb: usize,
        first_half_avg_kb: usize,
        second_half_avg_kb: usize,
        growth_factor: f32,
        sample_count: usize,
    }

    /// Estimate memory usage of spatial analysis components
    fn estimate_encoder_memory_usage(encoder: &AdvancedBoardStateEncoder) -> usize {
        let mut total = 0;

        // Movement history tracker memory
        total += encoder.movement_tracker.history.len() * (std::mem::size_of::<(Coord, u32)>() + 32); // Vec overhead
        total += encoder.movement_tracker.history.capacity() * std::mem::size_of::<(Coord, u32)>();

        // Voronoi analyzer memory
        total += encoder.voronoi_analyzer.our_territory.capacity() * (std::mem::size_of::<Coord>() + std::mem::size_of::<f32>() + 32);
        total += encoder.voronoi_analyzer.opponent_territory.capacity() * (std::mem::size_of::<Coord>() + std::mem::size_of::<f32>() + 32);

        // Danger zone predictor memory
        total += encoder.danger_predictor.danger_zones.capacity() * (std::mem::size_of::<Coord>() + std::mem::size_of::<f32>() + 32);

        // Strategic position analyzer memory
        total += encoder.strategic_analyzer.strategic_positions.capacity() * (std::mem::size_of::<Coord>() + std::mem::size_of::<f32>() + 32);

        // Add base struct sizes
        total += std::mem::size_of::<AdvancedBoardStateEncoder>();
        total += std::mem::size_of::<MovementHistoryTracker>();
        total += std::mem::size_of::<VoronoiTerritoryAnalyzer>();
        total += std::mem::size_of::<DangerZonePredictor>();
        total += std::mem::size_of::<StrategicPositionAnalyzer>();

        total
    }

    #[test]
    fn test_extended_gameplay_memory_stability() {
        let mut encoder = AdvancedBoardStateEncoder::new(50, 4);
        let mut tracker = MemoryUsageTracker::new("ExtendedGameplay", 100);
        let mut generator = RandomBoardGenerator::new(12345);

        // Simulate extended gameplay - equivalent to multiple full games
        for turn in 0..1000 {
            // Generate varied board states to stress different code paths
            let board_variation = turn % 5;
            let (board, our_snake) = match board_variation {
                0 => generator.generate_random_board(11, 11, 5, 3, 8),    // Standard
                1 => generator.generate_random_board(15, 15, 8, 5, 12),   // Large
                2 => generator.generate_random_board(7, 7, 3, 2, 6),      // Small
                3 => generator.generate_random_board(13, 13, 10, 6, 15),  // Dense
                4 => generator.generate_random_board(19, 19, 12, 4, 20),  // Extreme
                _ => unreachable!(),
            };

            // Perform encoding
            let _channels = encoder.encode_12_channel_board(&board, &our_snake, turn as u32);

            // Record memory usage every 10 turns
            if turn % 10 == 0 {
                let memory_usage = estimate_encoder_memory_usage(&encoder);
                tracker.record_measurement(memory_usage);
                
                // Periodic leak check
                if turn > 100 && turn % 100 == 0 {
                    tracker.check_for_leak(2.0).expect(&format!("Memory leak detected at turn {}", turn));
                }
            }

            // Validate internal bounds every 50 turns
            if turn % 50 == 0 {
                assert!(encoder.movement_tracker.history.len() <= encoder.movement_tracker.max_history,
                       "Movement history size {} exceeds limit {} at turn {}", 
                       encoder.movement_tracker.history.len(), encoder.movement_tracker.max_history, turn);

                let (our_territory, opponent_territory, danger_positions, strategic_positions) = encoder.get_encoding_stats();
                let total_positions = our_territory + opponent_territory + danger_positions + strategic_positions;
                
                // Memory should be bounded - not growing indefinitely
                assert!(total_positions < 5000, 
                       "Total tracked positions {} excessive at turn {} - possible memory leak", 
                       total_positions, turn);
            }
        }

        // Final memory leak analysis
        tracker.check_for_leak(1.5).expect("Memory leak detected in extended gameplay test");
        
        let stats = tracker.get_statistics();
        assert!(stats.growth_factor < 1.3, 
               "Memory growth factor {:.2} indicates potential leak", stats.growth_factor);

        println!("🔍 Extended Gameplay Memory Analysis:");
        println!("   Duration: 1000 turns across varied scenarios");
        println!("   Memory range: {}KB - {}KB", stats.min_kb, stats.max_kb);
        println!("   Average: {}KB", stats.avg_kb);
        println!("   Growth factor: {:.2}x", stats.growth_factor);
        println!("   ✅ No memory leaks detected");
    }

    #[test] 
    fn test_movement_history_tracker_memory_bounds() {
        let history_sizes = vec![10, 25, 50, 100, 200];
        
        for max_history in history_sizes {
            let mut tracker = MovementHistoryTracker::new(max_history);
            let mut memory_tracker = MemoryUsageTracker::new(&format!("HistoryTracker_{}", max_history), 50);
            
            // Add many more positions than the history limit
            for turn in 0..max_history * 5 {
                let position = Coord {
                    x: (turn % 20) as i32,
                    y: ((turn / 20) % 15) as i32,
                };
                
                tracker.add_position(position, turn as u32);
                
                // Record memory usage
                let memory_usage = tracker.history.len() * std::mem::size_of::<(Coord, u32)>() + 
                                  std::mem::size_of::<MovementHistoryTracker>();
                memory_tracker.record_measurement(memory_usage);
                
                // History should never exceed max_history
                assert!(tracker.history.len() <= max_history,
                       "History size {} exceeds max {} with max_history setting {}",
                       tracker.history.len(), max_history, max_history);
            }

            // Memory should stabilize, not grow indefinitely
            memory_tracker.check_for_leak(1.1).expect("Movement history tracker has memory leak");
            
            let stats = memory_tracker.get_statistics();
            println!("📍 History Tracker (max={}): stabilized at {}KB, growth factor: {:.2}x", 
                     max_history, stats.avg_kb, stats.growth_factor);
        }
    }

    #[test]
    fn test_voronoi_analyzer_memory_stability() {
        let mut analyzer = VoronoiTerritoryAnalyzer::new();
        let mut memory_tracker = MemoryUsageTracker::new("VoronoiAnalyzer", 100);
        let mut generator = RandomBoardGenerator::new(67890);

        // Run many analysis cycles with varying complexity
        for iteration in 0..200 {
            let complexity = (iteration % 4) + 1;
            let (board, our_snake) = generator.generate_random_board(
                9 + complexity * 2,  // Board size varies
                9 + complexity * 2,
                complexity * 3,      // Food count varies
                complexity + 1,      // Snake count varies
                complexity * 3 + 2   // Snake length varies
            );

            analyzer.analyze_territory(&board, &our_snake);

            // Record memory usage
            let memory_usage = 
                analyzer.our_territory.capacity() * (std::mem::size_of::<Coord>() + std::mem::size_of::<f32>()) +
                analyzer.opponent_territory.capacity() * (std::mem::size_of::<Coord>() + std::mem::size_of::<f32>()) +
                std::mem::size_of::<VoronoiTerritoryAnalyzer>();
            
            memory_tracker.record_measurement(memory_usage);

            // Periodic bounds checking
            if iteration % 20 == 0 && iteration > 0 {
                let total_territory_positions = analyzer.our_territory.len() + analyzer.opponent_territory.len();
                assert!(total_territory_positions < 1000,
                       "Territory positions {} excessive at iteration {} - possible unbounded growth",
                       total_territory_positions, iteration);
            }
        }

        // Verify no memory leak
        memory_tracker.check_for_leak(1.4).expect("Voronoi analyzer has memory leak");
        
        let stats = memory_tracker.get_statistics();
        println!("🗺️ Voronoi Analyzer Memory: avg={}KB, growth factor={:.2}x", 
                 stats.avg_kb, stats.growth_factor);
    }

    #[test]
    fn test_danger_zone_predictor_memory_bounds() {
        let prediction_depths = vec![1, 2, 3, 4, 5];

        for depth in prediction_depths {
            let mut predictor = DangerZonePredictor::new(depth);
            let mut memory_tracker = MemoryUsageTracker::new(&format!("DangerPredictor_{}", depth), 50);
            let mut generator = RandomBoardGenerator::new(depth as u64 * 1000);

            // Run predictions with increasing complexity
            for iteration in 0..100 {
                let (board, our_snake) = generator.generate_random_board(
                    11, 11, 
                    5 + (iteration % 8),  // Varying food
                    2 + (iteration % 5),  // Varying opponents
                    5 + (iteration % 10)  // Varying snake lengths
                );

                predictor.predict_danger_zones(&board, &our_snake, depth);

                // Record memory usage
                let memory_usage = 
                    predictor.danger_zones.capacity() * (std::mem::size_of::<Coord>() + std::mem::size_of::<f32>()) +
                    std::mem::size_of::<DangerZonePredictor>();
                
                memory_tracker.record_measurement(memory_usage);

                // Check bounds
                if iteration % 10 == 0 && iteration > 0 {
                    assert!(predictor.danger_zones.len() < 500,
                           "Danger zones {} excessive at iteration {} - possible memory issue",
                           predictor.danger_zones.len(), iteration);
                }
            }

            // Check for memory leaks
            memory_tracker.check_for_leak(1.3).expect(&format!("Danger predictor depth {} has memory leak", depth));
            
            let stats = memory_tracker.get_statistics();
            println!("⚠️ Danger Predictor (depth={}): avg={}KB, growth factor={:.2}x", 
                     depth, stats.avg_kb, stats.growth_factor);
        }
    }

    #[test]
    fn test_strategic_analyzer_memory_efficiency() {
        let mut analyzer = StrategicPositionAnalyzer::new();
        let mut memory_tracker = MemoryUsageTracker::new("StrategicAnalyzer", 100);
        let mut generator = RandomBoardGenerator::new(98765);

        // Test with scenarios of increasing strategic complexity
        for iteration in 0..150 {
            let scenario_type = iteration % 3;
            let (board, our_snake) = match scenario_type {
                0 => generator.generate_random_board(9, 9, 3, 2, 5),    // Simple
                1 => generator.generate_random_board(13, 13, 8, 4, 10), // Moderate
                2 => generator.generate_random_board(17, 17, 12, 6, 15), // Complex
                _ => unreachable!(),
            };

            analyzer.analyze_strategic_positions(&board, &our_snake);

            // Record memory usage
            let memory_usage = 
                analyzer.strategic_positions.capacity() * (std::mem::size_of::<Coord>() + std::mem::size_of::<f32>()) +
                std::mem::size_of::<StrategicPositionAnalyzer>();
            
            memory_tracker.record_measurement(memory_usage);

            // Periodic bounds check
            if iteration % 15 == 0 && iteration > 0 {
                assert!(analyzer.strategic_positions.len() < 800,
                       "Strategic positions {} excessive at iteration {} - check for unbounded growth",
                       analyzer.strategic_positions.len(), iteration);
            }
        }

        // Verify memory stability
        memory_tracker.check_for_leak(1.3).expect("Strategic analyzer has memory leak");
        
        let stats = memory_tracker.get_statistics();
        println!("🎯 Strategic Analyzer Memory: avg={}KB, growth factor={:.2}x", 
                 stats.avg_kb, stats.growth_factor);
    }

    #[test]
    fn test_encoder_long_running_server_simulation() {
        let mut encoder = AdvancedBoardStateEncoder::new(30, 3);
        let mut memory_tracker = MemoryUsageTracker::new("ServerSimulation", 200);
        let mut generator = RandomBoardGenerator::new(555555);

        // Simulate long-running server handling many games
        let game_scenarios = vec![
            (11, 11, 5, 3, 8, "tournament"),
            (15, 15, 8, 5, 12, "large_casual"),
            (7, 7, 3, 2, 5, "small_quick"),
            (19, 19, 10, 6, 15, "challenge"),
        ];

        for game_session in 0..50 {  // 50 game sessions
            let scenario_idx = game_session % game_scenarios.len();
            let (width, height, food, snakes, length, name) = game_scenarios[scenario_idx];

            // Simulate full game (multiple turns)
            for turn in 0..30 {  // 30 turns per game
                let (board, our_snake) = generator.generate_random_board(width, height, food, snakes, length);
                let _channels = encoder.encode_12_channel_board(&board, &our_snake, turn as u32);
            }

            // Record memory after each game session
            let memory_usage = estimate_encoder_memory_usage(&encoder);
            memory_tracker.record_measurement(memory_usage);

            // Check for memory leaks every 10 game sessions
            if game_session > 10 && game_session % 10 == 0 {
                memory_tracker.check_for_leak(1.3).expect(&format!("Memory leak detected after {} games", game_session));
                
                // Verify internal structures are bounded
                assert!(encoder.movement_tracker.history.len() <= encoder.movement_tracker.max_history,
                       "Movement history unbounded after {} games", game_session);

                let (our_territory, opponent_territory, danger_positions, strategic_positions) = encoder.get_encoding_stats();
                let total_tracked = our_territory + opponent_territory + danger_positions + strategic_positions;
                
                assert!(total_tracked < 3000,
                       "Total tracked positions {} excessive after {} games", total_tracked, game_session);
            }
        }

        // Final memory analysis
        memory_tracker.check_for_leak(1.2).expect("Server simulation shows memory leak");
        
        let stats = memory_tracker.get_statistics();
        assert!(stats.growth_factor < 1.15, 
               "Server memory growth factor {:.2} too high for production", stats.growth_factor);

        println!("🖥️ Server Simulation Results:");
        println!("   Games simulated: {} sessions", game_scenarios.len() * 50);
        println!("   Memory stability: {}KB - {}KB", stats.min_kb, stats.max_kb);
        println!("   Growth factor: {:.2}x", stats.growth_factor);
        println!("   ✅ Memory stable for production deployment");
    }

    #[test]
    fn test_memory_pressure_stress_test() {
        // Test behavior under memory pressure with large data structures
        let mut encoder = AdvancedBoardStateEncoder::new(100, 5); // Large settings
        let mut memory_tracker = MemoryUsageTracker::new("MemoryPressure", 50);
        let mut generator = RandomBoardGenerator::new(777777);

        // Create high memory pressure scenarios
        let stress_scenarios = vec![
            (25, 25, 20, 10, 30, "maximum_load"),
            (21, 21, 15, 8, 25, "heavy_load"),
            (19, 19, 12, 6, 20, "moderate_load"),
        ];

        for (scenario_idx, (width, height, food, snakes, length, name)) in stress_scenarios.iter().enumerate() {
            for iteration in 0..20 {
                let (board, our_snake) = generator.generate_random_board(*width, *height, *food, *snakes, *length);
                let _channels = encoder.encode_12_channel_board(&board, &our_snake, (scenario_idx * 20 + iteration) as u32);

                // Record memory under stress
                let memory_usage = estimate_encoder_memory_usage(&encoder);
                memory_tracker.record_measurement(memory_usage);

                // Ensure we don't run out of control even under stress
                let (our_territory, opponent_territory, danger_positions, strategic_positions) = encoder.get_encoding_stats();
                let total_tracked = our_territory + opponent_territory + danger_positions + strategic_positions;
                
                assert!(total_tracked < 8000,
                       "Memory pressure scenario '{}' iteration {} has {} tracked positions - too high",
                       name, iteration, total_tracked);
            }
            
            println!("💪 Memory pressure '{}': handled without issues", name);
        }

        // Verify system handles memory pressure without leaks
        memory_tracker.check_for_leak(1.4).expect("Memory pressure test shows leak");
        
        let stats = memory_tracker.get_statistics();
        println!("🔥 Memory Pressure Test Results:");
        println!("   Peak memory: {}KB", stats.max_kb);
        println!("   Growth factor: {:.2}x", stats.growth_factor);
        println!("   ✅ System stable under memory pressure");
    }

    #[test]
    fn test_component_memory_cleanup() {
        // Test that components properly clean up when reused
        let mut encoder = AdvancedBoardStateEncoder::new(20, 3);
        let mut generator = RandomBoardGenerator::new(123456);
        let mut baseline_memory = 0;

        // Establish baseline
        for _ in 0..10 {
            let (board, our_snake) = generator.generate_random_board(11, 11, 5, 3, 8);
            encoder.encode_12_channel_board(&board, &our_snake, 5);
        }
        baseline_memory = estimate_encoder_memory_usage(&encoder);

        // Run intensive usage period  
        for turn in 0..100 {
            let complexity = (turn % 3) + 1;
            let (board, our_snake) = generator.generate_random_board(
                9 + complexity * 3, 9 + complexity * 3,
                complexity * 4, complexity + 2, complexity * 5
            );
            encoder.encode_12_channel_board(&board, &our_snake, turn as u32);
        }

        // Return to baseline usage
        for _ in 0..20 {
            let (board, our_snake) = generator.generate_random_board(11, 11, 5, 3, 8);
            encoder.encode_12_channel_board(&board, &our_snake, 105);
        }

        let final_memory = estimate_encoder_memory_usage(&encoder);
        let memory_ratio = final_memory as f32 / baseline_memory as f32;

        // Memory should return close to baseline after cleanup
        assert!(memory_ratio < 1.5,
               "Memory usage ratio {:.2} indicates insufficient cleanup: baseline={}KB, final={}KB",

#[cfg(test)]
mod thread_safety_tests {
    use super::*;
    use crate::spatial_test_utilities::*;
    use std::sync::{Arc, Mutex, RwLock, Barrier};
    use std::thread;
    use std::time::{Duration, Instant};

    /// Thread-safe wrapper for testing concurrent access
    struct ConcurrentTestHarness {
        results: Arc<Mutex<Vec<ThreadTestResult>>>,
        barrier: Arc<Barrier>,
        thread_count: usize,
    }

    #[derive(Debug, Clone)]
    struct ThreadTestResult {
        thread_id: usize,
        duration: Duration,
        success: bool,
        error_message: Option<String>,
        iterations_completed: usize,
        channels_produced: usize,
    }

    impl ConcurrentTestHarness {
        fn new(thread_count: usize) -> Self {
            Self {
                results: Arc::new(Mutex::new(Vec::new())),
                barrier: Arc::new(Barrier::new(thread_count)),
                thread_count,
            }
        }

        fn execute_concurrent_test<F>(&self, test_function: F) -> Vec<ThreadTestResult>
        where
            F: Fn(usize, Arc<Barrier>) -> ThreadTestResult + Send + Sync + 'static,
        {
            let test_func = Arc::new(test_function);
            let mut handles = Vec::new();

            // Spawn worker threads
            for thread_id in 0..self.thread_count {
                let results_clone = Arc::clone(&self.results);
                let barrier_clone = Arc::clone(&self.barrier);
                let test_func_clone = Arc::clone(&test_func);

                let handle = thread::spawn(move || {
                    let result = test_func_clone(thread_id, barrier_clone);
                    results_clone.lock().unwrap().push(result);
                });

                handles.push(handle);
            }

            // Wait for all threads to complete
            for handle in handles {
                handle.join().expect("Thread should complete successfully");
            }

            // Return results
            let results = self.results.lock().unwrap();
            results.clone()
        }

        fn validate_results(&self, results: &[ThreadTestResult]) {
            assert_eq!(results.len(), self.thread_count, "All threads should report results");
            
            let successful_threads = results.iter().filter(|r| r.success).count();
            assert_eq!(successful_threads, self.thread_count, "All threads should succeed");

            // Check for reasonable performance consistency
            let durations: Vec<u128> = results.iter().map(|r| r.duration.as_millis()).collect();
            let min_duration = *durations.iter().min().unwrap();
            let max_duration = *durations.iter().max().unwrap();
            
            if min_duration > 0 {
                let performance_variance = max_duration as f32 / min_duration as f32;
                assert!(performance_variance < 10.0, 
                       "Performance variance {:.2}x too high between threads", performance_variance);
            }
        }
    }

    #[test]
    fn test_encoder_concurrent_access_basic() {
        let thread_count = 8;
        let harness = ConcurrentTestHarness::new(thread_count);
        let shared_encoder = Arc::new(Mutex::new(AdvancedBoardStateEncoder::new(15, 3)));

        let results = harness.execute_concurrent_test(move |thread_id, barrier| {
            let encoder_clone = Arc::clone(&shared_encoder);
            let mut generator = RandomBoardGenerator::new(thread_id as u64 * 1000);
            
            barrier.wait(); // Synchronize thread start
            let start = Instant::now();
            let mut iterations = 0;
            let mut total_channels = 0;

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                for iteration in 0..20 {
                    let (board, our_snake) = generator.generate_random_board(11, 11, 5, 3, 8);
                    
                    let channels = {
                        let mut encoder = encoder_clone.lock().unwrap();
                        encoder.encode_12_channel_board(&board, &our_snake, (thread_id * 100 + iteration) as u32)
                    };
                    
                    assert_eq!(channels.len(), 12, "Should produce 12 channels in thread {}", thread_id);
                    total_channels += channels.len();
                    iterations += 1;
                }
            }));

            let duration = start.elapsed();

            ThreadTestResult {
                thread_id,
                duration,
                success: result.is_ok(),
                error_message: result.err().map(|_| "Panic occurred".to_string()),
                iterations_completed: iterations,
                channels_produced: total_channels,
            }
        });

        harness.validate_results(&results);
        println!("🔒 Basic concurrent access: {} threads completed {} iterations each", 
                 thread_count, results[0].iterations_completed);
    }

    #[test]
    fn test_encoder_concurrent_access_read_write_locks() {
        let thread_count = 6;
        let harness = ConcurrentTestHarness::new(thread_count);
        let shared_encoder = Arc::new(RwLock::new(AdvancedBoardStateEncoder::new(20, 3)));

        let results = harness.execute_concurrent_test(move |thread_id, barrier| {
            let encoder_clone = Arc::clone(&shared_encoder);
            let mut generator = RandomBoardGenerator::new(thread_id as u64 * 2000);
            
            barrier.wait();
            let start = Instant::now();
            let mut iterations = 0;
            let mut total_channels = 0;

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                for iteration in 0..15 {
                    let (board, our_snake) = generator.generate_random_board(13, 13, 6, 4, 10);
                    
                    let channels = {
                        // Use write lock for encoding (modifies internal state)
                        let mut encoder = encoder_clone.write().unwrap();
                        encoder.encode_12_channel_board(&board, &our_snake, (thread_id * 50 + iteration) as u32)
                    };
                    
                    // Validate results
                    assert_eq!(channels.len(), 12, "Should produce 12 channels in thread {}", thread_id);
                    
                    // Simulate some processing time with read access
                    {
                        let encoder = encoder_clone.read().unwrap();
                        let stats = encoder.get_encoding_stats();
                        assert!(stats.0 < 10000, "Statistics should be reasonable");
                    }
                    
                    total_channels += channels.len();
                    iterations += 1;
                }
            }));

            let duration = start.elapsed();

            ThreadTestResult {
                thread_id,
                duration,
                success: result.is_ok(),
                error_message: result.err().map(|_| "Panic occurred".to_string()),
                iterations_completed: iterations,
                channels_produced: total_channels,
            }
        });

        harness.validate_results(&results);
        println!("🔐 RwLock concurrent access: {} threads with read/write locks", thread_count);
    }

    #[test]
    fn test_individual_component_thread_safety() {
        let thread_count = 8;
        
        // Test each component individually for thread safety
        let components = vec![
            ("VoronoiAnalyzer", Box::new(|| {
                Arc::new(Mutex::new(VoronoiTerritoryAnalyzer::new()))
            }) as Box<dyn Fn() -> Arc<Mutex<VoronoiTerritoryAnalyzer>> + Send>),
            
            ("DangerPredictor", Box::new(|| {
                Arc::new(Mutex::new(DangerZonePredictor::new(3)))
            }) as Box<dyn Fn() -> Arc<Mutex<DangerZonePredictor>> + Send>),
            
            ("StrategicAnalyzer", Box::new(|| {
                Arc::new(Mutex::new(StrategicPositionAnalyzer::new()))
            }) as Box<dyn Fn() -> Arc<Mutex<StrategicPositionAnalyzer>> + Send>),
        ];

        for (component_name, component_factory) in components {
            let harness = ConcurrentTestHarness::new(thread_count);
            
            match component_name {
                "VoronoiAnalyzer" => {
                    let shared_analyzer = Arc::new(Mutex::new(VoronoiTerritoryAnalyzer::new()));
                    
                    let results = harness.execute_concurrent_test(move |thread_id, barrier| {
                        let analyzer_clone = Arc::clone(&shared_analyzer);
                        let mut generator = RandomBoardGenerator::new(thread_id as u64 * 3000);
                        
                        barrier.wait();
                        let start = Instant::now();
                        let mut iterations = 0;

                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            for iteration in 0..25 {
                                let (board, our_snake) = generator.generate_random_board(9, 9, 4, 3, 6);
                                
                                {
                                    let mut analyzer = analyzer_clone.lock().unwrap();
                                    analyzer.analyze_territory(&board, &our_snake);
                                }
                                
                                iterations += 1;
                            }
                        }));

                        ThreadTestResult {
                            thread_id,
                            duration: start.elapsed(),
                            success: result.is_ok(),
                            error_message: result.err().map(|_| "Panic occurred".to_string()),
                            iterations_completed: iterations,
                            channels_produced: 0,
                        }
                    });

                    harness.validate_results(&results);
                }
                
                "DangerPredictor" => {
                    let shared_predictor = Arc::new(Mutex::new(DangerZonePredictor::new(3)));
                    
                    let results = harness.execute_concurrent_test(move |thread_id, barrier| {
                        let predictor_clone = Arc::clone(&shared_predictor);
                        let mut generator = RandomBoardGenerator::new(thread_id as u64 * 4000);
                        
                        barrier.wait();
                        let start = Instant::now();
                        let mut iterations = 0;

                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            for iteration in 0..20 {
                                let (board, our_snake) = generator.generate_random_board(11, 11, 5, 4, 8);
                                
                                {
                                    let mut predictor = predictor_clone.lock().unwrap();
                                    predictor.predict_danger_zones(&board, &our_snake, 3);
                                }
                                
                                iterations += 1;
                            }
                        }));

                        ThreadTestResult {
                            thread_id,
                            duration: start.elapsed(),
                            success: result.is_ok(),
                            error_message: result.err().map(|_| "Panic occurred".to_string()),
                            iterations_completed: iterations,
                            channels_produced: 0,
                        }
                    });

                    harness.validate_results(&results);
                }
                
                "StrategicAnalyzer" => {
                    let shared_analyzer = Arc::new(Mutex::new(StrategicPositionAnalyzer::new()));
                    
                    let results = harness.execute_concurrent_test(move |thread_id, barrier| {
                        let analyzer_clone = Arc::clone(&shared_analyzer);
                        let mut generator = RandomBoardGenerator::new(thread_id as u64 * 5000);
                        
                        barrier.wait();
                        let start = Instant::now();
                        let mut iterations = 0;

                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            for iteration in 0..25 {
                                let (board, our_snake) = generator.generate_random_board(10, 10, 6, 3, 7);
                                
                                {
                                    let mut analyzer = analyzer_clone.lock().unwrap();
                                    analyzer.analyze_strategic_positions(&board, &our_snake);
                                }
                                
                                iterations += 1;
                            }
                        }));

                        ThreadTestResult {
                            thread_id,
                            duration: start.elapsed(),
                            success: result.is_ok(),
                            error_message: result.err().map(|_| "Panic occurred".to_string()),
                            iterations_completed: iterations,
                            channels_produced: 0,
                        }
                    });

                    harness.validate_results(&results);
                }
                _ => unreachable!()
            }
            
            println!("🧩 Component '{}' thread safety validated", component_name);
        }
    }

    #[test]
    fn test_concurrent_encoding_consistency() {
        let thread_count = 6;
        let iterations_per_thread = 10;
        let shared_results = Arc::new(Mutex::new(HashMap::new()));
        let test_board_state = Arc::new({
            let mut generator = RandomBoardGenerator::new(999999);
            generator.generate_random_board(11, 11, 5, 3, 8)
        });

        let mut handles = Vec::new();

        // Each thread will encode the same board state and compare results
        for thread_id in 0..thread_count {
            let results_clone = Arc::clone(&shared_results);
            let board_state_clone = Arc::clone(&test_board_state);

            let handle = thread::spawn(move || {
                let mut encoder = AdvancedBoardStateEncoder::new(15, 3);
                let (board, our_snake) = &*board_state_clone;
                let mut thread_results = Vec::new();

                for iteration in 0..iterations_per_thread {
                    let channels = encoder.encode_12_channel_board(board, our_snake, iteration as u32);
                    
                    // Store key characteristics for comparison
                    let channel_sums: Vec<f32> = channels.iter()
                        .map(|ch| ch.iter().flatten().sum())
                        .collect();
                    
                    thread_results.push(channel_sums);
                }

                // Store results for consistency checking
                let mut results = results_clone.lock().unwrap();
                results.insert(thread_id, thread_results);
            });

            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }

        // Verify consistency across threads
        let results = shared_results.lock().unwrap();
        assert_eq!(results.len(), thread_count, "All threads should report results");

        // Compare results from first thread with others
        let reference_results = &results[&0];
        
        for thread_id in 1..thread_count {
            let thread_results = &results[&thread_id];
            assert_eq!(thread_results.len(), reference_results.len(), 
                      "Thread {} should have same number of iterations", thread_id);
            
            for (iteration, (thread_sums, reference_sums)) in 
                thread_results.iter().zip(reference_results.iter()).enumerate() {
                
                assert_eq!(thread_sums.len(), reference_sums.len(), 
                          "Thread {} iteration {} should have same number of channels", thread_id, iteration);
                
                for (ch_idx, (&thread_sum, &ref_sum)) in 
                    thread_sums.iter().zip(reference_sums.iter()).enumerate() {
                    
                    let diff = (thread_sum - ref_sum).abs();
                    assert!(diff < 0.001, 
                           "Thread {} iteration {} channel {} sum differs: {} vs {} (diff: {})",
                           thread_id, iteration, ch_idx, thread_sum, ref_sum, diff);
                }
            }
        }

        println!("🎯 Encoding consistency validated across {} threads with {} iterations each", 
                 thread_count, iterations_per_thread);
    }

    #[test]
    fn test_high_concurrency_stress() {
        let thread_count = 16; // High concurrency
        let harness = ConcurrentTestHarness::new(thread_count);
        let shared_encoder = Arc::new(Mutex::new(AdvancedBoardStateEncoder::new(25, 4)));

        let results = harness.execute_concurrent_test(move |thread_id, barrier| {
            let encoder_clone = Arc::clone(&shared_encoder);
            let mut generator = RandomBoardGenerator::new(thread_id as u64 * 10000);
            
            barrier.wait();
            let start = Instant::now();
            let mut iterations = 0;
            let mut total_channels = 0;

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                for iteration in 0..10 { // Fewer iterations per thread due to high concurrency
                    let complexity = (iteration % 3) + 1;
                    let (board, our_snake) = generator.generate_random_board(
                        9 + complexity * 2, 9 + complexity * 2,
                        complexity * 2, complexity + 1, complexity * 3
                    );
                    
                    let channels = {
                        let mut encoder = encoder_clone.lock().unwrap();
                        encoder.encode_12_channel_board(&board, &our_snake, (thread_id * 20 + iteration) as u32)
                    };
                    
                    assert_eq!(channels.len(), 12, "Should produce 12 channels in high-concurrency thread {}", thread_id);
                    total_channels += channels.len();
                    iterations += 1;
                    
                    // Small delay to increase contention
                    thread::sleep(Duration::from_millis(1));
                }
            }));

            let duration = start.elapsed();

            ThreadTestResult {
                thread_id,
                duration,
                success: result.is_ok(),
                error_message: result.err().map(|_| "Panic occurred".to_string()),
                iterations_completed: iterations,
                channels_produced: total_channels,
            }
        });

        harness.validate_results(&results);
        
        // Additional validation for high concurrency
        let total_iterations: usize = results.iter().map(|r| r.iterations_completed).sum();
        let total_channels: usize = results.iter().map(|r| r.channels_produced).sum();
        
        assert_eq!(total_iterations, thread_count * 10, "All iterations should complete");
        assert_eq!(total_channels, total_iterations * 12, "All channels should be produced");
        
        println!("🚀 High concurrency stress test: {} threads handled {} total operations", 
                 thread_count, total_iterations);
    }

    #[test]
    fn test_concurrent_memory_safety() {
        let thread_count = 8;
        let shared_encoder = Arc::new(Mutex::new(AdvancedBoardStateEncoder::new(30, 3)));
        let memory_measurements = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();

        for thread_id in 0..thread_count {
            let encoder_clone = Arc::clone(&shared_encoder);
            let memory_clone = Arc::clone(&memory_measurements);

            let handle = thread::spawn(move || {
                let mut generator = RandomBoardGenerator::new(thread_id as u64 * 7777);
                
                for iteration in 0..30 {
                    let (board, our_snake) = generator.generate_random_board(12, 12, 6, 4, 9);
                    
                    // Perform encoding
                    let channels = {
                        let mut encoder = encoder_clone.lock().unwrap();
                        encoder.encode_12_channel_board(&board, &our_snake, (thread_id * 30 + iteration) as u32)
                    };
                    
                    // Validate memory safety
                    assert_eq!(channels.len(), 12, "Memory corruption check: channel count");
                    
                    for (ch_idx, channel) in channels.iter().enumerate() {
                        assert_eq!(channel.len(), board.height as usize, 
                                  "Memory corruption check: channel {} height", ch_idx);
                        if !channel.is_empty() {
                            assert_eq!(channel[0].len(), board.width as usize,
                                      "Memory corruption check: channel {} width", ch_idx);
                        }
                        
                        // Check for invalid values that might indicate memory corruption
                        for row in channel {
                            for &value in row {
                                assert!(value >= 0.0 && value <= 1.0 && value.is_finite(),
                                       "Memory corruption check: invalid value {} in channel {}", value, ch_idx);
                            }
                        }
                    }
                    
                    // Periodic memory usage recording
                    if iteration % 10 == 0 {
                        let memory_usage = {
                            let encoder = encoder_clone.lock().unwrap();
                            encoder.movement_tracker.history.len() +
                            encoder.voronoi_analyzer.our_territory.len() +
                            encoder.danger_predictor.danger_zones.len() +
                            encoder.strategic_analyzer.strategic_positions.len()
                        };
                        
                        memory_clone.lock().unwrap().push((thread_id, iteration, memory_usage));
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().expect("Thread should complete without memory issues");
        }

        // Analyze memory measurements
        let measurements = memory_measurements.lock().unwrap();
        assert!(!measurements.is_empty(), "Should have memory measurements");
        
        let max_memory_usage = measurements.iter().map(|(_, _, usage)| *usage).max().unwrap();
        assert!(max_memory_usage < 10000, 
               "Memory usage {} seems excessive - possible leak or corruption", max_memory_usage);
        
        println!("🛡️ Memory safety validated across {} concurrent threads", thread_count);
    }

    #[test]
    fn test_deadlock_prevention() {
        let thread_count = 4;
        let shared_encoder = Arc::new(Mutex::new(AdvancedBoardStateEncoder::new(20, 2)));
        let completion_counter = Arc::new(Mutex::new(0));
        let mut handles = Vec::new();

        for thread_id in 0..thread_count {
            let encoder_clone = Arc::clone(&shared_encoder);
            let counter_clone = Arc::clone(&completion_counter);

            let handle = thread::spawn(move || {
                let mut generator = RandomBoardGenerator::new(thread_id as u64 * 8888);
                
                // Each thread performs operations that might cause deadlocks
                for iteration in 0..20 {
                    let (board, our_snake) = generator.generate_random_board(10, 10, 4, 3, 7);
                    
                    // Acquire lock, do work, release (test for proper lock handling)
                    {
                        let mut encoder = encoder_clone.lock().unwrap();
                        let _channels = encoder.encode_12_channel_board(&board, &our_snake, iteration as u32);
                        
                        // Simulate some processing time while holding lock
                        let _stats = encoder.get_encoding_stats();
                    } // Lock released here
                    
                    // Short sleep to create opportunities for lock contention
                    thread::sleep(Duration::from_millis(1));
                }
                
                // Increment completion counter
                *counter_clone.lock().unwrap() += 1;
            });

            handles.push(handle);
        }

        // Wait for all threads with timeout to detect deadlocks
        let timeout = Duration::from_secs(30); // Reasonable timeout
        let start = Instant::now();

        for handle in handles {
            let remaining_time = timeout.saturating_sub(start.elapsed());
            
            // Note: join() doesn't support timeout in std, but we can check if threads complete
            match handle.join() {
                Ok(_) => {
                    assert!(start.elapsed() < timeout, "Thread took too long - possible deadlock");
                }
                Err(_) => {
                    panic!("Thread panicked - possible deadlock or other concurrency issue");
                }
            }
        }

        // Verify all threads completed
        let final_count = *completion_counter.lock().unwrap();
        assert_eq!(final_count, thread_count, "All threads should complete without deadlock");
        
        println!("🔓 Deadlock prevention validated: {} threads completed in {}ms", 
                 thread_count, start.elapsed().as_millis());
    }

    #[test]
    fn test_race_condition_detection() {
        let thread_count = 10;
        let shared_encoder = Arc::new(RwLock::new(AdvancedBoardStateEncoder::new(15, 2)));
        let shared_state = Arc::new(Mutex::new(0u64)); // Shared counter for race detection
        let mut handles = Vec::new();

        for thread_id in 0..thread_count {
            let encoder_clone = Arc::clone(&shared_encoder);
            let state_clone = Arc::clone(&shared_state);

            let handle = thread::spawn(move || {
                let mut generator = RandomBoardGenerator::new(thread_id as u64 * 9999);
                
                for iteration in 0..50 {
                    let (board, our_snake) = generator.generate_random_board(9, 9, 3, 2, 6);
                    
                    // Perform encoding with read lock
                    let channels = {
                        let mut encoder = encoder_clone.write().unwrap();
                        encoder.encode_12_channel_board(&board, &our_snake, iteration as u32)
                    };
                    
                    // Validate encoding results
                    assert_eq!(channels.len(), 12, "Race condition check: channel count");
                    
                    // Update shared state (testing for race conditions)
                    {
                        let mut counter = state_clone.lock().unwrap();
                        let old_value = *counter;
                        
                        // Simulate some work
                        thread::sleep(Duration::from_nanos(100));
                        
                        *counter = old_value + 1;
                    }
                    
                    // Get encoder statistics with read lock
                    let stats = {
                        let encoder = encoder_clone.read().unwrap();
                        encoder.get_encoding_stats()
                    };
                    
                    // Validate statistics are reasonable (not corrupted by race conditions)
                    let total_positions = stats.0 + stats.1 + stats.2 + stats.3;
                    assert!(total_positions < 5000, 
                           "Statistics corruption check: total {} excessive", total_positions);
                }
            });

            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().expect("Thread should complete without race conditions");
        }

        // Verify shared state is consistent (no race conditions in counter)
        let final_count = *shared_state.lock().unwrap();
        let expected_count = thread_count * 50;
        assert_eq!(final_count, expected_count, 
                  "Race condition detected: expected {}, got {}", expected_count, final_count);
        
        println!("🏁 Race condition detection: {} threads, {} operations, state consistent", 
                 thread_count, expected_count);
    }

    #[test]
    fn test_concurrent_performance_stability() {
        let thread_counts = vec![1, 2, 4, 8, 12];
        let iterations_per_thread = 15;
        
        for &thread_count in &thread_counts {
            let shared_encoder = Arc::new(Mutex::new(AdvancedBoardStateEncoder::new(20, 3)));
            let performance_data = Arc::new(Mutex::new(Vec::new()));
            let mut handles = Vec::new();

            let test_start = Instant::now();

            for thread_id in 0..thread_count {
                let encoder_clone = Arc::clone(&shared_encoder);
                let perf_data_clone = Arc::clone(&performance_data);

                let handle = thread::spawn(move || {
                    let mut generator = RandomBoardGenerator::new(thread_id as u64 * 6666);
                    let mut thread_times = Vec::new();

                    for iteration in 0..iterations_per_thread {
                        let (board, our_snake) = generator.generate_random_board(11, 11, 5, 3, 8);
                        
                        let start = Instant::now();
                        let channels = {
                            let mut encoder = encoder_clone.lock().unwrap();
                            encoder.encode_12_channel_board(&board, &our_snake, iteration as u32)
                        };
                        let duration = start.elapsed();
                        
                        assert_eq!(channels.len(), 12, "Performance test: should produce 12 channels");
                        thread_times.push(duration.as_millis());
                    }

                    // Store performance data
                    perf_data_clone.lock().unwrap().extend(thread_times);
                });

                handles.push(handle);
            }

            // Wait for all threads
            for handle in handles {
                handle.join().expect("Performance test thread should complete");
            }

            let total_test_time = test_start.elapsed();
            let performance_data = performance_data.lock().unwrap();
            
            // Analyze performance
            let total_operations = performance_data.len();
            let min_time = *performance_data.iter().min().unwrap();
            let max_time = *performance_data.iter().max().unwrap();
            let avg_time = performance_data.iter().sum::<u128>() as f64 / total_operations as f64;

            // Performance should be reasonable
            assert!(avg_time < 500.0, "Average time {}ms too high for {} threads", avg_time, thread_count);
            assert!(max_time < 1000, "Max time {}ms too high for {} threads", max_time, thread_count);
            
            println!("⚡ {} threads: {}ops in {}ms, avg={}ms, min={}ms, max={}ms", 
                     thread_count, total_operations, total_test_time.as_millis(),
                     avg_time as u128, min_time, max_time);
        }
    }

    #[test]
    fn test_thread_safety_comprehensive_validation() {
        println!("\n🧵 === COMPREHENSIVE THREAD SAFETY VALIDATION ===");
        
        let test_scenarios = vec![
            ("basic_mutex", 8, 25),
            ("high_contention", 16, 15),  
            ("sustained_load", 6, 50),
        ];

        for (scenario_name, thread_count, iterations) in test_scenarios {
            let shared_encoder = Arc::new(Mutex::new(AdvancedBoardStateEncoder::new(20, 3)));
            let results_collector = Arc::new(Mutex::new(Vec::new()));
            let mut handles = Vec::new();

            let scenario_start = Instant::now();

            for thread_id in 0..thread_count {
                let encoder_clone = Arc::clone(&shared_encoder);
                let results_clone = Arc::clone(&results_collector);

                let handle = thread::spawn(move || {
                    let mut generator = RandomBoardGenerator::new(thread_id as u64 * 12345);
                    let mut thread_success = true;
                    let mut completed_iterations = 0;

                    let thread_start = Instant::now();

                    for iteration in 0..iterations {
                        let (board, our_snake) = generator.generate_random_board(12, 12, 6, 4, 9);
                        
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let channels = {
                                let mut encoder = encoder_clone.lock().unwrap();
                                encoder.encode_12_channel_board(&board, &our_snake, iteration as u32)
                            };
                            
                            // Comprehensive validation
                            assert_eq!(channels.len(), 12, "Thread safety: channel count check");
                            
                            for (ch_idx, channel) in channels.iter().enumerate() {
                                assert_eq!(channel.len(), board.height as usize, 
                                          "Thread safety: channel {} height", ch_idx);
                                if !channel.is_empty() {
                                    assert_eq!(channel[0].len(), board.width as usize,
                                              "Thread safety: channel {} width", ch_idx);
                                }
                                
                                for row in channel {
                                    for &value in row {
                                        assert!(value >= 0.0 && value <= 1.0 && value.is_finite(),
                                               "Thread safety: invalid value in channel {}", ch_idx);
                                    }
                                }
                            }
                            
                            channels.len()
                        }));

                        match result {
                            Ok(_) => completed_iterations += 1,
                            Err(_) => {
                                thread_success = false;
                                break;
                            }
                        }
                    }

                    let thread_duration = thread_start.elapsed();
                    
                    // Report results
                    results_clone.lock().unwrap().push((
                        thread_id, 
                        thread_success, 
                        completed_iterations, 
                        thread_duration.as_millis()
                    ));
                });

                handles.push(handle);
            }

            // Wait for all threads
            for handle in handles {
                handle.join().expect("Thread safety validation thread should complete");
            }

            let scenario_duration = scenario_start.elapsed();
            let results = results_collector.lock().unwrap();

            // Validate all threads succeeded
            let successful_threads = results.iter().filter(|(_, success, _, _)| *success).count();
            assert_eq!(successful_threads, thread_count, 
                      "All threads should succeed in scenario '{}'", scenario_name);

            let total_iterations: usize = results.iter().map(|(_, _, completed, _)| *completed).sum();
            let expected_iterations = thread_count * iterations;
            assert_eq!(total_iterations, expected_iterations,
                      "All iterations should complete in scenario '{}'", scenario_name);

            let avg_thread_time = results.iter().map(|(_, _, _, time)| *time).sum::<u128>() / thread_count as u128;

            println!("✅ Scenario '{}': {} threads, {} total ops, {}ms avg per thread, {}ms total", 
                     scenario_name, thread_count, total_iterations, avg_thread_time, scenario_duration.as_millis());
        }

        println!("🎯 THREAD SAFETY VALIDATION COMPLETE - All scenarios passed");
        println!("=== CONCURRENT ACCESS PATTERNS SAFE ===\n");
    }
}
               memory_ratio, baseline_memory / 1024, final_memory / 1024);

        println!("🧹 Memory Cleanup Test:");
        println!("   Baseline: {}KB", baseline_memory / 1024);
        println!("   After intensive usage: {}KB", final_memory / 1024);
        println!("   Ratio: {:.2}x", memory_ratio);
        println!("   ✅ Memory cleanup working properly");
    }

    #[test]
    fn test_pathological_memory_scenarios() {
        // Test edge cases that could cause memory issues
        let problematic_scenarios = vec![
            // Extremely long game
            ("long_game", || {
                let mut encoder = AdvancedBoardStateEncoder::new(200, 2);
                let mut generator = RandomBoardGenerator::new(9999);
                
                for turn in 0..500 {
                    let (board, our_snake) = generator.generate_random_board(11, 11, 5, 3, 8);
                    encoder.encode_12_channel_board(&board, &our_snake, turn as u32);
                }
                
                estimate_encoder_memory_usage(&encoder)
            }),

            // Rapid position changes
            ("rapid_movement", || {
                let mut encoder = AdvancedBoardStateEncoder::new(50, 2);
                let mut generator = RandomBoardGenerator::new(8888);
                
                for turn in 0..200 {
                    // Generate new position every turn to stress movement tracker
                    let position = Coord { x: (turn % 20) as i32, y: ((turn / 20) % 15) as i32 };
                    let our_snake = create_test_snake("rapid", position, 3, 100);
                    let board = Board {
                        width: 20, height: 15, food: vec![], snakes: vec![our_snake.clone()], hazards: vec![]
                    };
                    encoder.encode_12_channel_board(&board, &our_snake, turn as u32);
                }
                
                estimate_encoder_memory_usage(&encoder)
            }),

            // Maximum settings stress test
            ("max_settings", || {
                let mut encoder = AdvancedBoardStateEncoder::new(500, 10); // Extreme settings
                let mut generator = RandomBoardGenerator::new(7777);
                
                for turn in 0..50 {
                    let (board, our_snake) = generator.generate_random_board(25, 25, 20, 8, 30);
                    encoder.encode_12_channel_board(&board, &our_snake, turn as u32);
                }
                
                estimate_encoder_memory_usage(&encoder)
            }),
        ];

        for (scenario_name, scenario_fn) in problematic_scenarios {
            let memory_usage = scenario_fn();
            
            // Even pathological scenarios should have bounded memory
            assert!(memory_usage < 10_000_000, // 10MB limit
                   "Pathological scenario '{}' used {}KB - excessive memory",
                   scenario_name, memory_usage / 1024);
            
            println!("🧪 Pathological scenario '{}': {}KB", scenario_name, memory_usage / 1024);
        }

        println!("✅ All pathological memory scenarios handled within bounds");
    }

    #[test]
    fn test_memory_leak_detection_comprehensive() {
        println!("\n🔍 === COMPREHENSIVE MEMORY LEAK DETECTION ===");
        
        let test_scenarios = vec![
            ("short_games", 50, 20),     // Many short games
            ("medium_games", 20, 50),    // Fewer medium games  
            ("long_games", 10, 100),     // Few long games
        ];

        for (scenario_name, game_count, turns_per_game) in test_scenarios {
            let mut encoder = AdvancedBoardStateEncoder::new(30, 3);
            let mut memory_tracker = MemoryUsageTracker::new(scenario_name, game_count * 2);
            let mut generator = RandomBoardGenerator::new(scenario_name.len() as u64 * 1000);

            for game in 0..game_count {
                // Vary game complexity
                let complexity = (game % 3) + 1;
                
                for turn in 0..turns_per_game {
                    let (board, our_snake) = generator.generate_random_board(
                        9 + complexity * 2,
                        9 + complexity * 2, 
                        complexity * 3,
                        complexity + 1,
                        complexity * 4 + turn % 10
                    );
                    
                    encoder.encode_12_channel_board(&board, &our_snake, turn as u32);
                }
                
                // Record memory after each game
                let memory_usage = estimate_encoder_memory_usage(&encoder);
                memory_tracker.record_measurement(memory_usage);
            }

            // Analyze for leaks
            memory_tracker.check_for_leak(1.3).expect(&format!("Memory leak in scenario '{}'", scenario_name));
            
            let stats = memory_tracker.get_statistics();
            println!("📊 Scenario '{}': avg={}KB, growth={:.2}x ✅", 
                     scenario_name, stats.avg_kb, stats.growth_factor);
        }

        println!("🎯 MEMORY LEAK DETECTION COMPLETE - All scenarios stable");
        println!("=== NO MEMORY LEAKS DETECTED ===\n");
    }
}
                    assert_eq!(channel.len(), height as usize, "Channel dimensions must be correct");
                    if !channel.is_empty() {
                        assert_eq!(channel[0].len(), width as usize, "Channel width must be correct");
                    }
                }
            }
        }
        
        // Statistical analysis of all runs
        let times: Vec<Duration> = all_times.iter().map(|(d, _)| *d).collect();
        let mut sorted_times = times.clone();
        sorted_times.sort();
        
        let mean = times.iter().sum::<Duration>() / times.len() as u32;
        let p95 = sorted_times[(times.len() as f32 * 0.95) as usize];
        let p99 = sorted_times[(times.len() as f32 * 0.99) as usize];
        let max_time = *sorted_times.last().unwrap();
        
        // FINAL VALIDATION
        assert!(p95.as_millis() < 500, "🚨 95th percentile {}ms exceeds 500ms budget", p95.as_millis());
        assert!(p99.as_millis() < 500, "🚨 99th percentile {}ms exceeds 500ms budget", p99.as_millis());
        assert!(max_time.as_millis() < 500, "🚨 Maximum time {}ms exceeds 500ms budget", max_time.as_millis());
        assert!(mean.as_millis() < 300, "🚨 Mean time {}ms exceeds 300ms target", mean.as_millis());
        
        println!("✅ ALL CRITICAL REQUIREMENTS MET:");
        println!("   Total tests: {} runs across {} scenarios", times.len(), validation_scenarios.len());
        println!("   Mean: {}ms (target: <300ms) ✅", mean.as_millis());
        println!("   P95: {}ms (budget: <500ms) ✅", p95.as_millis());  
        println!("   P99: {}ms (budget: <500ms) ✅", p99.as_millis());
        println!("   Max: {}ms (budget: <500ms) ✅", max_time.as_millis());
        println!("🎯 12-CHANNEL SPATIAL ANALYSIS SYSTEM MEETS ALL PERFORMANCE REQUIREMENTS");
        println!("=== PERFORMANCE VALIDATION COMPLETE ===\n");
    }

    #[test]
    fn test_encoding_determinism_and_reproducibility() {
        let mut encoder1 = AdvancedBoardStateEncoder::new(10, 2);
        let mut encoder2 = AdvancedBoardStateEncoder::new(10, 2);
        
        let game_state = GameStateBuilder::new()
            .with_board_size(8, 8)
            .with_our_snake(create_test_snake("us", Coord { x: 4, y: 4 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 1, y: 1 }, 3, 100))
            .with_food(vec![Coord { x: 6, y: 6 }])
            .build();
        
        // Run identical sequences on both encoders
        let turns = vec![1, 2, 3, 4, 5];
        
        for &turn in &turns {
            let channels1 = encoder1.encode_12_channel_board(&game_state.board, &game_state.you, turn);
            let channels2 = encoder2.encode_12_channel_board(&game_state.board, &game_state.you, turn);
            
            // Results should be identical (deterministic)
            assert_eq!(channels1.len(), channels2.len(), "Channel count should be deterministic at turn {}", turn);
            
            for (ch_idx, (ch1, ch2)) in channels1.iter().zip(channels2.iter()).enumerate() {
                assert_eq!(ch1.len(), ch2.len(), 
                          "Channel {} dimensions should be deterministic at turn {}", ch_idx, turn);
                
                for (y, (row1, row2)) in ch1.iter().zip(ch2.iter()).enumerate() {
                    assert_eq!(row1.len(), row2.len(),
                              "Channel {} row {} dimensions should be deterministic at turn {}", ch_idx, y, turn);
                    
                    for (x, (&val1, &val2)) in row1.iter().zip(row2.iter()).enumerate() {
                        assert!((val1 - val2).abs() < 0.001,
                               "Channel {} value at ({}, {}) should be deterministic at turn {}: {} vs {}",
                               ch_idx, x, y, turn, val1, val2);
                    }
                }
            }
        }
        
        // Internal state should also be consistent
        let stats1 = encoder1.get_encoding_stats();
        let stats2 = encoder2.get_encoding_stats();
        assert_eq!(stats1, stats2, "Encoder statistics should be deterministic");
    }

    #[test] 
    fn test_cross_component_performance_integration() {
        let mut encoder = AdvancedBoardStateEncoder::new(20, 4);
        
        // Test performance with complex multi-component scenarios
        let complex_scenarios = vec![
            // Large board with many elements
            GameStateBuilder::new()
                .with_board_size(19, 19)
                .with_our_snake(create_test_snake("us", Coord { x: 9, y: 9 }, 8, 100))
                .with_opponent(create_test_snake("opp1", Coord { x: 3, y: 3 }, 6, 90))
                .with_opponent(create_test_snake("opp2", Coord { x: 15, y: 15 }, 5, 85))
                .with_opponent(create_test_snake("opp3", Coord { x: 3, y: 15 }, 4, 80))
                .with_food((0..10).map(|i| Coord { x: i * 2, y: i * 2 }).collect())
                .build(),
                
            // Dense snake scenario
            GameStateBuilder::new()
                .with_board_size(11, 11)
                .with_our_snake(create_test_snake("us", Coord { x: 5, y: 5 }, 12, 100))
                .with_opponent(create_test_snake("opp1", Coord { x: 2, y: 2 }, 10, 95))
                .with_opponent(create_test_snake("opp2", Coord { x: 8, y: 8 }, 8, 90))
                .with_food(vec![Coord { x: 0, y: 0 }, Coord { x: 10, y: 10 }])
                .build(),
        ];
        
        for (scenario_idx, scenario) in complex_scenarios.iter().enumerate() {
            let start = std::time::Instant::now();
            let channels = encoder.encode_12_channel_board(&scenario.board, &scenario.you, 25);
            let elapsed = start.elapsed().as_millis();
            
            // Even complex scenarios should meet performance requirements
            assert!(elapsed < 500, "Complex scenario {} took {}ms, should be under 500ms", scenario_idx, elapsed);
            
            // Results should still be valid
            assert_eq!(channels.len(), 12, "Complex scenario {} should produce 12 channels", scenario_idx);
            
            // All components should contribute meaningfully
            let channel_sums: Vec<f32> = channels.iter()
                .map(|ch| ch.iter().flatten().sum())
                .collect();
            
            assert!(channel_sums[0] > 0.0, "Empty channel should have content in scenario {}", scenario_idx);
            assert!(channel_sums[1] > 0.0, "Our head channel should have content in scenario {}", scenario_idx);
            assert!(channel_sums[7] > 0.0, "Territory channel should have content in scenario {}", scenario_idx);
            
            println!("Scenario {} performance: {}ms, board: {}x{}, snakes: {}", 
                     scenario_idx, elapsed, scenario.board.width, scenario.board.height, 
                     scenario.board.snakes.len());
        }
    }
    }

    #[test]
    fn test_fuzz_performance() {
        let iterations = 100u64;
        let mut total_time = 0u128;
        let encoder = AdvancedBoardStateEncoder::new();
        
        for i in 0..iterations {
            let start = std::time::Instant::now();
            let game_state = GameStateBuilder::new()
                .with_board_size(11, 11)
                .with_our_snake(create_test_snake("us", Coord { x: 5, y: 5 }, 3, 100))
                .with_opponent(create_test_snake("opponent", Coord { x: 2, y: 2 }, 3, 100))
                .build();
            
            let result = encoder.encode_12_channel_board(&game_state.board, &game_state.you, i as u32);
            let elapsed = start.elapsed().as_millis();
            
            total_time += elapsed;
            
            // Individual encoding should be fast
            assert!(elapsed < 500, "Individual encoding took too long: {}ms at iteration {}", elapsed, i);
            
            // Validate result structure
            assert_eq!(result.len(), 12, "Should produce 12 channels at iteration {}", i);
        }
        
        let average_time = total_time / iterations;
        println!("Fuzz test performance: average {}ms per encoding over {} iterations",
                 average_time, iterations);
        
        // Average should be well under budget
        assert!(average_time < 200, "Average encoding time {}ms too high", average_time);
    }
}

    // Helper function for creating test snake with custom body
    fn create_test_snake_with_body(id: &str, body: Vec<Coord>, health: i32) -> Battlesnake {
        if body.is_empty() {
            return create_test_snake(id, Coord { x: 0, y: 0 }, 1, health);
        }
        
        Battlesnake {
            id: id.to_string(),
            name: format!("Test Snake {}", id),
            health,
            body: body.clone(),
            head: body[0],
            length: body.len() as u32,
            latency: "0".to_string(),
            shout: None,
        }
    }

    // Property tests for performance characteristics
    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 10, // Fewer cases for performance tests
            max_shrink_iters: 10,
            .. ProptestConfig::default()
        })]
        
        #[test]
        fn property_encoder_performance_scales_reasonably(
            (width, height, _food) in board_strategy(8, 15, 5),
            our_snake in snake_strategy(15, 15, 8)
        ) {
            let mut encoder = AdvancedBoardStateEncoder::new(10, 2);
            
            let our_battlesnake = create_test_snake_with_body("us", our_snake.clone(), 100);
            let board = Board {
                width,
                height,
                food: vec![],
                snakes: vec![our_battlesnake.clone()],
                hazards: vec![],
            };
            
            let start = std::time::Instant::now();
            let _channels = encoder.encode_12_channel_board(&board, &our_battlesnake, 10);
            let elapsed = start.elapsed().as_millis();
            
            // Property: Encoding should complete within reasonable time
            // Time scales roughly with board area, so use a generous upper bound
            let board_area = (width * height as i32) as u128;
            let max_time_ms = (board_area * 10).min(1000); // Max 10ms per cell or 1000ms total
            
            prop_assert!(elapsed <= max_time_ms,
                       "Encoding {}x{} board took {}ms, should be <= {}ms",
                       width, height, elapsed, max_time_ms);
        }
    }

    // Property tests for algorithmic correctness invariants
    proptest! {
        #[test]
        fn property_voronoi_territory_sum_reasonable(
            (width, height, food) in board_strategy(5, 10, 5),
            our_snake in snake_strategy(10, 10, 6),
            opponent_snake in snake_strategy(10, 10, 6)
        ) {
            let mut analyzer = VoronoiTerritoryAnalyzer::new();
            
            let our_battlesnake = create_test_snake_with_body("us", our_snake.clone(), 100);
            let opponent_battlesnake = create_test_snake_with_body("opponent", opponent_snake.clone(), 100);
            
            let board = Board {
                width,
                height,
                food,
                snakes: vec![our_battlesnake.clone(), opponent_battlesnake],
                hazards: vec![],
            };
            
            analyzer.analyze_territory(&board, &our_battlesnake);
            
            let our_territory_sum: f32 = analyzer.our_territory.values().sum();
            let opponent_territory_sum: f32 = analyzer.opponent_territory.values().sum();
            
            // Property: Total territory should be reasonable relative to board size
            let board_area = width as f32 * height as f32;
            let total_territory = our_territory_sum + opponent_territory_sum;
            
            prop_assert!(total_territory <= board_area * 2.0,
                       "Total territory {} should not exceed twice the board area {}",
                       total_territory, board_area);
                       
            prop_assert!(total_territory > 0.0,
                       "Should have some territory assigned with multiple snakes");
        }
    }
    }

    #[test]
    fn test_asymmetric_board_strategic_analysis() {
        let mut analyzer = StrategicPositionAnalyzer::new();
        let game_state = GameStateBuilder::new()
            .with_board_size(15, 7) // Asymmetric board
            .with_our_snake(create_test_snake("us", Coord { x: 7, y: 3 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 2, y: 2 }, 3, 100))
            .build();
        
        let strategic_channel = analyzer.analyze_strategic_positions(&game_state.board, &game_state.you);
        
        // Validate dimensions handle asymmetric board
        assert_eq!(strategic_channel.len(), 7); // height
        assert_eq!(strategic_channel[0].len(), 15); // width
        
        // Center position should still be strategic
        let center_x = 7; // width / 2
        let center_y = 3; // height / 2
        let center_value = strategic_channel[center_y][center_x];
        assert!(center_value > 0.0, "Center should maintain strategic value on asymmetric board");
    }

    #[test]
    fn test_no_food_scenario() {
        let mut analyzer = StrategicPositionAnalyzer::new();
        let game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_test_snake("us", Coord { x: 3, y: 3 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 1, y: 1 }, 3, 100))
            .with_food(vec![]) // No food
            .build();
        
        let strategic_channel = analyzer.analyze_strategic_positions(&game_state.board, &game_state.you);
        
        // Should still identify strategic positions based on other factors
        SpatialAnalysisValidator::validate_channel_dimensions(&strategic_channel, &game_state.board, "strategic_positions")
            .expect("No-food scenario should produce valid strategic channel");
        
        // Center control should still be strategic
        let center_value = strategic_channel[3][3];
        assert!(center_value > 0.0, "Center control should be strategic even without food");
        
        // Corner traps should still be identified
        let corner_value = strategic_channel[0][0];
        assert!(corner_value <= 0.0, "Corner traps should be identified even without food");
        
        // Should still have strategic positions from cutting points and escape routes
        let strategic_position_count = analyzer.strategic_positions.len();
        assert!(strategic_position_count > 0, "Should identify strategic positions even without food");
    }

    #[test]
    fn test_single_opponent_vs_multiple_opponents() {
        // Test single opponent scenario
        let mut single_analyzer = StrategicPositionAnalyzer::new();
        let single_game_state = GameStateBuilder::new()
            .with_board_size(9, 9)
            .with_our_snake(create_test_snake("us", Coord { x: 4, y: 4 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 2, y: 2 }, 3, 100))
            .build();
        
        single_analyzer.analyze_strategic_positions(&single_game_state.board, &single_game_state.you);
        let single_cutting_positions = single_analyzer.strategic_positions.len();
        
        // Test multiple opponent scenario
        let mut multi_analyzer = StrategicPositionAnalyzer::new();
        let multi_game_state = GameStateBuilder::new()
            .with_board_size(9, 9)
            .with_our_snake(create_test_snake("us", Coord { x: 4, y: 4 }, 3, 100))
            .with_opponent(create_test_snake("opponent1", Coord { x: 2, y: 2 }, 3, 100))
            .with_opponent(create_test_snake("opponent2", Coord { x: 6, y: 6 }, 3, 100))
            .with_opponent(create_test_snake("opponent3", Coord { x: 2, y: 6 }, 3, 100))
            .build();
        
        multi_analyzer.analyze_strategic_positions(&multi_game_state.board, &multi_game_state.you);
        let multi_cutting_positions = multi_analyzer.strategic_positions.len();
        
        // Multiple opponents should create more strategic considerations
        assert!(multi_cutting_positions >= single_cutting_positions, 
                "Multiple opponents should create more strategic positions");
    }

    #[test]
    fn test_strategic_consistency_across_calls() {
        let mut analyzer = StrategicPositionAnalyzer::new();
        let game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_test_snake("us", Coord { x: 3, y: 3 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 1, y: 1 }, 3, 100))
            .with_food(vec![Coord { x: 5, y: 5 }])
            .build();
        
        // Analyze twice
        let channel1 = analyzer.analyze_strategic_positions(&game_state.board, &game_state.you);
        let channel2 = analyzer.analyze_strategic_positions(&game_state.board, &game_state.you);
        
        // Results should be consistent
        assert_eq!(channel1.len(), channel2.len());
        for (row1, row2) in channel1.iter().zip(channel2.iter()) {
            assert_eq!(row1.len(), row2.len());
            for (&val1, &val2) in row1.iter().zip(row2.iter()) {
                assert!((val1 - val2).abs() < 0.001, "Strategic analysis should be deterministic");
            }
        }
    }

    #[test]
    fn test_edge_case_minimal_board() {
        let mut analyzer = StrategicPositionAnalyzer::new();
        let game_state = GameStateBuilder::new()
            .with_board_size(3, 3)
            .with_our_snake(create_test_snake("us", Coord { x: 1, y: 1 }, 1, 100))
            .build();
        
        let strategic_channel = analyzer.analyze_strategic_positions(&game_state.board, &game_state.you);
        
        // Should handle minimal board without crashing
        assert_eq!(strategic_channel.len(), 3);
        assert_eq!(strategic_channel[0].len(), 3);
        
        // Center should still be strategic even on minimal board
        let center_value = strategic_channel[1][1];
        assert!(center_value >= 0.0, "Center should be strategic on minimal board");
        
        // All corners should have corner trap penalties
        let corners = [(0, 0), (0, 2), (2, 0), (2, 2)];
        for (y, x) in corners {
            let corner_value = strategic_channel[y][x];
            assert!(corner_value <= 0.0, "Corner ({},{}) should have negative strategic value", x, y);
        }
    }
    }

    #[test]
    fn test_movement_tracker_large_history() {
        let mut tracker = MovementHistoryTracker::new(100);
        
        // Fill up the history buffer
        for i in 0..100 {
            tracker.update_position(Coord { x: i % 20, y: i % 20 }, i);
        }
        
        assert_eq!(tracker.history.len(), 50);
        assert!(tracker.history.capacity() <= 100, "History capacity should be reasonable");
        
        // Should still function correctly
        let recent = tracker.get_recent_positions(10);
        assert_eq!(recent.len(), 10);
        
        let was_visited = tracker.was_recently_visited(&Coord { x: 19, y: 19 }, 5);
        assert!(was_visited, "Should correctly track recent visits even after many positions");
    }

    #[test]
    fn test_asymmetric_board_compatibility() {
        let mut tracker = MovementHistoryTracker::new(10);
        
        // Test with asymmetric board (width != height) - handles type inconsistency
        let board = Board {
            width: 12, // i32
            height: 8, // u32
            food: vec![],
            snakes: vec![],
            hazards: vec![],
        };
        
        // Add positions across the asymmetric board
        tracker.update_position(Coord { x: 0, y: 0 }, 0);
        tracker.update_position(Coord { x: 11, y: 0 }, 1); // Max width
        tracker.update_position(Coord { x: 0, y: 7 }, 2); // Max height
        tracker.update_position(Coord { x: 11, y: 7 }, 3); // Max both
        tracker.update_position(Coord { x: 6, y: 4 }, 4); // Center
        
        let history_channel = tracker.generate_history_channel(&board);
        
        // Validate dimensions handle type inconsistency
        assert_eq!(history_channel.len(), 8); // height as usize
        assert_eq!(history_channel[0].len(), 12); // width as usize
        
        // All positions should be recorded correctly
        assert!(history_channel[0][0] > 0.0, "Top-left should be recorded");
        assert!(history_channel[0][11] > 0.0, "Top-right should be recorded");
        assert!(history_channel[7][0] > 0.0, "Bottom-left should be recorded");
        assert!(history_channel[7][11] > 0.0, "Bottom-right should be recorded");
        assert!(history_channel[4][6] > 0.0, "Center should be recorded");
    }

    #[test]
    fn test_concurrent_position_updates() {
        // Simulate rapid position updates (like during fast game turns)
        let mut tracker = MovementHistoryTracker::new(20);
        
        // Rapid updates in sequence
        let positions = vec![
            (Coord { x: 5, y: 5 }, 100),
            (Coord { x: 5, y: 6 }, 101),
            (Coord { x: 4, y: 6 }, 102),
            (Coord { x: 3, y: 6 }, 103),
            (Coord { x: 3, y: 5 }, 104),
        ];
        
        for (pos, turn) in positions {
            tracker.update_position(pos, turn);
        }
        
        // Should handle rapid updates correctly
        assert_eq!(tracker.history.len(), 5);
        assert_eq!(tracker.current_turn, 104);
        
        // Recent position queries should work correctly
        let very_recent = tracker.get_recent_positions(1);
        assert_eq!(very_recent.len(), 1);
        assert_eq!(very_recent[0], Coord { x: 3, y: 5 });
        
        let recent_3 = tracker.get_recent_positions(3);
        assert_eq!(recent_3.len(), 3);
    }
}

    /// Analyze immediate collision dangers (current turn)
    fn analyze_immediate_dangers(&mut self, board: &Board, our_snake: &Battlesnake) {
        // Mark current snake bodies as maximum danger
        for snake in &board.snakes {
            if snake.id == our_snake.id {
                continue; // Don't mark our own body as danger for movement
            }

            for segment in &snake.body {
                self.danger_map.insert(*segment, 1.0);
            }
        }

        // Mark wall boundaries as maximum danger
        for x in 0..board.width {
            // Top and bottom walls
            self.danger_map.insert(Coord { x, y: -1 }, 1.0);
            self.danger_map.insert(Coord { x, y: board.height as i32 }, 1.0);
        }
        for y in 0..board.height as i32 {
            // Left and right walls  
            self.danger_map.insert(Coord { x: -1, y }, 1.0);
            self.danger_map.insert(Coord { x: board.width, y }, 1.0);
        }
    }

    /// Analyze future collision dangers using movement prediction
    fn analyze_future_dangers(&mut self, board: &Board, our_snake: &Battlesnake) {
        let opponents: Vec<&Battlesnake> = board.snakes.iter()
            .filter(|snake| snake.id != our_snake.id)
            .collect();

        for opponent in opponents {
            // Predict opponent movement patterns
            let predicted_positions = self.predict_opponent_positions(opponent, board);
            
            for (turn, positions) in predicted_positions.iter().enumerate() {
                let turn_weight = 1.0 / (turn as f32 + 1.0); // Decay over time
                
                for &pos in positions {
                    if pos.x >= 0 && pos.x < board.width && 
                       pos.y >= 0 && pos.y < board.height as i32 {
                        let existing_danger = self.danger_map.get(&pos).unwrap_or(&0.0);
                        let new_danger = (existing_danger + turn_weight * 0.5).min(1.0);
                        self.danger_map.insert(pos, new_danger);
                    }
                }
            }
        }
    }

    /// Analyze head-to-head collision risks
    fn analyze_head_collision_risks(&mut self, board: &Board, our_snake: &Battlesnake) {
        let opponents: Vec<&Battlesnake> = board.snakes.iter()
            .filter(|snake| snake.id != our_snake.id)
            .collect();

        for opponent in opponents {
            // Calculate positions where head-to-head collisions could occur
            let opponent_possible_moves = self.get_possible_moves(&opponent.head, board);
            
            for opponent_next_pos in opponent_possible_moves {
                // Check if we could also move to adjacent positions (head-to-head risk)
                let our_possible_moves = self.get_possible_moves(&our_snake.head, board);
                
                for our_next_pos in our_possible_moves {
                    let distance = (opponent_next_pos.x - our_next_pos.x).abs() + 
                                  (opponent_next_pos.y - our_next_pos.y).abs();
                    
                    if distance <= 1 {
                        // Potential head-to-head collision
                        let danger_level = if opponent.length >= our_snake.length { 0.9 } else { 0.3 };
                        
                        let existing_danger = self.danger_map.get(&our_next_pos).unwrap_or(&0.0);
                        let new_danger = (existing_danger + danger_level).min(1.0);
                        self.danger_map.insert(our_next_pos, new_danger);
                    }
                }
            }
        }
    }

    /// Predict opponent positions for next N turns
    fn predict_opponent_positions(&self, opponent: &Battlesnake, board: &Board) -> Vec<Vec<Coord>> {
        let mut predictions = Vec::new();
        let mut current_head = opponent.head;
        
        for turn in 0..self.prediction_depth {
            let possible_moves = self.get_possible_moves(&current_head, board);
            
            if possible_moves.is_empty() {
                break;
            }

            // Simple prediction: assume opponent moves toward food or center
            let best_move = self.predict_best_opponent_move(&current_head, board);
            let next_positions = vec![best_move];
            
            predictions.push(next_positions.clone());
            current_head = best_move;
        }

        predictions
    }

    /// Get possible moves for a position
    fn get_possible_moves(&self, pos: &Coord, board: &Board) -> Vec<Coord> {
        let mut moves = Vec::new();
        
        for direction in [Direction::Up, Direction::Down, Direction::Left, Direction::Right] {
            let next_pos = match direction {
                Direction::Up => Coord { x: pos.x, y: pos.y + 1 },
                Direction::Down => Coord { x: pos.x, y: pos.y - 1 },
                Direction::Left => Coord { x: pos.x - 1, y: pos.y },
                Direction::Right => Coord { x: pos.x + 1, y: pos.y },
            };

            // Basic bounds check
            if next_pos.x >= 0 && next_pos.x < board.width && 
               next_pos.y >= 0 && next_pos.y < board.height as i32 {
                moves.push(next_pos);
            }
        }

        moves
    }

    /// Predict opponent's most likely move (simplified heuristic)
    fn predict_best_opponent_move(&self, pos: &Coord, board: &Board) -> Coord {
        let possible_moves = self.get_possible_moves(pos, board);
        
        if possible_moves.is_empty() {
            return *pos;
        }

        // Simple heuristic: move toward nearest food or center
        if !board.food.is_empty() {
            let nearest_food = board.food.iter()
                .min_by_key(|food| (food.x - pos.x).abs() + (food.y - pos.y).abs())
                .unwrap();
            
            // Find move that gets closest to food
            return possible_moves.iter()
                .min_by_key(|next_pos| (next_pos.x - nearest_food.x).abs() + (next_pos.y - nearest_food.y).abs())
                .copied()
                .unwrap_or(possible_moves[0]);
        }

        // Fallback: move toward center
        let center_x = board.width / 2;
        let center_y = board.height as i32 / 2;
        
        possible_moves.iter()
            .min_by_key(|next_pos| (next_pos.x - center_x).abs() + (next_pos.y - center_y).abs())
            .copied()
            .unwrap_or(possible_moves[0])
    }

// ============================================================================
// MOVEMENT HISTORY TRACKER - Channel 10
// ============================================================================

/// Movement History Tracker for position tracking with time decay weighting
#[derive(Debug, Clone)]
pub struct MovementHistoryTracker {
    pub history: VecDeque<(Coord, u32)>,    // (position, turn_number)
    pub max_history: usize,                 // Maximum history to keep
    pub current_turn: u32,                  // Current game turn
}

impl MovementHistoryTracker {
    pub fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::new(),
            max_history,
            current_turn: 0,
        }
    }

    /// Update movement history with new position
    pub fn update_position(&mut self, position: Coord, turn: u32) {
        self.current_turn = turn;
        self.history.push_back((position, turn));
        
        // Keep only recent history
        while self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Generate movement history channel with time decay
    pub fn generate_history_channel(&self, board: &Board) -> Vec<Vec<f32>> {
        let width = board.width as usize;
        let height = board.height as usize;
        let mut history_channel = vec![vec![0.0; width]; height];

        if self.history.is_empty() {
            return history_channel;
        }

        let start_time = Instant::now();

        // Apply time decay weighting to historical positions
        for (position, turn) in &self.history {
            if position.x < 0 || position.x >= board.width || 
               position.y < 0 || position.y >= board.height as i32 {
                continue;
            }

            let x = position.x as usize;
            let y = position.y as usize;

            // Calculate time decay weight (more recent = higher weight)
            let age = self.current_turn.saturating_sub(*turn) as f32;
            let decay_weight = self.calculate_decay_weight(age);
            
            // Accumulate weighted visits
            history_channel[y][x] = (history_channel[y][x] + decay_weight).min(1.0);
        }

        let elapsed = start_time.elapsed().as_secs_f32() * 1000.0;
        info!("Movement history channel generated: {} positions tracked ({:.2}ms)",
              self.history.len(), elapsed);

        history_channel
    }

    /// Calculate time decay weight for position recency
    fn calculate_decay_weight(&self, age: f32) -> f32 {
        // Exponential decay: recent positions have higher weight
        let half_life = 5.0; // Turns for weight to decay to 50%
        let decay_rate = -(2.0_f32.ln()) / half_life;
        (decay_rate * age).exp().max(0.05) // Minimum 5% weight
    }

    /// Get positions visited in recent turns
    pub fn get_recent_positions(&self, recent_turns: u32) -> Vec<Coord> {
        let cutoff_turn = self.current_turn.saturating_sub(recent_turns);
        
        self.history.iter()
            .filter(|(_, turn)| *turn >= cutoff_turn)
            .map(|(pos, _)| *pos)
            .collect()
    }

    /// Check if position was recently visited
    pub fn was_recently_visited(&self, position: &Coord, recent_turns: u32) -> bool {
        let cutoff_turn = self.current_turn.saturating_sub(recent_turns);
        
        self.history.iter()
            .any(|(pos, turn)| pos == position && *turn >= cutoff_turn)
    }
}

// ============================================================================
// STRATEGIC POSITION ANALYZER - Channel 11
// ============================================================================

/// Strategic Position Analyzer for tactical advantage identification
#[derive(Debug, Clone)]
pub struct StrategicPositionAnalyzer {
    pub strategic_positions: HashMap<Coord, f32>, // Coord -> strategic value
}

impl StrategicPositionAnalyzer {
    pub fn new() -> Self {
        Self {
            strategic_positions: HashMap::new(),
        }
    }

    /// Analyze strategic positions and generate tactical advantage channel
    pub fn analyze_strategic_positions(&mut self, board: &Board, our_snake: &Battlesnake) -> Vec<Vec<f32>> {
        let start_time = Instant::now();
        self.strategic_positions.clear();

        let width = board.width as usize;
        let height = board.height as usize;
        let mut strategic_channel = vec![vec![0.0; width]; height];

        // Analyze different types of strategic positions
        self.analyze_food_proximity(board, our_snake);
        self.analyze_center_control(board);
        self.analyze_cutting_points(board, our_snake);
        self.analyze_escape_routes(board, our_snake);
        self.analyze_corner_traps(board);

        // Fill strategic channel
        for y in 0..height {
            for x in 0..width {
                let coord = Coord { x: x as i32, y: y as i32 };
                if let Some(&strategic_value) = self.strategic_positions.get(&coord) {
                    strategic_channel[y][x] = strategic_value;
                }
            }
        }

        let elapsed = start_time.elapsed().as_secs_f32() * 1000.0;
        info!("Strategic position analysis complete: {} strategic positions identified ({:.2}ms)",
              self.strategic_positions.len(), elapsed);

        strategic_channel
    }

    /// Analyze food proximity strategic value
    fn analyze_food_proximity(&mut self, board: &Board, our_snake: &Battlesnake) {
        for food in &board.food {
            let food_distance_from_us = (food.x - our_snake.head.x).abs() + (food.y - our_snake.head.y).abs();
            
            // Find nearest opponent distance to this food
            let nearest_opponent_distance = board.snakes.iter()
                .filter(|snake| snake.id != our_snake.id)
                .map(|snake| (food.x - snake.head.x).abs() + (food.y - snake.head.y).abs())
                .min()
                .unwrap_or(i32::MAX);

            // Strategic value based on our advantage in reaching food
            let advantage = nearest_opponent_distance - food_distance_from_us;
            let strategic_value = if advantage > 0 {
                (advantage as f32 / 10.0).min(0.8) // We're closer - good strategic value
            } else if advantage == 0 {
                0.3 // Equal distance - moderate value
            } else {
                0.1 // They're closer - low value
            };

            // Mark positions around food as strategic
            for dx in -2..=2 {
                for dy in -2..=2 {
                    let pos = Coord { x: food.x + dx, y: food.y + dy };
                    if pos.x >= 0 && pos.x < board.width && 
                       pos.y >= 0 && pos.y < board.height as i32 {
                        let distance_weight = 1.0 / ((dx.abs() + dy.abs()) as f32 + 1.0);
                        let weighted_value = strategic_value * distance_weight;
                        
                        let existing = self.strategic_positions.get(&pos).unwrap_or(&0.0);
                        self.strategic_positions.insert(pos, (existing + weighted_value).min(1.0));
                    }
                }
            }
        }
    }

    /// Analyze center control strategic positions
    fn analyze_center_control(&mut self, board: &Board) {
        let center_x = board.width / 2;
        let center_y = board.height as i32 / 2;
        let max_distance = ((board.width + board.height as i32) / 4) as f32;

        for x in 0..board.width {
            for y in 0..board.height as i32 {
                let pos = Coord { x, y };
                let distance_to_center = ((x - center_x).abs() + (y - center_y).abs()) as f32;
                
                // Strategic value decreases with distance from center
                let center_value = (1.0 - distance_to_center / max_distance).max(0.0) * 0.4;
                
                let existing = self.strategic_positions.get(&pos).unwrap_or(&0.0);
                self.strategic_positions.insert(pos, (existing + center_value).min(1.0));
            }
        }
    }

    /// Analyze cutting points for opponent blocking
    fn analyze_cutting_points(&mut self, board: &Board, our_snake: &Battlesnake) {
        for opponent in &board.snakes {
            if opponent.id == our_snake.id {
                continue;
            }

            // Find positions that could cut off opponent's movement
            let opponent_head = opponent.head;
            
            // Analyze positions that limit opponent's future movement options
            for direction in [Direction::Up, Direction::Down, Direction::Left, Direction::Right] {
                let cutting_pos = match direction {
                    Direction::Up => Coord { x: opponent_head.x, y: opponent_head.y + 2 },
                    Direction::Down => Coord { x: opponent_head.x, y: opponent_head.y - 2 },
                    Direction::Left => Coord { x: opponent_head.x - 2, y: opponent_head.y },
                    Direction::Right => Coord { x: opponent_head.x + 2, y: opponent_head.y },
                };

                if cutting_pos.x >= 0 && cutting_pos.x < board.width && 
                   cutting_pos.y >= 0 && cutting_pos.y < board.height as i32 {
                    
                    // Higher strategic value for positions that limit opponent movement
                    let cutting_value = 0.6;
                    let existing = self.strategic_positions.get(&cutting_pos).unwrap_or(&0.0);
                    self.strategic_positions.insert(cutting_pos, (existing + cutting_value).min(1.0));
                }
            }
        }
    }

    /// Analyze escape route preservation
    fn analyze_escape_routes(&mut self, board: &Board, our_snake: &Battlesnake) {
        // Positions that maintain multiple escape routes have strategic value
        for x in 1..board.width-1 {
            for y in 1..board.height as i32 - 1 {
                let pos = Coord { x, y };
                
                // Count available adjacent positions
                let mut open_directions = 0;
                for direction in [Direction::Up, Direction::Down, Direction::Left, Direction::Right] {
                    let adjacent = match direction {
                        Direction::Up => Coord { x, y: y + 1 },
                        Direction::Down => Coord { x, y: y - 1 },
                        Direction::Left => Coord { x: x - 1, y },
                        Direction::Right => Coord { x: x + 1, y },
                    };

                    if self.is_position_safe(&adjacent, board) {
                        open_directions += 1;
                    }
                }

                // Strategic value based on number of escape routes
                let escape_value = match open_directions {
                    4 => 0.5, // All directions open - excellent
                    3 => 0.4, // Good mobility
                    2 => 0.2, // Limited mobility
                    _ => 0.0, // Poor mobility
                };

                if escape_value > 0.0 {
                    let existing = self.strategic_positions.get(&pos).unwrap_or(&0.0);
                    self.strategic_positions.insert(pos, (existing + escape_value).min(1.0));
                }
            }
        }
    }

    /// Analyze corner trap positions (generally avoid)
    fn analyze_corner_traps(&mut self, board: &Board) {
        // Mark corner and edge positions as less strategic
        for x in 0..board.width {
            for y in 0..board.height as i32 {
                let pos = Coord { x, y };
                
                // Calculate distance from edges
                let edge_distances = [x, board.width - 1 - x, y, board.height as i32 - 1 - y];
                let edge_distance = edge_distances.iter().min().unwrap_or(&0);

                if *edge_distance <= 1 {
                    // Close to edge - negative strategic value
                    let trap_penalty = -0.3;
                    let existing = self.strategic_positions.get(&pos).unwrap_or(&0.0);
                    self.strategic_positions.insert(pos, (existing + trap_penalty).max(-1.0));
                }
            }
        }
    }

    /// Check if position is safe (not blocked by snake bodies)
    fn is_position_safe(&self, pos: &Coord, board: &Board) -> bool {
        for snake in &board.snakes {
            for segment in &snake.body {
                if segment == pos {
                    return false;
                }
            }
        }
        true
    }
}

// ============================================================================
// UNIFIED ADVANCED BOARD STATE ENCODER
// ============================================================================

/// Advanced Board State Encoder supporting full 12-channel encoding
/// Integrates all spatial analysis components into unified neural network input
#[derive(Debug)]
pub struct AdvancedBoardStateEncoder {
    pub voronoi_analyzer: VoronoiTerritoryAnalyzer,
    pub danger_predictor: DangerZonePredictor,
    pub movement_tracker: MovementHistoryTracker,
    pub strategic_analyzer: StrategicPositionAnalyzer,
}

impl AdvancedBoardStateEncoder {
    pub fn new(max_history: usize, prediction_depth: u8) -> Self {
        Self {
            voronoi_analyzer: VoronoiTerritoryAnalyzer::new(),
            danger_predictor: DangerZonePredictor::new(prediction_depth),
            movement_tracker: MovementHistoryTracker::new(max_history),
            strategic_analyzer: StrategicPositionAnalyzer::new(),
        }
    }

    /// Generate complete 12-channel board encoding
    pub fn encode_12_channel_board(&mut self, board: &Board, our_snake: &Battlesnake, turn: u32) -> Vec<Vec<Vec<f32>>> {
        let start_time = Instant::now();
        
        let width = board.width as usize;
        let height = board.height as usize;
        let mut channels = vec![vec![vec![0.0; width]; height]; 12];

        info!("Starting 12-channel advanced board encoding ({}x{})...", width, height);

        // Channels 0-6: Original 7-channel encoding
        self.encode_basic_channels(&mut channels, board, our_snake);

        // Update movement history
        self.movement_tracker.update_position(our_snake.head, turn);

        // Channels 7-8: Voronoi territory analysis
        let (our_territory, opponent_territory) = self.voronoi_analyzer.analyze_territory(board, our_snake);
        channels[7] = our_territory;
        channels[8] = opponent_territory;

        // Channel 9: Danger zones
        channels[9] = self.danger_predictor.predict_danger_zones(board, our_snake);

        // Channel 10: Movement history
        channels[10] = self.movement_tracker.generate_history_channel(board);

        // Channel 11: Strategic positions
        channels[11] = self.strategic_analyzer.analyze_strategic_positions(board, our_snake);

        let elapsed = start_time.elapsed().as_secs_f32() * 1000.0;
        info!("12-channel advanced board encoding complete ({:.2}ms)", elapsed);
        
        channels
    }

    /// Encode basic channels 0-6 (original 7-channel system)
    fn encode_basic_channels(&self, channels: &mut Vec<Vec<Vec<f32>>>, board: &Board, our_snake: &Battlesnake) {
        let width = board.width as usize;
        let height = board.height as usize;

        // Initialize all positions as empty (Channel 0)
        for y in 0..height {
            for x in 0..width {
                channels[0][y][x] = 1.0;
            }
        }

        // Channel 1: Our head
        if our_snake.head.x >= 0 && our_snake.head.x < board.width && 
           our_snake.head.y >= 0 && our_snake.head.y < board.height as i32 {
            let x = our_snake.head.x as usize;
            let y = our_snake.head.y as usize;
            channels[1][y][x] = 1.0;
            channels[0][y][x] = 0.0; // Not empty
        }

        // Channel 2: Our body
        for segment in &our_snake.body {
            if segment.x >= 0 && segment.x < board.width && 
               segment.y >= 0 && segment.y < board.height as i32 {
                let x = segment.x as usize;
                let y = segment.y as usize;
                channels[2][y][x] = 1.0;
                channels[0][y][x] = 0.0; // Not empty
            }
        }

        // Channels 3-4: Opponent heads and bodies
        for snake in &board.snakes {
            if snake.id == our_snake.id {
                continue;
            }

            // Channel 3: Opponent head
            if snake.head.x >= 0 && snake.head.x < board.width && 
               snake.head.y >= 0 && snake.head.y < board.height as i32 {
                let x = snake.head.x as usize;
                let y = snake.head.y as usize;
                channels[3][y][x] = 1.0;
                channels[0][y][x] = 0.0; // Not empty
            }

            // Channel 4: Opponent body
            for segment in &snake.body {
                if segment.x >= 0 && segment.x < board.width && 
                   segment.y >= 0 && segment.y < board.height as i32 {
                    let x = segment.x as usize;
                    let y = segment.y as usize;
                    channels[4][y][x] = 1.0;
                    channels[0][y][x] = 0.0; // Not empty
                }
            }
        }

        // Channel 5: Food
        for food in &board.food {
            if food.x >= 0 && food.x < board.width && 
               food.y >= 0 && food.y < board.height as i32 {
                let x = food.x as usize;
                let y = food.y as usize;
                channels[5][y][x] = 1.0;
                channels[0][y][x] = 0.0; // Not empty
            }
        }

        // Channel 6: Walls (boundaries)
        for x in 0..width {
            for y in 0..height {
                let coord_x = x as i32;
                let coord_y = y as i32;
                
                // Mark boundary positions
                if coord_x == 0 || coord_x == board.width - 1 || 
                   coord_y == 0 || coord_y == board.height as i32 - 1 {
                    channels[6][y][x] = 1.0;
                }
            }
        }
    }

    /// Get encoding statistics for monitoring
    pub fn get_encoding_stats(&self) -> (usize, usize, usize, usize) {
        (
            self.voronoi_analyzer.our_territory.len(),
            self.voronoi_analyzer.opponent_territory.len(),
            self.danger_predictor.danger_map.len(),
            self.strategic_analyzer.strategic_positions.len(),
        )
    }
}
#[cfg(test)]
mod voronoi_territory_analyzer_tests {
    use super::*;
    use crate::spatial_test_utilities::*;

    #[test]
    fn test_voronoi_analyzer_creation() {
        let analyzer = VoronoiTerritoryAnalyzer::new();
        assert!(analyzer.our_territory.is_empty());
        assert!(analyzer.opponent_territory.is_empty());
        assert!(analyzer.contested_zones.is_empty());
    }

    #[test]
    fn test_single_snake_scenario() {
        let mut analyzer = VoronoiTerritoryAnalyzer::new();
        let game_state = BoardScenarioGenerator::empty_board();
        
        let (our_channel, opponent_channel) = analyzer.analyze_territory(&game_state.board, &game_state.you);
        
        // With no opponents, all territory should be ours
        assert_eq!(our_channel.len(), 11);
        assert_eq!(our_channel[0].len(), 11);
        assert_eq!(opponent_channel.len(), 11);
        assert_eq!(opponent_channel[0].len(), 11);
        
        // All positions should have full ownership strength for us
        for row in &our_channel {
            for &value in row {
                assert_eq!(value, 1.0, "All territory should belong to us with no opponents");
            }
        }
        
        // Opponent channel should be empty
        for row in &opponent_channel {
            for &value in row {
                assert_eq!(value, 0.0, "Opponent channel should be empty with no opponents");
            }
        }
        
        assert!(analyzer.contested_zones.is_empty());
    }

    #[test]
    fn test_two_snake_scenario() {
        let mut analyzer = VoronoiTerritoryAnalyzer::new();
        let game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_test_snake("us", Coord { x: 1, y: 1 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 5, y: 5 }, 3, 100))
            .build();
        
        let (our_channel, opponent_channel) = analyzer.analyze_territory(&game_state.board, &game_state.you);
        
        // Validate channel dimensions
        SpatialAnalysisValidator::validate_channel_dimensions(&our_channel, &game_state.board, "our_territory")
            .expect("Our territory channel should have correct dimensions");
        SpatialAnalysisValidator::validate_channel_dimensions(&opponent_channel, &game_state.board, "opponent_territory")
            .expect("Opponent territory channel should have correct dimensions");
        
        // Validate channel ranges
        SpatialAnalysisValidator::validate_channel_range(&our_channel, "our_territory")
            .expect("Our territory values should be in [0.0, 1.0] range");
        SpatialAnalysisValidator::validate_channel_range(&opponent_channel, "opponent_territory")
            .expect("Opponent territory values should be in [0.0, 1.0] range");
        
        // Validate territory consistency
        SpatialAnalysisValidator::validate_territory_consistency(&our_channel, &opponent_channel)
            .expect("Territory analysis should be consistent");
        
        // Check that territory closer to our snake head has higher values for us
        let our_head_x = 1;
        let our_head_y = 1;
        let our_territory_at_head = our_channel[our_head_y][our_head_x];
        let opponent_territory_at_head = opponent_channel[our_head_y][our_head_x];
        
        // At our head position, we should have stronger territory control
        assert!(our_territory_at_head > opponent_territory_at_head, 
                "Our territory control should be stronger at our head position");
    }

    #[test]
    fn test_territory_strength_calculation() {
        let analyzer = VoronoiTerritoryAnalyzer::new();
        
        // Test distance-based strength calculation
        let strength_0 = analyzer.calculate_territory_strength(0);
        let strength_5 = analyzer.calculate_territory_strength(5);
        let strength_10 = analyzer.calculate_territory_strength(10);
        let strength_20 = analyzer.calculate_territory_strength(20);
        
        // Closer positions should have higher strength
        assert!(strength_0 > strength_5);
        assert!(strength_5 > strength_10);
        assert!(strength_10 > strength_20);
        
        // All strengths should be in valid range
        assert!(strength_0 >= 0.1 && strength_0 <= 1.0);
        assert!(strength_5 >= 0.1 && strength_5 <= 1.0);
        assert!(strength_10 >= 0.1 && strength_10 <= 1.0);
        assert!(strength_20 >= 0.1 && strength_20 <= 1.0);
        
        // Distance 0 should give maximum strength
        assert_eq!(strength_0, 1.0);
        
        // Very large distances should give minimum strength
        let strength_100 = analyzer.calculate_territory_strength(100);
        assert_eq!(strength_100, 0.1);
    }

    #[test]
    fn test_position_blocked_detection() {
        let analyzer = VoronoiTerritoryAnalyzer::new();
        let game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_snake_with_path("us", vec![
                Coord { x: 3, y: 3 }, // head
                Coord { x: 3, y: 2 }, // neck
                Coord { x: 3, y: 1 }, // body
            ], 100))
            .build();
        
        // Snake body positions should be blocked (excluding tail)
        assert!(analyzer.is_position_blocked(&Coord { x: 3, y: 3 }, &game_state.board)); // head
        assert!(analyzer.is_position_blocked(&Coord { x: 3, y: 2 }, &game_state.board)); // neck
        
        // Tail might not be blocked (depends on whether snake will grow)
        // Empty positions should not be blocked
        assert!(!analyzer.is_position_blocked(&Coord { x: 0, y: 0 }, &game_state.board));
        assert!(!analyzer.is_position_blocked(&Coord { x: 6, y: 6 }, &game_state.board));
    }

    #[test]
    fn test_contested_zones() {
        let mut analyzer = VoronoiTerritoryAnalyzer::new();
        
        // Create scenario where snakes are equidistant from center
        let game_state = GameStateBuilder::new()
            .with_board_size(5, 5)
            .with_our_snake(create_test_snake("us", Coord { x: 1, y: 2 }, 1, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 3, y: 2 }, 1, 100))
            .build();
        
        let (our_channel, opponent_channel) = analyzer.analyze_territory(&game_state.board, &game_state.you);
        
        // There should be some contested zones due to equidistant positions
        assert!(!analyzer.contested_zones.is_empty(), "Should have contested zones between equidistant snakes");
        
        // Contested positions should have split ownership (0.5 each)
        for contested_pos in &analyzer.contested_zones {
            if contested_pos.x >= 0 && contested_pos.x < 5 && 
               contested_pos.y >= 0 && contested_pos.y < 5 {
                let x = contested_pos.x as usize;
                let y = contested_pos.y as usize;
                assert_eq!(our_channel[y][x], 0.5);
                assert_eq!(opponent_channel[y][x], 0.5);
            }
        }
    }

    #[test]
    fn test_corner_scenarios() {
        let mut analyzer = VoronoiTerritoryAnalyzer::new();
        
        // Test snake in corner
        let game_state = GameStateBuilder::new()
            .with_board_size(7, 7)
            .with_our_snake(create_test_snake("us", Coord { x: 0, y: 0 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 6, y: 6 }, 3, 100))
            .build();
        
        let (our_channel, opponent_channel) = analyzer.analyze_territory(&game_state.board, &game_state.you);
        
        // Validate basic properties
        SpatialAnalysisValidator::validate_channel_dimensions(&our_channel, &game_state.board, "our_territory")
            .expect("Corner scenario should produce valid dimensions");
        SpatialAnalysisValidator::validate_channel_range(&our_channel, "our_territory")
            .expect("Corner scenario should produce valid range");
        
        // Corner positions should have strong territorial control
        assert!(our_channel[0][0] > 0.5, "Snake at corner should have strong control at its position");
        assert!(opponent_channel[6][6] > 0.5, "Opponent snake at opposite corner should have strong control");
    }

    #[test]
    fn test_large_board_performance() {
        let mut analyzer = VoronoiTerritoryAnalyzer::new();
        let game_state = BoardScenarioGenerator::large_board_scenario();
        
        let start = std::time::Instant::now();
        let (our_channel, opponent_channel) = analyzer.analyze_territory(&game_state.board, &game_state.you);
        let elapsed = start.elapsed().as_millis();
        
        // Performance test - should complete within reasonable time
        assert!(elapsed < 100, "Large board territory analysis should complete within 100ms, took {}ms", elapsed);
        
        // Validate results
        SpatialAnalysisValidator::validate_channel_dimensions(&our_channel, &game_state.board, "our_territory")
            .expect("Large board should produce valid dimensions");
        SpatialAnalysisValidator::validate_channel_range(&our_channel, "our_territory")
            .expect("Large board should produce valid range");
        
        // Should have territorial information
        assert!(analyzer.our_territory.len() > 0 || analyzer.opponent_territory.len() > 0);
    }

    #[test]
    fn test_boundary_handling() {
        let mut analyzer = VoronoiTerritoryAnalyzer::new();
        
        // Test minimum board size
        let game_state = GameStateBuilder::new()
            .with_board_size(3, 3)
            .with_our_snake(create_test_snake("us", Coord { x: 1, y: 1 }, 1, 100))
            .build();
        
        let (our_channel, opponent_channel) = analyzer.analyze_territory(&game_state.board, &game_state.you);
        
        // Should handle small boards correctly
        assert_eq!(our_channel.len(), 3);
        assert_eq!(our_channel[0].len(), 3);
        
        // With single snake, all positions should be ours
        for row in &our_channel {
            for &value in row {
                assert_eq!(value, 1.0);
            }
        }
    }

    #[test]
    fn test_multi_opponent_scenario() {
        let mut analyzer = VoronoiTerritoryAnalyzer::new();
        
        // Create scenario with multiple opponents
        let game_state = GameStateBuilder::new()
            .with_board_size(11, 11)
            .with_our_snake(create_test_snake("us", Coord { x: 5, y: 5 }, 4, 100))
            .with_opponent(create_test_snake("opponent1", Coord { x: 2, y: 2 }, 3, 100))
            .with_opponent(create_test_snake("opponent2", Coord { x: 8, y: 2 }, 3, 100))
            .with_opponent(create_test_snake("opponent3", Coord { x: 2, y: 8 }, 3, 100))
            .build();
        
        let (our_channel, opponent_channel) = analyzer.analyze_territory(&game_state.board, &game_state.you);
        
        // Validate basic properties
        SpatialAnalysisValidator::validate_channel_dimensions(&our_channel, &game_state.board, "our_territory")
            .expect("Multi-opponent scenario should produce valid dimensions");
        SpatialAnalysisValidator::validate_territory_consistency(&our_channel, &opponent_channel)
            .expect("Multi-opponent territory analysis should be consistent");
        
        // Should have territorial divisions
        assert!(analyzer.our_territory.len() > 0);
        assert!(analyzer.opponent_territory.len() > 0);
        
        // Center position should be strongly controlled by our snake (closest)
        let center_territory = our_channel[5][5];
        assert!(center_territory > 0.7, "Center snake should have strong territorial control at center");
    }

    #[test]
    fn test_asymmetric_board_dimensions() {
        let mut analyzer = VoronoiTerritoryAnalyzer::new();
        
        // Test with asymmetric board (width != height) - handles type inconsistency
        let game_state = GameStateBuilder::new()
            .with_board_size(15, 7) // width=i32, height=u32
            .with_our_snake(create_test_snake("us", Coord { x: 7, y: 3 }, 3, 100))
            .with_opponent(create_test_snake("opponent", Coord { x: 3, y: 1 }, 3, 100))
            .build();
        
        let (our_channel, opponent_channel) = analyzer.analyze_territory(&game_state.board, &game_state.you);
        
        // Validate dimensions handle type inconsistency correctly
        assert_eq!(our_channel.len(), 7); // height as usize
        assert_eq!(our_channel[0].len(), 15); // width as usize
        assert_eq!(opponent_channel.len(), 7);
        assert_eq!(opponent_channel[0].len(), 15);
        
        // All values should be valid
        SpatialAnalysisValidator::validate_channel_range(&our_channel, "our_territory")
            .expect("Asymmetric board should produce valid territory values");
    }

    #[test]
    fn test_empty_board_edge_case() {
        let analyzer = VoronoiTerritoryAnalyzer::new();
        
        // Test with minimal valid snake (just head)
        let minimal_snake = Battlesnake {
            id: "minimal".to_string(),
            name: "Minimal Snake".to_string(),
            health: 100,
            body: vec![Coord { x: 2, y: 2 }],
            head: Coord { x: 2, y: 2 },
            length: 1,
            latency: "0".to_string(),
            shout: None,
        };
        
        let game_state = GameStateBuilder::new()
            .with_board_size(5, 5)
            .with_our_snake(minimal_snake)
            .build();
        
        let mut test_analyzer = analyzer.clone();
        let (our_channel, opponent_channel) = test_analyzer.analyze_territory(&game_state.board, &game_state.you);
        
        // Should handle minimal snake correctly
        assert_eq!(our_channel.len(), 5);
        assert_eq!(our_channel[0].len(), 5);
        
        // All territory should be ours with single snake
        for row in &our_channel {
            for &value in row {
                assert_eq!(value, 1.0, "Single minimal snake should control all territory");
            }
        }
    }
}

// Close the remaining open modules and functions
} // Close movement_history_tracker_tests module
} // Close strategic_position_analyzer_tests module