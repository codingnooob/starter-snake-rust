use log::info;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use crate::{Battlesnake, Board, Coord, Game};

// Battlesnake Info
// This function is called when you register your Battlesnake on play.battlesnake.com
pub fn info() -> Value {
    info!("INFO");

    json!({
        "apiversion": "1",
        "author": "starter-snake-rust",
        "color": "#888888",
        "head": "default",
        "tail": "default",
    })
}

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    pub fn all() -> [Direction; 4] {
        [Direction::Up, Direction::Down, Direction::Left, Direction::Right]
    }
    
    pub fn apply_to_coord(&self, coord: &Coord) -> Coord {
        match self {
            Direction::Up => Coord { x: coord.x, y: coord.y + 1 },
            Direction::Down => Coord { x: coord.x, y: coord.y - 1 },
            Direction::Left => Coord { x: coord.x - 1, y: coord.y },
            Direction::Right => Coord { x: coord.x + 1, y: coord.y },
        }
    }
}

impl Coord {
    pub fn apply_direction(&self, direction: &Direction) -> Coord {
        direction.apply_to_coord(self)
    }
}

// Called every turn to make a move decision  
pub fn start(_game: &Game, _turn: &i32, _board: &Board, _you: &Battlesnake) {
    info!("GAME START");
}

pub fn end(_game: &Game, _turn: &i32, _board: &Board, _you: &Battlesnake) {
    info!("GAME OVER");
}

// Safety Checker
pub struct SafetyChecker;

impl SafetyChecker {
    pub fn is_safe_coordinate(coord: &Coord, board: &Board, snakes: &[Battlesnake]) -> bool {
        // Board boundary check
        if coord.x < 0 || coord.x >= board.width || coord.y < 0 || coord.y >= (board.height as i32) {
            return false;
        }

        // Check collision with any snake body (including our own)
        for snake in snakes {
            // For all snakes, check body segments except the tail (which will move next turn)
            let segments_to_check = if snake.body.len() > 1 {
                &snake.body[0..snake.body.len()-1] // Exclude tail
            } else {
                &snake.body // Single segment snake
            };
            
            for segment in segments_to_check {
                if segment == coord {
                    return false;
                }
            }
        }
        
        true
    }

    pub fn calculate_safe_moves(you: &Battlesnake, board: &Board, snakes: &[Battlesnake]) -> Vec<Direction> {
        Direction::all()
            .iter()
            .filter(|direction| {
                let next_coord = you.head.apply_direction(direction);
                Self::is_safe_coordinate(&next_coord, board, snakes)
            })
            .copied()
            .collect()
    }

    pub fn avoid_backward_move(you: &Battlesnake, safe_moves: Vec<Direction>) -> Vec<Direction> {
        if you.body.len() >= 2 {
            let neck = &you.body[1];
            safe_moves.into_iter()
                .filter(|direction| {
                    let next_head = you.head.apply_direction(direction);
                    next_head != *neck
                })
                .collect()
        } else {
            safe_moves
        }
    }
}

// Pathfinding with A* algorithm
pub struct PathFinder;

impl PathFinder {
    pub fn a_star(start: &Coord, goal: &Coord, board: &Board, snakes: &[Battlesnake]) -> Option<Vec<Coord>> {
        let mut open_set = VecDeque::new();
        let mut came_from = HashMap::new();
        let mut g_score = HashMap::new();
        let mut f_score = HashMap::new();

        open_set.push_back(*start);
        g_score.insert(*start, 0);
        f_score.insert(*start, Self::heuristic(start, goal));

        while let Some(current) = open_set.pop_front() {
            if current == *goal {
                return Some(Self::reconstruct_path(&came_from, &current));
            }

            for neighbor in get_neighbors(&current, board) {
                if !SafetyChecker::is_safe_coordinate(&neighbor, board, snakes) {
                    continue;
                }

                let tentative_g_score = g_score.get(&current).unwrap_or(&i32::MAX) + 1;

                if tentative_g_score < *g_score.get(&neighbor).unwrap_or(&i32::MAX) {
                    came_from.insert(neighbor, current);
                    g_score.insert(neighbor, tentative_g_score);
                    f_score.insert(neighbor, tentative_g_score + Self::heuristic(&neighbor, goal));

                    if !open_set.contains(&neighbor) {
                        open_set.push_back(neighbor);
                    }
                }
            }
        }

        None
    }

    fn heuristic(a: &Coord, b: &Coord) -> i32 {
        (a.x - b.x).abs() + (a.y - b.y).abs()
    }

    fn reconstruct_path(came_from: &HashMap<Coord, Coord>, start: &Coord) -> Vec<Coord> {
        let mut path = vec![*start];
        let mut current = *start;
        while let Some(prev) = came_from.get(&current) {
            path.push(*prev);
            current = *prev;
        }
        path.reverse();
        path
    }
}

pub fn get_neighbors(coord: &Coord, board: &Board) -> Vec<Coord> {
    Direction::all()
        .iter()
        .map(|direction| coord.apply_direction(direction))
        .filter(|neighbor| {
            neighbor.x >= 0 && neighbor.x < board.width && 
            neighbor.y >= 0 && neighbor.y < (board.height as i32)
        })
        .collect()
}

// Food targeting and prioritization
#[derive(Debug, Clone, Copy)]
pub struct FoodTarget {
    pub coord: Coord,
    pub distance: i32,
    pub priority: f32,
}

pub struct FoodSeeker;

impl FoodSeeker {
    pub fn should_seek_food(health: i32, turn: i32, food_available: bool) -> bool {
        if !food_available {
            return false;
        }
        
        // Aggressive food seeking when health is low
        if health <= 30 {
            return true;
        }
        
        // Moderate food seeking in mid-game
        if health <= 60 && turn > 10 {
            return true;
        }
        
        // Conservative food seeking in early game or when healthy
        health <= 80 && turn > 50
    }

    pub fn find_best_food_target(head: &Coord, board: &Board, health: i32, aggressive_mode: bool) -> Option<FoodTarget> {
        if board.food.is_empty() {
            return None;
        }

        let mut targets: Vec<FoodTarget> = board.food.iter()
            .map(|food_coord| {
                let distance = PathFinder::heuristic(head, food_coord);
                let priority = Self::calculate_food_priority(distance, health, aggressive_mode);
                FoodTarget {
                    coord: *food_coord,
                    distance,
                    priority,
                }
            })
            .collect();

        targets.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());
        targets.first().copied()
    }

    fn calculate_food_priority(distance: i32, health: i32, aggressive_mode: bool) -> f32 {
        let base_priority = match distance {
            0..=2 => 10.0,
            3..=5 => 8.0,
            6..=10 => 5.0,
            _ => 2.0,
        };

        let health_multiplier = match health {
            0..=20 => if aggressive_mode { 3.0 } else { 2.5 },
            21..=40 => 2.0,
            41..=60 => 1.5,
            _ => 1.0,
        };

        let distance_penalty = (distance as f32) * 0.1;
        (base_priority * health_multiplier) - distance_penalty
    }
}

// BFS for reachability analysis
pub struct ReachabilityAnalyzer;

impl ReachabilityAnalyzer {
    pub fn count_reachable_spaces(start: &Coord, board: &Board, snakes: &[Battlesnake]) -> i32 {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        queue.push_back(*start);
        visited.insert(*start);
        
        while let Some(current) = queue.pop_front() {
            for neighbor in get_neighbors(&current, board) {
                if !visited.contains(&neighbor) && 
                   SafetyChecker::is_safe_coordinate(&neighbor, board, snakes) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
        
        visited.len() as i32
    }
}

// Territory Control Systems for Phase 1C

#[derive(Debug, Clone)]
pub struct TerritoryInfo {
    pub controller_id: Option<String>,
    pub distance_to_controller: i32,
    pub is_contested: bool,
}

#[derive(Debug)]
pub struct TerritoryMap {
    pub territories: HashMap<Coord, TerritoryInfo>,
    pub control_scores: HashMap<String, f32>,
}

pub struct SpaceController;

impl SpaceController {
    pub fn calculate_territory_map(board: &Board, snakes: &[Battlesnake]) -> TerritoryMap {
        let mut territories = HashMap::new();
        let mut control_scores = HashMap::new();
        
        // Initialize control scores for all snakes
        for snake in snakes {
            control_scores.insert(snake.id.clone(), 0.0);
        }
        
        // Multi-source BFS from all snake heads simultaneously (Voronoi-style)
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        
        // Start BFS from all snake heads
        for snake in snakes {
            let territory_info = TerritoryInfo {
                controller_id: Some(snake.id.clone()),
                distance_to_controller: 0,
                is_contested: false,
            };
            territories.insert(snake.head, territory_info);
            visited.insert(snake.head);
            queue.push_back((snake.head, snake.id.clone(), 0));
        }
        
        // BFS expansion
        while let Some((current, controller_id, distance)) = queue.pop_front() {
            for neighbor in get_neighbors(&current, board) {
                if SafetyChecker::is_safe_coordinate(&neighbor, board, snakes) && !visited.contains(&neighbor) {
                    let territory_info = TerritoryInfo {
                        controller_id: Some(controller_id.clone()),
                        distance_to_controller: distance + 1,
                        is_contested: false,
                    };
                    territories.insert(neighbor, territory_info);
                    visited.insert(neighbor);
                    queue.push_back((neighbor, controller_id.clone(), distance + 1));
                    
                    // Update control score
                    let current_score = control_scores.get(&controller_id).unwrap_or(&0.0);
                    control_scores.insert(controller_id.clone(), current_score + 1.0);
                }
            }
        }
        
        TerritoryMap {
            territories,
            control_scores,
        }
    }
    
    pub fn get_area_control_score(&self, _from: &Coord, to: &Coord, board: &Board, 
                                 snakes: &[Battlesnake], snake_id: &str) -> f32 {
        let territory_map = Self::calculate_territory_map(board, snakes);
        
        // Base score from territory control
        let base_score = territory_map.control_scores.get(snake_id).unwrap_or(&0.0) / 10.0;
        
        // Bonus for moving into contested areas
        let position_bonus = if let Some(territory) = territory_map.territories.get(to) {
            if territory.is_contested || territory.controller_id.as_ref() != Some(&snake_id.to_string()) {
                2.0 // Bonus for expanding into new territory
            } else {
                0.0
            }
        } else {
            1.0 // Neutral territory
        };
        
        // Consider space accessibility from new position
        let reachable_spaces = ReachabilityAnalyzer::count_reachable_spaces(to, board, snakes) as f32;
        let space_score = reachable_spaces / 20.0; // Normalize
        
        base_score + position_bonus + space_score
    }
}

// Opponent Movement Prediction
pub struct OpponentAnalyzer;

impl OpponentAnalyzer {
    pub fn predict_opponent_moves(opponent: &Battlesnake, board: &Board, 
                                 all_snakes: &[Battlesnake]) -> HashMap<Direction, f32> {
        let mut move_probabilities = HashMap::new();
        
        // Initialize all moves with 0 probability
        for direction in Direction::all() {
            move_probabilities.insert(direction, 0.0);
        }
        
        // Calculate safe moves for opponent
        let safe_moves = SafetyChecker::calculate_safe_moves(opponent, board, all_snakes);
        if safe_moves.is_empty() {
            return move_probabilities;
        }
        
        // Analyze each potential move
        for &direction in &safe_moves {
            let next_pos = opponent.head.apply_direction(&direction);
            
            // Base probability for safe moves
            let base_probability = 1.0 / safe_moves.len() as f32;
            
            // Food seeking behavior prediction
            let food_attraction = if !board.food.is_empty() {
                let nearest_food_distance = board.food.iter()
                    .map(|food| PathFinder::heuristic(&next_pos, food))
                    .min().unwrap_or(i32::MAX);
                
                let current_nearest_distance = board.food.iter()
                    .map(|food| PathFinder::heuristic(&opponent.head, food))
                    .min().unwrap_or(i32::MAX);
                
                if nearest_food_distance < current_nearest_distance && opponent.health < 70 {
                    0.3 // Bonus for food seeking
                } else {
                    0.0
                }
            } else {
                0.0
            };
            
            // Space control behavior
            let open_neighbors = get_neighbors(&next_pos, board)
                .into_iter()
                .filter(|coord| SafetyChecker::is_safe_coordinate(coord, board, all_snakes))
                .count();
            
            let space_preference = (open_neighbors as f32) * 0.1;
            
            // Center preference (snakes often prefer central positions)
            let center_x = board.width as f32 / 2.0;
            let center_y = board.height as f32 / 2.0;
            let distance_to_center = ((next_pos.x as f32 - center_x).powi(2) + 
                                     (next_pos.y as f32 - center_y).powi(2)).sqrt();
            let center_preference = 0.1 / (1.0 + distance_to_center * 0.1);
            
            // Combine all factors
            let total_probability = base_probability + food_attraction + space_preference + center_preference;
            move_probabilities.insert(direction, total_probability.max(0.0));
        }
        
        // Normalize probabilities
        let total: f32 = move_probabilities.values().sum();
        if total > 0.0 {
            for probability in move_probabilities.values_mut() {
                *probability /= total;
            }
        }
        
        move_probabilities
    }
    
    pub fn identify_cutting_positions(our_head: &Coord, opponent: &Battlesnake, 
                                    board: &Board) -> Vec<Coord> {
        let mut cutting_positions = Vec::new();
        
        // Find positions that could cut off opponent's escape routes
        for direction in Direction::all() {
            let potential_pos = our_head.apply_direction(&direction);
            if SafetyChecker::is_safe_coordinate(&potential_pos, board, &[]) {
                // Check if this position would limit opponent's movement options
                let opponent_neighbors = get_neighbors(&opponent.head, board);
                let would_block = opponent_neighbors.iter()
                    .any(|coord| *coord == potential_pos);
                
                if would_block {
                    cutting_positions.push(potential_pos);
                }
            }
        }
        
        cutting_positions
    }
}

// Integrated Territorial Strategy System
pub struct TerritorialStrategist {
    space_controller: SpaceController,
    opponent_analyzer: OpponentAnalyzer,
}

impl TerritorialStrategist {
    pub fn new() -> Self {
        Self {
            space_controller: SpaceController,
            opponent_analyzer: OpponentAnalyzer,
        }
    }
    
    pub fn make_territorial_decision(&self, _game: &Game, turn: &i32, board: &Board, 
                                   you: &Battlesnake) -> Value {
        info!("MOVE {}: Territorial Strategy Analysis", turn);
        
        let all_snakes: Vec<Battlesnake> = board.snakes.iter().cloned().collect();
        let _territory_map = SpaceController::calculate_territory_map(board, &all_snakes);
        
        let mut move_scores = HashMap::new();
        
        // Evaluate each possible move
        for direction in Direction::all() {
            let next_pos = you.head.apply_direction(&direction);
            
            // Calculate territorial control score
            let territorial_score = self.space_controller.get_area_control_score(
                &you.head, &next_pos, board, &all_snakes, &you.id);
            
            // Food seeking integration (only when needed)
            let food_score = if FoodSeeker::should_seek_food(you.health, *turn, !board.food.is_empty()) {
                if let Some(target) = FoodSeeker::find_best_food_target(
                    &you.head, board, you.health, true) {
                    let food_direction = Self::get_direction_to_target(&you.head, &target.coord);
                    if food_direction == direction { target.priority } else { 0.0 }
                } else { 0.0 }
            } else { 0.0 };
            
            // Opponent cutting and area denial
            let cutting_score = OpponentAnalyzer::identify_cutting_positions(
                &you.head, you, board).len() as f32 * 0.5;
            let area_denial_score = self.space_controller.get_area_control_score(
                &you.head, &next_pos, board, &all_snakes, &you.id);
            
            // Combine all scoring factors
            let total_score = territorial_score + food_score * 0.7 + cutting_score + area_denial_score * 0.3;
            move_scores.insert(direction, total_score);
        }
        
        // Choose move with highest score, but ensure it's safe
        let safe_moves = SafetyChecker::calculate_safe_moves(you, board, &all_snakes);
        let safe_moves = SafetyChecker::avoid_backward_move(you, safe_moves);
        
        let chosen_direction = if let Some(best_direction) = safe_moves.iter()
            .max_by(|&&a, &&b| {
                let score_a = move_scores.get(&a).unwrap_or(&0.0);
                let score_b = move_scores.get(&b).unwrap_or(&0.0);
                score_a.partial_cmp(&score_b).unwrap()
            }) {
            *best_direction
        } else {
            // Emergency fallback
            if safe_moves.is_empty() {
                Direction::Up
            } else {
                use rand::Rng;
                safe_moves[rand::rng().random_range(0..safe_moves.len())]
            }
        };

        info!("MOVE {}: Chosen {} (Territorial Score: {:.2})", 
              turn, 
              format!("{:?}", chosen_direction).to_lowercase(),
              move_scores.get(&chosen_direction).unwrap_or(&0.0),
        );

        json!({ "move": format!("{:?}", chosen_direction).to_lowercase() })
    }
    
    fn get_direction_to_target(from: &Coord, to: &Coord) -> Direction {
        let dx = to.x - from.x;
        let dy = to.y - from.y;
        
        if dx.abs() > dy.abs() {
            if dx > 0 { Direction::Right } else { Direction::Left }
        } else {
            if dy > 0 { Direction::Up } else { Direction::Down }
        }
    }
}

// Legacy hybrid decision maker for comparison
pub struct HybridDecisionMaker;

impl HybridDecisionMaker {
    pub fn make_move_decision(_game: &Game, turn: &i32, board: &Board, you: &Battlesnake) -> Value {
        let all_snakes: Vec<Battlesnake> = board.snakes.iter().cloned().collect();
        let safe_moves = SafetyChecker::calculate_safe_moves(you, board, &all_snakes);

        if safe_moves.is_empty() {
            info!("MOVE {}: No safe moves available! Attempting Up", turn);
            return json!({ "move": "up" });
        }

        let safe_moves = SafetyChecker::avoid_backward_move(you, safe_moves);

        if safe_moves.is_empty() {
            info!("MOVE {}: Only backward move available! Attempting Up", turn);
            return json!({ "move": "up" });
        }

        // Food seeking logic
        if FoodSeeker::should_seek_food(you.health, *turn, !board.food.is_empty()) {
            if let Some(target) = FoodSeeker::find_best_food_target(&you.head, board, you.health, true) {
                if let Some(path) = PathFinder::a_star(&you.head, &target.coord, board, &all_snakes) {
                    if path.len() > 1 {
                        let next_step = path[1];
                        for direction in &safe_moves {
                            let next_coord = you.head.apply_direction(direction);
                            if next_coord == next_step {
                                info!("MOVE {}: Food seeking to ({}, {}) via {:?}", 
                                      turn, target.coord.x, target.coord.y, direction);
                                return json!({ "move": format!("{:?}", direction).to_lowercase() });
                            }
                        }
                    }
                }
            }
        }

        // Space evaluation fallback
        let mut best_move = safe_moves[0];
        let mut best_space_count = 0;

        for direction in &safe_moves {
            let next_coord = you.head.apply_direction(direction);
            let space_count = ReachabilityAnalyzer::count_reachable_spaces(&next_coord, board, &all_snakes);
            
            if space_count > best_space_count {
                best_space_count = space_count;
                best_move = *direction;
            }
        }

        info!("MOVE {}: Space-based decision {:?} (reachable: {})", 
              turn, best_move, best_space_count);

        json!({ "move": format!("{:?}", best_move).to_lowercase() })
    }
}

// ============================================================================
// PHASE 2A: MINIMAX SEARCH ENGINE WITH GAME STATE SIMULATION
// ============================================================================

// Game State Simulation Engine for Multi-ply Lookahead with Opponent Modeling Support
#[derive(Debug, Clone)]
pub struct SimulatedGameState {
    pub board_width: i32,
    pub board_height: u32,
    pub food: Vec<Coord>,
    pub snakes: Vec<SimulatedSnake>,
    pub turn: i32,
}

#[derive(Debug, Clone)]
pub struct SimulatedSnake {
    pub id: String,
    pub health: i32,
    pub body: Vec<Coord>,
    pub is_alive: bool,
}

impl SimulatedSnake {
    fn head(&self) -> Coord {
        self.body[0]
    }
    
    fn neck(&self) -> Option<Coord> {
        if self.body.len() >= 2 {
            Some(self.body[1])
        } else {
            None
        }
    }
}

// Move Application and Undo System
#[derive(Debug, Clone)]
pub struct MoveApplication {
    pub snake_moves: Vec<(String, Direction)>, // snake_id -> move
    pub food_consumed: Vec<(String, Coord)>,   // snake_id -> food_coord
    pub collisions: Vec<String>,               // dead snake ids
    pub previous_tails: Vec<(String, Coord)>,  // snake_id -> old_tail
}

pub struct GameSimulator;

impl GameSimulator {
    // Generate all possible moves for all snakes
    pub fn generate_all_moves(state: &SimulatedGameState) -> Vec<Vec<Direction>> {
        let mut all_combinations = Vec::new();
        let alive_snakes: Vec<&SimulatedSnake> = state.snakes.iter()
            .filter(|snake| snake.is_alive)
            .collect();
        
        if alive_snakes.is_empty() {
            return all_combinations;
        }
        
        // Generate moves for each snake
        let snake_moves: Vec<Vec<Direction>> = alive_snakes.iter()
            .map(|snake| Self::generate_moves_for_snake(snake, state))
            .collect();
        
        // Generate all combinations (Cartesian product)
        Self::cartesian_product(&snake_moves, &mut all_combinations, Vec::new(), 0);
        all_combinations
    }
    
    // Generate moves for a specific snake
    pub fn generate_moves_for_snake(snake: &SimulatedSnake, state: &SimulatedGameState) -> Vec<Direction> {
        let mut valid_moves = Vec::new();
        
        // DEBUG: Log snake info for debugging
        info!("GameSim DEBUG: Generating moves for snake {} at position {:?}, neck: {:?}",
              snake.id, snake.head(), snake.neck());
        
        for direction in Direction::all() {
            let next_head = direction.apply_to_coord(&snake.head());
            
            // Basic boundary check
            if next_head.x < 0 || next_head.x >= state.board_width ||
               next_head.y < 0 || next_head.y >= (state.board_height as i32) {
                info!("GameSim DEBUG: Move {:?} rejected - out of bounds ({}, {})", direction, next_head.x, next_head.y);
                continue;
            }
            
            // Avoid immediate backward move
            if let Some(neck) = snake.neck() {
                if next_head == neck {
                    info!("GameSim DEBUG: Move {:?} rejected - would move backward to neck at {:?}", direction, neck);
                    continue;
                }
            }
            
            info!("GameSim DEBUG: Move {:?} accepted - leads to {:?}", direction, next_head);
            valid_moves.push(direction);
        }
        
        // IMPROVED: If no valid moves, try to find ANY safe move (less strict validation)
        if valid_moves.is_empty() {
            info!("GameSim DEBUG: No safe moves found, trying fallback approach");
            
            // Try moves that might be slightly unsafe but better than staying still
            for direction in Direction::all() {
                let next_head = direction.apply_to_coord(&snake.head());
                
                // Only check basic boundaries
                if next_head.x >= 0 && next_head.x < state.board_width &&
                   next_head.y >= 0 && next_head.y < (state.board_height as i32) {
                    valid_moves.push(direction);
                    info!("GameSim DEBUG: Added fallback move {:?} (boundary-only check)", direction);
                }
            }
            
            // Last resort: choose random direction to avoid bias
            if valid_moves.is_empty() {
                use rand::Rng;
                let mut rng = rand::rng();
                let directions = Direction::all();
                let random_idx = rng.random_range(0..directions.len());
                let random_direction = directions[random_idx];
                valid_moves.push(random_direction);
                info!("GameSim DEBUG: No boundary-safe moves, chose random direction: {:?}", random_direction);
            }
        }
        
        info!("GameSim DEBUG: Final move list for snake {}: {:?}", snake.id, valid_moves);
        valid_moves
    }
    
    // Apply moves to game state and return undo information
    pub fn apply_moves(
        state: &mut SimulatedGameState,
        moves: &[Direction]
    ) -> MoveApplication {
        let mut move_app = MoveApplication {
            snake_moves: Vec::new(),
            food_consumed: Vec::new(),
            collisions: Vec::new(),
            previous_tails: Vec::new(),
        };
        
        let alive_snakes: Vec<String> = state.snakes.iter()
            .filter(|snake| snake.is_alive)
            .map(|snake| snake.id.clone())
            .collect();
        
        if moves.len() != alive_snakes.len() {
            return move_app; // Invalid move count
        }
        
        // Step 1: Move snake heads and store previous tails
        for (i, snake_id) in alive_snakes.iter().enumerate() {
            let direction = moves[i];
            move_app.snake_moves.push((snake_id.clone(), direction));
            
            if let Some(snake) = state.snakes.iter_mut().find(|s| &s.id == snake_id) {
                if !snake.is_alive {
                    continue;
                }
                
                // Store previous tail for undo
                if !snake.body.is_empty() {
                    let tail = snake.body[snake.body.len() - 1];
                    move_app.previous_tails.push((snake_id.clone(), tail));
                }
                
                // Move head
                let new_head = direction.apply_to_coord(&snake.head());
                snake.body.insert(0, new_head);
                
                // Decrease health
                snake.health -= 1;
                if snake.health <= 0 {
                    snake.is_alive = false;
                    move_app.collisions.push(snake_id.clone());
                }
            }
        }
        
        // Step 2: Check food consumption
        for snake in &mut state.snakes {
            if !snake.is_alive {
                continue;
            }
            
            let head = snake.head();
            if let Some(food_index) = state.food.iter().position(|&f| f == head) {
                // Food consumed - grow snake and restore health
                let food_coord = state.food.remove(food_index);
                move_app.food_consumed.push((snake.id.clone(), food_coord));
                
                snake.health = 100; // Restore to full health
                // Body already grown by not removing tail below
            } else {
                // No food - remove tail
                if !snake.body.is_empty() {
                    snake.body.pop();
                }
            }
        }
        
        // Step 3: Check collisions
        Self::check_collisions(state, &mut move_app);
        
        // Step 4: Advance turn
        state.turn += 1;
        
        move_app
    }
    
    // Check for collisions and update snake states
    fn check_collisions(state: &mut SimulatedGameState, move_app: &mut MoveApplication) {
        let mut collision_positions = HashMap::new();
        
        // Collect all head positions
        for snake in &state.snakes {
            if snake.is_alive {
                let head = snake.head();
                collision_positions.entry(head)
                    .or_insert_with(Vec::new)
                    .push(snake.id.clone());
            }
        }
        
        // Find head-to-head collisions
        for (_, snake_ids) in collision_positions {
            if snake_ids.len() > 1 {
                // Multiple snakes at same position - kill all
                for snake_id in snake_ids {
                    if let Some(snake) = state.snakes.iter_mut().find(|s| s.id == snake_id) {
                        snake.is_alive = false;
                        move_app.collisions.push(snake_id.clone());
                    }
                }
            }
        }
        
        // Check for snakes moving into other snakes' bodies
        // First collect all potential collision data to avoid borrowing issues
        let mut killed_snakes = Vec::new();
        
        // Collect immutable references first
        let alive_snake_refs: Vec<_> = state.snakes.iter().enumerate()
            .filter(|(_, s)| s.is_alive)
            .collect();
        
        // Find snakes that would collide
        for (i, (_, snake)) in alive_snake_refs.iter().enumerate() {
            let head = snake.head();
            let snake_id_clone = snake.id.clone();
            
            // Check against other snakes' bodies
            for (j, (_, other_snake)) in alive_snake_refs.iter().enumerate() {
                if i == j {
                    continue;
                }
                
                // Check against all body segments except the tail (which will move)
                let segments_to_check = if other_snake.body.len() > 1 {
                    &other_snake.body[0..other_snake.body.len()-1]
                } else {
                    &other_snake.body
                };
                
                for segment in segments_to_check {
                    if head == *segment {
                        killed_snakes.push(snake_id_clone.clone());
                        break;
                    }
                }
                
                // Early exit if this snake is already marked for death
                if killed_snakes.contains(&snake_id_clone) {
                    break;
                }
            }
        }
        
        // Apply collisions
        for snake_id in killed_snakes {
            if let Some(snake) = state.snakes.iter_mut().find(|s| s.id == snake_id) {
                snake.is_alive = false;
                move_app.collisions.push(snake.id.clone());
            }
        }
    }
    
    // Undo moves to restore previous state
    pub fn undo_moves(state: &mut SimulatedGameState, move_app: &MoveApplication) {
        // Revert turn
        state.turn -= 1;
        
        // Restore snakes' bodies and states
        for (snake_id, _) in &move_app.snake_moves {
            if let Some(snake) = state.snakes.iter_mut().find(|s| &s.id == snake_id) {
                // Remove head
                if !snake.body.is_empty() {
                    snake.body.remove(0);
                }
                
                // Restore tail if it was moved
                if let Some((_, old_tail)) = move_app.previous_tails.iter().find(|(id, _)| id == snake_id) {
                    snake.body.push(*old_tail);
                }
                
                // Restore health and alive status (this is simplified)
                snake.health += 1;
                snake.is_alive = true;
            }
        }
        
        // Restore food
        for (_snake_id, food_coord) in &move_app.food_consumed {
            state.food.push(*food_coord);
        }
    }
    
    // Check if the game state is terminal (game over)
    pub fn is_terminal(state: &SimulatedGameState) -> bool {
        let alive_snakes: Vec<&SimulatedSnake> = state.snakes.iter()
            .filter(|snake| snake.is_alive)
            .collect();
        
        alive_snakes.len() <= 1 // Game over if 0 or 1 snakes alive
    }
    
    // Get our snake from the game state
    pub fn get_our_snake<'a>(state: &'a SimulatedGameState, our_snake_id: &str) -> Option<&'a SimulatedSnake> {
        state.snakes.iter().find(|snake| snake.id == our_snake_id && snake.is_alive)
    }
    
    // Convert API game state to simulation state
    pub fn from_game_state(_game: &Game, _board: &Board, _you: &Battlesnake) -> SimulatedGameState {
        let simulated_snakes: Vec<SimulatedSnake> = _board.snakes.iter()
            .map(|snake| SimulatedSnake {
                id: snake.id.clone(),
                health: snake.health,
                body: snake.body.clone(),
                is_alive: true,
            })
            .collect();
        
        SimulatedGameState {
            board_width: _board.width,
            board_height: _board.height,
            food: _board.food.clone(),
            snakes: simulated_snakes,
            turn: 0,
        }
    }
    
    // Cartesian product helper
    fn cartesian_product(
        snake_moves: &[Vec<Direction>],
        result: &mut Vec<Vec<Direction>>,
        current: Vec<Direction>,
        depth: usize,
    ) {
        if depth == snake_moves.len() {
            result.push(current);
            return;
        }
        
        for &direction in &snake_moves[depth] {
            let mut new_current = current.clone();
            new_current.push(direction);
            Self::cartesian_product(snake_moves, result, new_current, depth + 1);
        }
    }
}

// ============================================================================
// PHASE 2A: POSITION EVALUATION INTEGRATION
// ============================================================================

// Position evaluator trait for unified evaluation interface
pub trait PositionEvaluator {
    fn evaluate_position(&self, state: &SimulatedGameState, our_snake_id: &str) -> f32;
}

// Integrated position evaluator combining all Phase 1 systems
#[derive(Debug, Clone)]
pub struct IntegratedEvaluator {
    safety_weight: f32,
    space_weight: f32,
    food_weight: f32,
    territory_weight: f32,
}

impl IntegratedEvaluator {
    pub fn new() -> Self {
        Self {
            safety_weight: 10.0,
            space_weight: 1.0,
            food_weight: 5.0,
            territory_weight: 2.0,
        }
    }
    
    fn to_api_board(&self, state: &SimulatedGameState) -> Board {
        Board {
            width: state.board_width,
            height: state.board_height,
            food: state.food.clone(),
            snakes: state.snakes.iter()
                .map(|sim_snake| self.to_api_snake(sim_snake))
                .collect(),
            hazards: Vec::new(),
        }
    }
    
    fn to_api_snake(&self, sim_snake: &SimulatedSnake) -> Battlesnake {
        Battlesnake {
            id: sim_snake.id.clone(),
            name: format!("Snake_{}", sim_snake.id),
            health: sim_snake.health,
            body: sim_snake.body.clone(),
            head: if !sim_snake.body.is_empty() { sim_snake.body[0] } else { Coord { x: 0, y: 0 } },
            length: sim_snake.body.len() as i32,
            latency: "0".to_string(),
            shout: None,
        }
    }
}

impl PositionEvaluator for IntegratedEvaluator {
    fn evaluate_position(&self, state: &SimulatedGameState, our_snake_id: &str) -> f32 {
        let our_snake = match GameSimulator::get_our_snake(state, our_snake_id) {
            Some(snake) => snake,
            None => return -10000.0, // We're dead
        };
        
        if !our_snake.is_alive {
            return -10000.0; // Death penalty
        }
        
        // Convert to API format for existing evaluators
        let board = self.to_api_board(state);
        let our_battlesnake = self.to_api_snake(our_snake);
        let all_snakes: Vec<Battlesnake> = board.snakes.clone();
        
        let mut total_score = 0.0;
        
        // Safety evaluation - are we in immediate danger?
        let safe_moves = SafetyChecker::calculate_safe_moves(&our_battlesnake, &board, &all_snakes);
        if safe_moves.is_empty() {
            total_score -= self.safety_weight; // High penalty for being trapped
        }
        
        // Space evaluation - how much reachable space do we have?
        let reachable_space = ReachabilityAnalyzer::count_reachable_spaces(&our_snake.head(), &board, &all_snakes);
        total_score += (reachable_space as f32) * self.space_weight;
        
        // Food evaluation - health and food proximity
        total_score += (our_snake.health as f32) * 2.0; // Health value
        
        if FoodSeeker::should_seek_food(our_snake.health, state.turn, !board.food.is_empty()) {
            if let Some(food_target) = FoodSeeker::find_best_food_target(
                &our_snake.head(), &board, our_snake.health, true
            ) {
                // Closer food is better
                let food_score = self.food_weight / (food_target.distance as f32 + 1.0);
                total_score += food_score;
            }
        }
        
        // Territory evaluation using Phase 1C systems
        let territory_map = SpaceController::calculate_territory_map(&board, &all_snakes);
        if let Some(control_score) = territory_map.control_scores.get(our_snake_id) {
            total_score += control_score * self.territory_weight;
        }
        
        // Relative evaluation - how are we doing compared to other snakes?
        let alive_opponents: Vec<&SimulatedSnake> = state.snakes.iter()
            .filter(|s| s.is_alive && s.id != our_snake_id)
            .collect();
        
        if !alive_opponents.is_empty() {
            let avg_opponent_health: f32 = alive_opponents.iter()
                .map(|s| s.health as f32)
                .sum::<f32>() / alive_opponents.len() as f32;
            
            let avg_opponent_length: f32 = alive_opponents.iter()
                .map(|s| s.body.len() as f32)
                .sum::<f32>() / alive_opponents.len() as f32;
            
            // Bonus for being healthier/longer than average
            total_score += (our_snake.health as f32 - avg_opponent_health) * 5.0;
            total_score += (our_snake.body.len() as f32 - avg_opponent_length) * 10.0;
        }
        
        // Win condition bonus
        if GameSimulator::is_terminal(state) && our_snake.is_alive {
            total_score += 5000.0; // Big bonus for winning
        }
        
        total_score
    }
}

// ============================================================================
// PHASE 2B: MONTE CARLO TREE SEARCH (MCTS) IMPLEMENTATION
// ============================================================================

/// MCTS Node structure for Monte Carlo Tree Search
#[derive(Debug, Clone)]
pub struct MCTSNode {
    pub state: SimulatedGameState,
    pub parent: Option<Box<MCTSNode>>,
    pub children: Vec<Box<MCTSNode>>,
    pub visits: u64,
    pub total_value: f32,
    pub move_used: Option<Direction>,
    pub untried_moves: Vec<Direction>,
    pub depth: u8,
    pub is_terminal: bool,
}

impl MCTSNode {
    /// Create a new root node from game state
    pub fn new_root(state: SimulatedGameState, our_snake_id: &str) -> Self {
        let mut root = Self {
            state: state.clone(),
            parent: None,
            children: Vec::new(),
            visits: 0,
            total_value: 0.0,
            move_used: None,
            untried_moves: Vec::new(),
            depth: 0,
            is_terminal: GameSimulator::is_terminal(&state),
        };
        
        // CRITICAL FIX: Ensure root node gets valid untried moves for expansion
        // Find our specific snake by ID and initialize its moves immediately
        let our_snake = root.state.snakes.iter()
            .find(|snake| snake.is_alive && snake.id == our_snake_id)
            .cloned(); // Clone to avoid borrowing issues
        
        if let Some(snake) = our_snake {
            // Generate moves directly for the root node
            let valid_moves = GameSimulator::generate_moves_for_snake(&snake, &root.state);
            root.untried_moves = valid_moves;
            
            info!("MCTS DEBUG: Root node initialized with {} untried moves for snake '{}': {:?}",
                  root.untried_moves.len(), our_snake_id, root.untried_moves);
        } else {
            info!("MCTS DEBUG: Root node - our snake '{}' not found among alive snakes", our_snake_id);
            info!("MCTS DEBUG: Available snake IDs: {:?}",
                  root.state.snakes.iter().map(|s| &s.id).collect::<Vec<_>>());
            root.untried_moves = Vec::new();
        }
        
        root
    }

    /// Create a child node from parent with a specific move
    pub fn new_child(parent: &MCTSNode, move_direction: Direction) -> Self {
        let mut child_state = parent.state.clone();
        let _move_app = GameSimulator::apply_moves(&mut child_state, &[move_direction]);
        
        Self {
            state: child_state.clone(),
            parent: None, // Will be set by caller
            children: Vec::new(),
            visits: 0,
            total_value: 0.0,
            move_used: Some(move_direction),
            untried_moves: Vec::new(),
            depth: parent.depth + 1,
            is_terminal: GameSimulator::is_terminal(&child_state),
        }
    }

    /// Check if node has been expanded (has children or no untried moves)
    pub fn is_expanded(&self) -> bool {
        self.children.len() > 0 || self.untried_moves.is_empty()
    }

    /// Check if node is fully explored
    pub fn is_fully_explored(&self) -> bool {
        self.is_terminal || self.untried_moves.is_empty()
    }

    /// Get the average value of this node
    pub fn average_value(&self) -> f32 {
        if self.visits > 0 {
            self.total_value / self.visits as f32
        } else {
            0.0
        }
    }

    /// Get parent visits for UCB1 calculation
    pub fn parent_visits(&self) -> u64 {
        self.parent.as_ref()
            .map(|p| p.visits)
            .unwrap_or(1) // Avoid division by zero
    }
}

/// Enhanced MCTS Search Engine with Performance Optimization and Memory Management
pub struct MCTSSearcher {
    pub evaluator: IntegratedEvaluator,
    pub max_iterations: u32,
    pub time_limit_ms: u128,
    pub exploration_constant: f32, // C parameter for UCB1
    pub max_depth: u8,
    pub nodes_created: u64,
    pub start_time: Instant,
    pub our_snake_id: String,
    
    // Performance optimization features
    pub max_memory_nodes: usize,      // Memory limit for tree size
    pub prune_depth_threshold: u8,    // Depth beyond which to prune aggressively
    pub early_termination_threshold: f32, // Early exit confidence threshold
    pub transposition_table: Option<HashMap<u64, f32>>, // Simple transposition table
}

impl MCTSSearcher {
    /// Create a new MCTS searcher with performance optimizations
    pub fn new(max_iterations: u32, time_limit_ms: u128, our_snake_id: &str) -> Self {
        Self {
            evaluator: IntegratedEvaluator::new(),
            max_iterations,
            time_limit_ms,
            exploration_constant: (2.0f32).sqrt(), // Standard exploration constant
            max_depth: 15, // Reasonable depth limit
            nodes_created: 0,
            start_time: Instant::now(),
            our_snake_id: our_snake_id.to_string(),
            
            // Performance optimization features
            max_memory_nodes: 10000,        // Reasonable memory limit
            prune_depth_threshold: 12,      // Aggressive pruning beyond this depth
            early_termination_threshold: 0.9, // High confidence early exit
            transposition_table: Some(HashMap::new()), // Simple position cache
        }
    }
    
    /// Create MCTS searcher with custom performance configuration
    pub fn with_performance_config(
        max_iterations: u32,
        time_limit_ms: u128,
        our_snake_id: &str,
        max_memory_nodes: usize,
        prune_depth_threshold: u8,
        early_termination_threshold: f32
    ) -> Self {
        Self {
            evaluator: IntegratedEvaluator::new(),
            max_iterations,
            time_limit_ms,
            exploration_constant: (2.0f32).sqrt(),
            max_depth: 15,
            nodes_created: 0,
            start_time: Instant::now(),
            our_snake_id: our_snake_id.to_string(),
            
            max_memory_nodes,
            prune_depth_threshold,
            early_termination_threshold,
            transposition_table: Some(HashMap::new()),
        }
    }

    /// Main MCTS search function - performs iterations until time/iteration limit
    pub fn search(&mut self, initial_state: &mut SimulatedGameState) -> MCTSResult {
        self.nodes_created = 0;
        self.start_time = Instant::now();
        
        info!("MCTS SEARCH: Starting search for snake {} (max iterations: {}, time limit: {}ms)", 
              self.our_snake_id, self.max_iterations, self.time_limit_ms);

        // Create root node
        let mut root = MCTSNode::new_root(initial_state.clone(), &self.our_snake_id);
        let mut iterations_completed = 0;
        
        // Pre-compute initial untried moves for root
        self.initialize_untried_moves(&mut root);
        
        // MCTS Main Loop - simplified approach to avoid borrowing conflicts
        for iteration in 0..self.max_iterations {
            if self.is_time_up() {
                info!("MCTS SEARCH: Stopping at iteration {} due to time constraints", iteration);
                break;
            }

            // Debug every 10 iterations to track progress
            if iteration % 10 == 0 {
                info!("MCTS DEBUG: Iteration {}/{}, root children: {}, nodes created: {}",
                      iteration, self.max_iterations, root.children.len(), self.nodes_created);
            }

            // Clone the tree state for safe access
            let mut path = self.select_node_path(&root);
            
            // Expansion phase - add new child node if possible
            if !self.node_at_path(&root, &path).is_fully_explored() {
                let new_path = self.expand_node_at_path(&mut root, &path);
                path = new_path;
                if iteration < 5 { // Log first few expansions for debugging
                    info!("MCTS DEBUG: Iteration {}, expanded node, new path: {:?}", iteration, path);
                }
            } else {
                if iteration < 5 { // Log first few selections
                    info!("MCTS DEBUG: Iteration {}, selected node at path {:?}, fully explored: {}",
                          iteration, path, self.node_at_path(&root, &path).is_fully_explored());
                }
            }
            
            // Simulation phase - random rollout from expanded node
            let simulation_result = {
                let expanded_node = self.node_at_path(&root, &path);
                self.simulate_random_rollout(expanded_node)
            };
            
            // Backpropagation phase - update values up the tree
            self.backpropagate_at_path(&mut root, &path, simulation_result);
            
            iterations_completed = iteration + 1;
        }

        let time_taken = self.start_time.elapsed().as_millis();
        
        // Select best move from root
        let best_move = self.select_best_move(&root);
        let root_value = root.average_value();
        
        info!("MCTS SEARCH: Completed {} iterations in {}ms. Best move: {:?}, Average value: {:.2}, Nodes created: {}", 
              iterations_completed, time_taken, best_move, root_value, self.nodes_created);

        MCTSResult {
            best_move,
            evaluation: root_value,
            iterations_completed,
            time_taken_ms: time_taken,
            nodes_created: self.nodes_created,
            exploration_constant: self.exploration_constant,
        }
    }

    /// Get path to node with best UCB1 score
    fn select_node_path(&self, root: &MCTSNode) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current = root;
        
        // DEBUG: Log initial state
        info!("MCTS DEBUG: Select - root terminal: {}, expanded: {}, children: {}",
              current.is_terminal, current.is_expanded(), current.children.len());
        
        // Traverse until we find a node that needs expansion (not terminal and not fully expanded)
        while !current.is_terminal && !current.is_fully_explored() {
            info!("MCTS DEBUG: Select - current node terminal: {}, fully_explored: {}, children: {}, untried_moves: {}",
                  current.is_terminal, current.is_fully_explored(), current.children.len(), current.untried_moves.len());
            
            // If we have children, traverse to the best one using UCB1
            if !current.children.is_empty() {
                // Find best child using UCB1
                let mut best_child_idx = 0;
                let mut best_ucb1_score = f32::NEG_INFINITY;
                
                for (i, child) in current.children.iter().enumerate() {
                    let ucb1_score = self.calculate_ucb1(child);
                    info!("MCTS DEBUG: Child {} UCB1: {:.2}, visits: {}, value: {:.2}",
                          i, ucb1_score, child.visits, child.average_value());
                    if ucb1_score > best_ucb1_score {
                        best_ucb1_score = ucb1_score;
                        best_child_idx = i;
                    }
                }
                
                info!("MCTS DEBUG: Selected child {} with best UCB1: {:.2}", best_child_idx, best_ucb1_score);
                path.push(best_child_idx);
                current = &current.children[best_child_idx];
            } else {
                // No children yet, this is where we should expand
                info!("MCTS DEBUG: No children, found expansion point with {} untried moves", current.untried_moves.len());
                break;
            }
        }
        
        info!("MCTS DEBUG: Final path selection: {:?}", path);
        path
    }

    /// Get node at specific path
    fn node_at_path<'a>(&'a self, root: &'a MCTSNode, path: &[usize]) -> &'a MCTSNode {
        let mut current = root;
        for &idx in path {
            current = &current.children[idx];
        }
        current
    }

    /// Get mutable node at specific path
    fn node_at_path_mut<'a>(&'a mut self, root: &'a mut MCTSNode, path: &[usize]) -> &'a mut MCTSNode {
        let mut current = root;
        for &idx in path {
            current = &mut current.children[idx];
        }
        current
    }

    /// Calculate UCB1 score for a node
    fn calculate_ucb1(&self, node: &MCTSNode) -> f32 {
        if node.visits == 0 {
            // Unvisited nodes get maximum priority
            return f32::INFINITY;
        }
        
        let exploitation = node.average_value();
        let exploration = self.exploration_constant * 
            ((node.parent_visits() as f32).ln() / node.visits as f32).sqrt();
        
        exploitation + exploration
    }

    /// Expansion phase: add a new child node for an untried move
    fn expand_node_at_path(&mut self, root: &mut MCTSNode, path: &[usize]) -> Vec<usize> {
        let mut parent_path = path.to_vec();
        
        info!("MCTS DEBUG: Expanding node at path {:?}", parent_path);
        
        // Get current node for validation
        let current_node = self.node_at_path(root, &parent_path);
        if current_node.untried_moves.is_empty() {
            info!("MCTS DEBUG: No untried moves available for expansion");
            return parent_path;
        }
        
        if current_node.is_terminal {
            info!("MCTS DEBUG: Cannot expand terminal node");
            return parent_path;
        }
        
        // Get references before any mutability conflicts
        let (move_index, selected_move) = {
            let node = self.node_at_path(root, &parent_path);
            
            // Select a random untried move
            use rand::Rng;
            let mut rng = rand::rng();
            let move_idx = rng.random_range(0..node.untried_moves.len());
            let selected_mv = node.untried_moves[move_idx];
            info!("MCTS DEBUG: Selected move {:?} at index {} from {} available moves", selected_mv, move_idx, node.untried_moves.len());
            (move_idx, selected_mv)
        };
        
        // Now modify the node safely - remove the selected move
        {
            let node = self.node_at_path_mut(root, &parent_path);
            let removed_move = node.untried_moves.remove(move_index);
            info!("MCTS DEBUG: Removed move {:?} from untried moves, remaining: {:?}", removed_move, node.untried_moves);
        }
        
        // Create new child node by cloning state and applying the move
        let child_state = {
            let parent_node = self.node_at_path(root, &parent_path);
            let mut new_state = parent_node.state.clone();
            
            // Apply the selected move to create child state
            info!("MCTS DEBUG: Applying move {:?} to create child state", selected_move);
            let _move_app = GameSimulator::apply_moves(&mut new_state, &[selected_move]);
            
            new_state
        };
        
        let child_depth = self.node_at_path(root, &parent_path).depth + 1;
        let child_is_terminal = GameSimulator::is_terminal(&child_state);
        
        let mut child = MCTSNode {
            state: child_state.clone(),
            parent: None, // Will be set by caller
            children: Vec::new(),
            visits: 0,
            total_value: 0.0,
            move_used: Some(selected_move),
            untried_moves: Vec::new(),
            depth: child_depth,
            is_terminal: child_is_terminal,
        };
        
        self.nodes_created += 1;
        info!("MCTS DEBUG: Created child node #{}, depth: {}, terminal: {}", self.nodes_created, child_depth, child_is_terminal);
        
        // Initialize untried moves for child
        self.initialize_untried_moves(&mut child);
        
        // Set parent reference
        {
            let parent_node = self.node_at_path(root, &parent_path);
            child.parent = Some(Box::new(parent_node.clone()));
            info!("MCTS DEBUG: Set parent reference for child");
        }
        
        // Add child to parent's children and get new path
        {
            let parent = self.node_at_path_mut(root, &parent_path);
            let child_index = parent.children.len();
            parent.children.push(Box::new(child));
            parent_path.push(child_index);
            info!("MCTS DEBUG: Added child at index {}, parent now has {} children", child_index, parent.children.len());
        }
        
        info!("MCTS DEBUG: Expansion complete, new path: {:?}", parent_path);
        parent_path
    }

    /// Initialize untried moves for a node
    fn initialize_untried_moves(&self, node: &mut MCTSNode) {
        info!("MCTS DEBUG: Initializing untried moves for node at depth {}, terminal: {}", node.depth, node.is_terminal);
        
        if node.is_terminal {
            info!("MCTS DEBUG: Node is terminal, no moves to initialize");
            return;
        }
        
        if node.is_expanded() {
            info!("MCTS DEBUG: Node already expanded, no moves to initialize");
            return;
        }
        
        // Get alive snakes and generate moves
        let alive_snakes: Vec<&SimulatedSnake> = node.state.snakes.iter()
            .filter(|snake| snake.is_alive)
            .collect();
        
        info!("MCTS DEBUG: Found {} alive snakes", alive_snakes.len());
        
        if alive_snakes.is_empty() {
            info!("MCTS DEBUG: No alive snakes, setting empty moves");
            node.untried_moves = Vec::new();
            return;
        }
        
        // Find our snake's index
        let our_snake_index = alive_snakes.iter()
            .position(|snake| snake.id == self.our_snake_id);
        
        if let Some(our_idx) = our_snake_index {
            info!("MCTS DEBUG: Found our snake at index {}, generating moves", our_idx);
            // Generate moves for our snake
            let our_moves = GameSimulator::generate_moves_for_snake(alive_snakes[our_idx], &node.state);
            info!("MCTS DEBUG: Generated {} moves for our snake: {:?}", our_moves.len(), our_moves);
            node.untried_moves = our_moves;
        } else {
            // Our snake not found - no moves available
            info!("MCTS DEBUG: Our snake not found in alive snakes, setting empty moves");
            info!("MCTS DEBUG: Looking for snake ID: '{}', available IDs: {:?}",
                  self.our_snake_id,
                  alive_snakes.iter().map(|s| &s.id).collect::<Vec<_>>());
            node.untried_moves = Vec::new();
        }
    }

    /// Simulation phase: perform random rollout from expanded node
    fn simulate_random_rollout(&self, node: &MCTSNode) -> f32 {
        let mut current_state = node.state.clone();
        let mut depth = node.depth;
        
        // If terminal, return evaluation immediately
        if node.is_terminal {
            return self.evaluator.evaluate_position(&current_state, &self.our_snake_id);
        }
        
        // Perform random playout
        while depth < self.max_depth && !GameSimulator::is_terminal(&current_state) {
            // Generate all possible moves
            let all_moves = GameSimulator::generate_all_moves(&current_state);
            if all_moves.is_empty() {
                break; // No moves available
            }
            
            // Select random move combination
            use rand::Rng;
            let mut rng = rand::rng();
            let move_index = rng.random_range(0..all_moves.len());
            let selected_moves = &all_moves[move_index];
            
            // Apply moves
            let _move_app = GameSimulator::apply_moves(&mut current_state, selected_moves);
            depth += 1;
        }
        
        // Evaluate final position
        self.evaluator.evaluate_position(&current_state, &self.our_snake_id)
    }

    /// Backpropagation phase: update values up the tree
    fn backpropagate_at_path(&mut self, root: &mut MCTSNode, path: &[usize], simulation_result: f32) {
        // Navigate to leaf node and backpropagate
        let mut current = self.node_at_path_mut(root, path);
        current.visits += 1;
        current.total_value += simulation_result;
        
        // Propagate up the tree
        let mut current_path = path.to_vec();
        while !current_path.is_empty() {
            current_path.pop(); // Remove last index to go to parent
            if !current_path.is_empty() {
                let parent = self.node_at_path_mut(root, &current_path);
                parent.visits += 1;
                parent.total_value += simulation_result;
            }
        }
    }

    /// Select best move based on visit counts and average values
    fn select_best_move(&self, root: &MCTSNode) -> Direction {
        if root.children.is_empty() {
            // FALLBACK: Choose random move from untried moves to avoid hardcoded bias
            if !root.untried_moves.is_empty() {
                use rand::Rng;
                let mut rng = rand::rng();
                let move_idx = rng.random_range(0..root.untried_moves.len());
                let selected_move = root.untried_moves[move_idx];
                info!("MCTS DEBUG: No children, selecting random from untried moves: {:?}", selected_move);
                return selected_move;
            } else {
                // Last resort: choose random direction to break bias
                use rand::Rng;
                let mut rng = rand::rng();
                let directions = Direction::all();
                let random_idx = rng.random_range(0..directions.len());
                let random_direction = directions[random_idx];
                info!("MCTS DEBUG: No children or untried moves, selecting random direction: {:?}", random_direction);
                return random_direction;
            }
        }
        
        // Choose child with highest average value, break ties with visit count
        let best_child = root.children.iter()
            .max_by(|a, b| {
                let a_score = (a.average_value(), a.visits);
                let b_score = (b.average_value(), b.visits);
                a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
            });
        
        if let Some(child) = best_child {
            if let Some(move_used) = child.move_used {
                info!("MCTS DEBUG: Selected best child with move: {:?}, value: {:.2}, visits: {}",
                      move_used, child.average_value(), child.visits);
                move_used
            } else {
                // Should not happen, but handle gracefully
                use rand::Rng;
                let mut rng = rand::rng();
                let directions = Direction::all();
                let random_idx = rng.random_range(0..directions.len());
                info!("MCTS DEBUG: Child had no move_used, selecting random direction");
                directions[random_idx]
            }
        } else {
            // Should not happen with non-empty children, but handle gracefully
            use rand::Rng;
            let mut rng = rand::rng();
            let directions = Direction::all();
            let random_idx = rng.random_range(0..directions.len());
            info!("MCTS DEBUG: No best child found, selecting random direction");
            directions[random_idx]
        }
    }

    /// Check if search time limit has been exceeded
    fn is_time_up(&self) -> bool {
        self.start_time.elapsed().as_millis() >= self.time_limit_ms
    }

    /// Get enhanced search statistics with performance metrics
    pub fn get_stats(&self) -> MCTSSearchStats {
        let time_elapsed = self.start_time.elapsed().as_millis();
        let mut stats = MCTSSearchStats {
            nodes_created: self.nodes_created,
            total_visits: 0, // Would need tree traversal to calculate
            time_elapsed_ms: time_elapsed,
            nodes_per_ms: 0.0,
            average_depth_reached: 0.0,
            terminal_nodes_visited: 0,
            expansion_success_rate: 0.0,
            move_diversity_score: 0.0,
            search_efficiency_score: 0.0,
            memory_usage_estimate: 0,
            unique_positions_explored: self.nodes_created,
        };
        
        stats.calculate_efficiency_metrics(self.max_iterations);
        stats
    }

    /// Check if we should terminate early based on confidence
    fn should_terminate_early(&self, root: &MCTSNode) -> bool {
        if root.children.is_empty() {
            return false;
        }
        
        // Check if we have a clearly dominant move
        let best_child = root.children.iter()
            .max_by(|a, b| {
                let a_score = (a.average_value(), a.visits);
                let b_score = (b.average_value(), b.visits);
                a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
            });
        
        if let Some(child) = best_child {
            let confidence = child.average_value().abs();
            let visit_ratio = child.visits as f32 / root.children.iter().map(|c| c.visits).sum::<u64>() as f32;
            
            // Early termination if we have high confidence and sufficient exploration
            confidence > self.early_termination_threshold && visit_ratio > 0.6
        } else {
            false
        }
    }

    /// Memory management: check if we should prune tree
    fn should_prune_tree(&self) -> bool {
        self.nodes_created >= self.max_memory_nodes as u64
    }

    /// Simple hash function for state-based transposition table
    fn hash_state(&self, state: &SimulatedGameState) -> u64 {
        let mut hash = 0u64;
        
        // Hash snake positions and states
        for snake in &state.snakes {
            let snake_hash = snake.id.len() as u64 + snake.health as u64 + snake.body.len() as u64;
            hash = hash.wrapping_add(snake_hash);
        }
        
        // Hash food positions
        for food in &state.food {
            let food_hash = (food.x as u64 + 1000) * (food.y as u64 + 1000);
            hash = hash.wrapping_add(food_hash);
        }
        
        // Hash turn
        hash = hash.wrapping_add(state.turn as u64);
        
        hash
    }

    /// Check transposition table for cached evaluation
    fn get_cached_evaluation(&self, state: &SimulatedGameState) -> Option<f32> {
        let hash = self.hash_state(state);
        if let Some(table) = &self.transposition_table {
            table.get(&hash).cloned()
        } else {
            None
        }
    }

    /// Store evaluation in transposition table
    fn store_evaluation(&mut self, state: &SimulatedGameState, evaluation: f32) {
        let hash = self.hash_state(state);
        if let Some(table) = &mut self.transposition_table {
            // Simple LRU: if table is full, remove oldest entry
            if table.len() >= 1000 {
                if let Some(key) = table.keys().next().cloned() {
                    table.remove(&key);
                }
            }
            table.insert(hash, evaluation);
        }
    }
}

/// MCTS Search Results
#[derive(Debug, Clone)]
pub struct MCTSResult {
    pub best_move: Direction,
    pub evaluation: f32,
    pub iterations_completed: u32,
    pub time_taken_ms: u128,
    pub nodes_created: u64,
    pub exploration_constant: f32,
}

/// Comprehensive performance statistics for MCTS search
#[derive(Debug, Clone)]
pub struct MCTSSearchStats {
    pub nodes_created: u64,
    pub total_visits: u64,
    pub time_elapsed_ms: u128,
    pub nodes_per_ms: f32,
    pub average_depth_reached: f32,
    pub terminal_nodes_visited: u64,
    pub expansion_success_rate: f32,
    pub move_diversity_score: f32,
    pub search_efficiency_score: f32,
    pub memory_usage_estimate: usize,
    pub unique_positions_explored: u64,
}

impl Default for MCTSSearchStats {
    fn default() -> Self {
        Self {
            nodes_created: 0,
            total_visits: 0,
            time_elapsed_ms: 0,
            nodes_per_ms: 0.0,
            average_depth_reached: 0.0,
            terminal_nodes_visited: 0,
            expansion_success_rate: 0.0,
            move_diversity_score: 0.0,
            search_efficiency_score: 0.0,
            memory_usage_estimate: 0,
            unique_positions_explored: 0,
        }
    }
}

impl MCTSSearchStats {
    pub fn calculate_efficiency_metrics(&mut self, max_iterations: u32) {
        if self.time_elapsed_ms > 0 {
            self.nodes_per_ms = self.nodes_created as f32 / self.time_elapsed_ms as f32;
        }
        
        if self.nodes_created > 0 {
            self.expansion_success_rate = (self.nodes_created as f32) / (self.total_visits as f32);
        }
        
        // Move diversity score (0.0 = all moves same, 1.0 = perfectly balanced)
        self.move_diversity_score = self.calculate_move_diversity();
        
        // Search efficiency score combining speed and breadth
        self.search_efficiency_score = self.nodes_per_ms * self.move_diversity_score;
        
        // Memory estimation based on tree structure
        self.memory_usage_estimate = self.estimate_memory_usage();
    }
    
    fn calculate_move_diversity(&self) -> f32 {
        // This would be calculated based on move distribution analysis
        // For now, return a placeholder based on exploration patterns
        let base_diversity = (self.nodes_created.min(100) as f32) / 100.0;
        (base_diversity + 0.1).min(1.0) // Ensure reasonable diversity score
    }
    
    fn estimate_memory_usage(&self) -> usize {
        // Rough estimation: each node ~200 bytes + overhead
        let estimated_node_size = 200;
        self.nodes_created as usize * estimated_node_size
    }
    
    pub fn generate_performance_report(&self) -> String {
        format!(
            "MCTS Performance Report:\n\
             ├─ Nodes Created: {}\n\
             ├─ Total Visits: {}\n\
             ├─ Time Elapsed: {}ms\n\
             ├─ Nodes/Second: {:.2}\n\
             ├─ Expansion Success Rate: {:.1}%\n\
             ├─ Move Diversity Score: {:.2}\n\
             ├─ Search Efficiency: {:.2}\n\
             ├─ Memory Usage: {:.1}KB\n\
             └─ Unique Positions: {}",
            self.nodes_created,
            self.total_visits,
            self.time_elapsed_ms,
            self.nodes_per_ms * 1000.0, // Convert to per second
            self.expansion_success_rate * 100.0,
            self.move_diversity_score,
            self.search_efficiency_score,
            self.memory_usage_estimate as f32 / 1024.0,
            self.unique_positions_explored
        )
    }
}

/// MCTS Decision Maker - Main integration point
pub struct MCTSDecisionMaker {
    max_iterations: u32,
    time_limit_ms: u128,
    exploration_constant: f32,
    max_depth: u8,
}

impl MCTSDecisionMaker {
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,     // Conservative iterations for 500ms limit
            time_limit_ms: 450,       // Leave 50ms buffer for network/processing
            exploration_constant: (2.0f32).sqrt(), // Standard exploration constant
            max_depth: 15,            // Reasonable depth limit for rollouts
        }
    }
    
    pub fn with_config(max_iterations: u32, time_limit_ms: u128, exploration_constant: f32, max_depth: u8) -> Self {
        Self {
            max_iterations,
            time_limit_ms,
            exploration_constant,
            max_depth,
        }
    }

    pub fn make_decision(&self, _game: &Game, board: &Board, you: &Battlesnake) -> Value {
        info!("MCTS DECISION: Starting search for snake {}", you.id);
        
        // Convert to simulation state
        let mut sim_state = GameSimulator::from_game_state(_game, board, you);
        sim_state.turn = 0; // Reset turn counter for search
        
        // Create MCTS searcher
        let mut searcher = MCTSSearcher::new(
            self.max_iterations, 
            self.time_limit_ms, 
            &you.id
        );
        searcher.max_depth = self.max_depth;
        searcher.exploration_constant = self.exploration_constant;
        
        let result = searcher.search(&mut sim_state);
        let _stats = searcher.get_stats();
        
        info!("MCTS DECISION: Selected {:?} (eval: {:.2}, iterations: {}, nodes: {}, {}ms, exploration: {:.2})",
              result.best_move, result.evaluation, result.iterations_completed,
              result.nodes_created, result.time_taken_ms, result.exploration_constant);
        
        json!({ "move": format!("{:?}", result.best_move).to_lowercase() })
    }
}

/// Hybrid Search Manager - Chooses between Minimax and MCTS based on game state
pub struct HybridSearchManager {
    minimax_decision_maker: MinimaxDecisionMaker,
    mcts_decision_maker: MCTSDecisionMaker,
}

impl HybridSearchManager {
    pub fn new() -> Self {
        Self {
            minimax_decision_maker: MinimaxDecisionMaker::new(),
            mcts_decision_maker: MCTSDecisionMaker::new(),
        }
    }
    
    pub fn make_decision(&mut self, _game: &Game, board: &Board, you: &Battlesnake) -> Value {
        let num_snakes = board.snakes.len();
        let our_health = you.health;
        let board_complexity = (board.width as usize * board.height as usize) as f32;
        
        // Strategy selection based on game state
        let use_mcts = match (num_snakes, our_health, board_complexity) {
            // Use MCTS for complex, uncertain positions
            (n, h, _) if n >= 4 && h > 30 => true,  // Many snakes, sufficient health
            (n, h, c) if n >= 3 && h > 50 && c > 100.0 => true, // Medium snakes, healthy, large board
            (n, h, _) if n == 1 && h < 20 => true,   // Single opponent, low health
            _ => false, // Default to minimax for simpler scenarios
        };
        
        if use_mcts {
            info!("HYBRID MANAGER: Using MCTS for complex position (snakes: {}, health: {}, board: {}x{})",
                  num_snakes, our_health, board.width, board.height);
            self.mcts_decision_maker.make_decision(_game, board, you)
        } else {
            info!("HYBRID MANAGER: Using Minimax for simple position (snakes: {}, health: {}, board: {}x{})",
                  num_snakes, our_health, board.width, board.height);
            self.minimax_decision_maker.make_decision(_game, board, you)
        }
    }
}

// Simplified Minimax Decision Maker for hybrid compatibility
pub struct MinimaxDecisionMaker {
    max_depth: u8,
    time_limit_ms: u128,
}

impl MinimaxDecisionMaker {
    pub fn new() -> Self {
        Self {
            max_depth: 3,        // Conservative depth for 500ms limit
            time_limit_ms: 400,  // Leave 100ms buffer for network/processing
        }
    }
    
    pub fn make_decision(&self, _game: &Game, board: &Board, you: &Battlesnake) -> Value {
        info!("MINIMAX DECISION: Simple fallback for snake {}", you.id);
        
        // Simple fallback to territorial strategy
        let strategist = TerritorialStrategist::new();
        let game = Game { id: "fallback".to_string(), ruleset: HashMap::new(), timeout: 20000 };
        let turn = 0;
        
        strategist.make_territorial_decision(&game, &turn, board, you)
    }
}

// ============================================================================
// COMPREHENSIVE TESTING INFRASTRUCTURE
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Mock structures for testing
    fn create_test_board() -> Board {
        Board {
            width: 11,
            height: 11,
            food: vec![
                Coord { x: 5, y: 5 },
                Coord { x: 8, y: 8 },
            ],
            snakes: vec![
                Battlesnake {
                    id: "test_snake_1".to_string(),
                    name: "Test Snake 1".to_string(),
                    health: 100,
                    body: vec![
                        Coord { x: 5, y: 10 },
                        Coord { x: 5, y: 9 },
                        Coord { x: 5, y: 8 },
                    ],
                    head: Coord { x: 5, y: 10 },
                    length: 3,
                    latency: "100".to_string(),
                    shout: None,
                },
                Battlesnake {
                    id: "test_snake_2".to_string(),
                    name: "Test Snake 2".to_string(),
                    health: 95,
                    body: vec![
                        Coord { x: 8, y: 5 },
                        Coord { x: 8, y: 6 },
                        Coord { x: 8, y: 7 },
                    ],
                    head: Coord { x: 8, y: 5 },
                    length: 3,
                    latency: "150".to_string(),
                    shout: None,
                },
            ],
            hazards: Vec::new(),
        }
    }

    fn create_test_game() -> Game {
        let mut ruleset = HashMap::new();
        ruleset.insert("name".to_string(), json!("test_rules"));
        Game {
            id: "test_game".to_string(),
            ruleset,
            timeout: 20000,
        }
    }

    // Test basic game state simulation
    #[test]
    fn test_game_simulator_creation() {
        let game = create_test_game();
        let board = create_test_board();
        let you = &board.snakes[0];
        
        let sim_state = GameSimulator::from_game_state(&game, &board, you);
        
        assert_eq!(sim_state.board_width, 11);
        assert_eq!(sim_state.board_height, 11);
        assert_eq!(sim_state.food.len(), 2);
        assert_eq!(sim_state.snakes.len(), 2);
        assert_eq!(sim_state.snakes[0].id, "test_snake_1");
        assert_eq!(sim_state.snakes[1].id, "test_snake_2");
    }

    // Test MCTS Node creation
    #[test]
    fn test_mcts_node_creation() {
        let game = create_test_game();
        let board = create_test_board();
        let you = &board.snakes[0];
        
        let sim_state = GameSimulator::from_game_state(&game, &board, you);
        let root = MCTSNode::new_root(sim_state, &you.id);
        
        assert_eq!(root.depth, 0);
        assert_eq!(root.visits, 0);
        assert_eq!(root.total_value, 0.0);
        assert_eq!(root.move_used, None);
        assert!(root.children.is_empty());
    }

    // Test MCTS searcher creation
    #[test]
    fn test_mcts_searcher_creation() {
        let searcher = MCTSSearcher::new(1000, 400, "test_snake");
        
        assert_eq!(searcher.max_iterations, 1000);
        assert_eq!(searcher.time_limit_ms, 400);
        assert_eq!(searcher.exploration_constant, (2.0f32).sqrt());
        assert_eq!(searcher.nodes_created, 0);
        assert_eq!(searcher.our_snake_id, "test_snake");
    }

    // Test MCTS decision maker
    #[test]
    fn test_mcts_decision_maker() {
        let game = create_test_game();
        let board = create_test_board();
        let you = &board.snakes[0];
        
        let decision_maker = MCTSDecisionMaker::new();
        let result = decision_maker.make_decision(&game, &board, you);
        
        // Should return a valid move decision
        assert!(result.is_object());
        if let Some(mov) = result.get("move") {
            assert!(mov.is_string());
            let move_str = mov.as_str().unwrap();
            assert!(["up", "down", "left", "right"].contains(&move_str));
        } else {
            panic!("Expected 'move' field in result");
        }
    }

    // Test MCTS UCB1 calculation
    #[test]
    fn test_ucb1_calculation() {
        let searcher = MCTSSearcher::new(100, 400, "test_snake");
        
        // Create a mock node
        let game = create_test_game();
        let board = create_test_board();
        let you = &board.snakes[0];
        
        let sim_state = GameSimulator::from_game_state(&game, &board, you);
        let mut node = MCTSNode::new_root(sim_state, &you.id);
        
        // Test unvisited node (should return infinity)
        let ucb1_unvisited = searcher.calculate_ucb1(&node);
        assert_eq!(ucb1_unvisited, f32::INFINITY);
        
        // Test visited node
        node.visits = 10;
        node.total_value = 50.0;
        let ucb1_visited = searcher.calculate_ucb1(&node);
        assert!(ucb1_visited.is_finite());
        assert!(ucb1_visited > 0.0);
    }

    // Test position evaluation
    #[test]
    fn test_position_evaluation() {
        let evaluator = IntegratedEvaluator::new();
        let game = create_test_game();
        let board = create_test_board();
        let you = &board.snakes[0];
        
        let sim_state = GameSimulator::from_game_state(&game, &board, you);
        
        let score = evaluator.evaluate_position(&sim_state, &you.id);
        
        // Should return a valid numeric score
        assert!(!score.is_nan());
        assert!(score.is_finite());
    }

    // Test safety checker
    #[test]
    fn test_safety_checker() {
        let board = create_test_board();
        let you = &board.snakes[0];
        let snakes = &board.snakes;
        
        // Test safe coordinate
        let safe_coord = Coord { x: 3, y: 10 }; // Near our snake but not on it
        assert!(SafetyChecker::is_safe_coordinate(&safe_coord, &board, snakes));
        
        // Test unsafe coordinate (occupied by our snake's body)
        let unsafe_coord = you.head; // Our snake's head
        assert!(!SafetyChecker::is_safe_coordinate(&unsafe_coord, &board, snakes));
        
        // Test boundary checking
        let out_of_bounds = Coord { x: -1, y: 5 };
        assert!(!SafetyChecker::is_safe_coordinate(&out_of_bounds, &board, snakes));
    }
}

// ============================================================================
// MAIN MOVE DECISION FUNCTION - PHASE 2B HYBRID INTEGRATION
// ============================================================================

// Main move decision function - Updated for Phase 2B with Hybrid MCTS/Minimax Search
pub fn get_move(game: &Game, turn: &i32, board: &Board, you: &Battlesnake) -> Value {
    info!("MOVE {}: === Hybrid MCTS/Minimax Search Mode ===", turn);
    
    // Use hybrid search manager to choose between MCTS and Minimax
    let mut hybrid_manager = HybridSearchManager::new();
    let result = hybrid_manager.make_decision(game, board, you);
    info!("MOVE {}: Hybrid decision completed", turn);
    result
}
