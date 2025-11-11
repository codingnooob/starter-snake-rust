use log::info;
use rand::Rng;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

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
    
    pub fn make_territorial_decision(&self, game: &Game, turn: &i32, board: &Board, 
                                   you: &Battlesnake) -> Value {
        info!("MOVE {}: Territorial Strategy Analysis", turn);
        
        let all_snakes: Vec<Battlesnake> = board.snakes.iter().cloned().collect();
        let territory_map = SpaceController::calculate_territory_map(board, &all_snakes);
        
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
                score_a.partial_cmp(score_b).unwrap()
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
    pub opponent_modeling: bool,           // Whether to use opponent modeling
    pub modeling_config: OpponentModelingConfig, // Current modeling configuration
    pub prediction_cache: OpponentPredictionCache, // Cache for predictions
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
    
    // Generate valid moves for a single snake
    fn generate_moves_for_snake(snake: &SimulatedSnake, state: &SimulatedGameState) -> Vec<Direction> {
        let mut valid_moves = Vec::new();
        
        for direction in Direction::all() {
            let next_head = direction.apply_to_coord(&snake.head());
            
            // Basic boundary check
            if next_head.x < 0 || next_head.x >= state.board_width ||
               next_head.y < 0 || next_head.y >= (state.board_height as i32) {
                continue;
            }
            
            // Avoid immediate backward move
            if let Some(neck) = snake.neck() {
                if next_head == neck {
                    continue;
                }
            }
            
            valid_moves.push(direction);
        }
        
        // Always ensure at least one move to prevent infinite loops
        if valid_moves.is_empty() {
            valid_moves.push(Direction::Up);
        }
        
        valid_moves
    }
    
    // Cartesian product for move combinations
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
                        if !move_app.collisions.contains(&snake_id) {
                            move_app.collisions.push(snake_id);
                        }
                    }
                }
            }
        }
        
        // Check body collisions - collect collision data first to avoid borrowing conflicts
        let mut body_collision_data = Vec::new();
        
        for snake in &state.snakes {
            if !snake.is_alive {
                continue;
            }
            
            let head = snake.head();
            let snake_id = snake.id.clone();
            
            // Check collision with any body (including own body from position 1 onwards)
            for other_snake in &state.snakes {
                if !other_snake.is_alive {
                    continue;
                }
                
                let body_to_check = if snake.id == other_snake.id {
                    // Own body - check from position 1 onwards (skip head)
                    if other_snake.body.len() > 1 {
                        &other_snake.body[1..]
                    } else {
                        &[]
                    }
                } else {
                    // Other snake - check entire body
                    &other_snake.body
                };
                
                if body_to_check.contains(&head) {
                    body_collision_data.push(snake_id.clone());
                    break;
                }
            }
        }
        
        // Apply body collision results
        for snake_id in body_collision_data {
            if let Some(snake) = state.snakes.iter_mut().find(|s| s.id == snake_id) {
                snake.is_alive = false;
                if !move_app.collisions.contains(&snake_id) {
                    move_app.collisions.push(snake_id);
                }
            }
        }
    }
    
    // Undo moves using MoveApplication
    pub fn undo_moves(state: &mut SimulatedGameState, move_app: &MoveApplication) {
        // Revert turn
        state.turn -= 1;
        
        // Restore food that was consumed
        for (_, food_coord) in &move_app.food_consumed {
            state.food.push(*food_coord);
        }
        
        // Restore snake states
        for (snake_id, _) in &move_app.snake_moves {
            if let Some(snake) = state.snakes.iter_mut().find(|s| &s.id == snake_id) {
                // Remove new head
                if !snake.body.is_empty() {
                    snake.body.remove(0);
                }
                
                // Restore tail if no food was consumed
                let consumed_food = move_app.food_consumed.iter()
                    .any(|(id, _)| id == snake_id);
                
                if !consumed_food {
                    if let Some((_, tail)) = move_app.previous_tails.iter()
                        .find(|(id, _)| id == snake_id) {
                        snake.body.push(*tail);
                    }
                }
                
                // Restore health
                snake.health += 1;
                if move_app.food_consumed.iter().any(|(id, _)| id == snake_id) {
                    snake.health -= 100; // Revert food health bonus
                }
                
                // Restore life if snake died this turn
                if move_app.collisions.contains(snake_id) {
                    snake.is_alive = true;
                }
            }
        }
    }
    
    // Convert from API structures to simulation state with opponent modeling support
    pub fn from_game_state(_game: &Game, board: &Board, _you: &Battlesnake) -> SimulatedGameState {
        let default_config = OpponentModelingConfig {
            mode: OpponentModelingMode::Rational,
            prediction_weight: 0.7,
            confidence_threshold: 0.5,
            cache_predictions: true,
            max_cache_size: 100,
            use_predictions_for_ordering: true,
            early_termination_threshold: 0.8,
        };
        
        SimulatedGameState {
            board_width: board.width,
            board_height: board.height,
            food: board.food.clone(),
            snakes: board.snakes.iter().map(|snake| SimulatedSnake {
                id: snake.id.clone(),
                health: snake.health,
                body: snake.body.clone(),
                is_alive: true,
            }).collect(),
            turn: 0, // Will be set by search algorithm
            opponent_modeling: true, // Enable opponent modeling by default
            modeling_config: default_config,
            prediction_cache: OpponentPredictionCache::new(100),
        }
    }
    
    // Check if game is terminal (one or no snakes alive)
    pub fn is_terminal(state: &SimulatedGameState) -> bool {
        let alive_count = state.snakes.iter().filter(|s| s.is_alive).count();
        alive_count <= 1
    }
    
    // Get our snake from simulation state
    pub fn get_our_snake<'a>(state: &'a SimulatedGameState, our_id: &str) -> Option<&'a SimulatedSnake> {
        state.snakes.iter().find(|snake| snake.id == our_id)
    }
    
    // ============================================================================
    // PHASE 2B: ADVANCED OPPONENT MODELING INTEGRATION
    // ============================================================================
    
    /// Generate prediction-based moves for a specific snake using OpponentAnalyzer
    pub fn generate_prediction_based_moves(
        snake: &SimulatedSnake,
        state: &SimulatedGameState,
        config: &OpponentModelingConfig
    ) -> Vec<(Direction, f32)> {
        let mut move_probabilities = Vec::new();
        
        // Convert to API format for OpponentAnalyzer
        let api_snake = Battlesnake {
            id: snake.id.clone(),
            name: format!("Snake_{}", snake.id),
            health: snake.health,
            body: snake.body.clone(),
            head: if !snake.body.is_empty() { snake.body[0] } else { Coord { x: 0, y: 0 } },
            length: snake.body.len() as i32,
            latency: "0".to_string(),
            shout: None,
        };
        
        let api_board = Board {
            width: state.board_width,
            height: state.board_height,
            food: state.food.clone(),
            snakes: state.snakes.iter().map(|s| Battlesnake {
                id: s.id.clone(),
                name: format!("Snake_{}", s.id),
                health: s.health,
                body: s.body.clone(),
                head: if !s.body.is_empty() { s.body[0] } else { Coord { x: 0, y: 0 } },
                length: s.body.len() as i32,
                latency: "0".to_string(),
                shout: None,
            }).collect(),
            hazards: Vec::new(),
        };
        
        let all_snakes: Vec<Battlesnake> = api_board.snakes.clone();
        
        match config.mode {
            OpponentModelingMode::Predicted => {
                // Use predictions exclusively
                let predictions = OpponentAnalyzer::predict_opponent_moves(&api_snake, &api_board, &all_snakes);
                for direction in Direction::all() {
                    let prob = predictions.get(&direction).unwrap_or(&0.0);
                    if *prob > 0.0 {
                        move_probabilities.push((direction, *prob));
                    }
                }
            },
            OpponentModelingMode::Hybrid => {
                // Combine predictions with rational analysis
                let predictions = OpponentAnalyzer::predict_opponent_moves(&api_snake, &api_board, &all_snakes);
                let rational_moves = Self::generate_moves_for_snake(snake, state);
                let rational_moves_count = rational_moves.len();
                
                for direction in &rational_moves {
                    let pred_prob = predictions.get(direction).unwrap_or(&0.0);
                    let rational_weight = 1.0 - config.prediction_weight;
                    let combined_prob = (config.prediction_weight * pred_prob) + (rational_weight / rational_moves_count as f32);
                    
                    if combined_prob > config.confidence_threshold {
                        move_probabilities.push((direction.clone(), combined_prob));
                    } else {
                        // Fallback to uniform rational distribution
                        move_probabilities.push((direction.clone(), rational_weight / rational_moves_count as f32));
                    }
                }
                
                for direction in &rational_moves {
                    let pred_prob = predictions.get(direction).unwrap_or(&0.0);
                    let rational_weight = 1.0 - config.prediction_weight;
                    let combined_prob = (config.prediction_weight * pred_prob) + (rational_weight / rational_moves.len() as f32);
                    
                    if combined_prob > config.confidence_threshold {
                        move_probabilities.push((*direction, combined_prob));
                    } else {
                        // Fallback to uniform rational distribution
                        move_probabilities.push((*direction, rational_weight / rational_moves.len() as f32));
                    }
                }
            },
            _ => {
                // Rational mode - use uniform distribution
                let rational_moves = Self::generate_moves_for_snake(snake, state);
                let uniform_prob = 1.0 / rational_moves.len() as f32;
                for direction in rational_moves {
                    move_probabilities.push((direction, uniform_prob));
                }
            }
        }
        
        // Normalize probabilities
        let total_prob: f32 = move_probabilities.iter().map(|(_, prob)| prob).sum();
        if total_prob > 0.0 {
            for (_, prob) in &mut move_probabilities {
                *prob /= total_prob;
            }
        }
        
        move_probabilities
    }
    
    /// Generate move combinations using opponent predictions
    pub fn generate_predicted_move_combinations(
        state: &SimulatedGameState,
        our_snake_id: &str,
        our_move: Direction,
        config: &OpponentModelingConfig
    ) -> Vec<(Vec<Direction>, f32)> {
        let alive_snakes: Vec<&SimulatedSnake> = state.snakes.iter()
            .filter(|snake| snake.is_alive)
            .collect();
        
        if alive_snakes.is_empty() {
            return Vec::new();
        }
        
        // Find our position in the alive snakes list
        let our_index = alive_snakes.iter().position(|s| s.id == our_snake_id);
        if our_index.is_none() {
            return Vec::new();
        }
        let our_index = our_index.unwrap();
        
        // Generate move combinations with probabilities
        let mut move_combinations = Vec::new();
        let mut snake_move_probs = Vec::new();
        
        for (i, snake) in alive_snakes.iter().enumerate() {
            if i == our_index {
                // Fixed move for us
                snake_move_probs.push(vec![(our_move, 1.0)]);
            } else {
                // Predicted moves for opponents
                let predictions = Self::generate_prediction_based_moves(snake, state, config);
                snake_move_probs.push(predictions);
            }
        }
        
        // Generate all combinations with combined probabilities
        Self::generate_probabilistic_combinations(&snake_move_probs, &mut move_combinations, Vec::new(), 1.0, 0);
        move_combinations
    }
    
    /// Helper to generate probabilistic combinations
    fn generate_probabilistic_combinations(
        snake_moves: &[Vec<(Direction, f32)>],
        result: &mut Vec<(Vec<Direction>, f32)>,
        current_moves: Vec<Direction>,
        current_prob: f32,
        depth: usize,
    ) {
        if depth == snake_moves.len() {
            result.push((current_moves, current_prob));
            return;
        }
        
        for (direction, prob) in &snake_moves[depth] {
            let mut new_moves = current_moves.clone();
            new_moves.push(*direction);
            let new_prob = current_prob * prob;
            Self::generate_probabilistic_combinations(snake_moves, result, new_moves, new_prob, depth + 1);
        }
    }
    
    /// Sample moves according to probability distribution
    pub fn sample_moves_by_probability(move_probs: &[(Direction, f32)]) -> Vec<Direction> {
        use rand::Rng;
        
        let mut rng = rand::rng();
        let mut selected_moves = Vec::new();
        
        // For expectiminimax, we might want to sample multiple times
        let sample_count = 1; // Could be configurable
        
        for _ in 0..sample_count {
            let rand_val: f32 = rng.random();
            let mut cumulative_prob = 0.0;
            
            for (direction, prob) in move_probs {
                cumulative_prob += prob;
                if rand_val <= cumulative_prob {
                    selected_moves.push(*direction);
                    break;
                }
            }
        }
        
        selected_moves
    }
    
    /// Calculate prediction confidence score for debugging
    pub fn calculate_prediction_confidence(
        snake: &SimulatedSnake,
        state: &SimulatedGameState,
        config: &OpponentModelingConfig
    ) -> f32 {
        let move_probs = Self::generate_prediction_based_moves(snake, state, config);
        
        if move_probs.is_empty() {
            return 0.0;
        }
        
        // Calculate entropy-based confidence (lower entropy = higher confidence)
        let entropy: f32 = move_probs.iter()
            .map(|(_, prob)| {
                if *prob > 0.0 {
                    -prob * prob.log2()
                } else {
                    0.0
                }
            })
            .sum();
        
        // Normalize entropy (max entropy for n moves is log2(n))
        let max_entropy = (move_probs.len() as f32).log2();
        if max_entropy > 0.0 {
            1.0 - (entropy / max_entropy)
        } else {
            1.0
        }
    }
}

// ============================================================================
// PHASE 2B: OPPONENT MODELING INTEGRATION UTILITIES
// ============================================================================

/// Configuration manager for opponent modeling settings
pub struct OpponentModelingManager {
    pub default_config: OpponentModelingConfig,
    pub debug_mode: bool,
    pub performance_tracking: bool,
}

impl OpponentModelingManager {
    pub fn new() -> Self {
        Self {
            default_config: OpponentModelingConfig {
                mode: OpponentModelingMode::Hybrid,
                prediction_weight: 0.7,
                confidence_threshold: 0.5,
                cache_predictions: true,
                max_cache_size: 100,
                use_predictions_for_ordering: true,
                early_termination_threshold: 0.8,
            },
            debug_mode: false,
            performance_tracking: true,
        }
    }
    
    /// Create configuration based on game state
    pub fn create_config_for_game(&self, board: &Board, snakes_count: usize, turn: i32) -> OpponentModelingConfig {
        let mut config = self.default_config.clone();
        
        // Adjust based on game complexity
        if snakes_count > 6 {
            // More complex games - be more conservative
            config.mode = OpponentModelingMode::Rational;
            config.prediction_weight = 0.5;
        } else if snakes_count <= 3 {
            // Simple games - can be more aggressive with predictions
            config.mode = OpponentModelingMode::Predicted;
            config.prediction_weight = 0.9;
        }
        
        // Adjust based on turn (early vs late game)
        if turn > 100 {
            // Late game - predictions more reliable
            config.prediction_weight = (config.prediction_weight + 0.1).min(1.0);
            config.confidence_threshold = (config.confidence_threshold - 0.1).max(0.3);
        }
        
        config
    }
    
    /// Enable/disable debug mode
    pub fn set_debug_mode(&mut self, enabled: bool) {
        self.debug_mode = enabled;
    }
    
    /// Log opponent modeling decision for debugging
    pub fn log_modeling_decision(&self, snake_id: &str, config: &OpponentModelingConfig, 
                               confidence: f32, moves_evaluated: usize) {
        if self.debug_mode {
            info!("OPPONENT MODELING: Snake {} using {:?} mode (confidence: {:.2}, moves: {})",
                  snake_id, config.mode, confidence, moves_evaluated);
        }
    }
    
    /// Performance metrics for opponent modeling
    pub fn get_modeling_metrics(&self) -> OpponentModelingMetrics {
        OpponentModelingMetrics {
            cache_hit_rate: 0.0, // Would be calculated from actual cache stats
            prediction_accuracy: 0.0, // Would be calculated from game results
            modeling_overhead_ms: 0.0, // Would be calculated from timing
            modes_used: vec![], // Would track which modes performed best
        }
    }
}

/// Performance metrics for opponent modeling system
#[derive(Debug, Clone)]
pub struct OpponentModelingMetrics {
    pub cache_hit_rate: f32,
    pub prediction_accuracy: f32,
    pub modeling_overhead_ms: f32,
    pub modes_used: Vec<OpponentModelingMode>,
}

impl OpponentModelingMetrics {
    pub fn new() -> Self {
        Self {
            cache_hit_rate: 0.0,
            prediction_accuracy: 0.0,
            modeling_overhead_ms: 0.0,
            modes_used: Vec::new(),
        }
    }
    
    /// Generate performance report
    pub fn generate_report(&self) -> String {
        format!("Opponent Modeling Performance Report:\n\
                Cache Hit Rate: {:.1}%\n\
                Prediction Accuracy: {:.1}%\n\
                Modeling Overhead: {:.2}ms\n\
                Modes Used: {:?}\n",
                self.cache_hit_rate * 100.0,
                self.prediction_accuracy * 100.0,
                self.modeling_overhead_ms,
                self.mode_used_summary())
    }
    
    fn mode_used_summary(&self) -> String {
        if self.modes_used.is_empty() {
            "None".to_string()
        } else {
            let mut counts = HashMap::new();
            for mode in &self.modes_used {
                *counts.entry(format!("{:?}", mode)).or_insert(0) += 1;
            }
            counts.iter()
                .map(|(mode, count)| format!("{}: {}", mode, count))
                .collect::<Vec<_>>()
                .join(", ")
        }
    }
}

/// Enhanced MinimaxDecisionMaker with opponent modeling support
pub struct EnhancedMinimaxDecisionMaker {
    max_depth: u8,
    time_limit_ms: u128,
    modeling_manager: OpponentModelingManager,
    metrics: OpponentModelingMetrics,
}

impl EnhancedMinimaxDecisionMaker {
    pub fn new() -> Self {
        Self {
            max_depth: 3,
            time_limit_ms: 400,
            modeling_manager: OpponentModelingManager::new(),
            metrics: OpponentModelingMetrics::new(),
        }
    }
    
    pub fn with_config(max_depth: u8, time_limit_ms: u128, modeling_config: OpponentModelingConfig) -> Self {
        let mut manager = OpponentModelingManager::new();
        manager.default_config = modeling_config;
        
        Self {
            max_depth,
            time_limit_ms,
            modeling_manager: manager,
            metrics: OpponentModelingMetrics::new(),
        }
    }
    
    pub fn make_decision(&mut self, game: &Game, board: &Board, you: &Battlesnake) -> Value {
        info!("ENHANCED MINIMAX DECISION: Starting search for snake {}", you.id);
        
        // Convert to simulation state with opponent modeling
        let mut sim_state = GameSimulator::from_game_state(game, board, you);
        sim_state.turn = 0;
        
        // Determine optimal modeling configuration for this game state
        let modeling_config = self.modeling_manager.create_config_for_game(
            board, board.snakes.len(), 0);
        
        // Enable opponent modeling if beneficial
        let use_modeling = self.should_use_opponent_modeling(&modeling_config, board, you);
        sim_state.opponent_modeling = use_modeling;
        sim_state.modeling_config = modeling_config;
        
        // Estimate max snakes for transposition table sizing
        let max_snakes = board.snakes.len().max(4);
        let tt_size = (board.width as usize * board.height as usize * max_snakes).min(50000);
        
        // Create enhanced searcher with opponent modeling
        let mut searcher = MinimaxSearcher::with_transposition_table(
            self.max_depth, self.time_limit_ms, tt_size, board, max_snakes);
        
        let start_time = Instant::now();
        let result = searcher.search_best_move(&mut sim_state, &you.id);
        let search_time = start_time.elapsed().as_millis();
        
        let tt_stats = searcher.get_transposition_stats();
        
        // Log comprehensive decision information
        info!("ENHANCED MINIMAX DECISION: Selected {:?} (eval: {:.2}, nodes: {}, TT hits: {}, TT hit rate: {:.1}%, {}ms, Opponent Model: {}, Confidence: {:.2})",
              result.best_move, result.evaluation, result.nodes_searched, 
              tt_stats.hits, tt_stats.hit_rate() * 100.0, search_time,
              result.opponent_model_used, result.prediction_confidence);
        
        // Performance analysis
        if self.modeling_manager.performance_tracking {
            self.analyze_performance(&result, &sim_state);
        }
        
        json!({ "move": format!("{:?}", result.best_move).to_lowercase() })
    }
    
    /// Determine if opponent modeling should be used for this game state
    fn should_use_opponent_modeling(&self, config: &OpponentModelingConfig, 
                                  board: &Board, you: &Battlesnake) -> bool {
        // Enable modeling if:
        // 1. Multiple opponents present
        // 2. Sufficient health to justify computation overhead
        // 3. Not in immediate danger
        // 4. Game state complex enough to benefit from modeling
        
        let opponent_count = board.snakes.len() - 1;
        let has_opponents = opponent_count > 0;
        let sufficient_health = you.health > 20;
        let complex_enough = board.snakes.len() >= 2;
        
        has_opponents && sufficient_health && complex_enough && config.mode != OpponentModelingMode::Rational
    }
    
    /// Analyze and track performance metrics
    fn analyze_performance(&mut self, result: &SearchResult, state: &SimulatedGameState) {
        // Update metrics based on search results
        if result.opponent_model_used {
            // Track modeling effectiveness
            info!("OPPONENT MODELING ANALYSIS: Used={}, Confidence={:.2}, Cached={}, Time={}ms",
                  result.opponent_model_used, result.prediction_confidence, 
                  result.predictions_cached, result.time_taken_ms);
        }
        
        // Log modeling configuration used
        if self.modeling_manager.debug_mode {
            info!("MODELING CONFIG: Mode={:?}, Weight={:.2}, Threshold={:.2}",
                  state.modeling_config.mode, state.modeling_config.prediction_weight,
                  state.modeling_config.confidence_threshold);
        }
    }
    
    /// Get current performance metrics
    pub fn get_metrics(&self) -> &OpponentModelingMetrics {
        &self.metrics
    }
    
    /// Enable debug mode for detailed logging
    pub fn enable_debug_mode(&mut self) {
        self.modeling_manager.set_debug_mode(true);
    }
    
    /// Generate comprehensive performance report
    pub fn generate_performance_report(&self) -> String {
        self.metrics.generate_report()
    }
}

// ============================================================================
// PHASE 2A: ZOBRIST HASHING FOR TRANSPOSITION TABLES
// ============================================================================

/// Zobrist hashing system for efficient game position identification
pub struct ZobristHasher {
    position_keys: Vec<Vec<u64>>,    // [x][y] -> random 64-bit value
    health_keys: Vec<Vec<u64>>,      // [snake_index][health] -> random value
    food_keys: Vec<u64>,             // [position_index] -> random value
    turn_keys: Vec<u64>,             // [turn_mod] -> random value
    snake_alive_keys: Vec<u64>,      // [snake_index] -> random value
    board_width: i32,
    board_height: i32,
    max_snakes: usize,
    max_health: i32,
}

impl ZobristHasher {
    pub fn new(board_width: i32, board_height: u32, max_snakes: usize, max_health: i32) -> Self {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        
        let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducible hashing
        
        let board_width_i32 = board_width as i32;
        let board_height_i32 = board_height as i32;
        
        // Initialize position keys for all board positions
        let mut position_keys = Vec::with_capacity(board_width_i32 as usize);
        for x in 0..board_width_i32 {
            let mut row = Vec::with_capacity(board_height_i32 as usize);
            for y in 0..board_height_i32 {
                row.push(rng.gen());
            }
            position_keys.push(row);
        }
        
        // Initialize health keys for each snake and health level
        let mut health_keys = Vec::with_capacity(max_snakes);
        for _snake_idx in 0..max_snakes {
            let mut health_row = Vec::with_capacity((max_health + 1) as usize);
            for health in 0..=max_health {
                health_row.push(rng.gen());
            }
            health_keys.push(health_row);
        }
        
        // Initialize food keys (one per board position)
        let mut food_keys = Vec::with_capacity((board_width_i32 * board_height_i32) as usize);
        for _ in 0..(board_width_i32 * board_height_i32) {
            food_keys.push(rng.gen());
        }
        
        // Turn keys (modulo to keep table size reasonable)
        let mut turn_keys = Vec::with_capacity(100); // 100 turn cycles
        for _ in 0..100 {
            turn_keys.push(rng.gen());
        }
        
        // Snake alive/dead keys
        let mut snake_alive_keys = Vec::with_capacity(max_snakes);
        for _ in 0..max_snakes {
            snake_alive_keys.push(rng.gen());
        }
        
        Self {
            position_keys,
            health_keys,
            food_keys,
            turn_keys,
            snake_alive_keys,
            board_width: board_width_i32,
            board_height: board_height_i32,
            max_snakes,
            max_health,
        }
    }
    
    /// Generate Zobrist hash for a game state
    pub fn hash_state(&self, state: &SimulatedGameState) -> u64 {
        let mut hash = 0u64;
        
        // Hash snake body positions and health
        for (snake_idx, snake) in state.snakes.iter().enumerate() {
            if snake_idx >= self.max_snakes {
                break;
            }
            
            // Hash each body segment
            for &coord in &snake.body {
                if coord.x >= 0 && coord.x < self.board_width && 
                   coord.y >= 0 && coord.y < self.board_height {
                    let pos_idx = (coord.y * self.board_width + coord.x) as usize;
                    hash ^= self.position_keys[coord.x as usize][coord.y as usize];
                }
            }
            
            // Hash health (if within bounds)
            if snake.health >= 0 && snake.health <= self.max_health {
                hash ^= self.health_keys[snake_idx][snake.health as usize];
            }
            
            // Hash alive/dead status
            if snake.is_alive {
                hash ^= self.snake_alive_keys[snake_idx];
            }
        }
        
        // Hash food positions
        for &food_coord in &state.food {
            if food_coord.x >= 0 && food_coord.x < self.board_width && 
               food_coord.y >= 0 && food_coord.y < self.board_height {
                let food_idx = (food_coord.y * self.board_width + food_coord.x) as usize;
                hash ^= self.food_keys[food_idx];
            }
        }
        
        // Hash turn (modulo to prevent overflow)
        let turn_idx = (state.turn as usize) % self.turn_keys.len();
        hash ^= self.turn_keys[turn_idx];
        
        hash
    }
    
    /// Create a hasher optimized for a specific board size
    pub fn for_board(board: &Board, max_snakes: usize) -> Self {
        Self::new(board.width, board.height, max_snakes, 100)
    }
}

// ============================================================================
// PHASE 2A: TRANSPOSITION TABLE IMPLEMENTATION
// ============================================================================

/// Entry types for transposition table storage
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EntryType {
    Exact,      // Exact evaluation
    LowerBound, // Alpha cutoff (lower bound)
    UpperBound, // Beta cutoff (upper bound)
}

/// Entry in the transposition table
#[derive(Debug, Clone)]
pub struct TranspositionEntry {
    pub depth: u8,
    pub evaluation: f32,
    pub entry_type: EntryType,
    pub best_move: Option<Direction>,
    pub created_turn: u32,
}

/// Performance statistics for transposition table
#[derive(Debug, Default, Clone)]
pub struct TranspositionStats {
    pub hits: u64,
    pub misses: u64,
    pub insertions: u64,
    pub collisions: u64,
    pub replacements: u64,
}

impl TranspositionStats {
    pub fn hit_rate(&self) -> f32 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f32 / (self.hits + self.misses) as f32
        }
    }
    
    pub fn reset(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.insertions = 0;
        self.collisions = 0;
        self.replacements = 0;
    }
}

/// Transposition table for caching minimax evaluations
pub struct TranspositionTable {
    table: HashMap<u64, TranspositionEntry>,
    max_size: usize,
    stats: TranspositionStats,
    current_turn: u32,
    replacement_policy: ReplacementPolicy,
}

/// Replacement policies for when table is full
#[derive(Debug, Clone, Copy)]
pub enum ReplacementPolicy {
    AlwaysReplace,     // Always replace existing entry
    DepthPreferred,    // Replace if new depth is greater or equal
    Aging,            // Use aging to prefer newer entries
}

impl TranspositionTable {
    pub fn new(max_size: usize) -> Self {
        Self {
            table: HashMap::with_capacity(max_size),
            max_size,
            stats: TranspositionStats::default(),
            current_turn: 0,
            replacement_policy: ReplacementPolicy::DepthPreferred,
        }
    }
    
    /// Set replacement policy
    pub fn with_policy(mut self, policy: ReplacementPolicy) -> Self {
        self.replacement_policy = policy;
        self
    }
    
    /// Update current turn for aging
    pub fn update_turn(&mut self, turn: u32) {
        self.current_turn = turn;
    }
    
    /// Lookup a position in the table
    pub fn lookup(&mut self, hash: u64, depth: u8) -> Option<&TranspositionEntry> {
        if let Some(entry) = self.table.get(&hash) {
            // Check if entry is still useful
            if entry.depth >= depth {
                self.stats.hits += 1;
                Some(entry)
            } else {
                self.stats.misses += 1;
                None
            }
        } else {
            self.stats.misses += 1;
            None
        }
    }
    
    /// Insert a new entry into the table
    pub fn insert(&mut self, hash: u64, entry: TranspositionEntry) {
        self.stats.insertions += 1;
        
        // Check if we need to make space
        if self.table.len() >= self.max_size {
            self.handle_table_full(hash, entry);
        } else {
            self.table.insert(hash, entry);
        }
    }
    
    /// Handle table full scenario with replacement policy
    fn handle_table_full(&mut self, hash: u64, new_entry: TranspositionEntry) {
        match self.replacement_policy {
            ReplacementPolicy::AlwaysReplace => {
                // Simple replacement - just insert
                self.table.insert(hash, new_entry);
                self.stats.replacements += 1;
            }
            ReplacementPolicy::DepthPreferred => {
                // Replace if new entry has equal or greater depth
                if let Some((&old_hash, _old_entry)) = self.table.iter()
                    .find(|(_, &ref old_entry)| old_entry.depth <= new_entry.depth) {
                    self.table.remove(&old_hash);
                    self.table.insert(hash, new_entry);
                    self.stats.replacements += 1;
                } else {
                    // No suitable entry to replace, skip insertion
                    self.stats.collisions += 1;
                }
            }
            ReplacementPolicy::Aging => {
                // Replace oldest entry
                let &oldest_hash = self.table.iter()
                    .min_by_key(|(_, entry)| entry.created_turn)
                    .map(|(key, _)| key)
                    .expect("Should have entries when table is full");
                self.table.remove(&oldest_hash);
                self.table.insert(hash, new_entry);
                self.stats.replacements += 1;
            }
        }
    }
    
    /// Get performance statistics
    pub fn get_stats(&self) -> &TranspositionStats {
        &self.stats
    }
    
    /// Clear the table and reset statistics
    pub fn clear(&mut self) {
        self.table.clear();
        self.stats.reset();
    }
    
    /// Get current table size
    pub fn size(&self) -> usize {
        self.table.len()
    }
    
    /// Get maximum table size
    pub fn capacity(&self) -> usize {
        self.max_size
    }
}

// ============================================================================
// STANDARDIZED EVALUATION FUNCTION INTERFACE (Phase 2A-2)
// ============================================================================

pub trait PositionEvaluator {
    fn evaluate_position(&self, state: &SimulatedGameState, our_snake_id: &str) -> f32;
}

// Integrated evaluation function combining all Phase 1 systems
pub struct IntegratedEvaluator {
    safety_weight: f32,
    food_weight: f32,
    territory_weight: f32,
    space_weight: f32,
}

impl IntegratedEvaluator {
    pub fn new() -> Self {
        Self {
            safety_weight: 1000.0,  // Safety is paramount
            food_weight: 50.0,      // Food seeking
            territory_weight: 20.0, // Territory control
            space_weight: 10.0,     // Space availability
        }
    }
    
    // Convert simulated state to API-compatible structures for Phase 1 systems
    fn to_api_board(&self, state: &SimulatedGameState) -> Board {
        Board {
            width: state.board_width,
            height: state.board_height,
            food: state.food.clone(),
            snakes: state.snakes.iter().map(|sim_snake| {
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
            }).collect(),
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
// PHASE 2B: ADVANCED OPPONENT MODELING INTEGRATION
// ============================================================================

/// Opponent modeling strategies for different adversarial scenarios
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OpponentModelingMode {
    Rational,     // Assumes perfect opponent optimization (current behavior)
    Predicted,    // Uses Phase 1C predictions exclusively
    Hybrid,       // Combines rational analysis with prediction weighting
    Expectiminimax, // Probabilistic evaluation with predictions
}

/// Configuration for opponent modeling integration
#[derive(Debug, Clone)]
pub struct OpponentModelingConfig {
    pub mode: OpponentModelingMode,
    pub prediction_weight: f32,     // Weight given to predictions vs rational analysis
    pub confidence_threshold: f32,  // Minimum confidence to use predictions
    pub cache_predictions: bool,    // Cache predictions during search
    pub max_cache_size: usize,      // Maximum number of cached predictions
    pub use_predictions_for_ordering: bool, // Use predictions for move ordering
    pub early_termination_threshold: f32,  // Early termination confidence
}

/// Cache for opponent predictions during search
#[derive(Debug, Clone)]
struct OpponentPredictionCache {
    predictions: HashMap<String, HashMap<Direction, f32>>,
    max_size: usize,
    access_count: u64,
}

impl OpponentPredictionCache {
    fn new(max_size: usize) -> Self {
        Self {
            predictions: HashMap::new(),
            max_size,
            access_count: 0,
        }
    }
    
    fn get_prediction(&mut self, snake_id: &str, board: &Board, all_snakes: &[Battlesnake]) -> Option<&HashMap<Direction, f32>> {
        self.access_count += 1;
        self.predictions.get(snake_id)
    }
    
    fn store_prediction(&mut self, snake_id: String, predictions: HashMap<Direction, f32>) {
        if self.predictions.len() >= self.max_size {
            // Simple eviction: remove oldest entry
            if let Some(key) = self.predictions.keys().next().cloned() {
                self.predictions.remove(&key);
            }
        }
        self.predictions.insert(snake_id, predictions);
    }
    
    fn clear(&mut self) {
        self.predictions.clear();
    }
    
    fn size(&self) -> usize {
        self.predictions.len()
    }
}

/// Search results with opponent modeling statistics
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best_move: Direction,
    pub evaluation: f32,
    pub nodes_searched: u64,
    pub time_taken_ms: u128,
    pub depth_reached: u8,
    pub depth_achieved: u8,        // Final achieved depth (deepest completed)
    pub iterations_completed: u8,  // Number of iterations completed
    pub tt_hits: u64,
    pub tt_misses: u64,
    pub opponent_model_used: bool,    // Whether opponent modeling was used
    pub prediction_confidence: f32,   // Average prediction confidence
    pub predictions_cached: usize,    // Number of predictions cached
}

/// Opponent Modeling Integration State
pub struct MinimaxSearcher {
    evaluator: IntegratedEvaluator,
    max_depth: u8,
    time_limit_ms: u128,
    nodes_searched: u64,
    start_time: Instant,
    transposition_table: TranspositionTable,
    zobrist_hasher: Option<ZobristHasher>,
    tt_hits: u64,
    tt_misses: u64,
    
    // Iterative Deepening Configuration
    time_safety_buffer_ms: u128,    // Buffer for API response constraints (default: 50ms)
    min_depth: u8,                  // Starting depth for iterative deepening (default: 1)
    time_per_depth_history: Vec<(u8, u128)>, // Track time per depth for estimation
    previous_best_move: Option<Direction>, // Best move from previous iteration
    iterations_completed: u8,       // Count of completed iterations
}

impl MinimaxSearcher {
    pub fn new(max_depth: u8, time_limit_ms: u128) -> Self {
        Self {
            evaluator: IntegratedEvaluator::new(),
            max_depth,
            time_limit_ms,
            nodes_searched: 0,
            start_time: Instant::now(),
            transposition_table: TranspositionTable::new(10000), // Default size
            zobrist_hasher: None,
            tt_hits: 0,
            tt_misses: 0,
            
            // Iterative Deepening Configuration
            time_safety_buffer_ms: 50,  // 50ms safety buffer for API response
            min_depth: 1,               // Start from depth 1
            time_per_depth_history: Vec::new(),
            previous_best_move: None,
            iterations_completed: 0,
        }
    }
    
    /// Create searcher with specific transposition table configuration
    pub fn with_transposition_table(max_depth: u8, time_limit_ms: u128,
                                   tt_size: usize, board: &Board, max_snakes: usize) -> Self {
        let hasher = ZobristHasher::for_board(board, max_snakes);
        
        Self {
            evaluator: IntegratedEvaluator::new(),
            max_depth,
            time_limit_ms,
            nodes_searched: 0,
            start_time: Instant::now(),
            transposition_table: TranspositionTable::new(tt_size).with_policy(ReplacementPolicy::DepthPreferred),
            zobrist_hasher: Some(hasher),
            tt_hits: 0,
            tt_misses: 0,
            
            // Iterative Deepening Configuration
            time_safety_buffer_ms: 50,  // 50ms safety buffer for API response
            min_depth: 1,               // Start from depth 1
            time_per_depth_history: Vec::new(),
            previous_best_move: None,
            iterations_completed: 0,
        }
    }
    
    /// Initialize Zobrist hasher if not already set
    fn ensure_hasher(&mut self, board: &Board, max_snakes: usize) {
        if self.zobrist_hasher.is_none() {
            self.zobrist_hasher = Some(ZobristHasher::for_board(board, max_snakes));
        }
    }
    
    // Main search entry point - Iterative Deepening Implementation
    pub fn search_best_move(&mut self, state: &mut SimulatedGameState, our_snake_id: &str) -> SearchResult {
        self.nodes_searched = 0;
        self.tt_hits = 0;
        self.tt_misses = 0;
        self.start_time = Instant::now();
        self.iterations_completed = 0;
        self.previous_best_move = None;
        self.time_per_depth_history.clear();
        
        // Update transposition table turn counter
        self.transposition_table.update_turn(state.turn as u32);
        
        info!("ITERATIVE DEEPENING: Starting search for snake {} (max depth: {})", our_snake_id, self.max_depth);
        
        // Generate our possible moves
        let our_snake_index = state.snakes.iter().position(|s| s.id == our_snake_id);
        if our_snake_index.is_none() {
            // Fallback - we're not in the game somehow
            return SearchResult {
                best_move: Direction::Up,
                evaluation: -10000.0,
                nodes_searched: 0,
                time_taken_ms: 0,
                depth_reached: 0,
                depth_achieved: 0,
                iterations_completed: 0,
                tt_hits: self.tt_hits,
                tt_misses: self.tt_misses,
                opponent_model_used: false,
                prediction_confidence: 0.0,
                predictions_cached: 0,
            };
        }
        
        let our_moves = GameSimulator::generate_moves_for_snake(&state.snakes[our_snake_index.unwrap()], state);
        let mut best_move = our_moves[0]; // Default fallback
        let mut best_evaluation = f32::NEG_INFINITY;
        let mut deepest_achieved = self.min_depth - 1; // Track deepest completed depth
        
        // Iterative deepening loop
        for current_depth in self.min_depth..=self.max_depth {
            if self.should_stop_iteration() {
                info!("ITERATIVE DEEPENING: Stopping at depth {} due to time constraints", current_depth);
                break;
            }
            
            let iteration_start = Instant::now();
            let (depth_best_move, depth_best_evaluation) = self.search_at_depth(
                &our_moves, state, our_snake_id, current_depth);
            
            let iteration_time = iteration_start.elapsed().as_millis();
            
            // Update best move if this depth produced a better evaluation
            if depth_best_evaluation > best_evaluation {
                best_evaluation = depth_best_evaluation;
                best_move = depth_best_move;
            }
            
            deepest_achieved = current_depth;
            self.iterations_completed += 1;
            self.previous_best_move = Some(best_move);
            
            // Store timing information for next iteration estimation
            self.time_per_depth_history.push((current_depth, iteration_time));
            if self.time_per_depth_history.len() > 10 {
                self.time_per_depth_history.remove(0); // Keep only last 10 measurements
            }
            
            info!("ITERATIVE DEEPENING: Depth {} completed in {}ms, best move: {:?}, evaluation: {:.2}",
                  current_depth, iteration_time, best_move, best_evaluation);
            
            // Clear transposition table between iterations to ensure fresh search
            // (or keep it for reuse - we'll keep it for better performance)
            // self.clear_transposition_table();
        }
        
        let time_taken = self.start_time.elapsed().as_millis();
        let tt_hit_rate = if self.tt_hits + self.tt_misses > 0 {
            self.tt_hits as f32 / (self.tt_hits + self.tt_misses) as f32
        } else {
            0.0
        };
        
        info!("ITERATIVE DEEPENING: Completed. Iterations: {}, Final depth: {}, Best move: {:?}, Evaluation: {:.2}, Nodes: {}, TT Hits: {}, TT Hit Rate: {:.1}%, Time: {}ms",
              self.iterations_completed, deepest_achieved, best_move, best_evaluation,
              self.nodes_searched, self.tt_hits, tt_hit_rate * 100.0, time_taken);
        
        SearchResult {
            best_move,
            evaluation: best_evaluation,
            nodes_searched: self.nodes_searched,
            time_taken_ms: time_taken,
            depth_reached: self.max_depth,
            depth_achieved: deepest_achieved,
            iterations_completed: self.iterations_completed,
            tt_hits: self.tt_hits,
            tt_misses: self.tt_misses,
            opponent_model_used: false, // TODO: track this during search
            prediction_confidence: 0.0, // TODO: calculate this during search
            predictions_cached: 0, // TODO: track cache usage
        }
    }
    
    // Search at a specific depth (single iteration)
    fn search_at_depth(&mut self, our_moves: &[Direction], state: &mut SimulatedGameState,
                      our_snake_id: &str, depth: u8) -> (Direction, f32) {
        let mut best_move = our_moves[0]; // Default fallback
        let mut best_evaluation = f32::NEG_INFINITY;
        
        // Order moves: previous best move first for better pruning
        let ordered_moves = self.order_moves(our_moves.to_vec(), self.previous_best_move);
        
        for &our_move in &ordered_moves {
            if self.is_time_up() {
                break;
            }
            
            // Create simplified move set for this search branch (our move + opponent moves)
            let move_combinations = self.generate_move_combinations(state, our_snake_id, our_move);
            
            let mut move_evaluation = f32::NEG_INFINITY;
            
            // Evaluate each combination where we make 'our_move'
            for moves in move_combinations {
                if self.is_time_up() {
                    break;
                }
                
                let move_app = GameSimulator::apply_moves(state, &moves);
                
                let evaluation = self.minimax(
                    state,
                    our_snake_id,
                    1, // Start at depth 1 since we just made our move
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    false, // Next level is minimizing (opponents turn)
                    depth, // Use current iteration depth
                );
                
                GameSimulator::undo_moves(state, &move_app);
                
                move_evaluation = move_evaluation.max(evaluation);
            }
            
            if move_evaluation > best_evaluation {
                best_evaluation = move_evaluation;
                best_move = our_move;
            }
        }
        
        (best_move, best_evaluation)
    }
    
    // Order moves with previous best move first for better alpha-beta pruning
    fn order_moves(&self, mut moves: Vec<Direction>, previous_best: Option<Direction>) -> Vec<Direction> {
        if let Some(prev_move) = previous_best {
            if let Some(pos) = moves.iter().position(|&m| m == prev_move) {
                moves.swap(0, pos);
            }
        }
        moves
    }
    
    // Check if we should stop current iteration based on time
    fn should_stop_iteration(&self) -> bool {
        let elapsed = self.start_time.elapsed().as_millis();
        let time_limit = self.time_limit_ms.saturating_sub(self.time_safety_buffer_ms);
        elapsed >= time_limit
    }
    
    /// Core minimax algorithm with alpha-beta pruning and transposition table
    fn minimax(&mut self,
               state: &mut SimulatedGameState,
               our_snake_id: &str,
               depth: u8,
               mut alpha: f32,
               mut beta: f32,
               maximizing: bool,
               max_depth: u8) -> f32 {
        
        self.nodes_searched += 1;
        
        // Terminal conditions
        if depth >= max_depth || GameSimulator::is_terminal(state) || self.is_time_up() {
            return self.evaluator.evaluate_position(state, our_snake_id);
        }
        
        // Check transposition table first
        if let Some(hasher) = &self.zobrist_hasher {
            let hash = hasher.hash_state(state);
            if let Some(entry) = self.transposition_table.lookup(hash, depth) {
                self.tt_hits += 1;
                
                // Use cached evaluation based on entry type
                match entry.entry_type {
                    EntryType::Exact => {
                        return entry.evaluation;
                    }
                    EntryType::LowerBound => {
                        // Alpha cutoff - we can prune if evaluation >= beta
                        if entry.evaluation >= beta {
                            return entry.evaluation;
                        }
                        alpha = alpha.max(entry.evaluation);
                    }
                    EntryType::UpperBound => {
                        // Beta cutoff - we can prune if evaluation <= alpha
                        if entry.evaluation <= alpha {
                            return entry.evaluation;
                        }
                        beta = beta.min(entry.evaluation);
                    }
                }
            } else {
                self.tt_misses += 1;
            }
        }
        
        let evaluation = if maximizing {
            // Maximizing player (us or beneficial moves)
            self.minimax_maximize(state, our_snake_id, depth, alpha, beta, max_depth)
        } else {
            // Minimizing player (opponents or unfavorable moves)
            self.minimax_minimize(state, our_snake_id, depth, alpha, beta, max_depth)
        };
        
        // Store result in transposition table
        if let Some(hasher) = &self.zobrist_hasher {
            let hash = hasher.hash_state(state);
            let entry_type = if evaluation <= alpha {
                EntryType::UpperBound // Beta didn't change
            } else if evaluation >= beta {
                EntryType::LowerBound // Alpha didn't change
            } else {
                EntryType::Exact // Both alpha and beta changed
            };
            
            self.transposition_table.insert(hash, TranspositionEntry {
                depth,
                evaluation,
                entry_type,
                best_move: None, // TODO: Store best move for this position
                created_turn: self.transposition_table.current_turn,
            });
        }
        
        evaluation
    }
    
    /// Maximizing player implementation
    fn minimax_maximize(&mut self,
                       state: &mut SimulatedGameState,
                       our_snake_id: &str,
                       depth: u8,
                       mut alpha: f32,
                       beta: f32,
                       max_depth: u8) -> f32 {
        let mut max_eval = f32::NEG_INFINITY;
        let move_combinations = GameSimulator::generate_all_moves(state);
        
        for moves in move_combinations {
            if self.is_time_up() {
                break;
            }
            
            let move_app = GameSimulator::apply_moves(state, &moves);
            let eval = self.minimax(state, our_snake_id, depth + 1, alpha, beta, false, max_depth);
            GameSimulator::undo_moves(state, &move_app);
            
            max_eval = max_eval.max(eval);
            alpha = alpha.max(eval);
            
            // Alpha-beta pruning
            if beta <= alpha {
                break;
            }
        }
        
        max_eval
    }
    
    /// Minimizing player implementation
    fn minimax_minimize(&mut self,
                       state: &mut SimulatedGameState,
                       our_snake_id: &str,
                       depth: u8,
                       alpha: f32,
                       mut beta: f32,
                       max_depth: u8) -> f32 {
        let mut min_eval = f32::INFINITY;
        let move_combinations = GameSimulator::generate_all_moves(state);
        
        for moves in move_combinations {
            if self.is_time_up() {
                break;
            }
            
            let move_app = GameSimulator::apply_moves(state, &moves);
            let eval = self.minimax(state, our_snake_id, depth + 1, alpha, beta, true, max_depth);
            GameSimulator::undo_moves(state, &move_app);
            
            min_eval = min_eval.min(eval);
            beta = beta.min(eval);
            
            // Alpha-beta pruning
            if beta <= alpha {
                break;
            }
        }
        
        min_eval
    }
    
    // Generate move combinations where our snake makes a specific move
    fn generate_move_combinations(&self, state: &SimulatedGameState, our_snake_id: &str, our_move: Direction) -> Vec<Vec<Direction>> {
        let alive_snakes: Vec<&SimulatedSnake> = state.snakes.iter()
            .filter(|snake| snake.is_alive)
            .collect();
        
        if alive_snakes.is_empty() {
            return Vec::new();
        }
        
        // Find our position in the alive snakes list
        let our_index = alive_snakes.iter().position(|s| s.id == our_snake_id);
        if our_index.is_none() {
            return Vec::new();
        }
        let our_index = our_index.unwrap();
        
        // Generate moves for other snakes
        let mut other_snake_moves: Vec<Vec<Direction>> = Vec::new();
        for (i, snake) in alive_snakes.iter().enumerate() {
            if i == our_index {
                other_snake_moves.push(vec![our_move]); // Fixed move for us
            } else {
                let moves = GameSimulator::generate_moves_for_snake(snake, state);
                other_snake_moves.push(moves);
            }
        }
        
        // Generate Cartesian product
        let mut combinations = Vec::new();
        Self::cartesian_product(&other_snake_moves, &mut combinations, Vec::new(), 0);
        combinations
    }
    
    // Helper for Cartesian product (similar to GameSimulator but standalone)
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
    
    // Check if search time limit has been exceeded
    fn is_time_up(&self) -> bool {
        self.start_time.elapsed().as_millis() >= self.time_limit_ms
    }
    
    /// Get transposition table statistics
    pub fn get_transposition_stats(&self) -> &TranspositionStats {
        self.transposition_table.get_stats()
    }
    
    /// Clear transposition table
    pub fn clear_transposition_table(&mut self) {
        self.transposition_table.clear();
    }
}

// ============================================================================
// PHASE 2A: MINIMAX DECISION MAKER (Integration Point)
// ============================================================================

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
    
    pub fn make_decision(&self, game: &Game, board: &Board, you: &Battlesnake) -> Value {
        info!("MINIMAX DECISION: Starting search for snake {}", you.id);
        
        // Convert to simulation state
        let mut sim_state = GameSimulator::from_game_state(game, board, you);
        sim_state.turn = 0; // Reset turn counter for search
        
        // Estimate max snakes for transposition table sizing
        let max_snakes = board.snakes.len().max(4);
        let tt_size = (board.width as usize * board.height as usize * max_snakes).min(50000);
        
        // Create searcher with transposition table
        let mut searcher = MinimaxSearcher::with_transposition_table(
            self.max_depth, self.time_limit_ms, tt_size, board, max_snakes);
        
        let result = searcher.search_best_move(&mut sim_state, &you.id);
        let tt_stats = searcher.get_transposition_stats();
        
        info!("MINIMAX DECISION: Selected {:?} (eval: {:.2}, nodes: {}, TT hits: {}, TT hit rate: {:.1}%, {}ms)",
              result.best_move, result.evaluation, result.nodes_searched, 
              tt_stats.hits, tt_stats.hit_rate() * 100.0, result.time_taken_ms);
        
        json!({ "move": format!("{:?}", result.best_move).to_lowercase() })
    }
}

// ============================================================================
// MAIN MOVE DECISION FUNCTION - PHASE 2A INTEGRATION
// ============================================================================

// Main move decision function - Updated for Phase 2A with Minimax Search
pub fn get_move(game: &Game, turn: &i32, board: &Board, you: &Battlesnake) -> Value {
    info!("MOVE {}: === Minimax Search Mode ===", turn);
    
    // Try minimax search with fallback to territorial strategist
    if board.snakes.len() <= 4 && you.health > 15 {
        // Use minimax for smaller games with sufficient health
        let minimax_decision_maker = MinimaxDecisionMaker::new();
        let result = minimax_decision_maker.make_decision(game, board, you);
        info!("MOVE {}: Minimax decision completed", turn);
        result
    } else {
        // Fallback to territorial strategist for complex scenarios
        info!("MOVE {}: Using territorial fallback (snakes: {}, health: {})",
              turn, board.snakes.len(), you.health);
        let territorial_strategist = TerritorialStrategist::new();
        territorial_strategist.make_territorial_decision(game, turn, board, you)
    }
}
