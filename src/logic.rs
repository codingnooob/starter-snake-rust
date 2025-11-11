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
            let territorial_score = SpaceController.get_area_control_score(
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
            let area_denial_score = SpaceController.get_area_control_score(
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

// Game State Simulation Engine for Multi-ply Lookahead
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
    
    // Convert from API structures to simulation state
    pub fn from_game_state(_game: &Game, board: &Board, _you: &Battlesnake) -> SimulatedGameState {
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
// PHASE 2A: MINIMAX SEARCH ALGORITHM WITH ALPHA-BETA PRUNING (2A-3 & 2A-4)
// ============================================================================

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best_move: Direction,
    pub evaluation: f32,
    pub nodes_searched: u64,
    pub time_taken_ms: u128,
    pub depth_reached: u8,
}

pub struct MinimaxSearcher {
    evaluator: IntegratedEvaluator,
    max_depth: u8,
    time_limit_ms: u128,
    nodes_searched: u64,
    start_time: Instant,
}

impl MinimaxSearcher {
    pub fn new(max_depth: u8, time_limit_ms: u128) -> Self {
        Self {
            evaluator: IntegratedEvaluator::new(),
            max_depth,
            time_limit_ms,
            nodes_searched: 0,
            start_time: Instant::now(),
        }
    }
    
    // Main search entry point
    pub fn search_best_move(&mut self, state: &mut SimulatedGameState, our_snake_id: &str) -> SearchResult {
        self.nodes_searched = 0;
        self.start_time = Instant::now();
        
        info!("MINIMAX SEARCH: Starting depth {} search for snake {}", self.max_depth, our_snake_id);
        
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
            };
        }
        
        let our_moves = GameSimulator::generate_moves_for_snake(&state.snakes[our_snake_index.unwrap()], state);
        let mut best_move = our_moves[0]; // Default fallback
        let mut best_evaluation = f32::NEG_INFINITY;
        
        for &our_move in &our_moves {
            if self.is_time_up() {
                info!("MINIMAX SEARCH: Time limit reached, stopping early");
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
                    false // Next level is minimizing (opponents turn)
                );
                
                GameSimulator::undo_moves(state, &move_app);
                
                move_evaluation = move_evaluation.max(evaluation);
            }
            
            if move_evaluation > best_evaluation {
                best_evaluation = move_evaluation;
                best_move = our_move;
            }
        }
        
        let time_taken = self.start_time.elapsed().as_millis();
        info!("MINIMAX SEARCH: Completed. Best move: {:?}, Evaluation: {:.2}, Nodes: {}, Time: {}ms",
              best_move, best_evaluation, self.nodes_searched, time_taken);
        
        SearchResult {
            best_move,
            evaluation: best_evaluation,
            nodes_searched: self.nodes_searched,
            time_taken_ms: time_taken,
            depth_reached: self.max_depth,
        }
    }
    
    // Core minimax algorithm with alpha-beta pruning
    fn minimax(&mut self,
               state: &mut SimulatedGameState,
               our_snake_id: &str,
               depth: u8,
               mut alpha: f32,
               mut beta: f32,
               maximizing: bool) -> f32 {
        
        self.nodes_searched += 1;
        
        // Terminal conditions
        if depth >= self.max_depth || GameSimulator::is_terminal(state) || self.is_time_up() {
            return self.evaluator.evaluate_position(state, our_snake_id);
        }
        
        if maximizing {
            // Maximizing player (us or beneficial moves)
            let mut max_eval = f32::NEG_INFINITY;
            let move_combinations = GameSimulator::generate_all_moves(state);
            
            for moves in move_combinations {
                if self.is_time_up() {
                    break;
                }
                
                let move_app = GameSimulator::apply_moves(state, &moves);
                let eval = self.minimax(state, our_snake_id, depth + 1, alpha, beta, false);
                GameSimulator::undo_moves(state, &move_app);
                
                max_eval = max_eval.max(eval);
                alpha = alpha.max(eval);
                
                // Alpha-beta pruning
                if beta <= alpha {
                    break;
                }
            }
            
            max_eval
        } else {
            // Minimizing player (opponents or unfavorable moves)
            let mut min_eval = f32::INFINITY;
            let move_combinations = GameSimulator::generate_all_moves(state);
            
            for moves in move_combinations {
                if self.is_time_up() {
                    break;
                }
                
                let move_app = GameSimulator::apply_moves(state, &moves);
                let eval = self.minimax(state, our_snake_id, depth + 1, alpha, beta, true);
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
        
        // Create searcher and find best move
        let mut searcher = MinimaxSearcher::new(self.max_depth, self.time_limit_ms);
        let result = searcher.search_best_move(&mut sim_state, &you.id);
        
        info!("MINIMAX DECISION: Selected {:?} (eval: {:.2}, nodes: {}, {}ms)",
              result.best_move, result.evaluation, result.nodes_searched, result.time_taken_ms);
        
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
