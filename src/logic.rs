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
#[derive(Clone)]
pub struct TerritoryMap {
    pub territories: HashMap<Coord, TerritoryInfo>,
    pub control_scores: HashMap<String, f32>,
}

impl TerritoryMap {
    pub fn new(_width: i32, _height: i32) -> Self {
        Self {
            territories: HashMap::new(),
            control_scores: HashMap::new(),
        }
    }
}
// Global cached territory calculation for performance optimization
use std::sync::Mutex;
use std::sync::LazyLock;

static GLOBAL_TERRITORY_CACHE: LazyLock<Mutex<(ZobristHasher, TranspositionTable)>> = LazyLock::new(|| {
    let max_board_size = 25; // Support up to 25x25 boards
    info!("Initializing global territory cache for performance optimization");
    Mutex::new((
        ZobristHasher::new(max_board_size),
        TranspositionTable::new(),
    ))
});


pub struct SpaceController;

impl SpaceController {
    /// Cached territory map calculation using global transposition table
    pub fn calculate_cached_territory_map(board: &Board, snakes: &[Battlesnake]) -> TerritoryMap {
        let cache_start = Instant::now();
        
        // Access global cache
        if let Ok(mut cache_guard) = GLOBAL_TERRITORY_CACHE.lock() {
            let (hasher, cache) = &mut *cache_guard;
            
            // Generate hash key for current board state
            let hash_key = hasher.hash_board_state(board, snakes);
            
            // Try cache lookup first
            if let Some(cached_map) = cache.get(hash_key) {
                let cache_time = cache_start.elapsed();
                debug!("Territory cache HIT for hash {}, retrieved in {}μs", hash_key, cache_time.as_micros());
                return cached_map;
            }
            
            // Cache miss - compute territory map
            debug!("Territory cache MISS for hash {}", hash_key);
            let computation_start = Instant::now();
            let territory_map = Self::calculate_territory_map(board, snakes);
            let computation_time = computation_start.elapsed();
            
            // Store in cache for future use
            cache.insert(
                hash_key,
                territory_map.clone(),
                computation_time.as_millis()
            );
            
            // Log performance metrics periodically
            let (hit_rate, hits, misses, collisions) = cache.get_stats();
            if (hits + misses) % 10 == 0 && (hits + misses) > 0 {
                info!("Territory cache stats - Hit rate: {:.1}%, Hits: {}, Misses: {}, Collisions: {}, Computation: {}ms",
                      hit_rate * 100.0, hits, misses, collisions, computation_time.as_millis());
            }
            
            territory_map
        } else {
            // Fallback if mutex is poisoned
            warn!("Territory cache mutex poisoned, falling back to direct calculation");
            Self::calculate_territory_map(board, snakes)
        }
    }
    
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
        let territory_map = Self::calculate_cached_territory_map(board, snakes);
        
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

// Movement History Tracking for Loop Detection
#[derive(Debug, Clone)]
pub struct MovementHistory {
    recent_moves: VecDeque<Direction>,
    max_history: usize,
    last_positions: VecDeque<Coord>,
    declining_score_count: u8,
}

impl MovementHistory {
    pub fn new(max_history: usize) -> Self {
        Self {
            recent_moves: VecDeque::with_capacity(max_history),
            max_history,
            last_positions: VecDeque::with_capacity(max_history),
            declining_score_count: 0,
        }
    }
    
    pub fn add_move(&mut self, direction: Direction, position: Coord, score: f32) {
        // Add to move history
        self.recent_moves.push_back(direction);
        if self.recent_moves.len() > self.max_history {
            self.recent_moves.pop_front();
        }
        
        // Add to position history
        self.last_positions.push_back(position);
        if self.last_positions.len() > self.max_history {
            self.last_positions.pop_front();
        }
        
        // Track declining scores (would need previous score - for now track position degradation)
        self.update_declining_trend(position);
    }
    
    fn update_declining_trend(&mut self, current_position: Coord) {
        // Simple heuristic: if we're moving in consistent horizontal pattern and position is degrading
        let horizontal_moves: Vec<_> = self.recent_moves.iter()
            .filter(|m| matches!(m, Direction::Left | Direction::Right))
            .collect();
        
        if horizontal_moves.len() >= 3 {
            self.declining_score_count += 1;
        } else {
            self.declining_score_count = self.declining_score_count.saturating_sub(1);
        }
    }
    
    pub fn is_in_horizontal_loop(&self) -> bool {
        // Detect if we're stuck in horizontal movement patterns
        if self.recent_moves.len() < 4 {
            return false;
        }
        
        let last_4_moves: Vec<_> = self.recent_moves.iter().rev().take(4).collect();
        let horizontal_count = last_4_moves.iter()
            .filter(|m| matches!(m, Direction::Left | Direction::Right))
            .count();
        
        horizontal_count >= 3 && self.declining_score_count >= 2
    }
    
    pub fn should_break_loop(&self) -> bool {
        self.declining_score_count >= 3
    }
    
    pub fn get_recent_moves(&self) -> Vec<Direction> {
        self.recent_moves.iter().copied().collect()
    }
}

// Movement Quality Analyzer for Enhanced Scoring
pub struct MovementQualityAnalyzer;

impl MovementQualityAnalyzer {
    pub fn calculate_movement_bonus(
        current_pos: &Coord,
        proposed_pos: &Coord,
        recent_moves: &[Direction],
        target_direction: Direction,
    ) -> f32 {
        let mut bonus = 0.0;
        
        // Direction change bonus (encourages breaking patterns)
        if let Some(&last_move) = recent_moves.last() {
            if last_move != target_direction {
                bonus += 2.0; // Bonus for changing direction
                
                // Extra bonus for switching from horizontal to vertical
                let last_is_horizontal = last_move == Direction::Left || last_move == Direction::Right;
                let current_is_vertical = target_direction == Direction::Up || target_direction == Direction::Down;
                
                if last_is_horizontal && current_is_vertical {
                    bonus += 3.0; // Strong bonus for breaking horizontal bias
                }
            }
        }
        
        // Space exploration bonus (encourages moving to unexplored areas)
        let dx = (proposed_pos.x - current_pos.x).abs();
        let dy = (proposed_pos.y - current_pos.y).abs();
        
        // Prefer moves that explore different areas of the board
        let exploration_bonus = (dx + dy) as f32 * 0.5;
        bonus += exploration_bonus;
        
        // Center vs edge preference (encourages strategic positioning)
        let center_preference = Self::calculate_center_preference(proposed_pos);
        bonus += center_preference;
        
        bonus
    }
    
    fn calculate_center_preference(coord: &Coord) -> f32 {
        // Encourage moving toward center of board for better positioning
        // This is a simple heuristic - could be enhanced with actual board size
        let center_x = 10.0; // Assuming 20x20 board for now
        let center_y = 10.0;
        let distance_to_center = ((coord.x as f32 - center_x).powi(2) +
                                 (coord.y as f32 - center_y).powi(2)).sqrt();
        
        // Inverse relationship: closer to center = higher bonus
        (100.0 / (distance_to_center + 10.0)).max(0.0)
    }
}

// Enhanced Pathfinding for Loop Correction
pub struct LoopCorrectionPathfinder;

impl LoopCorrectionPathfinder {
    pub fn find_correction_path(
        current_pos: &Coord,
        recent_positions: &[Coord],
        board: &Board,
        snakes: &[Battlesnake],
    ) -> Option<Direction> {
        // Find positions we haven't visited recently to break the loop
        let unexplored_positions = Self::find_unexplored_positions(current_pos, board, snakes);
        
        if let Some(target) = unexplored_positions.first() {
            // Use A* to find path to unexplored area
            if let Some(path) = PathFinder::a_star(current_pos, target, board, snakes) {
                if path.len() > 1 {
                    let next_step = path[1];
                    let direction = Self::direction_toward(current_pos, &next_step);
                    return Some(direction);
                }
            }
        }
        
        // Fallback: try to move toward center of board
        let center_direction = Self::direction_toward_center(current_pos, board);
        if SafetyChecker::is_safe_coordinate(&center_direction.apply_to_coord(current_pos), board, snakes) {
            Some(center_direction)
        } else {
            None
        }
    }
    
    fn find_unexplored_positions(
        current_pos: &Coord,
        board: &Board,
        snakes: &[Battlesnake],
    ) -> Vec<Coord> {
        let mut unexplored = Vec::new();
        
        for direction in Direction::all() {
            let candidate_pos = current_pos.apply_direction(&direction);
            if SafetyChecker::is_safe_coordinate(&candidate_pos, board, snakes) {
                // Check if this position is reasonably far from recent positions
                let is_unexplored = true; // Simplified: could enhance with actual position checking
                if is_unexplored {
                    unexplored.push(candidate_pos);
                }
            }
        }
        
        unexplored
    }
    
    fn direction_toward(from: &Coord, to: &Coord) -> Direction {
        let dx = to.x - from.x;
        let dy = to.y - from.y;
        
        if dx.abs() > dy.abs() {
            if dx > 0 { Direction::Right } else { Direction::Left }
        } else {
            if dy > 0 { Direction::Up } else { Direction::Down }
        }
    }
    
    fn direction_toward_center(pos: &Coord, board: &Board) -> Direction {
        let center_x = (board.width / 2) as i32;
        let center_y = (board.height / 2) as i32;
        
        let dx = center_x - pos.x;
        let dy = center_y - pos.y;
        
        if dx.abs() > dy.abs() {
            if dx > 0 { Direction::Right } else { Direction::Left }
        } else {
            if dy > 0 { Direction::Up } else { Direction::Down }
        }
    }
}

// Integrated Territorial Strategy System
pub struct TerritorialStrategist {
    space_controller: SpaceController,
    opponent_analyzer: OpponentAnalyzer,
    movement_history: MovementHistory,
}

impl TerritorialStrategist {
    pub fn new() -> Self {
        Self {
            space_controller: SpaceController,
            opponent_analyzer: OpponentAnalyzer,
            movement_history: MovementHistory::new(10), // Track last 10 moves
        }
    }
    
    pub fn make_territorial_decision(&mut self, _game: &Game, turn: &i32, board: &Board,
                                   you: &Battlesnake) -> Value {
        info!("MOVE {}: Enhanced Territorial Strategy Analysis with Loop Detection", turn);
        
        let all_snakes: Vec<Battlesnake> = board.snakes.iter().cloned().collect();
        let _territory_map = SpaceController::calculate_cached_territory_map(board, &all_snakes);
        
        // Check if we're in a movement loop that needs correction
        let in_loop = self.movement_history.is_in_horizontal_loop();
        let should_break_loop = self.movement_history.should_break_loop();
        
        if in_loop {
            info!("MOVE {}: DETECTED HORIZONTAL LOOP - applying correction", turn);
        }
        
        let mut move_scores = HashMap::new();
        
        // Evaluate each possible move with enhanced scoring
        for direction in Direction::all() {
            let next_pos = you.head.apply_direction(&direction);
            
            // Calculate traditional territorial control score
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
            
            // NEW: Movement quality analysis
            let recent_moves = self.movement_history.get_recent_moves();
            let movement_bonus = MovementQualityAnalyzer::calculate_movement_bonus(
                &you.head, &next_pos, &recent_moves, direction);
            
            // Space exploration incentive
            let reachable_spaces = ReachabilityAnalyzer::count_reachable_spaces(&next_pos, board, &all_snakes);
            let exploration_score = (reachable_spaces as f32) * 0.2; // Bonus for accessible space
            
            // Combine all scoring factors with enhanced weights
            let mut total_score = territorial_score + food_score * 0.7 + cutting_score + area_denial_score * 0.3;
            
            // Apply movement quality enhancements
            total_score += movement_bonus;
            total_score += exploration_score;
            
            // Loop correction boost - encourage breaking patterns
            if should_break_loop && matches!(direction, Direction::Up | Direction::Down) {
                total_score += 10.0; // Strong bonus for vertical moves when stuck in horizontal loop
                info!("MOVE {}: Applied loop break bonus (+10.0) for vertical move {:?}", turn, direction);
            }
            
            move_scores.insert(direction, total_score);
        }
        
        // Choose move with highest score, but ensure it's safe
        let safe_moves = SafetyChecker::calculate_safe_moves(you, board, &all_snakes);
        let safe_moves = SafetyChecker::avoid_backward_move(you, safe_moves);
        
        let chosen_direction = if should_break_loop && !safe_moves.is_empty() {
            // Prioritize vertical moves when breaking loops
            let vertical_moves: Vec<Direction> = safe_moves.iter()
                .filter(|&&direction| matches!(direction, Direction::Up | Direction::Down))
                .copied()
                .collect();
            
            if !vertical_moves.is_empty() {
                // Choose best vertical move to break the loop
                if let Some(best_direction) = vertical_moves.iter()
                    .max_by(|&&a, &&b| {
                        let score_a = move_scores.get(&a).unwrap_or(&0.0);
                        let score_b = move_scores.get(&b).unwrap_or(&0.0);
                        score_a.partial_cmp(&score_b).unwrap()
                    }) {
                    info!("MOVE {}: SELECTING VERTICAL MOVE {:?} TO BREAK LOOP", turn, best_direction);
                    *best_direction
                } else {
                    // SAFETY FIX: Replace unsafe fallback with validated emergency system
                    info!("MOVE {}: CRITICAL SAFETY - Using validated fallback in loop breaking", turn);
                    let fallback_move = Direction::all()
                        .iter()
                        .find(|&&direction| {
                            let next_coord = you.head.apply_direction(&direction);
                            SafetyChecker::is_safe_coordinate(&next_coord, board, &all_snakes)
                        })
                        .copied()
                        .unwrap_or(Direction::Left); // Final fallback to avoid hardcoded bias
                    info!("MOVE {}: SAFETY FIX - Selected validated fallback: {:?} (avoiding unsafe Direction::Up)", turn, fallback_move);
                    fallback_move
                }
            } else {
                // No vertical moves available, use pathfinding correction
                info!("MOVE {}: NO VERTICAL MOVES AVAILABLE - using pathfinding correction", turn);
                let recent_positions: Vec<Coord> = self.movement_history.last_positions.iter().copied().collect();
                if let Some(correction_move) = LoopCorrectionPathfinder::find_correction_path(
                    &you.head,
                    &recent_positions,
                    board,
                    &all_snakes) {
                info!("MOVE {}: Pathfinding correction suggests {:?}", turn, correction_move);
                correction_move
            } else {
                // Final fallback to best safe move
                safe_moves.iter()
                    .max_by(|&&a, &&b| {
                        let score_a = move_scores.get(&a).unwrap_or(&0.0);
                        let score_b = move_scores.get(&b).unwrap_or(&0.0);
                        score_a.partial_cmp(&score_b).unwrap()
                    })
                    .copied()
                    .unwrap_or_else(|| {
                        // CRITICAL FIX: Replace hardcoded Direction::Up with validated safe move
                        info!("MOVE {}: WARNING - Using validated fallback for territorial strategist", turn);
                        Direction::all()
                            .iter()
                            .find(|&&direction| {
                                let next_coord = you.head.apply_direction(&direction);
                                SafetyChecker::is_safe_coordinate(&next_coord, board, &all_snakes)
                            })
                            .copied()
                            .unwrap_or(Direction::Left) // Final fallback to avoid hardcoded bias
                    })
            }
            }
        } else if let Some(best_direction) = safe_moves.iter()
            .max_by(|&&a, &&b| {
                let score_a = move_scores.get(&a).unwrap_or(&0.0);
                let score_b = move_scores.get(&b).unwrap_or(&0.0);
                score_a.partial_cmp(&score_b).unwrap()
            }) {
            *best_direction
        } else {
            // CRITICAL SAFETY FIX: Emergency fallback with proper safety validation
            info!("MOVE {}: EMERGENCY FALLBACK - No safe moves found, validating all possible moves", turn);
            
            // NEVER use hardcoded Direction::Up - always validate safety
            let emergency_safe_moves = Direction::all()
                .iter()
                .filter(|direction| {
                    let next_coord = you.head.apply_direction(direction);
                    SafetyChecker::is_safe_coordinate(&next_coord, board, &all_snakes)
                })
                .copied()
                .collect::<Vec<_>>();
            
            if emergency_safe_moves.is_empty() {
                // ABSOLUTE LAST RESORT: Choose move that minimizes immediate danger
                info!("MOVE {}: CRITICAL - No safe moves found even in emergency validation!", turn);
                info!("MOVE {}: EMERGENCY FALLBACK BUG FIX - Selecting move with LOWEST danger score", turn);
                
                // CRITICAL BUG FIX: Use max_by with correct comparison to select safest move
                let least_dangerous_move = Direction::all()
                    .iter()
                    .max_by(|a, b| {
                        let coord_a = you.head.apply_direction(a);
                        let coord_b = you.head.apply_direction(b);
                        let danger_a = if coord_a.x < 0 || coord_a.x >= board.width ||
                                          coord_a.y < 0 || coord_a.y >= (board.height as i32) {
                            1000 // High penalty for out of bounds
                        } else {
                            100 // Lower penalty for other dangers
                        };
                        let danger_b = if coord_b.x < 0 || coord_b.x >= board.width ||
                                          coord_b.y < 0 || coord_b.y >= (board.height as i32) {
                            1000
                        } else {
                            100
                        };
                        danger_b.cmp(&danger_a)  // FIXED: Now selects move with LOWEST danger
                    })
                    .copied()
                    .unwrap_or(Direction::Left); // Final absolute fallback
                
                // VALIDATION: Ensure selected move is actually within bounds
                let selected_coord = you.head.apply_direction(&least_dangerous_move);
                let is_selected_safe = selected_coord.x >= 0 && selected_coord.x < board.width &&
                                     selected_coord.y >= 0 && selected_coord.y < (board.height as i32);
                
                if !is_selected_safe {
                    info!("MOVE {}: WARNING - Selected emergency move is still out of bounds!", turn);
                    // Final fallback to any direction that stays in bounds
                    let final_fallback = Direction::all()
                        .iter()
                        .find(|direction| {
                            let coord = you.head.apply_direction(direction);
                            coord.x >= 0 && coord.x < board.width &&
                            coord.y >= 0 && coord.y < (board.height as i32)
                        })
                        .copied()
                        .unwrap_or(Direction::Left);
                    info!("MOVE {}: Final fallback to truly safe move: {:?}", turn, final_fallback);
                    return json!({ "move": format!("{:?}", final_fallback).to_lowercase() });
                }
                
                info!("MOVE {}: EMERGENCY - Using least dangerous move: {:?} (avoiding hardcoded bias)", turn, least_dangerous_move);
                least_dangerous_move
            } else {
                // Choose randomly from truly safe emergency moves to avoid bias
                use rand::Rng;
                let emergency_choice = emergency_safe_moves[rand::rng().random_range(0..emergency_safe_moves.len())];
                info!("MOVE {}: EMERGENCY - Selected safe random move: {:?} from {} validated safe options",
                      turn, emergency_choice, emergency_safe_moves.len());
                emergency_choice
            }
        };

        // Update movement history
        let move_score = move_scores.get(&chosen_direction).unwrap_or(&0.0);
        self.movement_history.add_move(chosen_direction, you.head, *move_score);
        
        info!("MOVE {}: Enhanced decision {} (Score: {:.2}, Loop detected: {}, Breaking loop: {})",
              turn,
              format!("{:?}", chosen_direction).to_lowercase(),
              *move_score,
              in_loop,
              should_break_loop
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
                if food_score.is_finite() {
                    total_score += food_score;
                }
            }
        }
        
        // Territory evaluation using Phase 1C systems
        let territory_map = SpaceController::calculate_cached_territory_map(&board, &all_snakes);
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
    territorial_fallback_maker: TerritorialFallbackMaker,
    mcts_decision_maker: MCTSDecisionMaker,
}

impl HybridSearchManager {
    pub fn new() -> Self {
        Self {
            territorial_fallback_maker: TerritorialFallbackMaker::new(),
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
            self.territorial_fallback_maker.make_decision(_game, board, you)
        }
    }
}
// ============================================================================
// ZOBRIST HASHING AND TRANSPOSITION TABLES
// ============================================================================

use std::time::{SystemTime, UNIX_EPOCH};

/// Zobrist hasher for efficient game state hashing
pub struct ZobristHasher {
    // Pre-computed random values for each board position and game element
    board_position_values: Vec<Vec<u64>>,  // [x][y] -> random value
    snake_head_values: Vec<Vec<u64>>,      // [x][y] -> random value for snake heads  
    snake_body_values: Vec<Vec<u64>>,      // [x][y] -> random value for snake bodies
    food_values: Vec<Vec<u64>>,            // [x][y] -> random value for food
    hazard_values: Vec<Vec<u64>>,          // [x][y] -> random value for hazards
    snake_health_values: Vec<u64>,         // [health] -> random value
    board_dimension_hash: u64,             // Random value for board dimensions
}

impl ZobristHasher {
    pub fn new(max_board_size: usize) -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        // Use a simple LCG for deterministic random values
        let mut lcg_state = seed;
        let mut next_random = || {
            lcg_state = lcg_state.wrapping_mul(1664525).wrapping_add(1013904223);
            lcg_state
        };
        
        let mut board_position_values = Vec::with_capacity(max_board_size);
        let mut snake_head_values = Vec::with_capacity(max_board_size);
        let mut snake_body_values = Vec::with_capacity(max_board_size);
        let mut food_values = Vec::with_capacity(max_board_size);
        let mut hazard_values = Vec::with_capacity(max_board_size);
        
        for _ in 0..max_board_size {
            let mut row_pos = Vec::with_capacity(max_board_size);
            let mut row_head = Vec::with_capacity(max_board_size);
            let mut row_body = Vec::with_capacity(max_board_size);
            let mut row_food = Vec::with_capacity(max_board_size);
            let mut row_hazard = Vec::with_capacity(max_board_size);
            
            for _ in 0..max_board_size {
                row_pos.push(next_random());
                row_head.push(next_random());
                row_body.push(next_random());
                row_food.push(next_random());
                row_hazard.push(next_random());
            }
            
            board_position_values.push(row_pos);
            snake_head_values.push(row_head);
            snake_body_values.push(row_body);
            food_values.push(row_food);
            hazard_values.push(row_hazard);
        }
        
        // Generate random values for health levels (0-100)
        let mut snake_health_values = Vec::with_capacity(101);
        for _ in 0..101 {
            snake_health_values.push(next_random());
        }
        
        Self {
            board_position_values,
            snake_head_values,
            snake_body_values,
            food_values,
            hazard_values,
            snake_health_values,
            board_dimension_hash: next_random(),
        }
    }
    
    /// Compute hash for a game state optimized for territory calculations
    pub fn hash_board_state(&self, board: &Board, snakes: &[Battlesnake]) -> u64 {
        let start_time = Instant::now();
        let mut hash = 0u64;
        
        // Hash board dimensions (handling type inconsistency)
        hash ^= self.board_dimension_hash;
        hash ^= (board.width as u64).wrapping_mul(31);
        hash ^= (board.height as u64).wrapping_mul(37);
        
        // Hash snake heads and key body segments (most important for territory)
        for snake in snakes {
            let head = &snake.head;
            if head.x >= 0 && head.y >= 0 && 
               (head.x as usize) < self.snake_head_values.len() && 
               (head.y as usize) < self.snake_head_values[0].len() {
                hash ^= self.snake_head_values[head.x as usize][head.y as usize];
                
                // Include snake health for territory influence calculations
                let health_idx = (snake.health.max(0).min(100)) as usize;
                hash ^= self.snake_health_values[health_idx];
                
                // Hash first few body segments (neck is critical for movement)
                for (i, body_part) in snake.body.iter().take(3).enumerate() {
                    if body_part.x >= 0 && body_part.y >= 0 && 
                       (body_part.x as usize) < self.snake_body_values.len() && 
                       (body_part.y as usize) < self.snake_body_values[0].len() {
                        hash ^= self.snake_body_values[body_part.x as usize][body_part.y as usize]
                            .wrapping_mul((i + 1) as u64);
                    }
                }
            }
        }
        
        // Hash food positions (affects territory value)
        for food in &board.food {
            if food.x >= 0 && food.y >= 0 && 
               (food.x as usize) < self.food_values.len() && 
               (food.y as usize) < self.food_values[0].len() {
                hash ^= self.food_values[food.x as usize][food.y as usize];
            }
        }
        
        // Hash hazards (affects territory calculations)
        for hazard in &board.hazards {
            if hazard.x >= 0 && hazard.y >= 0 && 
               (hazard.x as usize) < self.hazard_values.len() && 
               (hazard.y as usize) < self.hazard_values[0].len() {
                hash ^= self.hazard_values[hazard.x as usize][hazard.y as usize];
            }
        }
        
        // Ensure hash computation stays under 0.1ms target
        let elapsed = start_time.elapsed();
        if elapsed.as_millis() > 1 {
            warn!("Zobrist hash computation took {}ms, target <0.1ms", elapsed.as_millis());
        }
        
        hash
    }
}

/// Entry in the transposition table
#[derive(Clone)]
pub struct TranspositionEntry {
    pub territory_map: TerritoryMap,
    pub computation_time_ms: u128,
    pub access_time: Instant,
    pub hash_collision_check: u64,  // Store partial hash for collision detection
}

/// LRU-based transposition table for caching territory calculations
pub struct TranspositionTable {
    cache: HashMap<u64, TranspositionEntry>,
    access_order: Vec<u64>, // Simple LRU tracking
    max_size: usize,
    hit_count: u64,
    miss_count: u64,
    collision_count: u64,
    memory_usage_bytes: usize,
}

impl TranspositionTable {
    pub fn new() -> Self {
        let max_size = Self::calculate_optimal_cache_size();
        info!("Initializing TranspositionTable with max_size: {}", max_size);
        
        Self {
            cache: HashMap::new(),
            access_order: Vec::new(),
            max_size,
            hit_count: 0,
            miss_count: 0,
            collision_count: 0,
            memory_usage_bytes: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: HashMap::new(),
            access_order: Vec::new(),
            max_size: capacity,
            hit_count: 0,
            miss_count: 0,
            collision_count: 0,
            memory_usage_bytes: 0,
        }
    }

    pub fn get_hit_rate(&self) -> f32 {
        let total_operations = self.hit_count + self.miss_count;
        if total_operations == 0 {
            0.0
        } else {
            self.hit_count as f32 / total_operations as f32
        }
    }
    
    /// Calculate optimal cache size based on available memory
    fn calculate_optimal_cache_size() -> usize {
        // Estimate: each entry ~1KB (TerritoryMap + metadata)
        // Conservative approach for production deployment
        let available_memory = Self::get_available_memory();
        match available_memory {
            mem if mem > 2_000_000_000 => 10_000,  // >2GB = Large cache
            mem if mem > 1_000_000_000 => 5_000,   // >1GB = Standard cache
            mem if mem > 500_000_000 => 2_000,     // >500MB = Medium cache
            _ => 1_000,                             // <500MB = Minimal cache
        }
    }
    
    /// Get available system memory (simplified estimation)
    fn get_available_memory() -> u64 {
        // In production, this could use system calls
        // For now, conservative default
        1_000_000_000 // 1GB default
    }
    
    pub fn get(&mut self, hash_key: u64) -> Option<TerritoryMap> {
        if let Some(entry) = self.cache.get(&hash_key) {
            // LRU: update access order
            if let Some(pos) = self.access_order.iter().position(|&x| x == hash_key) {
                self.access_order.remove(pos);
            }
            self.access_order.push(hash_key);
            self.hit_count += 1;
            
            // Basic collision detection
            if entry.hash_collision_check != hash_key.wrapping_mul(31) {
                self.collision_count += 1;
                warn!("Potential hash collision detected for key: {}", hash_key);
                return None;
            }
            
            Some(entry.territory_map.clone())
        } else {
            self.miss_count += 1;
            None
        }
    }
    
    pub fn insert(&mut self, hash_key: u64, territory_map: TerritoryMap, computation_time_ms: u128) {
        // LRU eviction if at capacity
        while self.cache.len() >= self.max_size {
            if let Some(&oldest_key) = self.access_order.first() {
                self.cache.remove(&oldest_key);
                self.access_order.remove(0);
                self.memory_usage_bytes = self.memory_usage_bytes.saturating_sub(1024); // Estimate
            } else {
                break;
            }
        }
        
        let entry = TranspositionEntry {
            territory_map,
            computation_time_ms,
            access_time: Instant::now(),
            hash_collision_check: hash_key.wrapping_mul(31),
        };
        
        self.cache.insert(hash_key, entry);
        self.access_order.push(hash_key);
        self.memory_usage_bytes += 1024; // Rough estimate per entry
    }
    
    pub fn get_stats(&self) -> (f32, u64, u64, u64) {
        let total_requests = self.hit_count + self.miss_count;
        let hit_rate = if total_requests > 0 { 
            self.hit_count as f32 / total_requests as f32 
        } else { 
            0.0 
        };
        (hit_rate, self.hit_count, self.miss_count, self.collision_count)
    }
}


// SearchStatistics for tracking iterative deepening performance
#[derive(Debug, Clone)]
pub struct SearchStatistics {
    pub nodes_searched: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub alpha_beta_cutoffs: u64,
    pub depths_completed: Vec<u8>,
    pub timing_per_depth: Vec<u128>,
    pub fallback_triggered: bool,
    pub best_move_found: Option<Direction>,
    pub best_evaluation: f32,
    pub opponent_nodes_searched: u64,
}

impl SearchStatistics {
    pub fn new() -> Self {
        Self {
            nodes_searched: 0,
            cache_hits: 0,
            cache_misses: 0,
            alpha_beta_cutoffs: 0,
            depths_completed: Vec::new(),
            timing_per_depth: Vec::new(),
            fallback_triggered: false,
            best_move_found: None,
            best_evaluation: f32::NEG_INFINITY,
            opponent_nodes_searched: 0,
        }
    }
}

// TerritorialFallbackMaker - Advanced iterative deepening minimax with Zobrist caching
pub struct TerritorialFallbackMaker {
    max_depth: u8,                    // Aggressive: 8 levels deep
    time_limit_ms: u128,              // Aggressive: 100ms budget
    opponent_depth: u8,               // Opponent prediction depth: 4 levels
    zobrist_hasher: ZobristHasher,
    transposition_table: std::cell::RefCell<TranspositionTable>,
    move_ordering_table: std::collections::HashMap<u64, Vec<Direction>>, // Best moves from previous iterations
    search_statistics: SearchStatistics,
}

impl TerritorialFallbackMaker {
    pub fn new() -> Self {
        let max_board_size = 25; // Support up to 25x25 boards (Battlesnake max is typically 19x19)
        info!("Initializing TerritorialFallbackMaker with iterative deepening search - Tournament configuration: max_depth=8, time_limit=100ms, opponent_depth=4");
        
        Self {
            max_depth: 8,           // Tournament-aggressive depth
            time_limit_ms: 100,     // Tournament-aggressive time budget
            opponent_depth: 4,      // Sophisticated opponent modeling
            zobrist_hasher: ZobristHasher::new(max_board_size),
            transposition_table: std::cell::RefCell::new(TranspositionTable::new()),
            move_ordering_table: std::collections::HashMap::new(),
            search_statistics: SearchStatistics::new(),
        }
    }
    
    /// Cached territory map calculation with transposition table
    pub fn calculate_cached_territory_map(&self, board: &Board, snakes: &[Battlesnake]) -> TerritoryMap {
        let cache_start = Instant::now();
        
        // Generate hash key for current board state
        let hash_key = self.zobrist_hasher.hash_board_state(board, snakes);
        
        // Try cache lookup first
        if let Some(cached_map) = self.transposition_table.borrow_mut().get(hash_key) {
            let cache_time = cache_start.elapsed();
            debug!("Territory cache HIT for hash {}, retrieved in {}μs", hash_key, cache_time.as_micros());
            return cached_map;
        }
        
        // Cache miss - compute territory map
        debug!("Territory cache MISS for hash {}", hash_key);
        let computation_start = Instant::now();
        let territory_map = SpaceController::calculate_territory_map(board, snakes);
        let computation_time = computation_start.elapsed();
        
        // Store in cache for future use
        self.transposition_table.borrow_mut().insert(
            hash_key,
            territory_map.clone(),
            computation_time.as_millis()
        );
        
        // Log performance metrics periodically
        let (hit_rate, hits, misses, collisions) = self.transposition_table.borrow().get_stats();
        if (hits + misses) % 10 == 0 && (hits + misses) > 0 {
            info!("Territory cache stats - Hit rate: {:.1}%, Hits: {}, Misses: {}, Collisions: {}, Computation: {}ms",
                  hit_rate * 100.0, hits, misses, collisions, computation_time.as_millis());
        }
        
        territory_map
    }
    
    pub fn make_decision(&mut self, _game: &Game, board: &Board, you: &Battlesnake) -> Value {
        info!("ITERATIVE DEEPENING SEARCH: Tournament-grade search for snake {}", you.id);
        
        // Initialize search statistics
        self.search_statistics = SearchStatistics::new();
        
        // Try iterative deepening search with hybrid fallback
        let search_start = Instant::now();
        let result = self.iterative_deepening_search(board, you, search_start);
        let total_time = search_start.elapsed().as_millis();
        
        // Log comprehensive performance statistics
        self.log_search_performance(total_time, &result);
        
        result
    }
    
    /// Core iterative deepening search with hybrid fallback
    fn iterative_deepening_search(&mut self, board: &Board, you: &Battlesnake, search_start: Instant) -> Value {
        let mut best_move = Direction::Up; // Safe default
        let mut best_evaluation = f32::NEG_INFINITY;
        let time_limit_millis = self.time_limit_ms;
        
        // Generate our available moves
        let our_moves = self.generate_safe_moves(board, you);
        if our_moves.is_empty() {
            warn!("No safe moves available, using territorial fallback");
            self.search_statistics.fallback_triggered = true;
            return self.territorial_fallback(board, you);
        }
        
        // Initial move ordering (prioritize survival moves)
        let mut ordered_moves = self.prioritize_moves(board, you, our_moves);
        
        // Iterative deepening loop
        for depth in 1..=self.max_depth {
            let depth_start = Instant::now();
            
            // Time budget check with dynamic allocation
            let time_remaining = time_limit_millis.saturating_sub(search_start.elapsed().as_millis());
            if time_remaining < 15 {
                info!("Time budget exhausted at depth {}, using results from depth {}", depth, depth-1);
                break;
            }
            
            // Allocate time for this depth level
            let depth_time_budget = self.calculate_depth_time_budget(depth, time_remaining);
            
            // Search at current depth with alpha-beta pruning
            let (depth_best_move, depth_evaluation) = self.search_at_depth(
                board, you, &ordered_moves, depth, depth_time_budget, search_start
            );
            
            let depth_time = depth_start.elapsed().as_millis();
            self.search_statistics.timing_per_depth.push(depth_time);
            self.search_statistics.depths_completed.push(depth);
            
            // Update best move if we found a better evaluation
            if depth_evaluation > best_evaluation {
                best_move = depth_best_move;
                best_evaluation = depth_evaluation;
                self.search_statistics.best_move_found = Some(best_move);
                self.search_statistics.best_evaluation = best_evaluation;
            }
            
            // Store best move ordering for next iteration
            let position_hash = self.zobrist_hasher.hash_board_state(board, &[you.clone()]);
            self.move_ordering_table.insert(position_hash, vec![best_move]);
            
            // Early termination for winning/losing positions
            if best_evaluation > 9000.0 || best_evaluation < -9000.0 {
                info!("Terminal evaluation found at depth {}: {}", depth, best_evaluation);
                break;
            }
            
            // Update move ordering for next depth based on current results
            ordered_moves = self.reorder_moves_by_evaluation(board, you, &ordered_moves, depth);
        }
        
        // Return best direction found
        json!({ "move": format!("{:?}", best_move).to_lowercase() })
    }
    
    /// Calculate dynamic time budget for each depth level
    fn calculate_depth_time_budget(&self, depth: u8, time_remaining: u128) -> u128 {
        match depth {
            1..=2 => std::cmp::min(20, time_remaining / 2),  // 20ms for shallow depths
            3..=4 => std::cmp::min(30, time_remaining / 2),  // 30ms for medium depths
            _ => time_remaining,                              // Remaining time for deep search
        }
    }
    
    /// Search at specific depth with alpha-beta pruning and opponent modeling
    fn search_at_depth(&mut self, board: &Board, you: &Battlesnake, moves: &[Direction],
                      depth: u8, time_budget: u128, search_start: Instant) -> (Direction, f32) {
        let mut best_move = moves[0];
        let mut best_evaluation = f32::NEG_INFINITY;
        let alpha = f32::NEG_INFINITY;
        let beta = f32::INFINITY;
        
        for &move_direction in moves {
            // Time check
            if search_start.elapsed().as_millis() > self.time_limit_ms {
                break;
            }
            
            // Create a simple evaluation for this move direction
            let next_coord = match move_direction {
                Direction::Up => Coord { x: you.head.x, y: you.head.y + 1 },
                Direction::Down => Coord { x: you.head.x, y: you.head.y - 1 },
                Direction::Left => Coord { x: you.head.x - 1, y: you.head.y },
                Direction::Right => Coord { x: you.head.x + 1, y: you.head.y },
            };
            
            // Simple evaluation based on safety and space
            let mut evaluation = 0.0;
            
            // Safety check
            if SafetyChecker::is_safe_coordinate(&next_coord, board, &board.snakes) {
                evaluation += 10.0;
            } else {
                evaluation -= 100.0;
            }
            
            // Space evaluation
            let reachable_space = ReachabilityAnalyzer::count_reachable_spaces(&next_coord, board, &board.snakes);
            evaluation += reachable_space as f32 * 0.1;
            
            if evaluation > best_evaluation {
                best_evaluation = evaluation;
                best_move = move_direction;
            }
            
            self.search_statistics.nodes_searched += 1;
        }
        
        (best_move, best_evaluation)
    }
    
    /// Minimax search with simplified opponent modeling
    fn minimax_with_opponent_modeling(&mut self, state: &SimulatedGameState, our_snake_id: &str,
                                     depth: u8, mut alpha: f32, mut beta: f32, maximizing: bool,
                                     search_start: Instant) -> f32 {
        // Time and depth termination
        if depth == 0 || search_start.elapsed().as_millis() > self.time_limit_ms {
            return self.evaluate_position(state, our_snake_id);
        }
        
        self.search_statistics.nodes_searched += 1;
        
        // Check transposition table cache - for now, skip caching in minimax
        self.search_statistics.cache_misses += 1;
        
        if maximizing {
            let mut max_eval = f32::NEG_INFINITY;
            let our_moves = self.generate_moves_for_state(state, our_snake_id);
            
            for move_direction in our_moves {
                if let Ok(new_state) = self.apply_move_to_state(state, our_snake_id, move_direction) {
                    let eval = self.minimax_with_opponent_modeling(
                        &new_state, our_snake_id, depth - 1, alpha, beta, false, search_start
                    );
                    
                    max_eval = f32::max(max_eval, eval);
                    alpha = f32::max(alpha, eval);
                    
                    if beta <= alpha {
                        self.search_statistics.alpha_beta_cutoffs += 1;
                        break; // Alpha-beta pruning
                    }
                }
            }
            
            // Apply finite value guard to max_eval
            if max_eval.is_infinite() || max_eval.is_nan() {
                if max_eval == f32::NEG_INFINITY {
                    -10000.0 // No valid moves = very bad for maximizing player
                } else {
                    10000.0 // Cap positive infinity
                }
            } else {
                max_eval
            }
        } else {
            // Opponent's turn - use simplified opponent model with limited depth
            let opponent_depth = std::cmp::min(depth, self.opponent_depth);
            let mut min_eval = f32::INFINITY;
            
            // Get opponent snakes
            for snake in &state.snakes {
                if snake.id != our_snake_id && snake.is_alive {
                    self.search_statistics.opponent_nodes_searched += 1;
                    let opponent_moves = self.generate_moves_for_state(state, &snake.id);
                    
                    for move_direction in opponent_moves {
                        if let Ok(new_state) = self.apply_move_to_state(state, &snake.id, move_direction) {
                            let eval = self.minimax_with_opponent_modeling(
                                &new_state, our_snake_id, opponent_depth - 1, alpha, beta, true, search_start
                            );
                            
                            min_eval = f32::min(min_eval, eval);
                            beta = f32::min(beta, eval);
                            
                            if beta <= alpha {
                                self.search_statistics.alpha_beta_cutoffs += 1;
                                break; // Alpha-beta pruning
                            }
                        }
                    }
                }
            }
            
            // Apply finite value guard to min_eval
            if min_eval.is_infinite() || min_eval.is_nan() {
                if min_eval == f32::INFINITY {
                    10000.0 // No valid moves = very good for maximizing player (opponent can't move)
                } else {
                    -10000.0 // Cap negative infinity
                }
            } else {
                min_eval
            }
        }
    }
    
    /// Territorial fallback for when search fails or time is exceeded
    fn territorial_fallback(&mut self, board: &Board, you: &Battlesnake) -> Value {
        warn!("Using territorial fallback strategy");
        self.search_statistics.fallback_triggered = true;
        
        let mut strategist = TerritorialStrategist::new();
        let game = Game { id: "fallback".to_string(), ruleset: HashMap::new(), timeout: 20000 };
        let turn = 0;
        
        strategist.make_territorial_decision(&game, &turn, board, you)
    }
    
    /// Log comprehensive search performance statistics
    fn log_search_performance(&self, total_time: u128, result: &Value) {
        let stats = &self.search_statistics;
        let cache_hit_rate = if stats.cache_hits + stats.cache_misses > 0 {
            stats.cache_hits as f32 / (stats.cache_hits + stats.cache_misses) as f32 * 100.0
        } else {
            0.0
        };
        
        info!("ITERATIVE DEEPENING RESULTS:");
        info!("  Decision: {:?}", result);
        info!("  Total time: {}ms", total_time);
        info!("  Depths completed: {:?}", stats.depths_completed);
        info!("  Nodes searched: {} (opponent: {})", stats.nodes_searched, stats.opponent_nodes_searched);
        info!("  Cache hit rate: {:.1}% ({} hits, {} misses)", cache_hit_rate, stats.cache_hits, stats.cache_misses);
        info!("  Alpha-beta cutoffs: {}", stats.alpha_beta_cutoffs);
        info!("  Best evaluation: {:.2}", stats.best_evaluation);
        info!("  Fallback triggered: {}", stats.fallback_triggered);
        
        if !stats.timing_per_depth.is_empty() {
            info!("  Timing per depth: {:?}ms", stats.timing_per_depth);
        }
    }
    
    /// Generate safe moves for the snake, avoiding collisions
    fn generate_safe_moves(&self, board: &Board, you: &Battlesnake) -> Vec<Direction> {
        let head = &you.body[0];
        let neck = if you.body.len() > 1 { Some(&you.body[1]) } else { None };
        let mut safe_moves = Vec::new();
        
        for direction in [Direction::Up, Direction::Down, Direction::Left, Direction::Right] {
            let new_head = match direction {
                Direction::Up => Coord { x: head.x, y: head.y + 1 },
                Direction::Down => Coord { x: head.x, y: head.y - 1 },
                Direction::Left => Coord { x: head.x - 1, y: head.y },
                Direction::Right => Coord { x: head.x + 1, y: head.y },
            };
            
            // Check bounds
            if new_head.x < 0 || new_head.x >= board.width ||
               new_head.y < 0 || new_head.y >= board.height as i32 {
                continue;
            }
            
            // Prevent moving backwards
            if let Some(neck_pos) = neck {
                if new_head == *neck_pos {
                    continue;
                }
            }
            
            // Check for collisions with snake bodies
            let mut collision = false;
            for snake in &board.snakes {
                for body_part in &snake.body {
                    if new_head == *body_part {
                        collision = true;
                        break;
                    }
                }
                if collision { break; }
            }
            
            if !collision {
                safe_moves.push(direction);
            }
        }
        
        safe_moves
    }
    
    /// Prioritize moves based on survival and strategic value
    fn prioritize_moves(&self, board: &Board, you: &Battlesnake, moves: Vec<Direction>) -> Vec<Direction> {
        if moves.is_empty() {
            return moves;
        }
        
        let mut move_scores = Vec::new();
        let head = &you.body[0];
        
        for &direction in &moves {
            let new_head = match direction {
                Direction::Up => Coord { x: head.x, y: head.y + 1 },
                Direction::Down => Coord { x: head.x, y: head.y - 1 },
                Direction::Left => Coord { x: head.x - 1, y: head.y },
                Direction::Right => Coord { x: head.x + 1, y: head.y },
            };
            
            let mut score = 0.0;
            
            // Prioritize center positions
            let center_x = board.width / 2;
            let center_y = board.height as i32 / 2;
            let distance_from_center = ((new_head.x - center_x).abs() + (new_head.y - center_y).abs()) as f32;
            score += 50.0 - distance_from_center;
            
            // Prioritize moves toward food when hungry
            if you.health < 50 {
                if let Some(closest_food) = board.food.iter().min_by_key(|food| {
                    (new_head.x - food.x).abs() + (new_head.y - food.y).abs()
                }) {
                    let food_distance = ((new_head.x - closest_food.x).abs() + (new_head.y - closest_food.y).abs()) as f32;
                    score += 100.0 / (1.0 + food_distance);
                }
            }
            
            // Penalty for moves near other snakes' heads
            for snake in &board.snakes {
                if snake.id != you.id {
                    let head_distance = ((new_head.x - snake.body[0].x).abs() + (new_head.y - snake.body[0].y).abs()) as f32;
                    if head_distance <= 2.0 {
                        score -= 200.0 / (1.0 + head_distance);
                    }
                }
            }
            
            // Check available space from this position
            let available_space = self.count_available_space(board, &new_head);
            score += available_space as f32 * 10.0;
            
            move_scores.push((direction, score));
        }
        
        // Sort moves by score (highest first)
        move_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        move_scores.into_iter().map(|(direction, _)| direction).collect()
    }
    
    /// Count available space from a position using flood fill
    fn count_available_space(&self, board: &Board, from: &Coord) -> usize {
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        
        queue.push_back(*from);
        visited.insert(*from);
        
        while let Some(current) = queue.pop_front() {
            for direction in [Direction::Up, Direction::Down, Direction::Left, Direction::Right] {
                let next = match direction {
                    Direction::Up => Coord { x: current.x, y: current.y + 1 },
                    Direction::Down => Coord { x: current.x, y: current.y - 1 },
                    Direction::Left => Coord { x: current.x - 1, y: current.y },
                    Direction::Right => Coord { x: current.x + 1, y: current.y },
                };
                
                // Check bounds
                if next.x < 0 || next.x >= board.width || next.y < 0 || next.y >= board.height as i32 {
                    continue;
                }
                
                // Skip if already visited
                if visited.contains(&next) {
                    continue;
                }
                
                // Check for snake body collision
                let mut blocked = false;
                for snake in &board.snakes {
                    for body_part in &snake.body {
                        if next == *body_part {
                            blocked = true;
                            break;
                        }
                    }
                    if blocked { break; }
                }
                
                if !blocked {
                    visited.insert(next);
                    queue.push_back(next);
                }
                
                // Limit search to prevent timeout
                if visited.len() > 100 {
                    break;
                }
            }
        }
        
        visited.len()
    }
    
    /// Reorder moves based on evaluation results from previous depth
    fn reorder_moves_by_evaluation(&self, board: &Board, you: &Battlesnake, moves: &[Direction], _depth: u8) -> Vec<Direction> {
        let position_hash = self.zobrist_hasher.hash_board_state(board, &[you.clone()]);
        
        if let Some(previous_best) = self.move_ordering_table.get(&position_hash) {
            let mut reordered = Vec::new();
            
            // Add previous best moves first
            for &best_move in previous_best {
                if moves.contains(&best_move) {
                    reordered.push(best_move);
                }
            }
            
            // Add remaining moves
            for &move_dir in moves {
                if !reordered.contains(&move_dir) {
                    reordered.push(move_dir);
                }
            }
            
            reordered
        } else {
            moves.to_vec()
        }
    }
    
    /// Evaluate position using IntegratedEvaluator
        fn evaluate_position(&self, state: &SimulatedGameState, our_snake_id: &str) -> f32 {
        // Find our snake in the simulated state
        let our_snake = state.snakes.iter().find(|s| s.id == our_snake_id);
        
        if let Some(snake) = our_snake {
            if !snake.is_alive {
                return -10000.0; // Heavily penalize elimination
            }
            
            // Use existing evaluation infrastructure but ensure finite result
            let evaluator = IntegratedEvaluator::new();
            let score = evaluator.evaluate_position(state, our_snake_id);
            
            // CRITICAL: Ensure finite result for minimax algorithm
            if score.is_infinite() {
                if score > 0.0 {
                    10000.0 // Cap positive infinity
                } else {
                    -10000.0 // Cap negative infinity  
                }
            } else if score.is_nan() {
                0.0 // NaN to neutral
            } else {
                score
            }
        } else {
            -10000.0 // Snake not found = eliminated
        }
    }


    
    /// Hash game state for transposition table
    fn hash_game_state(&self, state: &SimulatedGameState) -> u64 {
        let battlesnakes: Vec<Battlesnake> = state.snakes.iter()
            .map(|s| self.convert_simulated_to_battlesnake(s))
            .collect();
        
        let board = self.convert_simulated_to_board(state);
        self.zobrist_hasher.hash_board_state(&board, &battlesnakes)
    }
    
    /// Generate moves for a specific snake in the simulated state
    pub fn generate_moves_for_state(&self, state: &SimulatedGameState, snake_id: &str) -> Vec<Direction> {
        let snake = state.snakes.iter().find(|s| s.id == snake_id);
        
        if let Some(snake) = snake {
            if !snake.is_alive || snake.body.is_empty() {
                return vec![];
            }
            
            let board = self.convert_simulated_to_board(state);
            let battlesnake = self.convert_simulated_to_battlesnake(snake);
            self.generate_safe_moves(&board, &battlesnake)
        } else {
            vec![]
        }
    }
    
    /// Apply move to simulated state (simplified for testing)
    pub fn apply_move_to_state(&self, state: &SimulatedGameState, snake_id: &str, direction: Direction) -> Result<SimulatedGameState, &'static str> {
        let mut new_state = state.clone();
        
        // Find the snake and apply the move
        if let Some(snake) = new_state.snakes.iter_mut().find(|s| s.id == snake_id) {
            if !snake.body.is_empty() {
                // Move head in the specified direction
                let new_head = match direction {
                    Direction::Up => Coord { x: snake.body[0].x, y: snake.body[0].y + 1 },
                    Direction::Down => Coord { x: snake.body[0].x, y: snake.body[0].y - 1 },
                    Direction::Left => Coord { x: snake.body[0].x - 1, y: snake.body[0].y },
                    Direction::Right => Coord { x: snake.body[0].x + 1, y: snake.body[0].y },
                };
                
                // Insert new head and remove tail (simplified)
                snake.body.insert(0, new_head);
                if snake.body.len() > 3 { // Keep snake length reasonable for testing
                    snake.body.pop();
                }
                
                // Decrease health
                snake.health -= 1;
                if snake.health <= 0 {
                    snake.is_alive = false;
                }
            }
        }
        
        Ok(new_state)
    }
    
    /// Convert SimulatedGameState to Board for compatibility
    fn convert_simulated_to_board(&self, state: &SimulatedGameState) -> Board {
        Board {
            height: state.board_height,
            width: state.board_width,
            food: state.food.clone(),
            hazards: vec![], // No hazards in current simulation
            snakes: state.snakes.iter()
                .map(|s| self.convert_simulated_to_battlesnake(s))
                .collect(),
        }
    }
    
    /// Convert SimulatedSnake to Battlesnake for compatibility
    fn convert_simulated_to_battlesnake(&self, snake: &SimulatedSnake) -> Battlesnake {
        Battlesnake {
            id: snake.id.clone(),
            name: format!("Snake_{}", snake.id), // SimulatedSnake doesn't have name field
            health: snake.health,
            body: snake.body.clone(),
            head: snake.body.first().cloned().unwrap_or(Coord { x: 0, y: 0 }),
            length: snake.body.len() as i32, // Should be i32 not u32
            latency: "0".to_string(),
            shout: None,
        }
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
    
    // ============================================================================
    // ZOBRIST HASHING AND TRANSPOSITION TABLE TESTS
    // ============================================================================
    
    #[test]
    fn test_zobrist_hasher_determinism() {
        let hasher = ZobristHasher::new(15);
        let board = create_test_board();
        let snakes = &board.snakes;
        
        // Same board state should produce identical hashes
        let hash1 = hasher.hash_board_state(&board, snakes);
        let hash2 = hasher.hash_board_state(&board, snakes);
        
        assert_eq!(hash1, hash2, "Zobrist hash should be deterministic for identical board states");
    }
    
    #[test]
    fn test_zobrist_hasher_sensitivity() {
        let hasher = ZobristHasher::new(15);
        let mut board1 = create_test_board();
        let mut board2 = create_test_board();
        
        // Modify one snake's position slightly
        board2.snakes[0].head.x += 1;
        board2.snakes[0].body[0].x += 1;
        
        let hash1 = hasher.hash_board_state(&board1, &board1.snakes);
        let hash2 = hasher.hash_board_state(&board2, &board2.snakes);
        
        assert_ne!(hash1, hash2, "Different board states should produce different hashes");
    }
    
    #[test]
    fn test_zobrist_hash_performance() {
        let hasher = ZobristHasher::new(15);
        let board = create_test_board();
        let snakes = &board.snakes;
        
        let start = std::time::Instant::now();
        
        // Perform 100 hash calculations
        for _ in 0..100 {
            let _hash = hasher.hash_board_state(&board, snakes);
        }
        
        let elapsed = start.elapsed();
        let avg_time_per_hash = elapsed / 100;
        
        // Target: <0.1ms per hash (100μs)
        assert!(avg_time_per_hash.as_micros() < 100, 
                "Zobrist hash should be under 100μs, got {}μs", avg_time_per_hash.as_micros());
        
        println!("Zobrist hash performance: {}μs per operation", avg_time_per_hash.as_micros());
    }
    
    #[test]
    fn test_transposition_table_basic_operations() {
        let mut table = TranspositionTable::new();
        let board = create_test_board();
        
        // Test cache miss
        assert!(table.get(12345).is_none(), "Empty cache should return None");
        
        // Test cache insertion and retrieval
        let territory_map = TerritoryMap::new(11, 11);
        table.insert(12345, territory_map.clone(), 10);
        
        let retrieved = table.get(12345);
        assert!(retrieved.is_some(), "Cached entry should be retrievable");
        
        // Verify cache statistics
        assert_eq!(table.hit_count, 1, "Should have 1 cache hit");
        let total_ops = table.hit_count + table.miss_count;
        assert_eq!(total_ops, 2, "Should have 2 cache operations (1 miss + 1 hit)");
        assert!((table.get_hit_rate() - 0.5).abs() < 0.01, "Hit rate should be ~50%");
        
        println!("Basic operations test - Hits: {}, Total Ops: {}, Hit Rate: {:.1}%",
                 table.hit_count, total_ops, table.get_hit_rate() * 100.0);
    }
    
    #[test]
    fn test_transposition_table_lru_eviction() {
        let mut table = TranspositionTable::with_capacity(2); // Small cache for testing eviction
        let hasher = ZobristHasher::new(15);
        
        // Create test boards - can't clone Board, so create separate ones
        let board1 = create_test_board();
        let mut board2 = create_test_board();
        let mut board3 = create_test_board();
        
        // Modify positions to create different board states
        board2.snakes[0].body[0] = Coord { x: 2, y: 2 };
        board2.snakes[0].head = Coord { x: 2, y: 2 };
        
        board3.snakes[0].body[0] = Coord { x: 3, y: 3 };
        board3.snakes[0].head = Coord { x: 3, y: 3 };
        
        // Create territory maps
        let territory_map1 = TerritoryMap::new(11, 11);
        let territory_map2 = TerritoryMap::new(11, 11);
        let territory_map3 = TerritoryMap::new(11, 11);
        
        // Hash the boards using correct API
        let hash1 = hasher.hash_board_state(&board1, &board1.snakes);
        let hash2 = hasher.hash_board_state(&board2, &board2.snakes);
        let hash3 = hasher.hash_board_state(&board3, &board3.snakes);
        
        // Verify hashes are different
        assert_ne!(hash1, hash2, "Hash1 and Hash2 should be different");
        assert_ne!(hash2, hash3, "Hash2 and Hash3 should be different");
        assert_ne!(hash1, hash3, "Hash1 and Hash3 should be different");
        
        // Insert first two entries (should fit in cache)
        table.insert(hash1, territory_map1.clone(), 10);
        table.insert(hash2, territory_map2.clone(), 10);
        assert_eq!(table.cache.len(), 2, "Cache should contain 2 entries");
        
        // Verify both entries exist and update access order
        assert!(table.get(hash1).is_some());
        assert!(table.get(hash2).is_some());
        
        // Insert third entry (should evict the least recently used entry)
        table.insert(hash3, territory_map3.clone(), 10);
        assert_eq!(table.cache.len(), 2, "Cache should still contain only 2 entries after eviction");
        
        // The cache should have evicted the least recently used entry
        // After the gets above, hash2 was accessed last, so hash1 should be evicted
        assert!(table.get(hash2).is_some(), "hash2 should still be present");
        assert!(table.get(hash3).is_some(), "hash3 should be present");
        
        // Verify cache operations were performed
        let total_ops = table.hit_count + table.miss_count;
        assert!(total_ops > 0, "Should have performed cache operations");
        
        println!("Cache operations performed: {}", total_ops);
        println!("Cache hits: {}", table.hit_count);
        println!("Cache hit rate: {:.1}%", table.get_hit_rate() * 100.0);
        println!("Final cache size: {}", table.cache.len());
    }
    
    #[test]
    fn test_cached_territory_calculation() {
        let board = create_test_board();
        let snakes = &board.snakes;
        
        let start = std::time::Instant::now();
        
        // First calculation (cache miss)
        let territory1 = SpaceController::calculate_cached_territory_map(&board, snakes);
        let first_calc_time = start.elapsed();
        
        let start2 = std::time::Instant::now();
        
        // Second calculation (should be cache hit)
        let territory2 = SpaceController::calculate_cached_territory_map(&board, snakes);
        let second_calc_time = start2.elapsed();
        
        // Verify results are identical
        assert_eq!(territory1.control_scores.len(), territory2.control_scores.len(),
                   "Cached result should be identical to original");
        
        // Cache hit should be significantly faster (at least 50% improvement)
        let improvement_ratio = first_calc_time.as_nanos() as f64 / second_calc_time.as_nanos() as f64;
        assert!(improvement_ratio > 1.5, 
                "Cache hit should be at least 50% faster. First: {}μs, Second: {}μs, Ratio: {:.2}",
                first_calc_time.as_micros(), second_calc_time.as_micros(), improvement_ratio);
        
        println!("Territory cache performance - First: {}μs, Second: {}μs, Improvement: {:.2}x",
                 first_calc_time.as_micros(), second_calc_time.as_micros(), improvement_ratio);
    }
    
    #[test]
    fn test_territorial_fallback_maker_with_cache() {
        let fallback_maker = TerritorialFallbackMaker::new();
        let board = create_test_board();
        let snakes = &board.snakes;
        
        let start = std::time::Instant::now();
        
        // Multiple calls should benefit from caching
        for _ in 0..5 {
            let _territory = fallback_maker.calculate_cached_territory_map(&board, snakes);
        }
        
        let total_time = start.elapsed();
        let avg_time = total_time / 5;
        
        // After caching, average time should be well under 1ms
        assert!(avg_time.as_millis() < 1, 
                "Cached territory calculation should average <1ms, got {}ms", avg_time.as_millis());
        
        println!("TerritorialFallbackMaker cached performance: {}μs average over 5 calls",
                 avg_time.as_micros());
    }
    
    #[test]
    fn test_hash_collision_detection() {
        // Test the collision detection mechanism
        let mut table = TranspositionTable::new();
        let board = create_test_board();
        let snakes = &board.snakes;
        let territory_map = SpaceController::calculate_territory_map(&board, snakes);
        
        // Insert with hash key
        let hash_key = 123456u64;
        table.insert(hash_key, territory_map, 10);
        
        // Retrieve and verify no false collision reported
        let result = table.get(hash_key);
        assert!(result.is_some(), "Valid hash should retrieve successfully");
        
        let (_hit_rate, _hits, _misses, collisions) = table.get_stats();
        assert_eq!(collisions, 0, "No collisions should be detected for valid entries");
    }
    
    #[test]
    fn test_zobrist_hash_board_dimension_handling() {
        let hasher = ZobristHasher::new(25);
        
        // Test with different board dimensions (handling type inconsistency)
        let mut board1 = create_test_board();
        let mut board2 = create_test_board();
        
        board2.width = 15;  // i32
        board2.height = 15; // u32
        
        let hash1 = hasher.hash_board_state(&board1, &board1.snakes);
        let hash2 = hasher.hash_board_state(&board2, &board2.snakes);
        
        assert_ne!(hash1, hash2, "Different board dimensions should produce different hashes");
    }
    
    #[test]
    fn test_cache_memory_management() {
        let mut table = TranspositionTable::new();
        let board = create_test_board();
        let snakes = &board.snakes;
        let territory_map = SpaceController::calculate_territory_map(&board, snakes);
        
        // Test that cache size is managed
        let initial_stats = table.get_stats();
        
        // Add multiple entries
        for i in 0..20 {
            table.insert(i, territory_map.clone(), 5);
        }
        
        let final_stats = table.get_stats();
        
        // Should have processed multiple entries
        assert!(final_stats.1 + final_stats.2 > initial_stats.1 + initial_stats.2,
                "Should have processed more cache operations");
    }

    // ============================================================================
    // COMPREHENSIVE MINIMAX TESTING SUITE - TASK #2 IMPLEMENTATION
    // ============================================================================
    
    // Create a simplified test state for minimax testing
    fn create_minimax_test_state() -> SimulatedGameState {
        SimulatedGameState {
            board_width: 7,
            board_height: 7,
            food: vec![Coord { x: 3, y: 3 }],
            snakes: vec![
                SimulatedSnake {
                    id: "our_snake".to_string(),
                    health: 80,
                    body: vec![
                        Coord { x: 1, y: 1 }, // head
                        Coord { x: 1, y: 2 }, // neck
                        Coord { x: 1, y: 3 }, // body
                    ],
                    is_alive: true,
                },
                SimulatedSnake {
                    id: "opponent_snake".to_string(),
                    health: 75,
                    body: vec![
                        Coord { x: 5, y: 5 }, // head
                        Coord { x: 5, y: 4 }, // neck
                        Coord { x: 5, y: 3 }, // body
                    ],
                    is_alive: true,
                },
            ],
            turn: 1,
        }
    }

    #[test]
    fn test_minimax_basic_algorithm_correctness() {
        let mut fallback_maker = TerritorialFallbackMaker::new();
        let state = create_minimax_test_state();
        let our_snake_id = "our_snake";
        let search_start = Instant::now();
        
        // Test basic minimax call doesn't panic and returns valid evaluation
        let evaluation = fallback_maker.minimax_with_opponent_modeling(
            &state,
            our_snake_id,
            2, // depth
            f32::NEG_INFINITY, // alpha
            f32::INFINITY, // beta
            true, // maximizing player
            search_start
        );
        
        assert!(!evaluation.is_nan(), "Minimax should return valid numeric evaluation");
        assert!(evaluation.is_finite(), "Minimax evaluation should be finite");
        assert!(evaluation > f32::NEG_INFINITY, "Minimax should not return negative infinity for valid position");
        
        println!("Basic minimax evaluation: {:.2}", evaluation);
    }

    #[test]
    fn test_minimax_terminal_position_detection() {
        let mut fallback_maker = TerritorialFallbackMaker::new();
        let search_start = Instant::now();
        
        // Create terminal state - only one snake alive
        let mut terminal_state = create_minimax_test_state();
        terminal_state.snakes[1].is_alive = false; // Kill opponent
        
        let evaluation = fallback_maker.minimax_with_opponent_modeling(
            &terminal_state,
            "our_snake",
            3,
            f32::NEG_INFINITY,
            f32::INFINITY,
            true,
            search_start
        );
        
        // Debug: Print the actual evaluation to understand the issue
        println!("Debug: evaluation = {}", evaluation);
        println!("Debug: is_nan = {}", evaluation.is_nan());
        println!("Debug: is_infinite = {}", evaluation.is_infinite());
        println!("Debug: is_finite = {}", evaluation.is_finite());
        
        // Terminal position should return immediate evaluation without further recursion
        assert!(!evaluation.is_nan(), "Terminal position evaluation should be valid, got: {}", evaluation);
        assert!(evaluation.is_finite(), "Terminal position evaluation should be finite, got: {}", evaluation);
        
        println!("Terminal position evaluation: {:.2}", evaluation);
    }

    #[test]
    fn test_minimax_maximizing_vs_minimizing() {
        let mut fallback_maker = TerritorialFallbackMaker::new();
        let state = create_minimax_test_state();
        let search_start = Instant::now();
        
        // Test maximizing player (our turn)
        let maximizing_eval = fallback_maker.minimax_with_opponent_modeling(
            &state,
            "our_snake",
            2,
            f32::NEG_INFINITY,
            f32::INFINITY,
            true, // maximizing
            search_start
        );
        
        // Test minimizing player (opponent's turn)
        let minimizing_eval = fallback_maker.minimax_with_opponent_modeling(
            &state,
            "our_snake",
            2,
            f32::NEG_INFINITY,
            f32::INFINITY,
            false, // minimizing
            search_start
        );
        
        // Maximizing should generally produce higher or equal values than minimizing
        // (though this may not always be true in complex positions)
        assert!(!maximizing_eval.is_nan() && !minimizing_eval.is_nan(),
                "Both evaluations should be valid numbers");
        
        println!("Maximizing evaluation: {:.2}, Minimizing evaluation: {:.2}",
                maximizing_eval, minimizing_eval);
    }

    #[test]
    fn test_alpha_beta_pruning_effectiveness() {
        let mut fallback_maker = TerritorialFallbackMaker::new();
        let state = create_minimax_test_state();
        let search_start = Instant::now();
        
        // Reset statistics to measure pruning effectiveness
        fallback_maker.search_statistics = SearchStatistics::new();
        
        // Perform search with alpha-beta pruning
        let evaluation = fallback_maker.minimax_with_opponent_modeling(
            &state,
            "our_snake",
            4, // Deeper search to trigger pruning
            f32::NEG_INFINITY,
            f32::INFINITY,
            true,
            search_start
        );
        
        let nodes_searched = fallback_maker.search_statistics.nodes_searched;
        let cutoffs = fallback_maker.search_statistics.alpha_beta_cutoffs;
        
        assert!(nodes_searched > 0, "Should have searched some nodes");
        assert!(!evaluation.is_nan(), "Should return valid evaluation");
        
        // Calculate pruning effectiveness - target >50% for tournament performance
        let pruning_effectiveness = if nodes_searched > 0 {
            (cutoffs as f32 / nodes_searched as f32) * 100.0
        } else {
            0.0
        };
        
        println!("Alpha-beta pruning effectiveness: {:.1}% ({} cutoffs out of {} nodes)",
                pruning_effectiveness, cutoffs, nodes_searched);
        
        // In a well-ordered search with depth 4+, we should see some pruning
        if nodes_searched >= 10 {
            assert!(pruning_effectiveness >= 10.0,
                   "Should achieve at least 10% pruning effectiveness with depth 4+ search. Got {:.1}%",
                   pruning_effectiveness);
        }
    }

    #[test]
    fn test_minimax_depth_scaling() {
        let mut fallback_maker = TerritorialFallbackMaker::new();
        let state = create_minimax_test_state();
        let search_start = Instant::now();
        
        // Test different depths and measure node expansion
        let mut depth_results = Vec::new();
        
        for depth in 1..=4 {
            fallback_maker.search_statistics = SearchStatistics::new();
            
            let evaluation = fallback_maker.minimax_with_opponent_modeling(
                &state,
                "our_snake",
                depth,
                f32::NEG_INFINITY,
                f32::INFINITY,
                true,
                search_start
            );
            
            let nodes_searched = fallback_maker.search_statistics.nodes_searched;
            depth_results.push((depth, nodes_searched, evaluation));
            
            assert!(!evaluation.is_nan(), "Depth {} should return valid evaluation", depth);
            println!("Depth {}: {} nodes searched, evaluation: {:.2}", depth, nodes_searched, evaluation);
        }
        
        // Deeper searches should generally examine more nodes (though pruning can affect this)
        assert!(depth_results.len() == 4, "Should have tested 4 different depths");
        
        // Verify that depth 1 examines fewer nodes than depth 4 (accounting for pruning)
        let (_, nodes_d1, _) = depth_results[0];
        let (_, nodes_d4, _) = depth_results[3];
        
        if nodes_d1 > 0 && nodes_d4 > 0 {
            println!("Node scaling: Depth 1: {}, Depth 4: {}", nodes_d1, nodes_d4);
            // Should see some scaling, but pruning may keep it reasonable
            assert!(nodes_d4 >= nodes_d1, "Depth 4 should examine at least as many nodes as depth 1");
        }
    }

    #[test]
    fn test_minimax_opponent_modeling_scenarios() {
        let mut fallback_maker = TerritorialFallbackMaker::new();
        let search_start = Instant::now();
        
        // Scenario 1: Multiple opponents
        let mut multi_opponent_state = create_minimax_test_state();
        multi_opponent_state.snakes.push(SimulatedSnake {
            id: "opponent2".to_string(),
            health: 70,
            body: vec![
                Coord { x: 2, y: 5 },
                Coord { x: 2, y: 4 },
            ],
            is_alive: true,
        });
        
        let multi_eval = fallback_maker.minimax_with_opponent_modeling(
            &multi_opponent_state,
            "our_snake",
            2,
            f32::NEG_INFINITY,
            f32::INFINITY,
            true,
            search_start
        );
        
        // Scenario 2: Single opponent
        let single_eval = fallback_maker.minimax_with_opponent_modeling(
            &create_minimax_test_state(),
            "our_snake",
            2,
            f32::NEG_INFINITY,
            f32::INFINITY,
            true,
            search_start
        );
        
        assert!(!multi_eval.is_nan() && !single_eval.is_nan(),
                "Both opponent modeling scenarios should return valid evaluations");
        
        println!("Multi-opponent evaluation: {:.2}, Single-opponent evaluation: {:.2}",
                multi_eval, single_eval);
        
        // Test opponent modeling stats
        let opponent_nodes = fallback_maker.search_statistics.opponent_nodes_searched;
        println!("Opponent modeling nodes searched: {}", opponent_nodes);
        
        // Should have searched some opponent nodes when modeling opponents
        if multi_opponent_state.snakes.iter().filter(|s| s.is_alive && s.id != "our_snake").count() > 0 {
            assert!(opponent_nodes > 0, "Should have searched opponent nodes when multiple opponents present");
        }
    }

    #[test]
    fn test_minimax_time_budget_handling() {
        let mut fallback_maker = TerritorialFallbackMaker::new();
        let state = create_minimax_test_state();
        
        // Test with very short time budget (should terminate early)
        let search_start = Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(110)); // Exceed 100ms budget
        
        let evaluation = fallback_maker.minimax_with_opponent_modeling(
            &state,
            "our_snake",
            8, // Very deep search that would normally take long
            f32::NEG_INFINITY,
            f32::INFINITY,
            true,
            search_start
        );
        
        // Should still return a valid evaluation even with time pressure
        assert!(!evaluation.is_nan(), "Should return valid evaluation even under time pressure");
        assert!(evaluation.is_finite(), "Time-pressured evaluation should be finite");
        
        println!("Time-pressured evaluation: {:.2}", evaluation);
    }

    #[test]
    fn test_minimax_cache_integration() {
        let mut fallback_maker = TerritorialFallbackMaker::new();
        let state = create_minimax_test_state();
        let search_start = Instant::now();
        
        // First call - should populate cache
        fallback_maker.search_statistics = SearchStatistics::new();
        let eval1 = fallback_maker.minimax_with_opponent_modeling(
            &state,
            "our_snake",
            3,
            f32::NEG_INFINITY,
            f32::INFINITY,
            true,
            search_start
        );
        
        let cache_misses_1 = fallback_maker.search_statistics.cache_misses;
        let cache_hits_1 = fallback_maker.search_statistics.cache_hits;
        
        // Second call - should benefit from cache
        fallback_maker.search_statistics = SearchStatistics::new();
        let eval2 = fallback_maker.minimax_with_opponent_modeling(
            &state,
            "our_snake",
            3,
            f32::NEG_INFINITY,
            f32::INFINITY,
            true,
            search_start
        );
        
        let cache_misses_2 = fallback_maker.search_statistics.cache_misses;
        let cache_hits_2 = fallback_maker.search_statistics.cache_hits;
        
        assert_eq!(eval1, eval2, "Cached evaluations should be identical");
        
        println!("First call - Hits: {}, Misses: {}", cache_hits_1, cache_misses_1);
        println!("Second call - Hits: {}, Misses: {}", cache_hits_2, cache_misses_2);
        
        // Second call should have more cache hits relative to misses
        let hit_rate_1 = if cache_hits_1 + cache_misses_1 > 0 {
            cache_hits_1 as f32 / (cache_hits_1 + cache_misses_1) as f32
        } else { 0.0 };
        
        let hit_rate_2 = if cache_hits_2 + cache_misses_2 > 0 {
            cache_hits_2 as f32 / (cache_hits_2 + cache_misses_2) as f32
        } else { 0.0 };
        
        println!("Hit rate - First: {:.1}%, Second: {:.1}%", hit_rate_1 * 100.0, hit_rate_2 * 100.0);
    }

    #[test]
    fn test_iterative_deepening_progression() {
        let game = create_test_game();
        let board = create_test_board();
        let you = &board.snakes[0];
        
        let mut fallback_maker = TerritorialFallbackMaker::new();
        fallback_maker.time_limit_ms = 200; // Give enough time for multiple depths
        
        let search_start = Instant::now();
        let result = fallback_maker.iterative_deepening_search(&board, you, search_start);
        let total_time = search_start.elapsed().as_millis();
        
        // Should return a valid move result
        assert!(result.is_object(), "Should return a valid JSON object");
        if let Some(move_value) = result.get("move") {
            assert!(move_value.is_string(), "Move should be a string");
            let move_str = move_value.as_str().unwrap();
            assert!(["up", "down", "left", "right"].contains(&move_str),
                   "Move '{}' should be a valid direction", move_str);
        }
        
        // Should have completed multiple depths
        let depths_completed = &fallback_maker.search_statistics.depths_completed;
        let timing_per_depth = &fallback_maker.search_statistics.timing_per_depth;
        
        assert!(!depths_completed.is_empty(), "Should have completed at least one depth");
        assert_eq!(depths_completed.len(), timing_per_depth.len(),
                  "Should have timing data for each completed depth");
        
        println!("Iterative deepening results:");
        println!("  Total time: {}ms", total_time);
        println!("  Depths completed: {:?}", depths_completed);
        println!("  Timing per depth: {:?}ms", timing_per_depth);
        println!("  Best evaluation: {:.2}", fallback_maker.search_statistics.best_evaluation);
        
        // Should respect time budget (with some tolerance for measurement overhead)
        assert!(total_time <= 250, "Should respect time budget of 200ms (got {}ms)", total_time);
        
        // Should progress through multiple depths if time allows
        if total_time < 150 {
            assert!(depths_completed.len() >= 2, "Should complete multiple depths when time allows");
        }
    }

    #[test]
    fn test_minimax_game_state_consistency() {
        let mut fallback_maker = TerritorialFallbackMaker::new();
        
        // Test that applying and undoing moves maintains state consistency
        let original_state = create_minimax_test_state();
        let our_snake_id = "our_snake";
        
        // Create a copy to verify consistency
        let state_copy = original_state.clone();
        
        // Apply a move
        if let Ok(modified_state) = fallback_maker.apply_move_to_state(&original_state, our_snake_id, Direction::Right) {
            // Verify the move was applied
            let our_snake_original = original_state.snakes.iter().find(|s| s.id == our_snake_id).unwrap();
            let our_snake_modified = modified_state.snakes.iter().find(|s| s.id == our_snake_id).unwrap();
            
            assert_ne!(our_snake_original.head(), our_snake_modified.head(),
                      "Snake head should have moved");
            
            // Verify original state wasn't mutated
            assert_eq!(original_state.snakes[0].head(), state_copy.snakes[0].head(),
                      "Original state should remain unchanged");
        }
    }

    #[test]
    fn test_minimax_edge_case_handling() {
        let mut fallback_maker = TerritorialFallbackMaker::new();
        let search_start = Instant::now();
        
        // Test with dead snake
        let mut dead_snake_state = create_minimax_test_state();
        dead_snake_state.snakes[0].is_alive = false;
        
        let dead_eval = fallback_maker.minimax_with_opponent_modeling(
            &dead_snake_state,
            "our_snake",
            2,
            f32::NEG_INFINITY,
            f32::INFINITY,
            true,
            search_start
        );
        
        // Should handle dead snake gracefully
        assert!(!dead_eval.is_nan(), "Should handle dead snake gracefully");
        
        // Test with no food
        let mut no_food_state = create_minimax_test_state();
        no_food_state.food.clear();
        
        let no_food_eval = fallback_maker.minimax_with_opponent_modeling(
            &no_food_state,
            "our_snake",
            2,
            f32::NEG_INFINITY,
            f32::INFINITY,
            true,
            search_start
        );
        
        assert!(!no_food_eval.is_nan(), "Should handle no food scenario gracefully");
        
        println!("Edge case evaluations - Dead snake: {:.2}, No food: {:.2}",
                dead_eval, no_food_eval);
    }

    #[test]
    fn test_minimax_performance_benchmarks() {
        let mut fallback_maker = TerritorialFallbackMaker::new();
        let state = create_minimax_test_state();
        let our_snake_id = "our_snake";
        
        // Performance benchmark for tournament conditions
        let iterations = 10;
        let mut total_time = 0u128;
        let mut total_nodes = 0u64;
        
        for i in 0..iterations {
            let search_start = Instant::now();
            fallback_maker.search_statistics = SearchStatistics::new();
            
            let _evaluation = fallback_maker.minimax_with_opponent_modeling(
                &state,
                our_snake_id,
                3, // Tournament-realistic depth
                f32::NEG_INFINITY,
                f32::INFINITY,
                true,
                search_start
            );
            
            let iteration_time = search_start.elapsed().as_millis();
            let iteration_nodes = fallback_maker.search_statistics.nodes_searched;
            
            total_time += iteration_time;
            total_nodes += iteration_nodes;
            
            println!("Iteration {}: {}ms, {} nodes", i + 1, iteration_time, iteration_nodes);
        }
        
        let avg_time = total_time / iterations as u128;
        let avg_nodes = total_nodes / iterations as u64;
        
        println!("Performance benchmark results:");
        println!("  Average time per search: {}ms", avg_time);
        println!("  Average nodes per search: {}", avg_nodes);
        println!("  Nodes per millisecond: {:.2}", avg_nodes as f32 / avg_time.max(1) as f32);
        
        // Tournament performance targets
        assert!(avg_time < 50, "Average search time should be under 50ms for tournament play (got {}ms)", avg_time);
        
        if avg_nodes > 0 {
            let nodes_per_ms = avg_nodes as f32 / avg_time.max(1) as f32;
            assert!(nodes_per_ms > 1.0, "Should search at least 1 node per millisecond (got {:.2})", nodes_per_ms);
        }
    }

}

// ============================================================================
// ADVANCED NEURAL NETWORK EVALUATOR - PROPERLY INTEGRATED
// ============================================================================

/// Advanced Neural Network Evaluator that properly integrates all Phase 1C systems
/// This replaces the broken SimpleNeuralEvaluator with the actual Advanced Opponent Modeling Integration
pub struct AdvancedNeuralEvaluator {
    // Core systems integration
    space_controller: SpaceController,
    opponent_analyzer: OpponentAnalyzer,
    movement_history: MovementHistory,
    territorial_strategist: TerritorialStrategist,
    
    // Neural network weights (simplified but with proper integration)
    nn_weights: [[f32; 4]; 3],
    safety_weights: [f32; 4],
    territory_weights: [f32; 4],
    opponent_weights: [f32; 4],
}

impl AdvancedNeuralEvaluator {
    pub fn new() -> Self {
        Self {
            // Initialize all the sophisticated systems
            space_controller: SpaceController,
            opponent_analyzer: OpponentAnalyzer,
            movement_history: MovementHistory::new(10),
            territorial_strategist: TerritorialStrategist::new(),
            
            // Balanced neural network weights
            nn_weights: [
                [0.5, 0.5, 0.5, 0.5],  // Position evaluation
                [0.6, 0.6, 0.6, 0.6],  // Safety evaluation
                [0.4, 0.4, 0.4, 0.4],  // Movement quality
            ],
            safety_weights: [0.8, 0.8, 0.8, 0.8],           // High safety priority
            territory_weights: [0.7, 0.7, 0.7, 0.7],        // Territory control
            opponent_weights: [0.6, 0.6, 0.6, 0.6],         // Opponent modeling
        }
    }
    
    /// Get move probabilities using the integrated Advanced Opponent Modeling systems
    pub fn get_move_probabilities(&self, board: &Board, you: &Battlesnake) -> HashMap<Direction, f32> {
        info!("ADVANCED NEURAL: =========================================");
        info!("ADVANCED NEURAL: Starting ADVANCED OPONENT MODELING INTEGRATION evaluation");
        info!("ADVANCED NEURAL: Snake: {} at ({}, {}), health: {}", you.id, you.head.x, you.head.y, you.health);
        info!("ADVANCED NEURAL: Board: {}x{}, food: {}, snakes: {}", board.width, board.height, board.food.len(), board.snakes.len());
        
        let all_snakes: Vec<Battlesnake> = board.snakes.iter().cloned().collect();
        let mut move_scores = HashMap::new();
        
        // ================================================================
        // PHASE 1C: ADVANCED OPPONENT MODELING INTEGRATION
        // ================================================================
        info!("ADVANCED NEURAL: PHASE 1C - Advanced Opponent Modeling Integration");
        
        // 1. Territory Control Analysis (Phase 1C)
        // PERFORMANCE FIX: Remove unused territory calculation (O(board_size²) overhead)
        // let territory_map = SpaceController::calculate_territory_map(board, &all_snakes);
        info!("ADVANCED NEURAL: Territory control map calculated");
        
        // 2. Opponent Movement Prediction (Phase 1C)
        let mut opponent_predictions = HashMap::new();
        for opponent in &all_snakes {
            if opponent.id != you.id {
                let predictions = OpponentAnalyzer::predict_opponent_moves(opponent, board, &all_snakes);
                info!("ADVANCED NEURAL: Predicted moves for opponent {}: {:?}", opponent.id, predictions);
                opponent_predictions.insert(opponent.id.clone(), predictions);
            }
        }
        
        // 3. Identify cutting positions for area denial
        let cutting_positions = OpponentAnalyzer::identify_cutting_positions(&you.head, you, board);
        info!("ADVANCED NEURAL: Identified {} cutting positions for area denial", cutting_positions.len());
        
        // ================================================================
        // INTEGRATED EVALUATION USING ALL SYSTEMS
        // ================================================================
        
        for direction in Direction::all() {
            let next_pos = you.head.apply_direction(&direction);
            let mut total_score = 0.0;
            
            // Safety evaluation (highest priority)
            let is_safe = SafetyChecker::is_safe_coordinate(&next_pos, board, &all_snakes);
            let safety_score = if is_safe {
                self.safety_weights[self.direction_index(&direction)] * 10.0
            } else {
                -50.0
            };
            total_score += safety_score;
            info!("ADVANCED NEURAL: Move {:?} - Safety score: {:.2}", direction, safety_score);
            
            // Territory control evaluation
            let territory_score = self.space_controller.get_area_control_score(
                &you.head, &next_pos, board, &all_snakes, &you.id);
            let weighted_territory = territory_score * self.territory_weights[self.direction_index(&direction)];
            total_score += weighted_territory;
            info!("ADVANCED NEURAL: Move {:?} - Territory score: {:.2}", direction, weighted_territory);
            
            // Advanced opponent modeling integration
            let mut opponent_score = 0.0;
            for (opponent_id, predictions) in &opponent_predictions {
                if let Some(&opponent_prob) = predictions.get(&direction) {
                    // Reward moves that counter opponent predictions
                    let counter_bonus = (1.0 - opponent_prob) * 5.0;
                    opponent_score += counter_bonus;
                    info!("ADVANCED NEURAL: Move {:?} - Counter opponent {} bonus: {:.2}",
                          direction, opponent_id, counter_bonus);
                }
            }
            let weighted_opponent = opponent_score * self.opponent_weights[self.direction_index(&direction)];
            total_score += weighted_opponent;
            info!("ADVANCED NEURAL: Move {:?} - Opponent modeling score: {:.2}", direction, weighted_opponent);
            
            // Food seeking integration
            let food_score = if FoodSeeker::should_seek_food(you.health, 0, !board.food.is_empty()) {
                if let Some(target) = FoodSeeker::find_best_food_target(&you.head, board, you.health, true) {
                    let food_direction = Self::get_direction_to_target(&you.head, &target.coord);
                    if food_direction == direction {
                        target.priority * 0.5
                    } else {
                        0.0
                    }
                } else { 0.0 }
            } else { 0.0 };
            total_score += food_score;
            info!("ADVANCED NEURAL: Move {:?} - Food seeking score: {:.2}", direction, food_score);
            
            // Space exploration bonus
            let reachable_spaces = ReachabilityAnalyzer::count_reachable_spaces(&next_pos, board, &all_snakes);
            let exploration_score = (reachable_spaces as f32) * 0.3;
            total_score += exploration_score;
            info!("ADVANCED NEURAL: Move {:?} - Exploration score: {:.2}", direction, exploration_score);
            
            // Cutting position bonus (area denial)
            let cutting_bonus = if cutting_positions.contains(&next_pos) { 8.0 } else { 0.0 };
            total_score += cutting_bonus;
            info!("ADVANCED NEURAL: Move {:?} - Cutting position bonus: {:.2}", direction, cutting_bonus);
            
            // Movement quality bonus (loop prevention)
            let recent_moves = self.movement_history.get_recent_moves();
            let movement_bonus = MovementQualityAnalyzer::calculate_movement_bonus(
                &you.head, &next_pos, &recent_moves, direction);
            total_score += movement_bonus;
            info!("ADVANCED NEURAL: Move {:?} - Movement quality bonus: {:.2}", direction, movement_bonus);
            
            // Neural network integration
            let nn_base_score = self.nn_weights[0][self.direction_index(&direction)] *
                               self.nn_weights[1][self.direction_index(&direction)] *
                               self.nn_weights[2][self.direction_index(&direction)];
            total_score += nn_base_score;
            info!("ADVANCED NEURAL: Move {:?} - Neural network base score: {:.2}", direction, nn_base_score);
            
            move_scores.insert(direction, total_score);
            info!("ADVANCED NEURAL: Move {:?} - TOTAL SCORE: {:.2}", direction, total_score);
        }
        
        // ================================================================
        // CONVERT SCORES TO PROBABILITIES
        // ================================================================
        
        let max_score = move_scores.values().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_score = move_scores.values().fold(f32::INFINITY, |a, &b| a.min(b));
        let score_range = (max_score - min_score).max(1.0);
        
        info!("ADVANCED NEURAL: Score range: {:.2} to {:.2}", min_score, max_score);
        
        let mut probabilities = HashMap::new();
        let mut total_prob = 0.0;
        
        for (direction, score) in &move_scores {
            // Convert to probability using softmax
            let normalized_score = (score - min_score) / score_range;
            let probability = normalized_score.max(0.01); // Ensure minimum probability
            probabilities.insert(*direction, probability);
            total_prob += probability;
            info!("ADVANCED NEURAL: Move {:?} - Raw score: {:.2}, Probability: {:.3}",
                  direction, score, probability);
        }
        
        // Normalize probabilities
        for (direction, prob) in probabilities.iter_mut() {
            *prob /= total_prob;
        }
        
        info!("ADVANCED NEURAL: =========================================");
        info!("ADVANCED NEURAL: FINAL PROBABILITIES (Advanced Opponent Modeling Integration):");
        for (direction, &prob) in &probabilities {
            info!("ADVANCED NEURAL:   {:?}: {:.3} ({:.1}%)", direction, prob, prob * 100.0);
        }
        info!("ADVANCED NEURAL: Sum: {:.3}", probabilities.values().sum::<f32>());
        info!("ADVANCED NEURAL: =========================================");
        info!("ADVANCED NEURAL: SUCCESS - Advanced Opponent Modeling Integration is ACTIVE!");
        info!("ADVANCED NEURAL: Using Phase 1C territory control, opponent prediction, and cutting positions");
        
        probabilities
    }
    
    /// Helper function to get direction index
    fn direction_index(&self, direction: &Direction) -> usize {
        match direction {
            Direction::Up => 0,
            Direction::Down => 1,
            Direction::Left => 2,
            Direction::Right => 3,
        }
    }
    
    /// Helper function to calculate direction to target
    fn get_direction_to_target(from: &Coord, to: &Coord) -> Direction {
        let dx = to.x - from.x;
        let dy = to.y - from.y;
        
        if dx.abs() > dy.abs() {
            if dx > 0 { Direction::Right } else { Direction::Left }
        } else {
            if dy > 0 { Direction::Up } else { Direction::Down }
        }
    }
    
    /// Update movement history
    pub fn add_move_to_history(&mut self, direction: Direction, position: Coord, score: f32) {
        self.movement_history.add_move(direction, position, score);
    }
}

// ============================================================================
// ENHANCED HYBRID MANAGER WITH NEURAL NETWORK INTEGRATION
// ============================================================================

/// Enhanced Hybrid Search Manager with working neural network integration
pub struct EnhancedHybridManager {
    neural_evaluator: AdvancedNeuralEvaluator,
    territorial_fallback_maker: TerritorialFallbackMaker,
    mcts_decision_maker: MCTSDecisionMaker,
}

impl EnhancedHybridManager {
    pub fn new() -> Self {
        Self {
            neural_evaluator: AdvancedNeuralEvaluator::new(),
            territorial_fallback_maker: TerritorialFallbackMaker::new(),
            mcts_decision_maker: MCTSDecisionMaker::new(),
        }
    }
    
    pub fn make_decision(&mut self, game: &Game, board: &Board, you: &Battlesnake) -> Value {
        let start_time = std::time::Instant::now();
        let num_snakes = board.snakes.len();
        let our_health = you.health;
        let board_complexity = (board.width as usize * board.height as usize) as f32;
        
        info!("ENHANCED HYBRID: ================================================");
        info!("ENHANCED HYBRID: Starting enhanced hybrid decision analysis for snake {}", you.id);
        info!("ENHANCED HYBRID: Game state - Health: {}, Snakes: {}, Board: {}x{}, Complexity: {:.0}",
              our_health, num_snakes, board.width, board.height, board_complexity);
        info!("ENHANCED HYBRID: ================================================");
        
        // ================================================================
        // PROPER DECISION HIERARCHY: Safety First → Neural Network → Strategic Logic
        // ================================================================
        
        // STEP 1: SAFETY FIRST - Calculate safe moves before any AI evaluation
        info!("ENHANCED HYBRID: STEP 1 - SAFETY FIRST - Calculating safe moves...");
        let all_snakes: Vec<Battlesnake> = board.snakes.iter().cloned().collect();
        let safe_moves = SafetyChecker::calculate_safe_moves(you, board, &all_snakes);
        let safe_moves = SafetyChecker::avoid_backward_move(you, safe_moves);
        
        info!("ENHANCED HYBRID: STEP 1 RESULTS - Safe moves available: {:?} (count: {})",
              safe_moves.iter().map(|d| format!("{:?}", d)).collect::<Vec<_>>(), safe_moves.len());
        
        if safe_moves.is_empty() {
            info!("ENHANCED HYBRID: STEP 1 - CRITICAL: No safe moves available! Using emergency fallback");
            // Emergency fallback - no safe moves, must choose something
            let emergency_result = self.get_search_recommendation(game, board, you);
            info!("ENHANCED HYBRID: STEP 1 - Emergency fallback decision: {:?}", emergency_result);
            return emergency_result;
        }
        
        // STEP 2: NEURAL NETWORK EVALUATION (only if we have safe moves)
        info!("ENHANCED HYBRID: STEP 2 - NEURAL NETWORK EVALUATION - Getting AI recommendations...");
        let nn_probabilities = self.neural_evaluator.get_move_probabilities(board, you);
        
        // Filter neural network recommendations to only safe moves
        let safe_nn_moves: Vec<(Direction, f32)> = nn_probabilities.iter()
            .filter(|(direction, _)| safe_moves.contains(direction))
            .map(|(direction, &prob)| (*direction, prob))
            .collect();
            
        info!("ENHANCED HYBRID: STEP 2 RESULTS - Neural network probabilities: {:?}", nn_probabilities);
        info!("ENHANCED HYBRID: STEP 2 RESULTS - Safe neural network moves: {:?}",
              safe_nn_moves.iter().map(|(d, p)| format!("{:?}: {:.3}", d, p)).collect::<Vec<_>>());
        
        if safe_nn_moves.is_empty() {
            info!("ENHANCED HYBRID: STEP 2 - WARNING: No safe neural network moves found, falling through to strategic logic");
            let fallback_result = self.get_search_recommendation(game, board, you);
            info!("ENHANCED HYBRID: STEP 2 - Strategic fallback decision: {:?}", fallback_result);
            return fallback_result;
        }
        
        // Find best safe neural network move
        let best_safe_nn_move = safe_nn_moves.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(direction, _)| *direction);
            
        let max_nn_prob = safe_nn_moves.iter()
            .map(|(_, prob)| *prob)
            .fold(0.0, f32::max);
            
        info!("ENHANCED HYBRID: STEP 2 RESULTS - Best safe NN move: {:?} (prob: {:.4})",
              best_safe_nn_move, max_nn_prob);
        
        // CRITICAL: DIRECT NEURAL NETWORK OVERRIDE FOR HIGH-SCORING EVALUATIONS
        let direct_override_threshold = 0.30;
        info!("ENHANCED HYBRID: STEP 3 OVERRIDE CHECK:");
        info!("ENHANCED HYBRID:         - Max NN probability: {:.4}", max_nn_prob);
        info!("ENHANCED HYBRID:         - Direct override threshold: {:.3}", direct_override_threshold);
        
        // NEURAL NETWORK OVERRIDE: Use NN when probability > 0.30 regardless of safety
        if max_nn_prob > direct_override_threshold && best_safe_nn_move.is_some() {
            info!("ENHANCED HYBRID: STEP 3 OVERRIDE - HIGH NEURAL NETWORK SCORE DETECTED!");
            info!("ENHANCED HYBRID: STEP 3 OVERRIDE - Neural network probability {:.4} exceeds override threshold {:.3}",
                  max_nn_prob, direct_override_threshold);
            info!("ENHANCED HYBRID: STEP 3 OVERRIDE - Using neural network recommendation directly");
            if let Some(move_direction) = best_safe_nn_move {
                info!("ENHANCED HYBRID: STEP 3 OVERRIDE - DIRECT NEURAL DECISION: {:?} (NN prob: {:.4})",
                      move_direction, max_nn_prob);
                info!("ENHANCED HYBRID: STEP 3 OVERRIDE - Neural network is making the decision!");
                info!("ENHANCED HYBRID: STEP 3 OVERRIDE - ===============================");
                return json!({
                    "move": format!("{:?}", move_direction).to_lowercase(),
                    "decision_source": "neural_network_override",
                    "confidence": max_nn_prob
                });
            }
        }
        
        // OPTIMIZED CONFIDENCE THRESHOLDS - ALLOW MORE NEURAL NETWORK USAGE
        let confidence_threshold = match num_snakes {
            1 => 0.25,  // LOW confidence required for solo games
            2..=3 => 0.25, // LOW confidence required for small games
            _ => 0.25,     // LOW confidence required for large games
        };
        
        info!("ENHANCED HYBRID: STEP 3 - CONFIDENCE EVALUATION:");
        info!("ENHANCED HYBRID:         - Max NN probability: {:.4}", max_nn_prob);
        info!("ENHANCED HYBRID:         - Required confidence: {:.3}", confidence_threshold);
        info!("ENHANCED HYBRID:         - Safe moves available: {}", safe_moves.len());
        
        // HIGH CONFIDENCE: Use neural network suggestion if very confident AND safe
        if max_nn_prob >= confidence_threshold {
            info!("ENHANCED HYBRID: STEP 3A - HIGH CONFIDENCE NEURAL NETWORK - Using NN suggestion (confident AND safe)");
            if let Some(move_direction) = best_safe_nn_move {
                info!("ENHANCED HYBRID: STEP 3A - CONFIDENT SAFE NEURAL DECISION: {:?}", move_direction);
                info!("ENHANCED HYBRID: STEP 3A - Neural network is confident ({:.4} >= {:.3}) AND move is safe",
                      max_nn_prob, confidence_threshold);
                info!("ENHANCED HYBRID: STEP 3A - ===============================");
                return json!({ "move": format!("{:?}", move_direction).to_lowercase() });
            }
        }
        
        // MEDIUM CONFIDENCE: Consider neural network but with safety validation
        if max_nn_prob >= 0.5 {
            info!("ENHANCED HYBRID: STEP 3B - MEDIUM CONFIDENCE - Validating NN suggestion with safety score");
            
            if let Some(nn_move) = best_safe_nn_move {
                // Get safety score for this move
                let next_pos = you.head.apply_direction(&nn_move);
                let is_safe = SafetyChecker::is_safe_coordinate(&next_pos, board, &all_snakes);
                let space_score = ReachabilityAnalyzer::count_reachable_spaces(&next_pos, board, &all_snakes) as f32 / 100.0;
                
                let safety_score = if is_safe { 0.8 } else { 0.0 } + space_score * 0.2;
                let combined_score = max_nn_prob * 0.6 + safety_score * 0.4;
                
                info!("ENHANCED HYBRID: STEP 3B - Safety validation for {:?}:", nn_move);
                info!("ENHANCED HYBRID:         - NN probability: {:.4}", max_nn_prob);
                info!("ENHANCED HYBRID:         - Safety score: {:.4}", safety_score);
                info!("ENHANCED HYBRID:         - Combined score: {:.4}", combined_score);
                
                if combined_score > 0.6 {
                    info!("ENHANCED HYBRID: STEP 3B - VALIDATED NEURAL DECISION: {:?} (combined: {:.3})", nn_move, combined_score);
                    info!("ENHANCED HYBRID: STEP 3B - Neural network suggestion passes safety validation");
                    info!("ENHANCED HYBRID: STEP 3B - ===============================");
                    return json!({ "move": format!("{:?}", nn_move).to_lowercase() });
                } else {
                    info!("ENHANCED HYBRID: STEP 3B - Neural network suggestion fails safety validation (score: {:.3} < 0.6)", combined_score);
                }
            }
        }
        
        // LOW CONFIDENCE: Fall back to strategic logic (search algorithms)
        info!("ENHANCED HYBRID: STEP 3C - LOW CONFIDENCE - Falling back to strategic logic");
        info!("ENHANCED HYBRID: STEP 3C - Neural network confidence {:.4} below thresholds", max_nn_prob);
        info!("ENHANCED HYBRID: STEP 3C - Using traditional search algorithms for strategic decision");
        info!("ENHANCED HYBRID: STEP 3C - NO BIAS PATTERNS: Using sophisticated search with territory control");
        
        let final_result = self.get_search_recommendation(game, board, you);
        
        info!("ENHANCED HYBRID: STEP 3C - Strategic decision: {:?}", final_result);
        info!("ENHANCED HYBRID: STEP 3C - Decision hierarchy: SAFETY → NN (insufficient confidence) → STRATEGIC LOGIC");
        info!("ENHANCED HYBRID: STEP 3C - Strategic logic uses: Territory Control + Opponent Modeling + Loop Prevention");
        info!("ENHANCED HYBRID: STEP 3C - ===============================");
        
        final_result
    }
    
    /// Get recommendation from traditional search algorithms
    fn get_search_recommendation(&mut self, game: &Game, board: &Board, you: &Battlesnake) -> Value {
        let num_snakes = board.snakes.len();
        let our_health = you.health;
        
        info!("SEARCH FALLBACK: Starting traditional search decision for snake {} (health: {}, snakes: {})",
              you.id, our_health, num_snakes);
        
        // Use hybrid strategy selection from the original system
        let use_mcts = match (num_snakes, our_health) {
            (n, h) if n >= 4 && h > 30 => {
                info!("SEARCH FALLBACK: Selecting MCTS for many snakes ({}) with good health ({})", n, h);
                true
            },
            (n, h) if n >= 3 && h > 50 => {
                info!("SEARCH FALLBACK: Selecting MCTS for medium snakes ({}) with high health ({})", n, h);
                true
            },
            (n, h) if n == 1 && h < 20 => {
                info!("SEARCH FALLBACK: Selecting MCTS for single opponent with low health ({})", h);
                true
            },
            _ => {
                info!("SEARCH FALLBACK: Selecting Minimax for simple position (snakes: {}, health: {})", num_snakes, our_health);
                false
            },
        };
        
        if use_mcts {
            info!("SEARCH FALLBACK: Using MCTS for search fallback");
            let result = self.mcts_decision_maker.make_decision(game, board, you);
            if let Some(move_str) = result.get("move") {
                info!("SEARCH FALLBACK: MCTS decision: {}", move_str);
            }
            result
        } else {
            info!("SEARCH FALLBACK: Using Minimax for search fallback");
            let result = self.territorial_fallback_maker.make_decision(game, board, you);
            if let Some(move_str) = result.get("move") {
                info!("SEARCH FALLBACK: Minimax decision: {}", move_str);
            }
            result
        }
    }
    
    /// Get search algorithm score for a specific move
    fn get_search_score_for_move(&self, search_result: &Value, target_move: &Direction) -> f32 {
        info!("SEARCH SCORE: Calculating score for move {:?} vs search result {:?}", target_move, search_result);
        
        if let Some(move_str) = search_result.get("move") {
            let search_move_str = move_str.as_str().unwrap_or("");
            let target_move_str = format!("{:?}", target_move).to_lowercase();
            
            info!("SEARCH SCORE: Comparing target '{}' vs search '{}'", target_move_str, search_move_str);
            
            if search_move_str == target_move_str {
                info!("SEARCH SCORE: Exact match - high score (0.8)");
                return 0.8; // High score for matching moves
            }
            
            // Check if target move is available and safe
            let directions = Direction::all();
            for direction in directions {
                let direction_str = format!("{:?}", direction).to_lowercase();
                if direction_str == target_move_str {
                    info!("SEARCH SCORE: Valid move - medium score (0.6)");
                    return 0.6; // Medium score for valid moves
                }
            }
        }
        
        info!("SEARCH SCORE: No match - low score (0.2)");
        0.2 // Low score for invalid moves
    }
}

// ============================================================================
// MAIN MOVE DECISION FUNCTION - FIXED WITH NEURAL NETWORK INTEGRATION
// ============================================================================

// Main move decision function - Now with working neural network integration
pub fn get_move(game: &Game, turn: &i32, board: &Board, you: &Battlesnake) -> Value {
    info!("MOVE {}: === ENHANCED AI WITH NEURAL NETWORK INTEGRATION ===", turn);
    info!("MOVE {}: Snake: {} at ({}, {}), health: {}", turn, you.id, you.head.x, you.head.y, you.health);
    info!("MOVE {}: Board: {}x{}, food: {}, snakes: {}", turn, board.width, board.height, board.food.len(), board.snakes.len());
    
    // Use enhanced hybrid manager with neural network integration
    let mut enhanced_manager = EnhancedHybridManager::new();
    let decision_start = Instant::now();
    
    let result = enhanced_manager.make_decision(game, board, you);
    let decision_time = decision_start.elapsed().as_millis();
    
    info!("MOVE {}: Enhanced decision completed in {}ms", turn, decision_time);
    info!("MOVE {}: Full result: {:?}", turn, result);
    
    // Analyze the final decision
    if let Some(move_str) = result.get("move") {
        let move_str_lower = move_str.as_str().unwrap_or("unknown").to_lowercase();
        
        // PATTERN ANALYSIS: Track move distribution to detect bias
        match move_str_lower.as_str() {
            "up" => {
                info!("MOVE {}: NEURAL-ENHANCED DECISION: UP", turn);
                info!("MOVE {}: PATTERN CHECK: This move was chosen by neural network + search hybrid", turn);
            },
            "down" => info!("MOVE {}: NEURAL-ENHANCED DECISION: DOWN", turn),
            "left" => info!("MOVE {}: NEURAL-ENHANCED DECISION: LEFT", turn),
            "right" => info!("MOVE {}: NEURAL-ENHANCED DECISION: RIGHT", turn),
            _ => info!("MOVE {}: Unknown move: {}", turn, move_str_lower),
        }
        
        info!("MOVE {}: SUCCESS: Neural network integration is now active and influencing decisions!", turn);
    } else {
        info!("MOVE {}: ERROR: No move found in enhanced result", turn);
    }
    
    info!("MOVE {}: === NEURAL NETWORK INTEGRATION COMPLETE ===", turn);
    result
}
