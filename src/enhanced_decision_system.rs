// Enhanced Decision System - Final integration layer
// Replaces the neural network integration in logic.rs with the adaptive system

use crate::adaptive_neural_system::{AdaptiveNeuralSystem, GameOutcome};
use crate::neural_confidence_integration::DecisionOutcome;
use crate::confidence_validation::GameValidationContext;
use crate::main::{Board, Battlesnake, Coord};
use anyhow::{Result, anyhow};
use std::sync::{Arc, Mutex};
use log::{info, warn, debug, error};
use std::collections::HashMap;

/// Global enhanced decision system instance
static mut ENHANCED_SYSTEM: Option<Arc<Mutex<AdaptiveNeuralSystem>>> = None;
static SYSTEM_INIT: std::sync::Once = std::sync::Once::new();

/// Enhanced decision system that provides a drop-in replacement for existing logic
pub struct EnhancedDecisionSystem;

impl EnhancedDecisionSystem {
    /// Initialize the enhanced system (call once at startup)
    pub fn initialize() -> Result<()> {
        SYSTEM_INIT.call_once(|| {
            match AdaptiveNeuralSystem::new() {
                Ok(system) => {
                    unsafe {
                        ENHANCED_SYSTEM = Some(Arc::new(Mutex::new(system)));
                    }
                    info!("Enhanced neural decision system initialized successfully");
                }
                Err(e) => {
                    error!("Failed to initialize enhanced decision system: {}", e);
                    // Don't panic - allow fallback to basic decision making
                }
            }
        });
        
        Ok(())
    }

    /// Check if the enhanced system is available
    pub fn is_available() -> bool {
        unsafe {
            ENHANCED_SYSTEM.is_some()
        }
    }

    /// Make an enhanced move decision using the adaptive neural system
    /// This is the main interface that replaces the neural network calls in logic.rs
    pub fn choose_enhanced_move(
        board: &Board,
        you: &Battlesnake,
        turn: i32,
        safe_moves: &[String],
    ) -> Result<String> {
        if !Self::is_available() {
            return Err(anyhow!("Enhanced decision system not initialized"));
        }

        unsafe {
            if let Some(system_arc) = &ENHANCED_SYSTEM {
                let mut system = system_arc.lock().unwrap();
                let (chosen_move, decision_record) = system.make_enhanced_move_decision(
                    board, you, turn, safe_moves
                )?;
                
                info!("Enhanced decision: move '{}', confidence {:.3}, source: {:?}", 
                      chosen_move,
                      decision_record.confidence.unified_confidence,
                      decision_record.decision_source);
                
                Ok(chosen_move)
            } else {
                Err(anyhow!("Enhanced system not available"))
            }
        }
    }

    /// Record decision outcome for learning (call when you know how a move turned out)
    pub fn record_move_outcome(
        move_id: &str,
        outcome: MoveOutcome,
        context_after: Option<MoveContext>,
    ) -> Result<()> {
        if !Self::is_available() {
            return Ok(()); // Silently ignore if system not available
        }

        let decision_outcome = match outcome {
            MoveOutcome::Excellent => DecisionOutcome::Excellent,
            MoveOutcome::Good => DecisionOutcome::Good,
            MoveOutcome::Neutral => DecisionOutcome::Neutral,
            MoveOutcome::Poor => DecisionOutcome::Poor,
            MoveOutcome::Terrible => DecisionOutcome::Terrible,
        };

        let validation_context = context_after.map(|ctx| GameValidationContext {
            turn_number: ctx.turn_number,
            health_before: ctx.health_before,
            health_after: Some(ctx.health_after),
            length_before: ctx.length_before,
            length_after: Some(ctx.length_after),
            distance_to_food_before: ctx.distance_to_food_before,
            distance_to_food_after: Some(ctx.distance_to_food_after),
            danger_level: ctx.danger_level,
            alternative_moves_available: ctx.alternative_moves_available,
        });

        unsafe {
            if let Some(system_arc) = &ENHANCED_SYSTEM {
                let mut system = system_arc.lock().unwrap();
                system.record_decision_outcome(move_id, decision_outcome, validation_context)?;
            }
        }

        Ok(())
    }

    /// Record game outcome for higher-level learning
    pub fn record_game_outcome(outcome: GameResult, final_length: Option<usize>) -> Result<()> {
        if !Self::is_available() {
            return Ok(());
        }

        let game_outcome = match outcome {
            GameResult::Victory => GameOutcome::Victory,
            GameResult::Defeat => GameOutcome::Defeat,
            GameResult::Ongoing => GameOutcome::Ongoing,
        };

        let final_score = final_length.map(|len| len as f32);

        unsafe {
            if let Some(system_arc) = &ENHANCED_SYSTEM {
                let mut system = system_arc.lock().unwrap();
                system.record_game_outcome(game_outcome, final_score)?;
            }
        }

        Ok(())
    }

    /// Get system performance metrics for monitoring
    pub fn get_performance_metrics() -> Option<PerformanceMetrics> {
        if !Self::is_available() {
            return None;
        }

        unsafe {
            if let Some(system_arc) = &ENHANCED_SYSTEM {
                let system = system_arc.lock().unwrap();
                let status = system.get_system_status();
                
                return Some(PerformanceMetrics {
                    total_decisions: status.adaptation_state.total_decisions_made,
                    success_rate: status.success_rate,
                    confidence_accuracy: status.confidence_accuracy,
                    performance_trend: format!("{:?}", status.adaptation_state.performance_trend),
                    decisions_until_optimization: status.decisions_until_next_optimization,
                    optimization_in_progress: status.optimization_in_progress,
                });
            }
        }

        None
    }

    /// Force optimization cycle (for testing/debugging)
    pub fn force_optimization() -> Result<String> {
        if !Self::is_available() {
            return Err(anyhow!("Enhanced system not available"));
        }

        unsafe {
            if let Some(system_arc) = &ENHANCED_SYSTEM {
                let mut system = system_arc.lock().unwrap();
                let report = system.force_optimization()?;
                Ok(format!("Optimization complete: {:.3} -> {:.3} performance", 
                          report.performance_before, report.performance_after))
            } else {
                Err(anyhow!("Enhanced system not available"))
            }
        }
    }

    /// Export detailed system analysis for debugging
    pub fn export_system_analysis() -> Result<String> {
        if !Self::is_available() {
            return Err(anyhow!("Enhanced system not available"));
        }

        unsafe {
            if let Some(system_arc) = &ENHANCED_SYSTEM {
                let system = system_arc.lock().unwrap();
                system.export_system_analysis()
            } else {
                Err(anyhow!("Enhanced system not available"))
            }
        }
    }
}

/// Simplified move outcome for external integration
#[derive(Debug, Clone)]
pub enum MoveOutcome {
    Excellent, // Move led to significant advantage (food, good position)
    Good,      // Move was beneficial but not game-changing
    Neutral,   // Move had no clear positive/negative impact
    Poor,      // Move led to disadvantage but not critical
    Terrible,  // Move led to danger, collision, or major disadvantage
}

/// Context information after a move for outcome assessment
#[derive(Debug, Clone)]
pub struct MoveContext {
    pub turn_number: i32,
    pub health_before: i32,
    pub health_after: i32,
    pub length_before: usize,
    pub length_after: usize,
    pub distance_to_food_before: f32,
    pub distance_to_food_after: f32,
    pub danger_level: f32,
    pub alternative_moves_available: u32,
}

/// Game result for final outcome tracking
#[derive(Debug, Clone)]
pub enum GameResult {
    Victory, // Snake won the game
    Defeat,  // Snake lost/died
    Ongoing, // Game still in progress
}

/// Simplified performance metrics for monitoring
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_decisions: u64,
    pub success_rate: f32,
    pub confidence_accuracy: f32,
    pub performance_trend: String,
    pub decisions_until_optimization: u64,
    pub optimization_in_progress: bool,
}

/// Utility functions for logic.rs integration

/// Check if a move outcome can be automatically determined
pub fn assess_move_outcome_automatically(
    board_before: &Board,
    you_before: &Battlesnake,
    board_after: &Board,
    you_after: &Battlesnake,
    chosen_move: &str,
) -> Option<MoveOutcome> {
    // Automatic outcome assessment based on game state changes
    
    // Check if we died (ultimate failure)
    if you_after.body.is_empty() || you_after.health <= 0 {
        return Some(MoveOutcome::Terrible);
    }
    
    // Check if we got food (clear positive)
    if you_after.length > you_before.length {
        return Some(MoveOutcome::Good);
    }
    
    // Check if we moved into immediate danger
    let head_after = you_after.body[0];
    let immediate_danger = is_position_dangerous(&board_after, &head_after, you_after);
    let was_in_danger = if you_before.body.len() > 0 { 
        is_position_dangerous(&board_before, &you_before.body[0], you_before) 
    } else { false };
    
    if immediate_danger && !was_in_danger {
        return Some(MoveOutcome::Poor);
    }
    
    // Check if we escaped danger
    if was_in_danger && !immediate_danger {
        return Some(MoveOutcome::Good);
    }
    
    // Check health change
    let health_change = you_after.health - you_before.health;
    if health_change > 0 {
        return Some(MoveOutcome::Good); // Got food
    } else if health_change < -10 && you_after.health < 30 {
        return Some(MoveOutcome::Poor); // Significant health loss while already low
    }
    
    // If no clear indicators, return None for manual assessment
    None
}

/// Helper function to check if a position is dangerous
fn is_position_dangerous(board: &Board, pos: &Coord, you: &Battlesnake) -> bool {
    // Check bounds
    if pos.x < 0 || pos.x >= board.width || pos.y < 0 || pos.y >= board.height as i32 {
        return true;
    }
    
    // Check collision with our body (excluding tail which may move)
    let body_without_tail = if you.body.len() > 1 { 
        &you.body[..you.body.len()-1] 
    } else { 
        &you.body[..] 
    };
    
    if body_without_tail.contains(pos) {
        return true;
    }
    
    // Check collision with other snakes
    for snake in &board.snakes {
        if snake.id != you.id {
            // Other snake's body (excluding tail unless they ate food)
            let other_body = if snake.health == 100 || snake.length > you.length {
                &snake.body[..]  // Include tail if they might not move it
            } else if snake.body.len() > 1 {
                &snake.body[..snake.body.len()-1]  // Exclude tail
            } else {
                &snake.body[..]
            };
            
            if other_body.contains(pos) {
                return true;
            }
        }
    }
    
    false
}

/// Integration helper for logic.rs - provides fallback decision making
pub fn make_fallback_decision(safe_moves: &[String]) -> String {
    if safe_moves.is_empty() {
        // If no safe moves, just pick any valid direction as last resort
        warn!("No safe moves available - using fallback");
        return "up".to_string();
    }
    
    // Simple fallback: prefer moves in this order for basic behavior
    let preferred_moves = ["up", "left", "right", "down"];
    
    for preferred in &preferred_moves {
        if safe_moves.contains(&preferred.to_string()) {
            debug!("Fallback decision: {}", preferred);
            return preferred.to_string();
        }
    }
    
    // If none of the preferred moves are safe, pick the first safe move
    let fallback = safe_moves[0].clone();
    debug!("Fallback decision: {}", fallback);
    fallback
}

/// Helper to create a simple move context from board states
pub fn create_move_context(
    turn: i32,
    you_before: &Battlesnake,
    you_after: &Battlesnake,
    board_before: &Board,
    board_after: &Board,
) -> MoveContext {
    let food_distance_before = calculate_min_food_distance(&you_before.body[0], &board_before.food);
    let food_distance_after = calculate_min_food_distance(&you_after.body[0], &board_after.food);
    
    MoveContext {
        turn_number: turn,
        health_before: you_before.health,
        health_after: you_after.health,
        length_before: you_before.length,
        length_after: you_after.length,
        distance_to_food_before: food_distance_before,
        distance_to_food_after: food_distance_after,
        danger_level: calculate_position_danger_level(&you_after.body[0], board_after, you_after),
        alternative_moves_available: count_safe_moves(board_after, you_after),
    }
}

fn calculate_min_food_distance(head: &Coord, food: &[Coord]) -> f32 {
    if food.is_empty() {
        return 100.0; // Large distance if no food
    }
    
    food.iter()
        .map(|f| ((head.x - f.x).abs() + (head.y - f.y).abs()) as f32)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(100.0)
}

fn calculate_position_danger_level(pos: &Coord, board: &Board, you: &Battlesnake) -> f32 {
    let mut danger_score = 0.0;
    
    // Proximity to walls
    let wall_distance = std::cmp::min(
        std::cmp::min(pos.x, board.width - pos.x - 1),
        std::cmp::min(pos.y, board.height as i32 - pos.y - 1)
    );
    
    if wall_distance <= 1 {
        danger_score += 0.3;
    }
    
    // Proximity to other snakes
    for snake in &board.snakes {
        if snake.id != you.id {
            let min_distance = snake.body.iter()
                .map(|segment| ((pos.x - segment.x).abs() + (pos.y - segment.y).abs()) as f32)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(100.0);
                
            if min_distance <= 2.0 {
                danger_score += (3.0 - min_distance) * 0.2;
            }
        }
    }
    
    // Proximity to our own body
    let self_distance = you.body.iter()
        .map(|segment| ((pos.x - segment.x).abs() + (pos.y - segment.y).abs()) as f32)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(100.0);
        
    if self_distance <= 1.0 {
        danger_score += 0.4;
    }
    
    danger_score.min(1.0)
}

fn count_safe_moves(board: &Board, you: &Battlesnake) -> u32 {
    if you.body.is_empty() {
        return 0;
    }
    
    let head = you.body[0];
    let possible_moves = [
        Coord { x: head.x, y: head.y + 1 }, // up
        Coord { x: head.x, y: head.y - 1 }, // down
        Coord { x: head.x - 1, y: head.y }, // left
        Coord { x: head.x + 1, y: head.y }, // right
    ];
    
    possible_moves.iter()
        .filter(|pos| !is_position_dangerous(board, pos, you))
        .count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move_outcome_assessment() {
        // Test cases would go here
        // This is a placeholder for comprehensive testing
        assert!(true);
    }

    #[test] 
    fn test_danger_level_calculation() {
        // Test danger level calculation
        // This is a placeholder for comprehensive testing
        assert!(true);
    }

    #[test]
    fn test_fallback_decision_making() {
        let safe_moves = vec!["up".to_string(), "left".to_string()];
        let decision = make_fallback_decision(&safe_moves);
        assert!(safe_moves.contains(&decision));
        
        // Test with empty moves
        let empty_moves: Vec<String> = vec![];
        let fallback_decision = make_fallback_decision(&empty_moves);
        assert_eq!(fallback_decision, "up");
    }
}