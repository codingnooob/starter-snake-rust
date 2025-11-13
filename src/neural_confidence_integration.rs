// Neural Network Confidence Integration
// Replaces the three inconsistent confidence calculation methods
// with the unified system based on actual neural network outputs

use crate::unified_confidence::{UnifiedConfidenceCalculator, NeuralConfidence, ConfidenceLevel};
use crate::main::{Board, Battlesnake};
use anyhow::{Result, anyhow};
use ndarray::Array2;
use log::{info, warn, debug, error};
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};

/// Enhanced Neural Network Evaluator with Proper Confidence Calculation
/// Replaces the broken confidence systems in neural_network.rs and neural_network_integration.rs
pub struct EnhancedNeuralEvaluator {
    confidence_calculator: Arc<RwLock<UnifiedConfidenceCalculator>>,
    metrics_collector: Arc<RwLock<ConfidenceMetricsCollector>>,
    decision_history: Arc<RwLock<Vec<NeuralDecisionRecord>>>,
}

/// Metrics collection for monitoring and validation
#[derive(Debug, Clone)]
pub struct ConfidenceMetricsCollector {
    pub total_decisions: u64,
    pub neural_network_decisions: u64,
    pub high_confidence_decisions: u64,
    pub medium_confidence_decisions: u64,
    pub low_confidence_decisions: u64,
    pub safety_fallbacks: u64,
    pub confidence_score_history: Vec<f32>,
    pub decision_outcome_correlation: Vec<(f32, bool)>, // (confidence, was_good_move)
}

/// Record of a neural network decision for analysis and learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralDecisionRecord {
    pub timestamp: String,
    pub confidence: NeuralConfidence,
    pub chosen_move: String,
    pub alternative_moves: Vec<String>,
    pub board_hash: u64,
    pub game_context: GameContextSnapshot,
    pub decision_outcome: Option<DecisionOutcome>, // Filled in retrospectively
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameContextSnapshot {
    pub our_health: i32,
    pub our_length: usize,
    pub turn_number: i32,
    pub num_opponents: usize,
    pub food_available: usize,
    pub board_size: (i32, u32), // Note: keeping original type inconsistency for compatibility
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionOutcome {
    Excellent, // Led to food, avoided danger, gained advantage
    Good,      // Safe move, reasonable choice
    Neutral,   // No immediate consequences
    Poor,      // Missed opportunity, suboptimal
    Terrible,  // Led to collision, death, or major disadvantage
}

/// Neural Network Decision Result with comprehensive information
#[derive(Debug, Clone)]
pub struct NeuralDecisionResult {
    pub recommended_move: String,
    pub confidence: NeuralConfidence,
    pub decision_source: DecisionSource,
    pub all_move_evaluations: Vec<MoveEvaluation>,
    pub reasoning: String,
    pub should_log_decision: bool,
}

#[derive(Debug, Clone)]
pub enum DecisionSource {
    NeuralNetworkHighConfidence,
    NeuralNetworkMediumConfidence,
    HeuristicFallback(String), // Reason for fallback
    SafetyOverride(String),    // Reason for safety intervention
}

#[derive(Debug, Clone)]
pub struct MoveEvaluation {
    pub move_name: String,
    pub neural_probability: f32,
    pub position_score: f32,
    pub win_probability: f32,
    pub safety_score: f32,
    pub is_safe: bool,
}

impl Default for ConfidenceMetricsCollector {
    fn default() -> Self {
        Self {
            total_decisions: 0,
            neural_network_decisions: 0,
            high_confidence_decisions: 0,
            medium_confidence_decisions: 0,
            low_confidence_decisions: 0,
            safety_fallbacks: 0,
            confidence_score_history: Vec::new(),
            decision_outcome_correlation: Vec::new(),
        }
    }
}

impl EnhancedNeuralEvaluator {
    pub fn new() -> Self {
        Self {
            confidence_calculator: Arc::new(RwLock::new(UnifiedConfidenceCalculator::new())),
            metrics_collector: Arc::new(RwLock::new(ConfidenceMetricsCollector::default())),
            decision_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Main decision-making function that replaces the broken confidence systems
    /// This is the unified entry point that replaces:
    /// - neural_network_integration.rs estimate_confidence()
    /// - simple_neural_integration.rs confidence calculations
    /// - logic.rs AdvancedNeuralEvaluator confidence logic
    pub fn make_neural_decision(
        &self,
        board: &Board,
        our_snake: &Battlesnake,
        safe_moves: &[String],
        neural_outputs: &NeuralNetworkOutputs,
    ) -> Result<NeuralDecisionResult> {
        let mut metrics = self.metrics_collector.write();
        metrics.total_decisions += 1;

        // 1. Calculate confidence for each neural network output
        let move_confidence = if let Some(ref move_probs) = neural_outputs.move_probabilities {
            self.confidence_calculator
                .read()
                .calculate_move_confidence(move_probs)?
        } else {
            return Err(anyhow!("No move probabilities available from neural network"));
        };

        let position_confidence = if let Some(position_score) = neural_outputs.position_score {
            self.confidence_calculator
                .read()
                .calculate_value_confidence(position_score, "position")?
        } else {
            return self.handle_missing_neural_output("position evaluation", safe_moves);
        };

        let outcome_confidence = if let Some(outcome_prob) = neural_outputs.win_probability {
            self.confidence_calculator
                .read()
                .calculate_value_confidence(outcome_prob, "outcome")?
        } else {
            return self.handle_missing_neural_output("game outcome", safe_moves);
        };

        // 2. Calculate multi-model consistency
        let consistency_score = if let Some(ref move_probs) = neural_outputs.move_probabilities {
            let move_probs_slice = move_probs.row(0).as_slice().unwrap();
            self.confidence_calculator.read().calculate_consistency_confidence(
                neural_outputs.position_score.unwrap_or(0.5),
                move_probs_slice,
                neural_outputs.win_probability.unwrap_or(0.5),
            )
        } else {
            0.5 // Default consistency when data is missing
        };

        // 3. Create unified confidence combining all models
        let unified_confidence = self.create_unified_confidence(
            &move_confidence,
            &position_confidence,
            &outcome_confidence,
            consistency_score,
        );

        // 4. Make decision based on confidence
        let decision_result = self.make_decision_from_confidence(
            &unified_confidence,
            neural_outputs,
            safe_moves,
            board,
            our_snake,
        )?;

        // 5. Record decision for learning and analysis
        self.record_decision(&decision_result, board, our_snake, &unified_confidence);

        // 6. Update metrics
        self.update_metrics(&mut metrics, &unified_confidence, &decision_result);

        Ok(decision_result)
    }

    /// Create a unified confidence score combining all neural network models
    fn create_unified_confidence(
        &self,
        move_conf: &NeuralConfidence,
        position_conf: &NeuralConfidence,
        outcome_conf: &NeuralConfidence,
        consistency: f32,
    ) -> NeuralConfidence {
        // Weighted combination of all confidence sources
        let combined_confidence = (
            move_conf.unified_confidence * 0.4 +        // Move prediction is most important
            position_conf.unified_confidence * 0.3 +    // Position evaluation
            outcome_conf.unified_confidence * 0.2 +     // Game outcome
            consistency * 0.1                           // Inter-model consistency
        ).min(1.0);

        // Take the most conservative confidence level
        let conservative_level = match (
            &move_conf.confidence_level,
            &position_conf.confidence_level,
            &outcome_conf.confidence_level,
        ) {
            (ConfidenceLevel::High, ConfidenceLevel::High, ConfidenceLevel::High) => ConfidenceLevel::High,
            (ConfidenceLevel::Low, _, _) | (_, ConfidenceLevel::Low, _) | (_, _, ConfidenceLevel::Low) => ConfidenceLevel::Low,
            _ => ConfidenceLevel::Medium,
        };

        // Generate comprehensive reasoning
        let reasoning = format!(
            "Unified: {:.3} (move: {:.3}, pos: {:.3}, out: {:.3}, consistency: {:.3})",
            combined_confidence,
            move_conf.unified_confidence,
            position_conf.unified_confidence,
            outcome_conf.unified_confidence,
            consistency
        );

        // Always require safety validation - NEVER bypass safety checks
        let should_use_neural = combined_confidence > 0.3; // Reasonable threshold
        let always_check_safety = true; // CRITICAL: Always validate safety

        NeuralConfidence {
            entropy_confidence: move_conf.entropy_confidence,
            probability_confidence: move_conf.probability_confidence,
            deviation_confidence: (position_conf.deviation_confidence + outcome_conf.deviation_confidence) / 2.0,
            consistency_confidence: consistency,
            unified_confidence: combined_confidence,
            confidence_level: conservative_level,
            should_use_neural_network: should_use_neural,
            should_apply_safety_check: always_check_safety,
            recommendation_reason: reasoning,
            raw_metrics: move_conf.raw_metrics.clone(),
        }
    }

    /// Make the final decision based on unified confidence
    fn make_decision_from_confidence(
        &self,
        confidence: &NeuralConfidence,
        neural_outputs: &NeuralNetworkOutputs,
        safe_moves: &[String],
        board: &Board,
        our_snake: &Battlesnake,
    ) -> Result<NeuralDecisionResult> {
        let move_names = ["up", "down", "left", "right"];
        
        // Get move probabilities
        let move_probs = neural_outputs.move_probabilities.as_ref()
            .ok_or_else(|| anyhow!("No move probabilities available"))?;
        let probs_slice = move_probs.row(0).as_slice().unwrap();

        // Create move evaluations
        let mut move_evaluations: Vec<MoveEvaluation> = move_names.iter()
            .enumerate()
            .map(|(i, &move_name)| {
                let is_safe = safe_moves.contains(&move_name.to_string());
                MoveEvaluation {
                    move_name: move_name.to_string(),
                    neural_probability: probs_slice[i],
                    position_score: neural_outputs.position_score.unwrap_or(0.5),
                    win_probability: neural_outputs.win_probability.unwrap_or(0.5),
                    safety_score: if is_safe { 1.0 } else { 0.0 },
                    is_safe,
                }
            })
            .collect();

        // Sort by neural network preference
        move_evaluations.sort_by(|a, b| b.neural_probability.partial_cmp(&a.neural_probability).unwrap());

        // Decision logic based on confidence level
        let (recommended_move, decision_source, reasoning) = match confidence.confidence_level {
            ConfidenceLevel::High | ConfidenceLevel::Medium if confidence.should_use_neural_network => {
                // Use neural network but ALWAYS validate safety
                let neural_preferred = &move_evaluations[0];
                
                if neural_preferred.is_safe {
                    // Neural network choice is safe - use it
                    let source = match confidence.confidence_level {
                        ConfidenceLevel::High => DecisionSource::NeuralNetworkHighConfidence,
                        _ => DecisionSource::NeuralNetworkMediumConfidence,
                    };
                    let reason = format!(
                        "Neural network chose {} (prob: {:.3}, confidence: {:.3}) - SAFE",
                        neural_preferred.move_name, neural_preferred.neural_probability, confidence.unified_confidence
                    );
                    (neural_preferred.move_name.clone(), source, reason)
                } else {
                    // Neural network choice is unsafe - override with safety
                    let safe_preferred = move_evaluations.iter()
                        .find(|eval| eval.is_safe)
                        .ok_or_else(|| anyhow!("No safe moves available"))?;
                    
                    let reason = format!(
                        "Neural network chose {} but UNSAFE - safety override to {}",
                        neural_preferred.move_name, safe_preferred.move_name
                    );
                    (
                        safe_preferred.move_name.clone(),
                        DecisionSource::SafetyOverride(reason.clone()),
                        reason
                    )
                }
            }
            _ => {
                // Low confidence or neural network not recommended - fallback to heuristics
                let fallback_reason = format!(
                    "Low neural confidence ({:.3}) - falling back to heuristic selection",
                    confidence.unified_confidence
                );
                
                // Choose the safest move (in practice, this would integrate with existing heuristics)
                let heuristic_choice = safe_moves.first()
                    .ok_or_else(|| anyhow!("No safe moves available for heuristic fallback"))?;
                
                (
                    heuristic_choice.clone(),
                    DecisionSource::HeuristicFallback(fallback_reason.clone()),
                    fallback_reason
                )
            }
        };

        Ok(NeuralDecisionResult {
            recommended_move,
            confidence: confidence.clone(),
            decision_source,
            all_move_evaluations: move_evaluations,
            reasoning,
            should_log_decision: true, // Always log for analysis
        })
    }

    /// Handle missing neural network outputs gracefully
    fn handle_missing_neural_output(&self, model_type: &str, safe_moves: &[String]) -> Result<NeuralDecisionResult> {
        warn!("Neural network {} output missing - falling back to heuristics", model_type);
        
        let fallback_move = safe_moves.first()
            .ok_or_else(|| anyhow!("No safe moves available for fallback"))?
            .clone();

        // Create minimal confidence for fallback
        let fallback_confidence = NeuralConfidence {
            entropy_confidence: 0.0,
            probability_confidence: 0.0,
            deviation_confidence: 0.0,
            consistency_confidence: 0.0,
            unified_confidence: 0.0,
            confidence_level: ConfidenceLevel::Low,
            should_use_neural_network: false,
            should_apply_safety_check: true,
            recommendation_reason: format!("Missing {} output", model_type),
            raw_metrics: crate::unified_confidence::RawConfidenceMetrics {
                entropy: None,
                max_probability: None,
                deviation_from_neutral: None,
                prediction_spread: None,
                model_uncertainty: 1.0,
            },
        };

        Ok(NeuralDecisionResult {
            recommended_move: fallback_move,
            confidence: fallback_confidence,
            decision_source: DecisionSource::HeuristicFallback(format!("Missing {}", model_type)),
            all_move_evaluations: vec![],
            reasoning: format!("Neural network {} unavailable", model_type),
            should_log_decision: true,
        })
    }

    /// Record decision for learning and analysis
    fn record_decision(
        &self,
        decision: &NeuralDecisionResult,
        board: &Board,
        our_snake: &Battlesnake,
        confidence: &NeuralConfidence,
    ) {
        let record = NeuralDecisionRecord {
            timestamp: chrono::Utc::now().to_rfc3339(),
            confidence: confidence.clone(),
            chosen_move: decision.recommended_move.clone(),
            alternative_moves: decision.all_move_evaluations.iter()
                .map(|eval| eval.move_name.clone())
                .collect(),
            board_hash: self.calculate_board_hash(board),
            game_context: GameContextSnapshot {
                our_health: our_snake.health,
                our_length: our_snake.body.len(),
                turn_number: board.turn,
                num_opponents: board.snakes.len() - 1, // Exclude ourselves
                food_available: board.food.len(),
                board_size: (board.width, board.height),
            },
            decision_outcome: None, // Will be filled in retrospectively
        };

        self.decision_history.write().push(record);

        // Keep only recent decisions to prevent unbounded memory growth
        let mut history = self.decision_history.write();
        if history.len() > 10000 {
            history.drain(0..5000); // Keep the most recent 5000
        }
    }

    /// Update metrics for monitoring
    fn update_metrics(
        &self,
        metrics: &mut ConfidenceMetricsCollector,
        confidence: &NeuralConfidence,
        decision: &NeuralDecisionResult,
    ) {
        if confidence.should_use_neural_network {
            metrics.neural_network_decisions += 1;
        }

        match confidence.confidence_level {
            ConfidenceLevel::High => metrics.high_confidence_decisions += 1,
            ConfidenceLevel::Medium => metrics.medium_confidence_decisions += 1,
            ConfidenceLevel::Low => metrics.low_confidence_decisions += 1,
        }

        if matches!(decision.decision_source, DecisionSource::SafetyOverride(_)) {
            metrics.safety_fallbacks += 1;
        }

        metrics.confidence_score_history.push(confidence.unified_confidence);

        // Keep only recent history to prevent memory growth
        if metrics.confidence_score_history.len() > 1000 {
            metrics.confidence_score_history.drain(0..500);
        }
    }

    /// Calculate a simple hash of the board state for tracking
    fn calculate_board_hash(&self, board: &Board) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        
        // Hash key board features
        board.width.hash(&mut hasher);
        board.height.hash(&mut hasher);
        board.turn.hash(&mut hasher);
        
        // Hash food positions
        for food in &board.food {
            food.x.hash(&mut hasher);
            food.y.hash(&mut hasher);
        }
        
        // Hash snake positions (simplified)
        for snake in &board.snakes {
            snake.health.hash(&mut hasher);
            snake.body.len().hash(&mut hasher);
            if !snake.body.is_empty() {
                snake.body[0].x.hash(&mut hasher);
                snake.body[0].y.hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Get current metrics for monitoring
    pub fn get_metrics(&self) -> ConfidenceMetricsCollector {
        self.metrics_collector.read().clone()
    }

    /// Get recent decision history for analysis
    pub fn get_recent_decisions(&self, count: usize) -> Vec<NeuralDecisionRecord> {
        let history = self.decision_history.read();
        let start = history.len().saturating_sub(count);
        history[start..].to_vec()
    }

    /// Update confidence calculator configuration
    pub fn update_confidence_config(&self, config: crate::unified_confidence::ConfidenceConfig) {
        *self.confidence_calculator.write() = UnifiedConfidenceCalculator::with_config(config);
        info!("Updated neural network confidence configuration");
    }

    /// Export metrics and decisions for analysis
    pub fn export_analysis_data(&self) -> Result<String> {
        let metrics = self.get_metrics();
        let recent_decisions = self.get_recent_decisions(100);

        let analysis_data = serde_json::json!({
            "metrics": {
                "total_decisions": metrics.total_decisions,
                "neural_network_decisions": metrics.neural_network_decisions,
                "high_confidence_decisions": metrics.high_confidence_decisions,
                "medium_confidence_decisions": metrics.medium_confidence_decisions,
                "low_confidence_decisions": metrics.low_confidence_decisions,
                "safety_fallbacks": metrics.safety_fallbacks,
                "neural_utilization_rate": if metrics.total_decisions > 0 {
                    metrics.neural_network_decisions as f64 / metrics.total_decisions as f64
                } else { 0.0 },
                "average_confidence": if metrics.confidence_score_history.is_empty() { 0.0 } else {
                    metrics.confidence_score_history.iter().sum::<f32>() / metrics.confidence_score_history.len() as f32
                }
            },
            "recent_decisions": recent_decisions
        });

        serde_json::to_string_pretty(&analysis_data)
            .map_err(|e| anyhow!("Failed to serialize analysis data: {}", e))
    }
}

/// Neural Network Outputs structure for the unified system
#[derive(Debug, Clone)]
pub struct NeuralNetworkOutputs {
    pub position_score: Option<f32>,
    pub move_probabilities: Option<Array2<f32>>,
    pub win_probability: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use crate::main::{Board, Battlesnake, Coord};

    fn create_test_board() -> Board {
        Board {
            height: 11,
            width: 11,
            food: vec![Coord { x: 5, y: 5 }],
            snakes: vec![],
            turn: 10,
        }
    }

    fn create_test_snake() -> Battlesnake {
        Battlesnake {
            id: "test".to_string(),
            name: "test".to_string(),
            health: 100,
            body: vec![Coord { x: 5, y: 6 }, Coord { x: 5, y: 7 }],
            head: Coord { x: 5, y: 6 },
            length: 2,
            latency: "0".to_string(),
            shout: None,
        }
    }

    #[test]
    fn test_unified_confidence_integration() {
        let evaluator = EnhancedNeuralEvaluator::new();
        let board = create_test_board();
        let snake = create_test_snake();
        let safe_moves = vec!["up".to_string(), "left".to_string()];

        // Test with low confidence (realistic current model outputs)
        let neural_outputs = NeuralNetworkOutputs {
            position_score: Some(0.52), // Close to neutral
            move_probabilities: Some(array![[0.26, 0.24, 0.25, 0.25]]), // Near random
            win_probability: Some(0.54), // Close to neutral
        };

        let result = evaluator.make_neural_decision(&board, &snake, &safe_moves, &neural_outputs).unwrap();

        // Should fallback to heuristics due to low confidence
        assert!(matches!(result.decision_source, DecisionSource::HeuristicFallback(_)));
        assert!(!result.confidence.should_use_neural_network);
    }

    #[test]
    fn test_safety_override_mechanism() {
        let evaluator = EnhancedNeuralEvaluator::new();
        let board = create_test_board();
        let snake = create_test_snake();
        let safe_moves = vec!["up".to_string(), "left".to_string()]; // "down" is not safe

        // Test with high confidence but unsafe neural choice
        let neural_outputs = NeuralNetworkOutputs {
            position_score: Some(0.85), // High confidence
            move_probabilities: Some(array![[0.1, 0.8, 0.05, 0.05]]), // Strongly prefers "down"
            win_probability: Some(0.80), // High confidence
        };

        let result = evaluator.make_neural_decision(&board, &snake, &safe_moves, &neural_outputs).unwrap();

        // Should override unsafe neural choice
        assert!(matches!(result.decision_source, DecisionSource::SafetyOverride(_)));
        assert!(safe_moves.contains(&result.recommended_move));
        assert_ne!(result.recommended_move, "down"); // Should not choose the unsafe move
    }

    #[test]
    fn test_high_confidence_safe_decision() {
        let evaluator = EnhancedNeuralEvaluator::new();
        let board = create_test_board();
        let snake = create_test_snake();
        let safe_moves = vec!["up".to_string(), "left".to_string(), "right".to_string()];

        // Test with high confidence and safe neural choice
        let neural_outputs = NeuralNetworkOutputs {
            position_score: Some(0.85), // High confidence
            move_probabilities: Some(array![[0.7, 0.1, 0.1, 0.1]]), // Strongly prefers "up" (safe)
            win_probability: Some(0.80), // High confidence
        };

        let result = evaluator.make_neural_decision(&board, &snake, &safe_moves, &neural_outputs).unwrap();

        // Should use neural network choice
        assert!(matches!(result.decision_source, DecisionSource::NeuralNetworkHighConfidence));
        assert_eq!(result.recommended_move, "up");
        assert!(result.confidence.should_apply_safety_check); // Safety always enabled
    }
}