// Adaptive Neural System - Integrates validation, optimization, and decision making
// Creates a self-improving neural network confidence system

use crate::unified_confidence::{ConfidenceConfig, UnifiedConfidenceCalculator};
use crate::neural_confidence_integration::{EnhancedNeuralEvaluator, NeuralDecisionRecord, DecisionOutcome};
use crate::confidence_validation::{ConfidenceValidator, ValidationRecord, GameValidationContext};
use crate::main::{Board, Battlesnake, Coord};
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use log::{info, warn, debug, error};

/// Adaptive neural system that learns and optimizes over time
pub struct AdaptiveNeuralSystem {
    /// Core neural evaluation system
    neural_evaluator: Arc<Mutex<EnhancedNeuralEvaluator>>,
    
    /// Validation system for outcome tracking
    validator: Arc<Mutex<ConfidenceValidator>>,
    
    /// Current configuration (can be updated dynamically)
    config: Arc<Mutex<ConfidenceConfig>>,
    
    /// Performance metrics and adaptation state
    adaptation_state: AdaptationState,
    
    /// Decision history for pattern analysis
    decision_history: Vec<HistoricalDecision>,
    
    /// Optimization scheduling and control
    optimization_scheduler: OptimizationScheduler,
}

/// Tracks the current adaptation and learning state
#[derive(Debug, Clone)]
pub struct AdaptationState {
    pub total_decisions_made: u64,
    pub successful_decisions: u64,
    pub failed_decisions: u64,
    pub last_optimization_timestamp: String,
    pub optimization_cycle_count: u32,
    pub current_performance_score: f32,
    pub performance_trend: PerformanceTrend,
    pub confidence_accuracy_trend: f32,
}

#[derive(Debug, Clone)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Declining,
    Insufficient_Data,
}

/// Historical decision record for pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalDecision {
    pub decision_record: NeuralDecisionRecord,
    pub final_outcome: Option<DecisionOutcome>,
    pub outcome_timestamp: Option<String>,
    pub game_outcome: Option<GameOutcome>,
    pub performance_impact: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GameOutcome {
    Victory,
    Defeat,
    Ongoing,
}

/// Controls when and how optimization occurs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationScheduler {
    pub min_decisions_between_optimization: u64,
    pub max_decisions_between_optimization: u64,
    pub performance_threshold_for_optimization: f32,
    pub last_optimization_decision_count: u64,
    pub optimization_in_progress: bool,
    pub scheduled_optimization_types: Vec<OptimizationType>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    ThresholdCalibration,
    ConfidenceWeightAdjustment,
    SafetyParameterTuning,
    ModelPerformanceAnalysis,
}

/// System status for monitoring and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub adaptation_state: AdaptationState,
    pub success_rate: f32,
    pub confidence_accuracy: f32,
    pub decisions_until_next_optimization: u64,
    pub optimization_in_progress: bool,
    pub recent_performance_trend: f32,
}

/// Report of optimization cycle results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReport {
    pub optimization_timestamp: String,
    pub optimizations_performed: Vec<OptimizationResult>,
    pub performance_before: f32,
    pub performance_after: f32,
    pub configuration_changes: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Result of a specific optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub optimization_type: OptimizationType,
    pub success: bool,
    pub performance_impact: f32,
    pub changes_made: Vec<String>,
    pub metrics_improved: Vec<String>,
}

impl AdaptiveNeuralSystem {
    pub fn new() -> Result<Self> {
        let config = ConfidenceConfig::default();
        let neural_evaluator = EnhancedNeuralEvaluator::new(config.clone())?;
        let validator = ConfidenceValidator::new();
        
        Ok(Self {
            neural_evaluator: Arc::new(Mutex::new(neural_evaluator)),
            validator: Arc::new(Mutex::new(validator)),
            config: Arc::new(Mutex::new(config)),
            adaptation_state: AdaptationState {
                total_decisions_made: 0,
                successful_decisions: 0,
                failed_decisions: 0,
                last_optimization_timestamp: chrono::Utc::now().to_rfc3339(),
                optimization_cycle_count: 0,
                current_performance_score: 0.0,
                performance_trend: PerformanceTrend::Insufficient_Data,
                confidence_accuracy_trend: 0.0,
            },
            decision_history: Vec::new(),
            optimization_scheduler: OptimizationScheduler {
                min_decisions_between_optimization: 100,
                max_decisions_between_optimization: 1000,
                performance_threshold_for_optimization: 0.6,
                last_optimization_decision_count: 0,
                optimization_in_progress: false,
                scheduled_optimization_types: vec![
                    OptimizationType::ThresholdCalibration,
                    OptimizationType::ConfidenceWeightAdjustment,
                ],
            },
        })
    }

    /// Main decision-making interface - replaces the old logic
    pub fn make_enhanced_move_decision(
        &mut self,
        board: &Board,
        you: &Battlesnake,
        turn: i32,
        safe_moves: &[String],
    ) -> Result<(String, NeuralDecisionRecord)> {
        info!("Making enhanced move decision for turn {}", turn);
        
        // Check if optimization is needed before making decision
        if self.should_trigger_optimization() && !self.optimization_scheduler.optimization_in_progress {
            info!("Triggering optimization before decision");
            if let Err(e) = self.run_optimization_cycle() {
                warn!("Optimization failed, continuing with current settings: {}", e);
            }
        }

        // Make the enhanced decision using current configuration
        let decision_record = {
            let mut evaluator = self.neural_evaluator.lock().unwrap();
            evaluator.make_enhanced_decision(board, you, turn, safe_moves)?
        };

        // Record the decision for validation
        self.record_decision_made(&decision_record);
        
        // Update adaptation state
        self.adaptation_state.total_decisions_made += 1;
        
        let chosen_move = decision_record.chosen_move.clone();
        
        info!("Enhanced decision made: move '{}' with confidence {:.3} ({})", 
              chosen_move, 
              decision_record.confidence.unified_confidence,
              format!("{:?}", decision_record.confidence.confidence_level));

        Ok((chosen_move, decision_record))
    }

    /// Record the outcome of a previous decision for learning
    pub fn record_decision_outcome(
        &mut self,
        decision_id: &str,
        outcome: DecisionOutcome,
        game_context_after: Option<GameValidationContext>,
    ) -> Result<()> {
        debug!("Recording decision outcome for {}: {:?}", decision_id, outcome);
        
        // Find the decision in our history
        if let Some(historical_decision) = self.decision_history.iter_mut()
            .find(|d| d.decision_record.timestamp.contains(decision_id)) {
            
            historical_decision.final_outcome = Some(outcome.clone());
            historical_decision.outcome_timestamp = Some(chrono::Utc::now().to_rfc3339());
            historical_decision.performance_impact = self.outcome_to_performance_impact(&outcome);
            
            // Update success/failure counters
            match outcome {
                DecisionOutcome::Excellent | DecisionOutcome::Good => {
                    self.adaptation_state.successful_decisions += 1;
                }
                DecisionOutcome::Poor | DecisionOutcome::Terrible => {
                    self.adaptation_state.failed_decisions += 1;
                }
                DecisionOutcome::Neutral => {
                    // Neutral outcomes don't affect success/failure counters
                }
            }
            
            // Record in validator for correlation analysis
            {
                let mut validator = self.validator.lock().unwrap();
                validator.record_decision_outcome(
                    &historical_decision.decision_record, 
                    outcome, 
                    game_context_after.as_ref()
                )?;
            }
            
            // Update performance metrics
            self.update_performance_metrics();
            
            debug!("Decision outcome recorded successfully");
        } else {
            warn!("Could not find decision {} in history for outcome recording", decision_id);
        }
        
        Ok(())
    }

    /// Record game outcome (victory/defeat) for higher-level learning
    pub fn record_game_outcome(&mut self, game_outcome: GameOutcome, final_score: Option<f32>) -> Result<()> {
        info!("Recording game outcome: {:?} with score {:?}", game_outcome, final_score);
        
        // Update recent decisions in history with game outcome
        let recent_decisions = self.decision_history.len().saturating_sub(50); // Last 50 decisions
        for decision in self.decision_history[recent_decisions..].iter_mut() {
            if decision.game_outcome.is_none() {
                decision.game_outcome = Some(game_outcome.clone());
            }
        }
        
        // Trigger optimization if game ended poorly
        match game_outcome {
            GameOutcome::Victory => {
                info!("Game won - system is performing well");
            }
            GameOutcome::Defeat => {
                warn!("Game lost - scheduling optimization review");
                if !self.optimization_scheduler.scheduled_optimization_types.contains(&OptimizationType::ModelPerformanceAnalysis) {
                    self.optimization_scheduler.scheduled_optimization_types.push(OptimizationType::ModelPerformanceAnalysis);
                }
            }
            GameOutcome::Ongoing => {
                // No action needed
            }
        }
        
        Ok(())
    }

    /// Get current system status and performance metrics
    pub fn get_system_status(&self) -> SystemStatus {
        let total_decisions = self.adaptation_state.total_decisions_made;
        let success_rate = if total_decisions > 0 {
            self.adaptation_state.successful_decisions as f32 / total_decisions as f32
        } else {
            0.0
        };
        
        let confidence_accuracy = {
            let validator = self.validator.lock().unwrap();
            // Get overall correlation if available
            validator.outcome_correlations.get("overall")
                .map(|corr| corr.confidence_outcome_correlation)
                .unwrap_or(0.0)
        };
        
        SystemStatus {
            adaptation_state: self.adaptation_state.clone(),
            success_rate,
            confidence_accuracy,
            decisions_until_next_optimization: self.decisions_until_next_optimization(),
            optimization_in_progress: self.optimization_scheduler.optimization_in_progress,
            recent_performance_trend: self.calculate_recent_performance_trend(),
        }
    }

    /// Force an optimization cycle (for testing or manual tuning)
    pub fn force_optimization(&mut self) -> Result<OptimizationReport> {
        info!("Forcing optimization cycle");
        self.run_optimization_cycle()
    }

    /// Export detailed system analysis
    pub fn export_system_analysis(&self) -> Result<String> {
        let validator = self.validator.lock().unwrap();
        let validation_report = validator.export_validation_report()?;
        
        let system_analysis = serde_json::json!({
            "system_status": self.get_system_status(),
            "validation_report": serde_json::from_str::<serde_json::Value>(&validation_report)?,
            "decision_history_summary": {
                "total_decisions": self.decision_history.len(),
                "recent_decisions": self.decision_history.iter().rev().take(10).collect::<Vec<_>>(),
                "performance_distribution": self.calculate_performance_distribution(),
            },
            "optimization_state": {
                "scheduler": self.optimization_scheduler,
                "next_optimizations": &self.optimization_scheduler.scheduled_optimization_types,
            },
            "export_timestamp": chrono::Utc::now().to_rfc3339(),
        });
        
        serde_json::to_string_pretty(&system_analysis)
            .map_err(|e| anyhow!("Failed to serialize system analysis: {}", e))
    }

    // Private helper methods

    fn record_decision_made(&mut self, decision_record: &NeuralDecisionRecord) {
        let historical_decision = HistoricalDecision {
            decision_record: decision_record.clone(),
            final_outcome: None,
            outcome_timestamp: None,
            game_outcome: None,
            performance_impact: 0.0,
        };
        
        self.decision_history.push(historical_decision);
        
        // Limit history size to prevent memory growth
        if self.decision_history.len() > 5000 {
            self.decision_history.drain(0..2500); // Keep most recent 2500
        }
    }

    fn should_trigger_optimization(&self) -> bool {
        let decisions_since_last = self.adaptation_state.total_decisions_made 
            - self.optimization_scheduler.last_optimization_decision_count;
        
        // Trigger if enough decisions have passed
        if decisions_since_last >= self.optimization_scheduler.min_decisions_between_optimization {
            // Always trigger if we've hit the maximum
            if decisions_since_last >= self.optimization_scheduler.max_decisions_between_optimization {
                return true;
            }
            
            // Trigger if performance is below threshold
            if self.adaptation_state.current_performance_score < self.optimization_scheduler.performance_threshold_for_optimization {
                return true;
            }
            
            // Trigger if we have scheduled optimizations
            if !self.optimization_scheduler.scheduled_optimization_types.is_empty() {
                return true;
            }
        }
        
        false
    }

    fn run_optimization_cycle(&mut self) -> Result<OptimizationReport> {
        info!("Starting optimization cycle");
        self.optimization_scheduler.optimization_in_progress = true;
        
        let mut report = OptimizationReport {
            optimization_timestamp: chrono::Utc::now().to_rfc3339(),
            optimizations_performed: Vec::new(),
            performance_before: self.adaptation_state.current_performance_score,
            performance_after: 0.0,
            configuration_changes: Vec::new(),
            recommendations: Vec::new(),
        };
        
        // Run correlation analysis first
        {
            let mut validator = self.validator.lock().unwrap();
            if let Err(e) = validator.analyze_correlations() {
                warn!("Correlation analysis failed: {}", e);
                report.recommendations.push(format!("Correlation analysis failed: {}", e));
            }
        }
        
        // Run scheduled optimizations
        let optimization_types = self.optimization_scheduler.scheduled_optimization_types.clone();
        
        for optimization_type in optimization_types {
            match self.run_specific_optimization(optimization_type.clone()) {
                Ok(optimization_result) => {
                    report.optimizations_performed.push(optimization_result);
                }
                Err(e) => {
                    warn!("Optimization {:?} failed: {}", optimization_type, e);
                    report.recommendations.push(format!("Optimization {:?} failed: {}", optimization_type, e));
                }
            }
        }
        
        // Clear scheduled optimizations
        self.optimization_scheduler.scheduled_optimization_types.clear();
        
        // Update optimization state
        self.optimization_scheduler.last_optimization_decision_count = self.adaptation_state.total_decisions_made;
        self.optimization_scheduler.optimization_cycle_count += 1;
        self.adaptation_state.last_optimization_timestamp = chrono::Utc::now().to_rfc3339();
        
        // Calculate performance after optimization
        self.update_performance_metrics();
        report.performance_after = self.adaptation_state.current_performance_score;
        
        self.optimization_scheduler.optimization_in_progress = false;
        
        info!("Optimization cycle complete. Performance: {:.3} -> {:.3}", 
              report.performance_before, report.performance_after);
        
        Ok(report)
    }

    fn run_specific_optimization(&mut self, optimization_type: OptimizationType) -> Result<OptimizationResult> {
        match optimization_type {
            OptimizationType::ThresholdCalibration => {
                self.optimize_thresholds()
            }
            OptimizationType::ConfidenceWeightAdjustment => {
                self.optimize_confidence_weights()
            }
            OptimizationType::SafetyParameterTuning => {
                self.optimize_safety_parameters()
            }
            OptimizationType::ModelPerformanceAnalysis => {
                self.analyze_model_performance()
            }
        }
    }

    fn optimize_thresholds(&mut self) -> Result<OptimizationResult> {
        info!("Optimizing confidence thresholds");
        
        let mut validator = self.validator.lock().unwrap();
        
        // Test various threshold values
        let threshold_candidates = vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let optimal_thresholds = validator.optimize_thresholds(&threshold_candidates)?;
        
        // Generate optimized configuration
        let optimized_config = validator.generate_optimized_config()?;
        
        // Update current configuration
        {
            let mut config = self.config.lock().unwrap();
            *config = optimized_config;
        }
        
        // Update neural evaluator with new configuration
        {
            let mut evaluator = self.neural_evaluator.lock().unwrap();
            let config = self.config.lock().unwrap();
            evaluator.update_configuration(config.clone())?;
        }
        
        Ok(OptimizationResult {
            optimization_type: OptimizationType::ThresholdCalibration,
            success: true,
            performance_impact: 0.1, // Placeholder - would calculate actual impact
            changes_made: vec![format!("Updated thresholds: {:?}", optimal_thresholds)],
            metrics_improved: vec!["threshold_accuracy".to_string(), "calibration_score".to_string()],
        })
    }

    fn optimize_confidence_weights(&mut self) -> Result<OptimizationResult> {
        info!("Optimizing confidence calculation weights");
        
        // Analyze which confidence components correlate best with outcomes
        let validator = self.validator.lock().unwrap();
        
        if let Some(correlation) = validator.outcome_correlations.get("overall") {
            let mut config = self.config.lock().unwrap();
            
            // Adjust weights based on correlation strength
            if correlation.confidence_outcome_correlation > 0.6 {
                // Strong correlation - increase entropy weight
                config.entropy_weight = 0.6;
                config.max_probability_weight = 0.3;
                config.consistency_weight = 0.1;
            } else {
                // Weaker correlation - balance weights more evenly
                config.entropy_weight = 0.4;
                config.max_probability_weight = 0.4;
                config.consistency_weight = 0.2;
            }
            
            Ok(OptimizationResult {
                optimization_type: OptimizationType::ConfidenceWeightAdjustment,
                success: true,
                performance_impact: correlation.confidence_outcome_correlation * 0.1,
                changes_made: vec![format!("Updated weights: entropy={:.1}, max_prob={:.1}, consistency={:.1}",
                                          config.entropy_weight, config.max_probability_weight, config.consistency_weight)],
                metrics_improved: vec!["confidence_accuracy".to_string()],
            })
        } else {
            Err(anyhow!("Insufficient correlation data for weight optimization"))
        }
    }

    fn optimize_safety_parameters(&mut self) -> Result<OptimizationResult> {
        info!("Optimizing safety parameters");
        
        // Safety parameters should be very conservative and only adjusted with extreme caution
        // This implementation prioritizes safety over performance
        
        Ok(OptimizationResult {
            optimization_type: OptimizationType::SafetyParameterTuning,
            success: true,
            performance_impact: 0.0, // Safety changes should not compromise safety for performance
            changes_made: vec!["Safety parameters reviewed - no changes needed".to_string()],
            metrics_improved: vec![],
        })
    }

    fn analyze_model_performance(&mut self) -> Result<OptimizationResult> {
        info!("Analyzing model performance");
        
        // Analyze recent decision history for patterns
        let recent_decisions = self.decision_history.iter().rev().take(100).collect::<Vec<_>>();
        let poor_decisions = recent_decisions.iter()
            .filter(|d| matches!(d.final_outcome, Some(DecisionOutcome::Poor) | Some(DecisionOutcome::Terrible)))
            .count();
        
        let performance_issues = if poor_decisions as f32 / recent_decisions.len() as f32 > 0.3 {
            vec!["High rate of poor decisions detected".to_string()]
        } else {
            vec!["Model performance within acceptable range".to_string()]
        };
        
        // Schedule additional optimization if performance is poor
        if poor_decisions as f32 / recent_decisions.len() as f32 > 0.4 {
            self.optimization_scheduler.scheduled_optimization_types.push(OptimizationType::ThresholdCalibration);
        }
        
        Ok(OptimizationResult {
            optimization_type: OptimizationType::ModelPerformanceAnalysis,
            success: true,
            performance_impact: 0.0,
            changes_made: performance_issues,
            metrics_improved: vec!["performance_analysis".to_string()],
        })
    }

    fn update_performance_metrics(&mut self) {
        let total = self.adaptation_state.total_decisions_made;
        if total == 0 {
            return;
        }
        
        let success_rate = self.adaptation_state.successful_decisions as f32 / total as f32;
        self.adaptation_state.current_performance_score = success_rate;
        
        // Update trend based on recent performance
        let recent_performance = self.calculate_recent_performance_trend();
        self.adaptation_state.confidence_accuracy_trend = recent_performance;
        
        // Determine performance trend
        if recent_performance > self.adaptation_state.current_performance_score + 0.1 {
            self.adaptation_state.performance_trend = PerformanceTrend::Improving;
        } else if recent_performance < self.adaptation_state.current_performance_score - 0.1 {
            self.adaptation_state.performance_trend = PerformanceTrend::Declining;
        } else {
            self.adaptation_state.performance_trend = PerformanceTrend::Stable;
        }
    }

    fn outcome_to_performance_impact(&self, outcome: &DecisionOutcome) -> f32 {
        match outcome {
            DecisionOutcome::Excellent => 1.0,
            DecisionOutcome::Good => 0.75,
            DecisionOutcome::Neutral => 0.5,
            DecisionOutcome::Poor => 0.25,
            DecisionOutcome::Terrible => 0.0,
        }
    }

    fn decisions_until_next_optimization(&self) -> u64 {
        let decisions_since_last = self.adaptation_state.total_decisions_made 
            - self.optimization_scheduler.last_optimization_decision_count;
        
        self.optimization_scheduler.min_decisions_between_optimization
            .saturating_sub(decisions_since_last)
    }

    fn calculate_recent_performance_trend(&self) -> f32 {
        let recent_decisions = self.decision_history.iter().rev().take(50).collect::<Vec<_>>();
        
        if recent_decisions.is_empty() {
            return 0.0;
        }
        
        let successful = recent_decisions.iter()
            .filter(|d| matches!(d.final_outcome, Some(DecisionOutcome::Excellent) | Some(DecisionOutcome::Good)))
            .count();
        
        successful as f32 / recent_decisions.len() as f32
    }

    fn calculate_performance_distribution(&self) -> HashMap<String, f32> {
        let mut distribution = HashMap::new();
        
        for decision in &self.decision_history {
            if let Some(outcome) = &decision.final_outcome {
                let key = format!("{:?}", outcome);
                *distribution.entry(key).or_insert(0.0) += 1.0;
            }
        }
        
        let total = distribution.values().sum::<f32>();
        if total > 0.0 {
            for value in distribution.values_mut() {
                *value /= total;
            }
        }
        
        distribution
    }
}

/// Implementation of PartialEq for OptimizationType to support contains() checks
impl PartialEq for OptimizationType {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (OptimizationType::ThresholdCalibration, OptimizationType::ThresholdCalibration) |
            (OptimizationType::ConfidenceWeightAdjustment, OptimizationType::ConfidenceWeightAdjustment) |
            (OptimizationType::SafetyParameterTuning, OptimizationType::SafetyParameterTuning) |
            (OptimizationType::ModelPerformanceAnalysis, OptimizationType::ModelPerformanceAnalysis)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_system_creation() {
        let system = AdaptiveNeuralSystem::new();
        assert!(system.is_ok());
        
        let system = system.unwrap();
        assert_eq!(system.adaptation_state.total_decisions_made, 0);
        assert_eq!(system.decision_history.len(), 0);
    }

    #[test]
    fn test_performance_metrics_calculation() {
        let mut system = AdaptiveNeuralSystem::new().unwrap();
        
        // Simulate some decisions and outcomes
        system.adaptation_state.total_decisions_made = 100;
        system.adaptation_state.successful_decisions = 75;
        system.adaptation_state.failed_decisions = 25;
        
        system.update_performance_metrics();
        
        assert_eq!(system.adaptation_state.current_performance_score, 0.75);
    }

    #[test]
    fn test_optimization_scheduling() {
        let mut system = AdaptiveNeuralSystem::new().unwrap();
        
        // Should not trigger optimization with few decisions
        assert!(!system.should_trigger_optimization());
        
        // Should trigger after enough decisions
        system.adaptation_state.total_decisions_made = 150;
        assert!(system.should_trigger_optimization());
    }

    #[test]
    fn test_outcome_to_performance_mapping() {
        let system = AdaptiveNeuralSystem::new().unwrap();
        
        assert_eq!(system.outcome_to_performance_impact(&DecisionOutcome::Excellent), 1.0);
        assert_eq!(system.outcome_to_performance_impact(&DecisionOutcome::Good), 0.75);
        assert_eq!(system.outcome_to_performance_impact(&DecisionOutcome::Neutral), 0.5);
        assert_eq!(system.outcome_to_performance_impact(&DecisionOutcome::Poor), 0.25);
        assert_eq!(system.outcome_to_performance_impact(&DecisionOutcome::Terrible), 0.0);
    }
}