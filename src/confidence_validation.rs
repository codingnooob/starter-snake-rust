// Confidence Validation Framework
// Correlates confidence scores with actual move quality outcomes
// Provides empirical feedback to improve confidence thresholds

use crate::neural_confidence_integration::{NeuralDecisionRecord, DecisionOutcome, EnhancedNeuralEvaluator};
use crate::unified_confidence::{ConfidenceLevel, ConfidenceConfig};
use crate::{Board, Battlesnake, Coord};
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use log::{info, warn, debug};

/// Confidence validation system that learns from actual game outcomes
pub struct ConfidenceValidator {
    validation_data: Vec<ValidationRecord>,
    outcome_correlations: HashMap<String, ConfidenceCorrelation>,
    threshold_performance: ThresholdPerformanceTracker,
    calibration_metrics: CalibrationMetrics,
}

/// Individual validation record linking confidence to actual outcome
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRecord {
    pub decision_id: String,
    pub confidence_score: f32,
    pub confidence_level: ConfidenceLevel,
    pub predicted_move_quality: f32, // From neural network
    pub actual_move_outcome: DecisionOutcome,
    pub outcome_score: f32, // Quantified outcome quality (0.0 = terrible, 1.0 = excellent)
    pub game_context: GameValidationContext,
    pub validation_timestamp: String,
}

/// Context information for validation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameValidationContext {
    pub turn_number: i32,
    pub health_before: i32,
    pub health_after: Option<i32>,
    pub length_before: usize,
    pub length_after: Option<usize>,
    pub distance_to_food_before: f32,
    pub distance_to_food_after: Option<f32>,
    pub danger_level: f32, // 0.0 = safe, 1.0 = very dangerous
    pub alternative_moves_available: u32,
}

/// Correlation statistics between confidence and outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceCorrelation {
    pub total_samples: u32,
    pub confidence_outcome_correlation: f32, // Pearson correlation coefficient
    pub high_confidence_accuracy: f32, // % of high confidence predictions that were good
    pub medium_confidence_accuracy: f32,
    pub low_confidence_accuracy: f32,
    pub false_positive_rate: f32, // High confidence but poor outcome
    pub false_negative_rate: f32, // Low confidence but good outcome
    pub calibration_score: f32, // How well confidence matches actual probability of success
}

/// Tracks performance of different confidence thresholds
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThresholdPerformanceTracker {
    pub threshold_tests: HashMap<String, ThresholdTestResult>,
    pub optimal_thresholds: HashMap<String, f32>,
    pub threshold_stability: HashMap<String, f32>, // How stable the optimal threshold is
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdTestResult {
    pub threshold_value: f32,
    pub true_positives: u32,  // High confidence, good outcome
    pub false_positives: u32, // High confidence, poor outcome
    pub true_negatives: u32,  // Low confidence, poor outcome (correctly avoided)
    pub false_negatives: u32, // Low confidence, good outcome (missed opportunity)
    pub precision: f32,       // TP / (TP + FP)
    pub recall: f32,          // TP / (TP + FN)
    pub f1_score: f32,        // 2 * (precision * recall) / (precision + recall)
    pub accuracy: f32,        // (TP + TN) / (TP + FP + TN + FN)
}

/// Overall calibration metrics for the confidence system
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    pub expected_calibration_error: f32, // ECE - measures calibration quality
    pub maximum_calibration_error: f32,  // MCE - worst calibration bin
    pub brier_score: f32,                // Measures both calibration and discrimination
    pub reliability_score: f32,          // How reliable confidence scores are
    pub discrimination_score: f32,       // How well confidence discriminates good/poor outcomes
}

impl ConfidenceValidator {
    pub fn new() -> Self {
        Self {
            validation_data: Vec::new(),
            outcome_correlations: HashMap::new(),
            threshold_performance: ThresholdPerformanceTracker::default(),
            calibration_metrics: CalibrationMetrics::default(),
        }
    }

    /// Record a decision and its eventual outcome for validation
    pub fn record_decision_outcome(
        &mut self,
        decision_record: &NeuralDecisionRecord,
        actual_outcome: DecisionOutcome,
        game_context_after: Option<&GameValidationContext>,
    ) -> Result<()> {
        // Convert categorical outcome to numeric score
        let outcome_score = self.outcome_to_score(&actual_outcome);
        
        // Extract game context from the decision record
        let context_before = &decision_record.game_context;
        let validation_context = self.create_validation_context(
            context_before,
            game_context_after,
            &decision_record.chosen_move,
        )?;

        let validation_record = ValidationRecord {
            decision_id: format!("{}_{}", decision_record.timestamp, decision_record.board_hash),
            confidence_score: decision_record.confidence.unified_confidence,
            confidence_level: decision_record.confidence.confidence_level.clone(),
            predicted_move_quality: decision_record.confidence.unified_confidence, // Use confidence as quality prediction
            actual_move_outcome: actual_outcome.clone(),
            outcome_score,
            game_context: validation_context,
            validation_timestamp: chrono::Utc::now().to_rfc3339(),
        };

        self.validation_data.push(validation_record);
        
        // Limit validation data size to prevent memory growth
        if self.validation_data.len() > 10000 {
            self.validation_data.drain(0..5000); // Keep most recent 5000
        }

        debug!("Recorded validation outcome: confidence {:.3}, outcome {:?}, score {:.3}",
               decision_record.confidence.unified_confidence, actual_outcome, outcome_score);

        Ok(())
    }

    /// Analyze all recorded data and update correlations
    pub fn analyze_correlations(&mut self) -> Result<()> {
        if self.validation_data.len() < 10 {
            warn!("Insufficient data for correlation analysis: {} samples", self.validation_data.len());
            return Ok(());
        }

        info!("Analyzing confidence correlations with {} samples", self.validation_data.len());

        // Calculate overall correlation
        let overall_correlation = self.calculate_correlation()?;
        self.outcome_correlations.insert("overall".to_string(), overall_correlation);

        // Calculate correlations by confidence level
        for level in [ConfidenceLevel::High, ConfidenceLevel::Medium, ConfidenceLevel::Low] {
            let level_data: Vec<_> = self.validation_data.iter()
                .filter(|record| matches!(&record.confidence_level, level))
                .collect();
            
            if level_data.len() >= 5 {
                let level_correlation = self.calculate_correlation_for_subset(&level_data)?;
                let level_name = format!("{:?}", level).to_lowercase();
                self.outcome_correlations.insert(level_name, level_correlation);
            }
        }

        // Update calibration metrics
        self.update_calibration_metrics()?;

        info!("Correlation analysis complete");
        Ok(())
    }

    /// Test different threshold values to find optimal settings
    pub fn optimize_thresholds(&mut self, threshold_candidates: &[f32]) -> Result<HashMap<String, f32>> {
        info!("Testing {} threshold candidates for optimization", threshold_candidates.len());
        
        let mut optimal_thresholds = HashMap::new();
        
        for &threshold in threshold_candidates {
            let test_result = self.test_threshold(threshold)?;
            let threshold_key = format!("threshold_{:.2}", threshold);
            self.threshold_performance.threshold_tests.insert(threshold_key.clone(), test_result.clone());
            
            // Track best F1 score for this threshold type
            if let Some(current_best) = self.threshold_performance.optimal_thresholds.get("f1_optimal") {
                if test_result.f1_score > self.threshold_performance.threshold_tests
                    .get(&format!("threshold_{:.2}", current_best))
                    .map(|r| r.f1_score)
                    .unwrap_or(0.0)
                {
                    optimal_thresholds.insert("f1_optimal".to_string(), threshold);
                }
            } else {
                optimal_thresholds.insert("f1_optimal".to_string(), threshold);
            }
            
            // Track best accuracy for this threshold type  
            if let Some(current_best) = self.threshold_performance.optimal_thresholds.get("accuracy_optimal") {
                if test_result.accuracy > self.threshold_performance.threshold_tests
                    .get(&format!("threshold_{:.2}", current_best))
                    .map(|r| r.accuracy)
                    .unwrap_or(0.0)
                {
                    optimal_thresholds.insert("accuracy_optimal".to_string(), threshold);
                }
            } else {
                optimal_thresholds.insert("accuracy_optimal".to_string(), threshold);
            }
        }

        self.threshold_performance.optimal_thresholds.extend(optimal_thresholds.clone());
        
        info!("Threshold optimization complete. Optimal thresholds: {:?}", optimal_thresholds);
        Ok(optimal_thresholds)
    }

    /// Generate updated confidence configuration based on validation results
    pub fn generate_optimized_config(&self) -> Result<ConfidenceConfig> {
        let mut config = ConfidenceConfig::default();
        
        // Update thresholds based on empirical results
        if let Some(&optimal_threshold) = self.threshold_performance.optimal_thresholds.get("f1_optimal") {
            // Apply optimal threshold to move prediction
            config.move_prediction_thresholds.high_confidence_max_prob_threshold = optimal_threshold;
            config.move_prediction_thresholds.medium_confidence_max_prob_threshold = optimal_threshold * 0.8;
            
            // Scale entropy thresholds based on correlation strength
            if let Some(correlation) = self.outcome_correlations.get("overall") {
                let correlation_strength = correlation.confidence_outcome_correlation.abs();
                
                // Stronger correlation allows for more aggressive thresholds
                config.move_prediction_thresholds.high_confidence_entropy_threshold = 0.7 - (correlation_strength * 0.2);
                config.move_prediction_thresholds.medium_confidence_entropy_threshold = 0.9 - (correlation_strength * 0.1);
            }
        }

        // Update deviation thresholds for position/outcome models
        if let Some(correlation) = self.outcome_correlations.get("overall") {
            let calibration_factor = correlation.calibration_score;
            
            // Better calibration allows for more confident thresholds
            config.position_evaluation_thresholds.high_confidence_deviation = 0.15 * (1.0 - calibration_factor * 0.3);
            config.position_evaluation_thresholds.medium_confidence_deviation = 0.08 * (1.0 - calibration_factor * 0.2);
            
            config.game_outcome_thresholds.high_confidence_deviation = 0.15 * (1.0 - calibration_factor * 0.3);
            config.game_outcome_thresholds.medium_confidence_deviation = 0.08 * (1.0 - calibration_factor * 0.2);
        }

        // Adjust weights based on which confidence metrics perform best
        if let Some(high_conf_correlation) = self.outcome_correlations.get("high") {
            if high_conf_correlation.confidence_outcome_correlation > 0.7 {
                // High confidence is very reliable - increase its weight
                config.entropy_weight = 0.5;
                config.max_probability_weight = 0.4;
                config.consistency_weight = 0.1;
            }
        }

        info!("Generated optimized configuration based on {} validation samples", self.validation_data.len());
        Ok(config)
    }

    /// Get outcome correlations (public accessor)
    pub fn get_outcome_correlations(&self) -> &HashMap<String, ConfidenceCorrelation> {
        &self.outcome_correlations
    }

    /// Export validation analysis for review and debugging
    pub fn export_validation_report(&self) -> Result<String> {
        let report = serde_json::json!({
            "validation_summary": {
                "total_samples": self.validation_data.len(),
                "analysis_timestamp": chrono::Utc::now().to_rfc3339(),
                "data_time_range": {
                    "earliest": self.validation_data.first().map(|r| &r.validation_timestamp),
                    "latest": self.validation_data.last().map(|r| &r.validation_timestamp)
                }
            },
            "correlations": &self.outcome_correlations,
            "calibration_metrics": &self.calibration_metrics,
            "threshold_performance": &self.threshold_performance,
            "sample_records": self.validation_data.iter().take(10).collect::<Vec<_>>(), // Include 10 sample records
            "recommendations": self.generate_recommendations()
        });

        serde_json::to_string_pretty(&report)
            .map_err(|e| anyhow!("Failed to serialize validation report: {}", e))
    }

    /// Generate actionable recommendations based on analysis
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check overall correlation strength
        if let Some(overall) = self.outcome_correlations.get("overall") {
            if overall.confidence_outcome_correlation < 0.3 {
                recommendations.push("LOW CORRELATION WARNING: Confidence scores show weak correlation with outcomes. Consider retraining neural networks or adjusting confidence calculation.".to_string());
            } else if overall.confidence_outcome_correlation > 0.7 {
                recommendations.push("STRONG CORRELATION: Confidence system is well-calibrated. Consider increasing neural network utilization.".to_string());
            }

            if overall.false_positive_rate > 0.3 {
                recommendations.push("HIGH FALSE POSITIVE RATE: Too many high-confidence predictions result in poor outcomes. Consider raising confidence thresholds.".to_string());
            }

            if overall.false_negative_rate > 0.3 {
                recommendations.push("HIGH FALSE NEGATIVE RATE: Too many good moves are marked as low confidence. Consider lowering confidence thresholds.".to_string());
            }
        }

        // Check calibration
        if self.calibration_metrics.expected_calibration_error > 0.1 {
            recommendations.push("POOR CALIBRATION: Confidence scores do not match actual success probabilities. Recalibrate confidence thresholds.".to_string());
        }

        // Check sample size
        if self.validation_data.len() < 100 {
            recommendations.push("INSUFFICIENT DATA: Need more validation samples for reliable analysis. Current samples may not be representative.".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("System appears well-calibrated based on current data. Monitor performance and continue collecting validation data.".to_string());
        }

        recommendations
    }

    // Private helper methods

    fn outcome_to_score(&self, outcome: &DecisionOutcome) -> f32 {
        match outcome {
            DecisionOutcome::Excellent => 1.0,
            DecisionOutcome::Good => 0.75,
            DecisionOutcome::Neutral => 0.5,
            DecisionOutcome::Poor => 0.25,
            DecisionOutcome::Terrible => 0.0,
        }
    }

    fn create_validation_context(
        &self,
        context_before: &crate::neural_confidence_integration::GameContextSnapshot,
        context_after: Option<&GameValidationContext>,
        chosen_move: &str,
    ) -> Result<GameValidationContext> {
        Ok(GameValidationContext {
            turn_number: context_before.turn_number,
            health_before: context_before.our_health,
            health_after: context_after.and_then(|c| c.health_after),
            length_before: context_before.our_length,
            length_after: context_after.and_then(|c| c.length_after),
            distance_to_food_before: 5.0, // Placeholder - would calculate from board state
            distance_to_food_after: context_after.and_then(|c| c.distance_to_food_after),
            danger_level: 0.5, // Placeholder - would calculate from board analysis
            alternative_moves_available: 3, // Placeholder - would count safe moves
        })
    }

    fn calculate_correlation(&self) -> Result<ConfidenceCorrelation> {
        self.calculate_correlation_for_subset(&self.validation_data.iter().collect::<Vec<_>>())
    }

    fn calculate_correlation_for_subset(&self, data: &[&ValidationRecord]) -> Result<ConfidenceCorrelation> {
        if data.is_empty() {
            return Err(anyhow!("No data for correlation calculation"));
        }

        let n = data.len() as f32;
        
        // Calculate Pearson correlation coefficient
        let confidence_mean = data.iter().map(|r| r.confidence_score).sum::<f32>() / n;
        let outcome_mean = data.iter().map(|r| r.outcome_score).sum::<f32>() / n;
        
        let numerator: f32 = data.iter()
            .map(|r| (r.confidence_score - confidence_mean) * (r.outcome_score - outcome_mean))
            .sum();
            
        let conf_variance: f32 = data.iter()
            .map(|r| (r.confidence_score - confidence_mean).powi(2))
            .sum();
            
        let outcome_variance: f32 = data.iter()
            .map(|r| (r.outcome_score - outcome_mean).powi(2))
            .sum();
        
        let correlation = if conf_variance > 0.0 && outcome_variance > 0.0 {
            numerator / (conf_variance * outcome_variance).sqrt()
        } else {
            0.0
        };

        // Calculate accuracy by confidence level
        let high_conf_data: Vec<_> = data.iter().filter(|r| matches!(r.confidence_level, ConfidenceLevel::High)).collect();
        let medium_conf_data: Vec<_> = data.iter().filter(|r| matches!(r.confidence_level, ConfidenceLevel::Medium)).collect();
        let low_conf_data: Vec<_> = data.iter().filter(|r| matches!(r.confidence_level, ConfidenceLevel::Low)).collect();

        let high_accuracy = if !high_conf_data.is_empty() {
            high_conf_data.iter().map(|r| if r.outcome_score > 0.5 { 1.0 } else { 0.0 }).sum::<f32>() / high_conf_data.len() as f32
        } else { 0.0 };

        let medium_accuracy = if !medium_conf_data.is_empty() {
            medium_conf_data.iter().map(|r| if r.outcome_score > 0.5 { 1.0 } else { 0.0 }).sum::<f32>() / medium_conf_data.len() as f32
        } else { 0.0 };

        let low_accuracy = if !low_conf_data.is_empty() {
            low_conf_data.iter().map(|r| if r.outcome_score > 0.5 { 1.0 } else { 0.0 }).sum::<f32>() / low_conf_data.len() as f32
        } else { 0.0 };

        // Calculate false positive/negative rates
        let high_conf_poor_outcomes = high_conf_data.iter().filter(|r| r.outcome_score <= 0.5).count() as f32;
        let false_positive_rate = if !high_conf_data.is_empty() {
            high_conf_poor_outcomes / high_conf_data.len() as f32
        } else { 0.0 };

        let low_conf_good_outcomes = low_conf_data.iter().filter(|r| r.outcome_score > 0.5).count() as f32;
        let false_negative_rate = if !low_conf_data.is_empty() {
            low_conf_good_outcomes / low_conf_data.len() as f32
        } else { 0.0 };

        // Simple calibration score (perfect calibration = 1.0)
        let calibration_score = 1.0 - (correlation - high_accuracy).abs();

        Ok(ConfidenceCorrelation {
            total_samples: n as u32,
            confidence_outcome_correlation: correlation,
            high_confidence_accuracy: high_accuracy,
            medium_confidence_accuracy: medium_accuracy,
            low_confidence_accuracy: low_accuracy,
            false_positive_rate,
            false_negative_rate,
            calibration_score: calibration_score.max(0.0),
        })
    }

    fn test_threshold(&self, threshold: f32) -> Result<ThresholdTestResult> {
        let mut tp = 0u32; // True positives: high confidence, good outcome
        let mut fp = 0u32; // False positives: high confidence, poor outcome  
        let mut tn = 0u32; // True negatives: low confidence, poor outcome
        let mut fn_ = 0u32; // False negatives: low confidence, good outcome

        for record in &self.validation_data {
            let is_high_confidence = record.confidence_score >= threshold;
            let is_good_outcome = record.outcome_score > 0.5;

            match (is_high_confidence, is_good_outcome) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, false) => tn += 1,
                (false, true) => fn_ += 1,
            }
        }

        let total = tp + fp + tn + fn_;
        
        let precision = if tp + fp > 0 { tp as f32 / (tp + fp) as f32 } else { 0.0 };
        let recall = if tp + fn_ > 0 { tp as f32 / (tp + fn_) as f32 } else { 0.0 };
        let f1_score = if precision + recall > 0.0 { 
            2.0 * (precision * recall) / (precision + recall) 
        } else { 0.0 };
        let accuracy = if total > 0 { (tp + tn) as f32 / total as f32 } else { 0.0 };

        Ok(ThresholdTestResult {
            threshold_value: threshold,
            true_positives: tp,
            false_positives: fp,
            true_negatives: tn,
            false_negatives: fn_,
            precision,
            recall,
            f1_score,
            accuracy,
        })
    }

    fn update_calibration_metrics(&mut self) -> Result<()> {
        // Calculate Expected Calibration Error (ECE) - simplified version
        let bin_count = 10;
        let mut bin_accuracies = vec![0.0; bin_count];
        let mut bin_confidences = vec![0.0; bin_count];
        let mut bin_counts = vec![0; bin_count];

        for record in &self.validation_data {
            let bin_idx = ((record.confidence_score * bin_count as f32).floor() as usize).min(bin_count - 1);
            bin_confidences[bin_idx] += record.confidence_score;
            bin_accuracies[bin_idx] += if record.outcome_score > 0.5 { 1.0 } else { 0.0 };
            bin_counts[bin_idx] += 1;
        }

        let mut ece = 0.0f32;
        let mut mce = 0.0f32;
        let total_samples = self.validation_data.len() as f32;

        for i in 0..bin_count {
            if bin_counts[i] > 0 {
                let avg_confidence = bin_confidences[i] / bin_counts[i] as f32;
                let avg_accuracy = bin_accuracies[i] / bin_counts[i] as f32;
                let calibration_error = (avg_confidence - avg_accuracy).abs();
                
                ece += (bin_counts[i] as f32 / total_samples) * calibration_error;
                mce = mce.max(calibration_error);
            }
        }

        // Calculate Brier score (simplified)
        let brier_score = self.validation_data.iter()
            .map(|r| (r.confidence_score - if r.outcome_score > 0.5 { 1.0 } else { 0.0 }).powi(2))
            .sum::<f32>() / self.validation_data.len() as f32;

        self.calibration_metrics.expected_calibration_error = ece;
        self.calibration_metrics.maximum_calibration_error = mce;
        self.calibration_metrics.brier_score = brier_score;
        self.calibration_metrics.reliability_score = 1.0 - ece; // Simple reliability measure
        self.calibration_metrics.discrimination_score = if let Some(overall) = self.outcome_correlations.get("overall") {
            overall.confidence_outcome_correlation.abs()
        } else { 0.0 };

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outcome_to_score_conversion() {
        let validator = ConfidenceValidator::new();
        
        assert_eq!(validator.outcome_to_score(&DecisionOutcome::Excellent), 1.0);
        assert_eq!(validator.outcome_to_score(&DecisionOutcome::Good), 0.75);
        assert_eq!(validator.outcome_to_score(&DecisionOutcome::Neutral), 0.5);
        assert_eq!(validator.outcome_to_score(&DecisionOutcome::Poor), 0.25);
        assert_eq!(validator.outcome_to_score(&DecisionOutcome::Terrible), 0.0);
    }

    #[test]
    fn test_threshold_optimization() {
        let mut validator = ConfidenceValidator::new();
        
        // Add some test validation data
        for i in 0..100 {
            let confidence = (i as f32) / 100.0;
            let outcome = if confidence > 0.6 { 
                DecisionOutcome::Good 
            } else { 
                DecisionOutcome::Poor 
            };
            
            // Create minimal validation record
            let record = ValidationRecord {
                decision_id: format!("test_{}", i),
                confidence_score: confidence,
                confidence_level: if confidence > 0.7 { ConfidenceLevel::High } else { ConfidenceLevel::Low },
                predicted_move_quality: confidence,
                actual_move_outcome: outcome.clone(),
                outcome_score: validator.outcome_to_score(&outcome),
                game_context: GameValidationContext {
                    turn_number: i,
                    health_before: 100,
                    health_after: None,
                    length_before: 3,
                    length_after: None,
                    distance_to_food_before: 5.0,
                    distance_to_food_after: None,
                    danger_level: 0.5,
                    alternative_moves_available: 3,
                },
                validation_timestamp: chrono::Utc::now().to_rfc3339(),
            };
            
            validator.validation_data.push(record);
        }
        
        let thresholds = vec![0.3, 0.5, 0.7, 0.9];
        let optimal = validator.optimize_thresholds(&thresholds).unwrap();
        
        assert!(!optimal.is_empty());
        assert!(optimal.contains_key("f1_optimal") || optimal.contains_key("accuracy_optimal"));
    }
}