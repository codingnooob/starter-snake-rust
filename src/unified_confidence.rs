// Unified Neural Network Confidence System
// Addresses the root cause: current models produce near-random outputs
// Provides proper confidence measurement for current and future models

use ndarray::Array2;
use anyhow::{Result, anyhow};
use log::{info, warn, debug};
use serde::{Serialize, Deserialize};

/// Unified confidence calculation for all neural network models
#[derive(Debug, Clone)]
pub struct UnifiedConfidenceCalculator {
    config: ConfidenceConfig,
}

/// Configuration for confidence thresholds and calculation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceConfig {
    // Entropy-based confidence (for move predictions)
    pub entropy_weight: f32,
    pub max_probability_weight: f32,
    pub consistency_weight: f32,
    
    // Deviation-based confidence (for position/outcome)
    pub deviation_weight: f32,
    pub extreme_value_bonus: f32,
    
    // Empirically-based thresholds (calibrated to actual model performance)
    pub move_prediction_thresholds: PredictionThresholds,
    pub position_evaluation_thresholds: ValueThresholds,
    pub game_outcome_thresholds: ValueThresholds,
    
    // Safety and fallback settings
    pub safety_override_enabled: bool,
    pub minimum_confidence_for_override: f32,
    pub fallback_when_models_uncertain: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionThresholds {
    pub high_confidence_entropy_threshold: f32,      // Normalized entropy < this = high conf
    pub medium_confidence_entropy_threshold: f32,    // Entropy between thresholds = medium
    pub high_confidence_max_prob_threshold: f32,     // Max probability > this = high conf
    pub medium_confidence_max_prob_threshold: f32,   // Max prob between thresholds = medium
    pub random_baseline: f32,                        // Baseline random performance (0.25 for 4 moves)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueThresholds {
    pub high_confidence_deviation: f32,              // Distance from neutral > this = high conf
    pub medium_confidence_deviation: f32,            // Deviation between thresholds = medium
    pub neutral_point: f32,                          // Neutral value (0.5 for [0,1] outputs)
    pub extreme_value_threshold: f32,                // Values beyond this get confidence boost
}

/// Comprehensive confidence metrics for a neural network prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfidence {
    // Raw confidence scores
    pub entropy_confidence: f32,        // Based on prediction entropy (higher = more confident)
    pub probability_confidence: f32,    // Based on max probability
    pub deviation_confidence: f32,      // Based on distance from neutral
    pub consistency_confidence: f32,    // Agreement between different metrics
    
    // Combined confidence score (0.0 to 1.0)
    pub unified_confidence: f32,
    
    // Confidence level classification
    pub confidence_level: ConfidenceLevel,
    
    // Decision recommendation
    pub should_use_neural_network: bool,
    pub should_apply_safety_check: bool,
    pub recommendation_reason: String,
    
    // Debug information
    pub raw_metrics: RawConfidenceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    High,    // Use neural network with minimal safety checks
    Medium,  // Use neural network with full safety validation
    Low,     // Fallback to search algorithms
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawConfidenceMetrics {
    pub entropy: Option<f32>,
    pub max_probability: Option<f32>,
    pub deviation_from_neutral: Option<f32>,
    pub prediction_spread: Option<f32>,
    pub model_uncertainty: f32,
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self {
            entropy_weight: 0.4,
            max_probability_weight: 0.4,
            consistency_weight: 0.2,
            deviation_weight: 0.7,
            extreme_value_bonus: 0.3,
            
            // Calibrated to actual model performance from analysis
            move_prediction_thresholds: PredictionThresholds {
                // Current models have mean entropy ~0.999, max prob ~0.265
                // Setting realistic thresholds for near-random models
                high_confidence_entropy_threshold: 0.7,    // Normalized entropy < 0.7
                medium_confidence_entropy_threshold: 0.9,  // Between 0.7-0.9
                high_confidence_max_prob_threshold: 0.35,  // Max prob > 0.35 (well above 0.265 mean)
                medium_confidence_max_prob_threshold: 0.28, // Between 0.28-0.35
                random_baseline: 0.25,
            },
            
            position_evaluation_thresholds: ValueThresholds {
                // Current models have mean deviation ~0.040 from neutral
                high_confidence_deviation: 0.15,          // Deviation > 0.15 (well above 0.040 mean)
                medium_confidence_deviation: 0.08,        // Between 0.08-0.15  
                neutral_point: 0.5,
                extreme_value_threshold: 0.3,             // Values < 0.2 or > 0.8
            },
            
            game_outcome_thresholds: ValueThresholds {
                // Current models have mean deviation ~0.046 from neutral
                high_confidence_deviation: 0.15,          // Similar to position evaluation
                medium_confidence_deviation: 0.08,
                neutral_point: 0.5,
                extreme_value_threshold: 0.3,
            },
            
            safety_override_enabled: true,
            minimum_confidence_for_override: 0.8,         // Very high bar for bypassing safety
            fallback_when_models_uncertain: true,
        }
    }
}

impl UnifiedConfidenceCalculator {
    pub fn new() -> Self {
        Self {
            config: ConfidenceConfig::default(),
        }
    }
    
    pub fn with_config(config: ConfidenceConfig) -> Self {
        Self { config }
    }
    
    /// Calculate confidence for move prediction outputs (4-dimensional probabilities)
    pub fn calculate_move_confidence(
        &self,
        move_probabilities: &Array2<f32>,
    ) -> Result<NeuralConfidence> {
        if move_probabilities.dim().1 != 4 {
            return Err(anyhow!("Expected 4 move probabilities, got {}", move_probabilities.dim().1));
        }
        
        let probs = move_probabilities.row(0);
        let probs_slice = probs.as_slice().unwrap();
        
        debug!("Calculating move confidence for probabilities: {:?}", probs_slice);
        
        // 1. Calculate entropy-based confidence
        let entropy = self.calculate_entropy(probs_slice);
        let max_entropy = (probs_slice.len() as f32).ln();
        let normalized_entropy = entropy / max_entropy;
        let entropy_confidence = 1.0 - normalized_entropy; // Higher entropy = lower confidence
        
        // 2. Calculate max probability confidence
        let max_prob = probs_slice.iter().fold(0.0f32, |a: f32, &b| a.max(b));
        let prob_confidence = if max_prob > self.config.move_prediction_thresholds.random_baseline {
            (max_prob - self.config.move_prediction_thresholds.random_baseline) / 
            (1.0 - self.config.move_prediction_thresholds.random_baseline)
        } else {
            0.0
        };
        
        // 3. Calculate prediction spread (how different from uniform)
        let uniform_prob = 1.0 / probs_slice.len() as f32;
        let spread = probs_slice.iter()
            .map(|&p| (p - uniform_prob).abs())
            .sum::<f32>() / probs_slice.len() as f32;
        let spread_confidence = spread * 4.0; // Normalize to [0,1] range
        
        // 4. Combine confidence scores
        let unified_confidence = (
            entropy_confidence * self.config.entropy_weight +
            prob_confidence * self.config.max_probability_weight +
            spread_confidence * self.config.consistency_weight
        ).min(1.0);
        
        // 5. Determine confidence level
        let confidence_level = self.classify_move_confidence(normalized_entropy, max_prob);
        
        // 6. Generate recommendations
        let (should_use_neural, should_check_safety, reason) = 
            self.generate_move_recommendations(unified_confidence, &confidence_level, max_prob);
        
        let raw_metrics = RawConfidenceMetrics {
            entropy: Some(normalized_entropy),
            max_probability: Some(max_prob),
            deviation_from_neutral: None,
            prediction_spread: Some(spread),
            model_uncertainty: normalized_entropy,
        };
        
        Ok(NeuralConfidence {
            entropy_confidence,
            probability_confidence: prob_confidence,
            deviation_confidence: 0.0, // Not applicable for move predictions
            consistency_confidence: spread_confidence,
            unified_confidence,
            confidence_level,
            should_use_neural_network: should_use_neural,
            should_apply_safety_check: should_check_safety,
            recommendation_reason: reason,
            raw_metrics,
        })
    }
    
    /// Calculate confidence for single-value outputs (position evaluation, game outcome)
    pub fn calculate_value_confidence(
        &self,
        value: f32,
        model_type: &str, // "position" or "outcome"
    ) -> Result<NeuralConfidence> {
        let thresholds = match model_type {
            "position" => &self.config.position_evaluation_thresholds,
            "outcome" => &self.config.game_outcome_thresholds,
            _ => return Err(anyhow!("Unknown model type: {}", model_type)),
        };
        
        debug!("Calculating {} confidence for value: {}", model_type, value);
        
        // 1. Calculate deviation from neutral
        let deviation = (value - thresholds.neutral_point).abs();
        let max_possible_deviation = 0.5; // For [0,1] range
        let deviation_confidence = (deviation / max_possible_deviation).min(1.0);
        
        // 2. Check for extreme values (bonus confidence for strong predictions)
        let extreme_bonus = if deviation > thresholds.extreme_value_threshold {
            self.config.extreme_value_bonus
        } else {
            0.0
        };
        
        // 3. Model uncertainty (how close to neutral = how uncertain)
        let model_uncertainty = 1.0 - deviation_confidence;
        
        // 4. Combined confidence
        let unified_confidence = (
            deviation_confidence * self.config.deviation_weight + extreme_bonus
        ).min(1.0);
        
        // 5. Classify confidence level
        let confidence_level = self.classify_value_confidence(deviation, thresholds);
        
        // 6. Generate recommendations
        let (should_use_neural, should_check_safety, reason) = 
            self.generate_value_recommendations(unified_confidence, &confidence_level, deviation, model_type);
        
        let raw_metrics = RawConfidenceMetrics {
            entropy: None,
            max_probability: None,
            deviation_from_neutral: Some(deviation),
            prediction_spread: None,
            model_uncertainty,
        };
        
        Ok(NeuralConfidence {
            entropy_confidence: 0.0, // Not applicable for single values
            probability_confidence: 0.0, // Not applicable
            deviation_confidence,
            consistency_confidence: 0.0, // Not applicable
            unified_confidence,
            confidence_level,
            should_use_neural_network: should_use_neural,
            should_apply_safety_check: should_check_safety,
            recommendation_reason: reason,
            raw_metrics,
        })
    }
    
    /// Calculate multi-model consistency confidence
    pub fn calculate_consistency_confidence(
        &self,
        position_score: f32,
        move_probs: &[f32],
        outcome_prob: f32,
    ) -> f32 {
        // Check if models agree: positive position should correlate with high win probability
        let pos_normalized = position_score - 0.5; // Center around 0
        let outcome_normalized = outcome_prob - 0.5;
        let correlation = pos_normalized * outcome_normalized;
        
        // Strong disagreement should result in very low consistency
        let position_outcome_consistency = if correlation < -0.1 {
            // Strong negative correlation = highly inconsistent
            0.0
        } else if correlation > 0.1 {
            // Strong positive correlation = highly consistent
            1.0
        } else {
            // Weak correlation = moderate consistency
            0.5
        };
        
        // Check if strongest move aligns with position and outcome
        let max_move_idx = move_probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        let max_move_prob = move_probs[max_move_idx];
        let move_strength = (max_move_prob - 0.25) * 4.0; // Normalize move confidence
        let position_strength = (position_score - 0.5) * 2.0; // Normalize position
        let outcome_strength = (outcome_prob - 0.5) * 2.0; // Normalize outcome
        
        // All three should agree for high consistency
        let signs_agree = (position_strength > 0.0) == (outcome_strength > 0.0) && 
                         (move_strength > 0.1) == (position_strength > 0.0);
        
        let move_consistency = if signs_agree && move_strength.abs() > 0.1 {
            0.8 // All models agree on direction
        } else if !signs_agree {
            0.2 // Models disagree on direction = inconsistent
        } else {
            0.5 // Neutral/mixed signals
        };
        
        let combined_consistency: f32 = (position_outcome_consistency + move_consistency) / 2.0;
        
        // Ensure consistency is properly bounded and handles negative correlations
        combined_consistency.max(0.0).min(1.0)
    }
    
    // Private helper methods
    
    fn calculate_entropy(&self, probabilities: &[f32]) -> f32 {
        // Handle edge case: all zeros should have maximum entropy (minimum confidence)
        let non_zero_count = probabilities.iter().filter(|&&p| p > 0.0).count();
        if non_zero_count == 0 {
            // All zeros = maximum uncertainty = maximum entropy
            return (probabilities.len() as f32).ln();
        }
        
        -probabilities.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f32>()
    }
    
    fn classify_move_confidence(&self, normalized_entropy: f32, max_prob: f32) -> ConfidenceLevel {
        let thresholds = &self.config.move_prediction_thresholds;
        
        if normalized_entropy < thresholds.high_confidence_entropy_threshold ||
           max_prob > thresholds.high_confidence_max_prob_threshold {
            ConfidenceLevel::High
        } else if normalized_entropy < thresholds.medium_confidence_entropy_threshold ||
                  max_prob > thresholds.medium_confidence_max_prob_threshold {
            ConfidenceLevel::Medium
        } else {
            ConfidenceLevel::Low
        }
    }
    
    fn classify_value_confidence(&self, deviation: f32, thresholds: &ValueThresholds) -> ConfidenceLevel {
        if deviation > thresholds.high_confidence_deviation {
            ConfidenceLevel::High
        } else if deviation > thresholds.medium_confidence_deviation {
            ConfidenceLevel::Medium
        } else {
            ConfidenceLevel::Low
        }
    }
    
    fn generate_move_recommendations(
        &self,
        confidence: f32,
        level: &ConfidenceLevel,
        max_prob: f32,
    ) -> (bool, bool, String) {
        match level {
            ConfidenceLevel::High => (
                true,
                true, // Always apply safety checks - NEVER bypass safety
                format!("High confidence ({}), max_prob: {:.3}", confidence, max_prob)
            ),
            ConfidenceLevel::Medium => (
                true,
                true,
                format!("Medium confidence ({}), using neural with safety validation", confidence)
            ),
            ConfidenceLevel::Low => (
                false,
                false,
                format!("Low confidence ({}), falling back to search algorithms", confidence)
            ),
        }
    }
    
    fn generate_value_recommendations(
        &self,
        confidence: f32,
        level: &ConfidenceLevel,
        deviation: f32,
        model_type: &str,
    ) -> (bool, bool, String) {
        match level {
            ConfidenceLevel::High => (
                true,
                true, // Always apply safety checks
                format!("High {} confidence ({}), deviation: {:.3}", model_type, confidence, deviation)
            ),
            ConfidenceLevel::Medium => (
                true,
                true,
                format!("Medium {} confidence ({})", model_type, confidence)
            ),
            ConfidenceLevel::Low => (
                false,
                false,
                format!("Low {} confidence ({}), value too close to neutral", model_type, confidence)
            ),
        }
    }
    
    /// Update configuration with empirical data
    pub fn update_thresholds_from_analysis(&mut self, _analysis_data: &str) -> Result<()> {
        // This would parse the neural_output_analysis.json and adjust thresholds
        // based on actual model performance
        info!("Updating confidence thresholds based on empirical analysis");
        
        // For now, we use the thresholds calibrated to the analysis results
        // In a production system, this would dynamically adjust based on ongoing performance
        
        Ok(())
    }
    
    /// Get current configuration
    pub fn get_config(&self) -> &ConfidenceConfig {
        &self.config
    }
    
    /// Set new configuration
    pub fn set_config(&mut self, config: ConfidenceConfig) {
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_move_confidence_calculation() {
        let calculator = UnifiedConfidenceCalculator::new();
        
        // Test with near-random probabilities (like current models)
        let random_probs = array![[0.26, 0.24, 0.25, 0.25]];
        let confidence = calculator.calculate_move_confidence(&random_probs).unwrap();
        
        assert!(matches!(confidence.confidence_level, ConfidenceLevel::Low));
        assert!(!confidence.should_use_neural_network);
        assert!(confidence.unified_confidence < 0.3);
    }
    
    #[test]
    fn test_confident_move_prediction() {
        let calculator = UnifiedConfidenceCalculator::new();
        
        // Test with confident prediction
        let confident_probs = array![[0.7, 0.1, 0.1, 0.1]];
        let confidence = calculator.calculate_move_confidence(&confident_probs).unwrap();
        
        assert!(matches!(confidence.confidence_level, ConfidenceLevel::High));
        assert!(confidence.should_use_neural_network);
        assert!(confidence.unified_confidence > 0.6);
    }
    
    #[test]
    fn test_value_confidence_calculation() {
        let calculator = UnifiedConfidenceCalculator::new();
        
        // Test with neutral value (like current models)
        let neutral_confidence = calculator.calculate_value_confidence(0.52, "position").unwrap();
        assert!(matches!(neutral_confidence.confidence_level, ConfidenceLevel::Low));
        
        // Test with extreme value
        let extreme_confidence = calculator.calculate_value_confidence(0.85, "position").unwrap();
        assert!(matches!(extreme_confidence.confidence_level, ConfidenceLevel::High));
    }
    
    #[test]
    fn test_safety_always_enabled() {
        let calculator = UnifiedConfidenceCalculator::new();
        
        // Even with high confidence, safety should always be enabled
        let high_probs = array![[0.9, 0.03, 0.03, 0.04]];
        let confidence = calculator.calculate_move_confidence(&high_probs).unwrap();
        
        assert!(confidence.should_use_neural_network);
        assert!(confidence.should_apply_safety_check); // Safety NEVER bypassed
    }
}
