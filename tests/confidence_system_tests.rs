// Comprehensive Test Suite for Neural Network Confidence System
// Tests edge cases, threshold behavior, and system robustness

use starter_snake_rust::unified_confidence::*;
use starter_snake_rust::neural_confidence_integration::*;
use starter_snake_rust::{Board, Battlesnake, Coord};
use ndarray::array;
use anyhow::Result;

#[cfg(test)]
mod confidence_calculation_tests {
    use super::*;

    /// Test suite for basic confidence calculation functionality
    mod basic_functionality {
        use super::*;

        #[test]
        fn test_move_confidence_with_random_predictions() {
            let calculator = UnifiedConfidenceCalculator::new();
            
            // Test current model behavior: near-random predictions
            let random_probs = array![[0.26, 0.24, 0.25, 0.25]];
            let confidence = calculator.calculate_move_confidence(&random_probs).unwrap();
            
            assert!(matches!(confidence.confidence_level, ConfidenceLevel::Low));
            assert!(!confidence.should_use_neural_network);
            assert!(confidence.unified_confidence < 0.4);
            assert!(confidence.raw_metrics.entropy.unwrap() > 0.95); // High entropy = low confidence
        }

        #[test]
        fn test_move_confidence_with_confident_predictions() {
            let calculator = UnifiedConfidenceCalculator::new();
            
            // Test with very confident prediction
            let confident_probs = array![[0.85, 0.05, 0.05, 0.05]];
            let confidence = calculator.calculate_move_confidence(&confident_probs).unwrap();
            
            assert!(matches!(confidence.confidence_level, ConfidenceLevel::High));
            assert!(confidence.should_use_neural_network);
            assert!(confidence.should_apply_safety_check); // Safety always required
            assert!(confidence.unified_confidence > 0.7);
        }

        #[test]
        fn test_position_confidence_with_neutral_values() {
            let calculator = UnifiedConfidenceCalculator::new();
            
            // Test current model behavior: values close to neutral (0.5)
            let neutral_confidence = calculator.calculate_value_confidence(0.52, "position").unwrap();
            
            assert!(matches!(neutral_confidence.confidence_level, ConfidenceLevel::Low));
            assert!(!neutral_confidence.should_use_neural_network);
            assert!(neutral_confidence.unified_confidence < 0.3);
        }

        #[test]
        fn test_position_confidence_with_extreme_values() {
            let calculator = UnifiedConfidenceCalculator::new();
            
            // Test with extreme position value (high confidence)
            let extreme_confidence = calculator.calculate_value_confidence(0.9, "position").unwrap();
            
            assert!(matches!(extreme_confidence.confidence_level, ConfidenceLevel::High));
            assert!(extreme_confidence.should_use_neural_network);
            assert!(extreme_confidence.unified_confidence > 0.6);
        }
    }

    /// Test suite for edge cases and error conditions
    mod edge_cases {
        use super::*;

        #[test]
        fn test_invalid_move_probabilities_shape() {
            let calculator = UnifiedConfidenceCalculator::new();
            
            // Wrong number of move probabilities
            let invalid_probs = array![[0.5, 0.5]]; // Only 2 moves instead of 4
            let result = calculator.calculate_move_confidence(&invalid_probs);
            
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("Expected 4 move probabilities"));
        }

        #[test]
        fn test_extreme_probability_values() {
            let calculator = UnifiedConfidenceCalculator::new();
            
            // Test with probabilities that sum to more than 1.0
            let extreme_probs = array![[1.0, 0.5, 0.3, 0.2]]; // Sum = 2.0
            let confidence = calculator.calculate_move_confidence(&extreme_probs).unwrap();
            
            // System should handle gracefully by normalizing
            assert!(confidence.unified_confidence >= 0.0);
            assert!(confidence.unified_confidence <= 1.0);
        }

        #[test]
        fn test_zero_probabilities() {
            let calculator = UnifiedConfidenceCalculator::new();
            
            // Test with all zero probabilities
            let zero_probs = array![[0.0, 0.0, 0.0, 0.0]];
            let confidence = calculator.calculate_move_confidence(&zero_probs).unwrap();
            
            // Should result in low confidence
            assert!(matches!(confidence.confidence_level, ConfidenceLevel::Low));
            assert!(!confidence.should_use_neural_network);
        }

        #[test]
        fn test_out_of_range_position_values() {
            let calculator = UnifiedConfidenceCalculator::new();
            
            // Test values outside [0,1] range
            let high_value = calculator.calculate_value_confidence(1.5, "position").unwrap();
            let low_value = calculator.calculate_value_confidence(-0.5, "position").unwrap();
            
            // System should handle gracefully
            assert!(high_value.unified_confidence >= 0.0);
            assert!(high_value.unified_confidence <= 1.0);
            assert!(low_value.unified_confidence >= 0.0);
            assert!(low_value.unified_confidence <= 1.0);
        }

        #[test]
        fn test_invalid_model_type() {
            let calculator = UnifiedConfidenceCalculator::new();
            
            let result = calculator.calculate_value_confidence(0.5, "invalid_type");
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("Unknown model type"));
        }
    }

    /// Test suite for threshold behavior and configuration
    mod threshold_behavior {
        use super::*;

        #[test]
        fn test_confidence_threshold_boundaries() {
            let calculator = UnifiedConfidenceCalculator::new();
            let config = calculator.get_config();
            
            // Test right at high confidence threshold
            let threshold_entropy = config.move_prediction_thresholds.high_confidence_entropy_threshold;
            let at_threshold_probs = create_probabilities_with_entropy(threshold_entropy);
            let confidence = calculator.calculate_move_confidence(&at_threshold_probs).unwrap();
            
            // Should be at boundary between medium and high
            assert!(matches!(confidence.confidence_level, ConfidenceLevel::Medium | ConfidenceLevel::High));
        }

        #[test]
        fn test_custom_configuration() {
            let mut custom_config = ConfidenceConfig::default();
            custom_config.move_prediction_thresholds.high_confidence_entropy_threshold = 0.5; // More lenient
            custom_config.move_prediction_thresholds.high_confidence_max_prob_threshold = 0.4; // More lenient
            
            let calculator = UnifiedConfidenceCalculator::with_config(custom_config);
            
            // Test that custom thresholds affect classification
            let medium_probs = array![[0.45, 0.2, 0.2, 0.15]]; // Max prob = 0.45 > 0.4 threshold
            let confidence = calculator.calculate_move_confidence(&medium_probs).unwrap();
            
            assert!(matches!(confidence.confidence_level, ConfidenceLevel::High));
        }

        #[test]
        fn test_calibrated_thresholds_for_current_models() {
            let calculator = UnifiedConfidenceCalculator::new();
            
            // Test with realistic current model outputs (from our analysis)
            // Mean max prob ~0.265, mean entropy ~0.999
            let realistic_probs = array![[0.265, 0.24, 0.25, 0.245]];
            let confidence = calculator.calculate_move_confidence(&realistic_probs).unwrap();
            
            // With calibrated thresholds, this should be low confidence
            assert!(matches!(confidence.confidence_level, ConfidenceLevel::Low));
            assert!(!confidence.should_use_neural_network);
        }
    }

    /// Test suite for multi-model consistency
    mod consistency_tests {
        use super::*;

        #[test]
        fn test_consistent_model_outputs() {
            let calculator = UnifiedConfidenceCalculator::new();
            
            // High position score should correlate with high win probability
            let consistency = calculator.calculate_consistency_confidence(0.8, &[0.6, 0.2, 0.1, 0.1], 0.75);
            
            // Should detect consistency
            assert!(consistency > 0.6);
        }

        #[test]
        fn test_inconsistent_model_outputs() {
            let calculator = UnifiedConfidenceCalculator::new();
            
            // High position score but low win probability (inconsistent)
            let consistency = calculator.calculate_consistency_confidence(0.8, &[0.6, 0.2, 0.1, 0.1], 0.2);
            
            // Should detect inconsistency
            assert!(consistency < 0.5);
        }

        #[test]
        fn test_neutral_model_outputs() {
            let calculator = UnifiedConfidenceCalculator::new();
            
            // All models neutral/random
            let consistency = calculator.calculate_consistency_confidence(0.5, &[0.25, 0.25, 0.25, 0.25], 0.5);
            
            // Should be moderate consistency (no strong signals either way)
            assert!(consistency > 0.3);
            assert!(consistency < 0.7);
        }
    }

    // Helper function to create probabilities with specific entropy
    fn create_probabilities_with_entropy(target_entropy_normalized: f32) -> ndarray::Array2<f32> {
        // Create probabilities that approximate the target normalized entropy
        if target_entropy_normalized > 0.9 {
            // Very high entropy (random)
            array![[0.25, 0.25, 0.25, 0.25]]
        } else if target_entropy_normalized > 0.6 {
            // Medium entropy
            array![[0.4, 0.3, 0.2, 0.1]]
        } else {
            // Low entropy (confident)
            array![[0.7, 0.15, 0.1, 0.05]]
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Test the complete decision-making pipeline
    pub mod decision_pipeline {
        use super::*;

        pub fn create_test_scenario() -> (Board, Battlesnake, Vec<String>) {
            let board = Board {
                height: 11,
                width: 11,
                food: vec![Coord { x: 5, y: 5 }],
                snakes: vec![],
                hazards: vec![],
                turn: 10,
            };
            
            let snake = Battlesnake {
                id: "test".to_string(),
                name: "test".to_string(),
                health: 100,
                body: vec![Coord { x: 5, y: 6 }, Coord { x: 5, y: 7 }],
                head: Coord { x: 5, y: 6 },
                length: 2,
                latency: "0".to_string(),
                shout: None,
            };
            
            let safe_moves = vec!["up".to_string(), "left".to_string(), "right".to_string()];
            
            (board, snake, safe_moves)
        }

        #[test]
        fn test_complete_decision_pipeline_low_confidence() {
            let evaluator = EnhancedNeuralEvaluator::new();
            let (board, snake, safe_moves) = integration_tests::decision_pipeline::create_test_scenario();
            
            // Low confidence neural outputs (realistic current model behavior)
            let neural_outputs = NeuralNetworkOutputs {
                position_score: Some(0.52), // Close to neutral
                move_probabilities: Some(array![[0.26, 0.24, 0.25, 0.25]]), // Near random
                win_probability: Some(0.54), // Close to neutral
            };

            let result = evaluator.make_neural_decision(&board, &snake, 10, &safe_moves, &neural_outputs).unwrap();

            // Should fallback to heuristics
            assert!(matches!(result.decision_source, DecisionSource::HeuristicFallback(_)));
            assert!(safe_moves.contains(&result.recommended_move));
            assert!(result.confidence.unified_confidence < 0.5);
        }

        #[test]
        fn test_complete_decision_pipeline_high_confidence_safe() {
            let evaluator = EnhancedNeuralEvaluator::new();
            let (board, snake, safe_moves) = integration_tests::decision_pipeline::create_test_scenario();
            
            // High confidence neural outputs with safe choice
            let neural_outputs = NeuralNetworkOutputs {
                position_score: Some(0.85), // High confidence
                move_probabilities: Some(array![[0.7, 0.1, 0.1, 0.1]]), // Strongly prefers "up" (safe)
                win_probability: Some(0.80), // High confidence
            };

            let result = evaluator.make_neural_decision(&board, &snake, 10, &safe_moves, &neural_outputs).unwrap();

            // Should use neural network choice
            assert!(matches!(result.decision_source, DecisionSource::NeuralNetworkHighConfidence));
            assert_eq!(result.recommended_move, "up");
            assert!(result.confidence.should_apply_safety_check); // Safety always enabled
            assert!(result.confidence.unified_confidence > 0.6);
        }

        #[test]
        fn test_safety_override_with_high_confidence() {
            let evaluator = EnhancedNeuralEvaluator::new();
            let (board, snake, safe_moves) = integration_tests::decision_pipeline::create_test_scenario();
            
            // High confidence neural outputs but preferring unsafe move
            let neural_outputs = NeuralNetworkOutputs {
                position_score: Some(0.85), // High confidence
                move_probabilities: Some(array![[0.1, 0.8, 0.05, 0.05]]), // Strongly prefers "down" (unsafe)
                win_probability: Some(0.80), // High confidence
            };

            let result = evaluator.make_neural_decision(&board, &snake, 10, &safe_moves, &neural_outputs).unwrap();

            // Should override with safety
            assert!(matches!(result.decision_source, DecisionSource::SafetyOverride(_)));
            assert!(safe_moves.contains(&result.recommended_move));
            assert_ne!(result.recommended_move, "down"); // Should not choose unsafe move
        }

        #[test]
        fn test_missing_neural_outputs_graceful_handling() {
            let evaluator = EnhancedNeuralEvaluator::new();
            let (board, snake, safe_moves) = integration_tests::decision_pipeline::create_test_scenario();
            
            // Missing position score
            let neural_outputs = NeuralNetworkOutputs {
                position_score: None, // Missing
                move_probabilities: Some(array![[0.7, 0.1, 0.1, 0.1]]),
                win_probability: Some(0.80),
            };

            let result = evaluator.make_neural_decision(&board, &snake, 10, &safe_moves, &neural_outputs).unwrap();

            // Should handle gracefully and fallback
            assert!(matches!(result.decision_source, DecisionSource::HeuristicFallback(_)));
            assert!(safe_moves.contains(&result.recommended_move));
        }

        #[test]
        fn test_no_safe_moves_error_handling() {
            let evaluator = EnhancedNeuralEvaluator::new();
            let (board, snake, _) = create_test_scenario();
            
            let empty_safe_moves = vec![]; // No safe moves available
            let neural_outputs = NeuralNetworkOutputs {
                position_score: Some(0.85),
                move_probabilities: Some(array![[0.7, 0.1, 0.1, 0.1]]),
                win_probability: Some(0.80),
            };

            let result = evaluator.make_neural_decision(&board, &snake, 10, &empty_safe_moves, &neural_outputs);

            // Should return error when no safe moves available
            assert!(result.is_err());
        }
    }

    /// Test metrics collection and analysis
    mod metrics_tests {
        use super::*;

        #[test]
        fn test_metrics_collection() {
            let evaluator = EnhancedNeuralEvaluator::new();
            let (board, snake, safe_moves) = integration_tests::decision_pipeline::create_test_scenario();
            
            // Make several decisions to test metrics
            for i in 0..5 {
                let confidence_level = if i % 2 == 0 { 0.8 } else { 0.2 }; // Alternate high/low confidence
                
                let neural_outputs = NeuralNetworkOutputs {
                    position_score: Some(confidence_level),
                    move_probabilities: Some(if confidence_level > 0.5 {
                        array![[0.7, 0.1, 0.1, 0.1]]
                    } else {
                        array![[0.26, 0.24, 0.25, 0.25]]
                    }),
                    win_probability: Some(confidence_level),
                };

                let _ = evaluator.make_neural_decision(&board, &snake, 10, &safe_moves, &neural_outputs).unwrap();
            }

            let metrics = evaluator.get_metrics();
            assert_eq!(metrics.total_decisions, 5);
            assert!(metrics.high_confidence_decisions > 0);
            assert!(metrics.low_confidence_decisions > 0);
            assert_eq!(metrics.confidence_score_history.len(), 5);
        }

        #[test]
        fn test_decision_history_tracking() {
            let evaluator = EnhancedNeuralEvaluator::new();
            let (board, snake, safe_moves) = integration_tests::decision_pipeline::create_test_scenario();
            
            let neural_outputs = NeuralNetworkOutputs {
                position_score: Some(0.85),
                move_probabilities: Some(array![[0.7, 0.1, 0.1, 0.1]]),
                win_probability: Some(0.80),
            };

            let _ = evaluator.make_neural_decision(&board, &snake, 10, &safe_moves, &neural_outputs).unwrap();
            
            let recent_decisions = evaluator.get_recent_decisions(10);
            assert_eq!(recent_decisions.len(), 1);
            
            let decision = &recent_decisions[0];
            assert_eq!(decision.chosen_move, "up");
            assert!(decision.confidence.unified_confidence > 0.5);
            assert_eq!(decision.game_context.our_health, 100);
        }

        #[test]
        fn test_analysis_data_export() {
            let evaluator = EnhancedNeuralEvaluator::new();
            let (board, snake, safe_moves) = integration_tests::decision_pipeline::create_test_scenario();
            
            let neural_outputs = NeuralNetworkOutputs {
                position_score: Some(0.85),
                move_probabilities: Some(array![[0.7, 0.1, 0.1, 0.1]]),
                win_probability: Some(0.80),
            };

            let _ = evaluator.make_neural_decision(&board, &snake, 10, &safe_moves, &neural_outputs).unwrap();
            
            let analysis_data = evaluator.export_analysis_data().unwrap();
            assert!(analysis_data.contains("total_decisions"));
            assert!(analysis_data.contains("neural_utilization_rate"));
            assert!(analysis_data.contains("recent_decisions"));
            
            // Should be valid JSON
            let parsed: serde_json::Value = serde_json::from_str(&analysis_data).unwrap();
            assert!(parsed["metrics"]["total_decisions"].as_u64().unwrap() > 0);
        }
    }
}

/// Stress tests for system robustness
#[cfg(test)]
mod stress_tests {
    use super::*;

    #[test]
    fn test_high_volume_decision_making() {
        let evaluator = EnhancedNeuralEvaluator::new();
        let (board, snake, safe_moves) = integration_tests::decision_pipeline::create_test_scenario();
        
        // Make 1000 decisions to test performance and memory usage
        for i in 0..1000 {
            let confidence_val = (i as f32 / 1000.0).min(1.0);
            
            let neural_outputs = NeuralNetworkOutputs {
                position_score: Some(0.5 + confidence_val * 0.3),
                move_probabilities: Some(array![[
                    0.25 + confidence_val * 0.5,
                    0.25 - confidence_val * 0.15,
                    0.25 - confidence_val * 0.15,
                    0.25 - confidence_val * 0.2
                ]]),
                win_probability: Some(0.5 + confidence_val * 0.3),
            };

            let result = evaluator.make_neural_decision(&board, &snake, 10, &safe_moves, &neural_outputs);
            assert!(result.is_ok());
        }

        let metrics = evaluator.get_metrics();
        assert_eq!(metrics.total_decisions, 1000);
        assert!(metrics.confidence_score_history.len() <= 1000); // Should cap history size
        
        let recent_decisions = evaluator.get_recent_decisions(50);
        assert!(recent_decisions.len() <= 50);
    }

    #[test] 
    fn test_concurrent_decision_making() {
        use std::sync::Arc;
        use std::thread;
        
        let evaluator = Arc::new(EnhancedNeuralEvaluator::new());
        let (board, snake, safe_moves) = integration_tests::decision_pipeline::create_test_scenario();
        
        let mut handles = vec![];
        
        // Spawn 10 threads making concurrent decisions
        for thread_id in 0..10 {
            let evaluator_clone = Arc::clone(&evaluator);
            let board_clone = board.clone();
            let snake_clone = snake.clone();
            let safe_moves_clone = safe_moves.clone();
            
            let handle = thread::spawn(move || {
                for i in 0..10 {
                    let neural_outputs = NeuralNetworkOutputs {
                        position_score: Some(0.5 + (thread_id as f32 * 0.1)),
                        move_probabilities: Some(array![[0.4, 0.3, 0.2, 0.1]]),
                        win_probability: Some(0.5 + (i as f32 * 0.05)),
                    };

                    let result = evaluator_clone.make_neural_decision(
                        &board_clone, 
                        &snake_clone, 
                        10,
                        &safe_moves_clone, 
                        &neural_outputs
                    );
                    assert!(result.is_ok());
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        let metrics = evaluator.get_metrics();
        assert_eq!(metrics.total_decisions, 100); // 10 threads × 10 decisions each
    }

    fn create_test_scenario() -> (Board, Battlesnake, Vec<String>) {
        let board = Board {
            height: 11,
            width: 11,
            food: vec![Coord { x: 5, y: 5 }],
            snakes: vec![],
            hazards: vec![],
            turn: 10,
        };
        
        let snake = Battlesnake {
            id: "test".to_string(),
            name: "test".to_string(),
            health: 100,
            body: vec![Coord { x: 5, y: 6 }, Coord { x: 5, y: 7 }],
            head: Coord { x: 5, y: 6 },
            length: 2,
            latency: "0".to_string(),
            shout: None,
        };
        
        let safe_moves = vec!["up".to_string(), "left".to_string(), "right".to_string()];
        
        (board, snake, safe_moves)
    }
}