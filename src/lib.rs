// Library interface for starter-snake-rust
// Enables integration tests to import modules

pub mod types;
pub mod advanced_spatial_analysis;
pub mod logic;
pub mod neural_network;
pub mod unified_confidence;
pub mod neural_confidence_integration;
pub mod confidence_validation;
pub mod adaptive_neural_system;
pub mod enhanced_decision_system;

// Re-export core types for easier access
pub use types::{Board, Battlesnake, Coord, Game, GameState};

// Re-export advanced spatial analysis components for neural network integration
pub use advanced_spatial_analysis::{AdvancedBoardStateEncoder, VoronoiTerritoryAnalyzer,
                                   DangerZonePredictor, MovementHistoryTracker, StrategicPositionAnalyzer};