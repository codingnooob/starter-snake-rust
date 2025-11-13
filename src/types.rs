// Core types for Battlesnake game
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Deserialize, Serialize, Debug)]
pub struct Game {
    pub id: String,
    // Make ruleset optional for CLI compatibility - default to empty HashMap
    #[serde(default)]
    pub ruleset: HashMap<String, Value>,
    // Make timeout optional for CLI compatibility - default to 500ms
    #[serde(default = "default_timeout")]
    pub timeout: u32,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Board {
    pub height: u32,
    pub width: i32,
    pub food: Vec<Coord>,
    pub snakes: Vec<Battlesnake>,
    // Make hazards optional for CLI compatibility - default to empty Vec
    #[serde(default)]
    pub hazards: Vec<Coord>,
    // Adding turn field for test compatibility
    #[serde(default)]
    pub turn: i32,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Battlesnake {
    pub id: String,
    // Make name optional for CLI compatibility - default to empty string
    #[serde(default)]
    pub name: String,
    pub health: i32,
    pub body: Vec<Coord>,
    // Make head optional - compute from body[0] if missing
    #[serde(default)]
    pub head: Coord,
    // Make length optional - compute from body.len() if missing
    #[serde(default)]
    pub length: i32,
    // Make latency optional for CLI compatibility - default to empty string
    #[serde(default)]
    pub latency: String,
    pub shout: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coord {
    pub x: i32,
    pub y: i32,
}

// Default implementation for Coord
impl Default for Coord {
    fn default() -> Self {
        Coord { x: 0, y: 0 }
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub struct GameState {
    pub game: Game,
    // Make turn optional for CLI compatibility - default to 0
    #[serde(default)]
    pub turn: i32,
    pub board: Board,
    pub you: Battlesnake,
}

// Default timeout function for serde
fn default_timeout() -> u32 {
    500
}

// Helper constructor functions for testing
impl Board {
    pub fn with_turn(mut self, turn: i32) -> Self {
        self.turn = turn;
        self
    }
}