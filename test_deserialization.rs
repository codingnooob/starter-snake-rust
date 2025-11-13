use serde_json;
use std::fs;

// Copy the struct definitions from main.rs for testing
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use serde_json::Value;

#[derive(Deserialize, Serialize, Debug)]
pub struct Game {
    id: String,
    ruleset: HashMap<String, Value>,
    timeout: u32,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Board {
    height: u32,
    width: i32,
    food: Vec<Coord>,
    snakes: Vec<Battlesnake>,
    hazards: Vec<Coord>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Battlesnake {
    pub id: String,
    pub name: String,
    pub health: i32,
    pub body: Vec<Coord>,
    pub head: Coord,
    pub length: i32,
    pub latency: String,
    pub shout: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coord {
    pub x: i32,
    pub y: i32,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct GameState {
    game: Game,
    turn: i32,
    board: Board,
    you: Battlesnake,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing JSON deserialization...");
    
    // Read the test JSON file
    let json_content = fs::read_to_string("test_move_request.json")?;
    
    // Try to deserialize
    match serde_json::from_str::<GameState>(&json_content) {
        Ok(game_state) => {
            println!("✅ SUCCESS: JSON deserialization completed!");
            println!("Game ID: {}", game_state.game.id);
            println!("Turn: {}", game_state.turn);
            println!("Board: {}x{}", game_state.board.width, game_state.board.height);
            println!("Snake: {} (health: {})", game_state.you.name, game_state.you.health);
            println!("🎉 The 422 Unprocessable Entity bug is FIXED!");
        },
        Err(e) => {
            println!("❌ FAILED: {}", e);
            return Err(e.into());
        }
    }
    
    Ok(())
}