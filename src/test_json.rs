// Simple JSON deserialization test
use crate::{GameState};

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_battlesnake_json_deserialization() {
        let json = r#"{
            "game": {
                "id": "test-game-123",
                "ruleset": {
                    "name": "standard",
                    "version": "v1.0.0"
                },
                "timeout": 10000
            },
            "turn": 0,
            "board": {
                "width": 11,
                "height": 11,
                "food": [{"x": 5, "y": 5}],
                "snakes": [{
                    "id": "neural-test-snake",
                    "name": "Neural Network Test",
                    "health": 100,
                    "body": [{"x": 5, "y": 8}, {"x": 5, "y": 9}, {"x": 5, "y": 10}],
                    "head": {"x": 5, "y": 8},
                    "length": 3,
                    "latency": "100",
                    "shout": null
                }],
                "hazards": []
            },
            "you": {
                "id": "neural-test-snake",
                "name": "Neural Network Test",
                "health": 100,
                "body": [{"x": 5, "y": 8}, {"x": 5, "y": 9}, {"x": 5, "y": 10}],
                "head": {"x": 5, "y": 8},
                "length": 3,
                "latency": "100",
                "shout": null
            }
        }"#;

        // This should now work without the "missing field `turn`" error
        let result = serde_json::from_str::<GameState>(json);
        
        match result {
            Ok(game_state) => {
                println!("✅ SUCCESS: JSON deserialization works!");
                println!("Turn: {}", game_state.turn);
                assert_eq!(game_state.turn, 0);
                assert_eq!(game_state.you.name, "Neural Network Test");
            },
            Err(e) => {
                println!("❌ FAILED: {}", e);
                panic!("JSON deserialization should work now that Board.turn field is removed");
            }
        }
    }
}