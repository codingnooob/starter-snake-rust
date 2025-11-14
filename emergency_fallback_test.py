#!/usr/bin/env python3
"""
EMERGENCY FALLBACK BUG FIX VALIDATION TEST

This test specifically targets the emergency fallback logic that was fixed.
It creates scenarios where no safe moves are available to trigger the emergency fallback.
"""

import requests
import json
import time

def create_corner_trap_scenario():
    """Create a game state where snake is trapped in corner with no safe moves"""
    return {
        "game": {
            "id": "emergency_test",
            "ruleset": {"name": "solo"},
            "timeout": 20000
        },
        "turn": 50,
        "board": {
            "width": 11,
            "height": 11,
            "food": [{"x": 0, "y": 0}],  # Food in corner
            "snakes": [
                {
                    "id": "test_snake",
                    "name": "Emergency Test Snake",
                    "health": 50,
                    "body": [
                        {"x": 0, "y": 0},  # Head at corner
                        {"x": 0, "y": 1},  # Body segments blocking most moves
                        {"x": 1, "y": 0},
                        {"x": 2, "y": 0}
                    ],
                    "head": {"x": 0, "y": 0},
                    "length": 4,
                    "latency": "100",
                    "shout": None
                }
            ],
            "hazards": []
        },
        "you": {
            "id": "test_snake",
            "name": "Emergency Test Snake",
            "health": 50,
            "body": [
                {"x": 0, "y": 0},  # Head at corner
                {"x": 0, "y": 1},  # Body segments blocking most moves
                {"x": 1, "y": 0},
                {"x": 2, "y": 0}
            ],
            "head": {"x": 0, "y": 0},
            "length": 4,
            "latency": "100",
            "shout": None
        }
    }

def test_emergency_fallback(server_url="http://localhost:8001"):
    """Test the emergency fallback logic by sending a trap scenario"""
    
    print("EMERGENCY FALLBACK BUG FIX VALIDATION TEST")
    print("=" * 50)
    
    # Test scenario 1: Corner trap (should trigger emergency fallback)
    print("\nTest 1: Corner Trap Scenario")
    print("Snake at (0,0) corner with body blocking most moves")
    
    game_state = create_corner_trap_scenario()
    
    try:
        # Test the move endpoint
        response = requests.post(f"{server_url}/move", json=game_state, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            move = result.get("move", "unknown")
            
            print(f"Move response: {move}")
            
            # Validate the response contains expected emergency fallback indicators
            if "confidence" in result:
                print(f"Emergency fallback activated (confidence: {result['confidence']:.3f})")
                print(f"Decision source: {result.get('decision_source', 'unknown')}")
            
            # Check if the move is valid
            valid_moves = ["up", "down", "left", "right"]
            if move in valid_moves:
                print(f"Valid move returned: {move}")
            else:
                print(f"Invalid move returned: {move}")
                
        else:
            print(f"HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    
    print("\n" + "=" * 50)
    print("EMERGENCY FALLBACK TEST COMPLETE")

if __name__ == "__main__":
    test_emergency_fallback()