#!/usr/bin/env python3
import requests
import json
import time
import sys

def test_move_endpoint():
    """Test the move endpoint and analyze response structure"""
    base_url = "http://localhost:8888"
    
    # Test payload
    test_data = {
        "game": {
            "id": "test-game-debug",
            "ruleset": {"name": "standard"},
            "timeout": 500
        },
        "turn": 1,
        "board": {
            "height": 11,
            "width": 11,
            "food": [{"x": 5, "y": 5}],
            "snakes": [{
                "id": "test-snake",
                "name": "Test Snake",
                "health": 100,
                "body": [
                    {"x": 5, "y": 10},
                    {"x": 5, "y": 9},
                    {"x": 5, "y": 8}
                ],
                "head": {"x": 5, "y": 10},
                "length": 3,
                "latency": "100",
                "shout": None
            }],
            "hazards": []
        },
        "you": {
            "id": "test-snake",
            "name": "Test Snake",
            "health": 100,
            "body": [
                {"x": 5, "y": 10},
                {"x": 5, "y": 9},
                {"x": 5, "y": 8}
            ],
            "head": {"x": 5, "y": 10},
            "length": 3,
            "latency": "100",
            "shout": None
        }
    }
    
    print("=== INTEGRATION DIAGNOSIS TEST ===")
    print(f"Testing {base_url}/move endpoint")
    print(f"Timestamp: {time.time()}")
    
    # Test with JSON content type explicitly
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        print("\n--- Sending request ---")
        response = requests.post(
            f"{base_url}/move",
            headers=headers,
            json=test_data,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Raw Response: {response.text}")
        
        # Parse JSON response
        try:
            response_json = response.json()
            print(f"Parsed JSON: {json.dumps(response_json, indent=2)}")
            
            # Validate response structure
            if "move" in response_json:
                move_value = response_json["move"]
                print(f"✓ Move field found: '{move_value}'")
                print(f"✓ Move type: {type(move_value)}")
                
                # Check for valid move values
                valid_moves = ["up", "down", "left", "right"]
                if move_value in valid_moves:
                    print(f"✓ Valid move: {move_value}")
                    print("✓ INTEGRATION SUCCESS: Move decision is properly formatted and ready for game transmission")
                    return True
                else:
                    print(f"✗ Invalid move value: {move_value}")
                    print(f"✗ Expected one of: {valid_moves}")
                    return False
            else:
                print("✗ ERROR: No 'move' field in response")
                print(f"Available fields: {list(response_json.keys()) if isinstance(response_json, dict) else 'Not a dictionary'}")
                return False
                
        except json.JSONDecodeError as e:
            print(f"✗ ERROR: Failed to parse JSON response: {e}")
            print(f"Raw response was: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ ERROR: Request failed: {e}")
        return False
    except Exception as e:
        print(f"✗ ERROR: Unexpected error: {e}")
        return False

def test_index_endpoint():
    """Test the index endpoint to verify server is running"""
    base_url = "http://localhost:8888"
    
    try:
        print("\n--- Testing index endpoint ---")
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"Index Status: {response.status_code}")
        print(f"Index Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Index test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting Integration Diagnosis...")
    
    # Test index endpoint first
    index_ok = test_index_endpoint()
    
    if index_ok:
        # Test move endpoint
        move_ok = test_move_endpoint()
        
        if move_ok:
            print("\n🎯 DIAGNOSIS RESULT: API RESPONSE INTEGRATION IS WORKING CORRECTLY")
            print("   If the live game still shows 'up' movements, the issue is likely:")
            print("   1. Game server caching previous responses")
            print("   2. Network/proxy intercepting responses") 
            print("   3. Game server not sending requests to our endpoint")
            sys.exit(0)
        else:
            print("\n❌ DIAGNOSIS RESULT: API RESPONSE INTEGRATION HAS ISSUES")
            sys.exit(1)
    else:
        print("\n❌ DIAGNOSIS RESULT: Server is not responding properly")
        sys.exit(1)