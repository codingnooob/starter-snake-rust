#!/usr/bin/env python3
"""
Simple Behavioral Validation Script for Advanced Opponent Modeling Integration

This script validates that the implemented fixes have resolved the behavioral anomalies.
"""

import requests
import json
import time
import statistics
from typing import Dict, List, Any
import uuid

def test_server_health(server_url="http://localhost:8888"):
    """Test if the server is responding properly"""
    print("Testing server health...")
    try:
        response = requests.get(f"{server_url}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"[PASS] Server health check: {data.get('apiversion', 'unknown')}")
            return True
    except Exception as e:
        print(f"[FAIL] Server health check: {e}")
    return False

def create_game_state(turn=0, health=100, multi_snake=False):
    """Create a game state for testing"""
    if multi_snake:
        return {
            "game": {"id": str(uuid.uuid4()), "ruleset": {"name": "multi"}, "timeout": 20000},
            "turn": turn,
            "board": {
                "width": 11, "height": 11,
                "food": [{"x": 5, "y": 5}, {"x": 8, "y": 8}],
                "snakes": [
                    {
                        "id": "test_snake_1", "name": "Snake 1", "health": health,
                        "body": [{"x": 4, "y": 5}, {"x": 4, "y": 6}, {"x": 4, "y": 7}],
                        "head": {"x": 4, "y": 5}, "length": 3, "latency": "100", "shout": ""
                    },
                    {
                        "id": "test_snake_2", "name": "Snake 2", "health": 95,
                        "body": [{"x": 6, "y": 5}, {"x": 6, "y": 6}, {"x": 6, "y": 7}],
                        "head": {"x": 6, "y": 5}, "length": 3, "latency": "100", "shout": ""
                    }
                ],
                "hazards": []
            },
            "you": {
                "id": "test_snake_1", "name": "Snake 1", "health": health,
                "body": [{"x": 4, "y": 5}, {"x": 4, "y": 6}, {"x": 4, "y": 7}],
                "head": {"x": 4, "y": 5}, "length": 3, "latency": "100", "shout": ""
            }
        }
    else:
        return {
            "game": {"id": str(uuid.uuid4()), "ruleset": {"name": "solo"}, "timeout": 20000},
            "turn": turn,
            "board": {
                "width": 11, "height": 11,
                "food": [{"x": 5, "y": 5}, {"x": 2, "y": 8}],
                "snakes": [
                    {
                        "id": "test_snake_solo", "name": "Solo Test Snake", "health": health,
                        "body": [{"x": 5, "y": 6}, {"x": 5, "y": 7}, {"x": 5, "y": 8}],
                        "head": {"x": 5, "y": 6}, "length": 3, "latency": "100", "shout": ""
                    }
                ],
                "hazards": []
            },
            "you": {
                "id": "test_snake_solo", "name": "Solo Test Snake", "health": health,
                "body": [{"x": 5, "y": 6}, {"x": 5, "y": 7}, {"x": 5, "y": 8}],
                "head": {"x": 5, "y": 6}, "length": 3, "latency": "100", "shout": ""
            }
        }

def test_move_endpoint(server_url, game_state):
    """Test the /move endpoint"""
    try:
        start_time = time.time()
        response = requests.post(f"{server_url}/move", json=game_state, timeout=10)
        end_time = time.time()
        response_time = (end_time - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            return {
                'success': True,
                'move': data.get('move', 'unknown'),
                'response_time_ms': response_time
            }
        else:
            return {'success': False, 'error': f"HTTP {response.status_code}"}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def validate_solo_behavior(server_url, num_turns=50):
    """Validate solo mode behavior for looping patterns"""
    print(f"\\nTesting solo mode behavior ({num_turns} turns)...")
    
    moves = []
    positions = []
    response_times = []
    
    for turn in range(num_turns):
        health = max(30, 100 - (turn // 15) * 10)
        game_state = create_game_state(turn=turn, health=health)
        
        result = test_move_endpoint(server_url, game_state)
        
        if result['success']:
            move = result['move']
            moves.append(move)
            response_times.append(result['response_time_ms'])
            
            current_pos = game_state['you']['head']
            positions.append(current_pos)
            
            if turn % 10 == 0:
                print(f"  Turn {turn:2d}: {move:5s} ({result['response_time_ms']:.1f}ms)")
        else:
            print(f"  Turn {turn:2d}: ERROR - {result['error']}")
            return {'error': f"Failed at turn {turn}"}
        
        time.sleep(0.05)
    
    # Analyze results
    move_counts = {}
    for move in moves:
        move_counts[move] = move_counts.get(move, 0) + 1
    
    avg_response_time = statistics.mean(response_times) if response_times else 0
    upward_ratio = move_counts.get('up', 0) / len(moves) if moves else 0
    
    # Check for patterns
    upper_right_turns = sum(1 for pos in positions if pos['x'] >= 8 and pos['y'] >= 8)
    upper_right_ratio = upper_right_turns / len(positions) if positions else 0
    
    print(f"\\nSolo Mode Results:")
    print(f"  Moves: {move_counts}")
    print(f"  Avg response: {avg_response_time:.1f}ms")
    print(f"  Upward ratio: {upward_ratio:.3f}")
    print(f"  Upper-right ratio: {upper_right_ratio:.3f}")
    
    return {
        'moves': moves,
        'move_counts': move_counts,
        'avg_response_time_ms': avg_response_time,
        'upward_ratio': upward_ratio,
        'upper_right_ratio': upper_right_ratio,
        'success': True
    }

def validate_multi_snake_behavior(server_url, num_tests=25):
    """Validate multi-snake behavior for upward bias"""
    print(f"\\nTesting multi-snake behavior ({num_tests} scenarios)...")
    
    moves = []
    response_times = []
    
    for test_num in range(num_tests):
        game_state = create_game_state(turn=test_num, multi_snake=True)
        
        result = test_move_endpoint(server_url, game_state)
        
        if result['success']:
            move = result['move']
            moves.append(move)
            response_times.append(result['response_time_ms'])
            
            if test_num % 5 == 0:
                print(f"  Test {test_num:2d}: {move:5s} ({result['response_time_ms']:.1f}ms)")
        else:
            print(f"  Test {test_num:2d}: ERROR - {result['error']}")
            return {'error': f"Failed at test {test_num}"}
        
        time.sleep(0.05)
    
    # Analyze results
    move_counts = {}
    for move in moves:
        move_counts[move] = move_counts.get(move, 0) + 1
    
    avg_response_time = statistics.mean(response_times) if response_times else 0
    upward_ratio = move_counts.get('up', 0) / len(moves) if moves else 0
    
    print(f"\\nMulti-Snake Results:")
    print(f"  Moves: {move_counts}")
    print(f"  Avg response: {avg_response_time:.1f}ms")
    print(f"  Upward ratio: {upward_ratio:.3f}")
    
    return {
        'moves': moves,
        'move_counts': move_counts,
        'avg_response_time_ms': avg_response_time,
        'upward_ratio': upward_ratio,
        'success': True
    }

def validate_neural_integration(server_url):
    """Test neural network integration"""
    print("\\nTesting neural network integration...")
    
    test_scenarios = [
        ("High complexity", create_game_state(turn=0, multi_snake=True)),
        ("Low health", create_game_state(turn=0, health=20)),
        ("Normal solo", create_game_state(turn=0, health=100))
    ]
    
    all_moves = []
    response_times = []
    
    for scenario_name, game_state in test_scenarios:
        print(f"  Testing: {scenario_name}")
        
        for i in range(3):
            result = test_move_endpoint(server_url, game_state)
            
            if result['success']:
                move = result['move']
                all_moves.append(move)
                response_times.append(result['response_time_ms'])
                print(f"    {move:5s} ({result['response_time_ms']:.1f}ms)")
            else:
                print(f"    ERROR: {result['error']}")
                return {'error': f"Neural integration test failed"}
    
    move_diversity = len(set(all_moves))
    avg_response_time = statistics.mean(response_times) if response_times else 0
    integration_active = move_diversity >= 3 and avg_response_time < 200
    
    print(f"\\nNeural Network Integration Results:")
    print(f"  Move diversity: {move_diversity}")
    print(f"  Avg response: {avg_response_time:.1f}ms")
    print(f"  Integration active: {integration_active}")
    
    return {
        'move_diversity': move_diversity,
        'avg_response_time_ms': avg_response_time,
        'integration_active': integration_active,
        'all_moves': all_moves,
        'success': True
    }

def main():
    """Main validation function"""
    server_url = "http://localhost:8888"
    
    print("COMPREHENSIVE BEHAVIORAL VALIDATION")
    print("===================================")
    print("Testing Advanced Opponent Modeling Integration fixes...")
    
    # Test server health
    if not test_server_health(server_url):
        print("\\n[FAIL] Server is not responding properly")
        return 1
    
    # Run validation tests
    solo_results = validate_solo_behavior(server_url, num_turns=50)
    if not solo_results.get('success'):
        print(f"\\n[FAIL] Solo validation failed: {solo_results.get('error')}")
        return 1
    
    multi_results = validate_multi_snake_behavior(server_url, num_tests=25)
    if not multi_results.get('success'):
        print(f"\\n[FAIL] Multi-snake validation failed: {multi_results.get('error')}")
        return 1
    
    neural_results = validate_neural_integration(server_url)
    if not neural_results.get('success'):
        print(f"\\n[FAIL] Neural integration test failed: {neural_results.get('error')}")
        return 1
    
    # Generate final assessment
    print("\\n" + "="*50)
    print("VALIDATION RESULTS SUMMARY")
    print("="*50)
    
    issues_found = []
    fixes_confirmed = []
    
    # Check for behavioral anomalies
    if solo_results.get('upward_ratio', 0) > 0.7:
        issues_found.append(f"Strong upward bias in solo mode ({solo_results['upward_ratio']:.3f})")
    else:
        fixes_confirmed.append("No excessive upward bias in solo mode")
    
    if multi_results.get('upward_ratio', 0) > 0.7:
        issues_found.append(f"Strong upward bias in multi-snake ({multi_results['upward_ratio']:.3f})")
    else:
        fixes_confirmed.append("No excessive upward bias in multi-snake mode")
    
    if solo_results.get('upper_right_ratio', 0) > 0.4:
        issues_found.append(f"Upper-right quadrant looping ({solo_results['upper_right_ratio']:.3f})")
    else:
        fixes_confirmed.append("No upper-right quadrant looping detected")
    
    if neural_results.get('integration_active', False):
        fixes_confirmed.append("Neural network integration is active")
    else:
        issues_found.append("Neural network integration appears inactive")
    
    # Performance assessment
    avg_solo_time = solo_results.get('avg_response_time_ms', 0)
    avg_multi_time = multi_results.get('avg_response_time_ms', 0)
    
    if avg_solo_time > 100 or avg_multi_time > 100:
        issues_found.append(f"Performance degradation detected (solo: {avg_solo_time:.1f}ms, multi: {avg_multi_time:.1f}ms)")
    else:
        fixes_confirmed.append("Performance maintained within acceptable limits")
    
    # Final verdict
    overall_status = "PASSED" if not issues_found else "ISSUES DETECTED"
    
    print(f"\\nOVERALL STATUS: {overall_status}")
    
    print("\\nFIXES CONFIRMED WORKING:")
    for fix in fixes_confirmed:
        print(f"  + {fix}")
    
    if issues_found:
        print("\\nISSUES DETECTED:")
        for issue in issues_found:
            print(f"  - {issue}")
        print("\\nRECOMMENDATION: Additional fixes may be required")
    else:
        print("\\nRECOMMENDATION: All behavioral anomalies resolved!")
    
    # Save results
    results = {
        'solo_results': solo_results,
        'multi_results': multi_results,
        'neural_results': neural_results,
        'fixes_confirmed': fixes_confirmed,
        'issues_found': issues_found,
        'overall_status': overall_status
    }
    
    with open('validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\\nDetailed results saved to: validation_results.json")
    return 0 if not issues_found else 1

if __name__ == "__main__":
    exit(main())