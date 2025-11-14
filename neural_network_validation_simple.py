#!/usr/bin/env python3
"""
Simple Neural Network Performance Validation
Tests neural network performance without Unicode issues
"""

import json
import time
import requests
import statistics
from datetime import datetime

def test_neural_network_performance():
    """Test current neural network performance"""
    print("NEURAL NETWORK PERFORMANCE VALIDATION")
    print("=" * 50)
    
    # Server URL (updated to port 8001)
    server_url = "http://localhost:8001"
    
    # Test scenarios
    scenarios = [
        {
            "name": "simple_scenario",
            "data": {
                "game": {"id": "neural-validation", "ruleset": {"name": "standard"}, "timeout": 500},
                "turn": 1,
                "board": {
                    "width": 11, "height": 11,
                    "food": [{"x": 8, "y": 8}],
                    "snakes": [{
                        "id": "test-snake", "name": "Test Snake",
                        "body": [{"x": 5, "y": 5}, {"x": 5, "y": 4}],
                        "head": {"x": 5, "y": 5}, "health": 90, "length": 2
                    }], "hazards": []
                },
                "you": {
                    "id": "test-snake", "name": "Test Snake", 
                    "body": [{"x": 5, "y": 5}, {"x": 5, "y": 4}],
                    "head": {"x": 5, "y": 5}, "health": 90, "length": 2
                }
            }
        },
        {
            "name": "complex_scenario",
            "data": {
                "game": {"id": "complex-validation", "ruleset": {"name": "standard"}, "timeout": 500},
                "turn": 45,
                "board": {
                    "width": 11, "height": 11,
                    "food": [{"x": 1, "y": 9}, {"x": 10, "y": 1}],
                    "snakes": [
                        {
                            "id": "test-snake", "name": "Test Snake",
                            "body": [{"x": 3, "y": 7}, {"x": 3, "y": 6}, {"x": 3, "y": 5}],
                            "head": {"x": 3, "y": 7}, "health": 75, "length": 3
                        },
                        {
                            "id": "opponent", "name": "Opponent",
                            "body": [{"x": 8, "y": 3}, {"x": 8, "y": 2}],
                            "head": {"x": 8, "y": 3}, "health": 80, "length": 2
                        }
                    ], "hazards": []
                },
                "you": {
                    "id": "test-snake", "name": "Test Snake",
                    "body": [{"x": 3, "y": 7}, {"x": 3, "y": 6}, {"x": 3, "y": 5}],
                    "head": {"x": 3, "y": 7}, "health": 75, "length": 3
                }
            }
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\nTesting {scenario['name']}...")
        response_times = []
        
        for i in range(15):  # 15 iterations for better accuracy
            try:
                start_time = time.time()
                response = requests.post(
                    f"{server_url}/move", 
                    json=scenario['data'], 
                    timeout=10
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    response_time_ms = (end_time - start_time) * 1000
                    response_times.append(response_time_ms)
                    
                    # Check response content for neural network indicators
                    if i == 0:  # Log first response for debugging
                        result = response.json()
                        move = result.get("move", "unknown")
                        print(f"  Response: {move}")
                
                time.sleep(0.05)  # Small delay between requests
                
            except Exception as e:
                print(f"  Request {i+1} failed: {e}")
                continue
        
        if response_times:
            results[scenario['name']] = {
                'response_times': response_times,
                'average': statistics.mean(response_times),
                'median': statistics.median(response_times),
                'min': min(response_times),
                'max': max(response_times),
                'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0,
                'success_rate': len(response_times) / 15 * 100
            }
            
            avg = results[scenario['name']]['average']
            print(f"  SUCCESS: Average {avg:.2f}ms ({len(response_times)}/15 successful)")
        else:
            print(f"  FAILED: All requests failed")
    
    # Overall analysis
    if results:
        all_times = []
        for scenario_data in results.values():
            all_times.extend(scenario_data['response_times'])
        
        if all_times:
            overall_avg = statistics.mean(all_times)
            overall_std = statistics.stdev(all_times)
            
            print(f"\nOVERALL PERFORMANCE ANALYSIS:")
            print(f"  Average Response Time: {overall_avg:.2f}ms")
            print(f"  Standard Deviation: {overall_std:.2f}ms")
            print(f"  Range: {min(all_times):.2f}ms - {max(all_times):.2f}ms")
            
            # Performance target analysis
            target_8ms = 8.6  # Target from documentation
            target_5ms = 5.0  # Original target
            
            print(f"\nTARGET ANALYSIS:")
            print(f"  8.6ms Target Gap: {overall_avg - target_8ms:.2f}ms")
            print(f"  Performance vs 8.6ms: {overall_avg/target_8ms:.1f}x target")
            
            if overall_avg <= target_8ms:
                print(f"  STATUS: MEETS 8.6ms TARGET!")
            elif overall_avg <= target_5ms * 2:  # Within 2x original target
                print(f"  STATUS: ACCEPTABLE (within 2x 5ms target)")
            else:
                print(f"  STATUS: ABOVE TARGET")
            
            # Check for neural network characteristics
            cv = (overall_std / overall_avg) * 100
            print(f"\nNEURAL NETWORK INDICATORS:")
            print(f"  Coefficient of Variation: {cv:.1f}%")
            if cv < 20:
                print(f"  High consistency - likely neural inference")
            else:
                print(f"  Variable performance - may be fallback mode")
            
            # Save results
            output = {
                'validation_timestamp': datetime.now().isoformat(),
                'overall_average_ms': overall_avg,
                'overall_std_ms': overall_std,
                'scenarios': results,
                'target_8_6ms_met': overall_avg <= target_8ms,
                'coefficient_of_variation': cv
            }
            
            with open('neural_validation_results.json', 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"\nResults saved to neural_validation_results.json")
            return overall_avg <= target_8ms
    
    return False

if __name__ == "__main__":
    test_neural_network_performance()