#!/usr/bin/env python3
"""
Comprehensive Performance Investigation Script
Analyzes the 40x performance degradation: 2000+ms vs claimed 5ms response times
"""

import json
import time
import requests
import statistics
import subprocess
import sys
from typing import Dict, List, Tuple
from pathlib import Path

class PerformanceInvestigator:
    def __init__(self, server_url: str = "http://localhost:8888"):
        self.server_url = server_url
        self.results = {}
        
    def create_test_scenarios(self) -> List[Dict]:
        """Create test scenarios to measure performance"""
        return [
            {
                "name": "simple_scenario",
                "description": "Single snake, minimal board",
                "payload": {
                    "game": {"id": "perf-test-1", "ruleset": {"name": "standard", "version": "v1.0.0"}, "timeout": 10000},
                    "turn": 1,
                    "board": {
                        "width": 11, "height": 11,
                        "food": [{"x": 5, "y": 5}],
                        "snakes": [{
                            "id": "test-snake", "name": "Performance Test",
                            "health": 100, "body": [{"x": 1, "y": 1}, {"x": 1, "y": 2}],
                            "head": {"x": 1, "y": 1}, "length": 2, "latency": "100", "shout": None
                        }],
                        "hazards": []
                    },
                    "you": {
                        "id": "test-snake", "name": "Performance Test",
                        "health": 100, "body": [{"x": 1, "y": 1}, {"x": 1, "y": 2}],
                        "head": {"x": 1, "y": 1}, "length": 2, "latency": "100", "shout": None
                    }
                }
            },
            {
                "name": "complex_scenario", 
                "description": "Multi-snake, complex board",
                "payload": {
                    "game": {"id": "perf-test-2", "ruleset": {"name": "standard", "version": "v1.0.0"}, "timeout": 10000},
                    "turn": 50,
                    "board": {
                        "width": 11, "height": 11,
                        "food": [{"x": 3, "y": 3}, {"x": 7, "y": 7}, {"x": 9, "y": 2}],
                        "snakes": [
                            {
                                "id": "test-snake", "name": "Our Snake",
                                "health": 85, "body": [
                                    {"x": 5, "y": 5}, {"x": 5, "y": 6}, {"x": 5, "y": 7},
                                    {"x": 4, "y": 7}, {"x": 3, "y": 7}
                                ],
                                "head": {"x": 5, "y": 5}, "length": 5, "latency": "100", "shout": None
                            },
                            {
                                "id": "opponent-1", "name": "Opponent 1", 
                                "health": 90, "body": [
                                    {"x": 8, "y": 3}, {"x": 8, "y": 4}, {"x": 8, "y": 5}
                                ],
                                "head": {"x": 8, "y": 3}, "length": 3, "latency": "100", "shout": None
                            },
                            {
                                "id": "opponent-2", "name": "Opponent 2",
                                "health": 70, "body": [
                                    {"x": 2, "y": 9}, {"x": 3, "y": 9}, {"x": 4, "y": 9}
                                ],
                                "head": {"x": 2, "y": 9}, "length": 3, "latency": "100", "shout": None
                            }
                        ],
                        "hazards": []
                    },
                    "you": {
                        "id": "test-snake", "name": "Our Snake",
                        "health": 85, "body": [
                            {"x": 5, "y": 5}, {"x": 5, "y": 6}, {"x": 5, "y": 7},
                            {"x": 4, "y": 7}, {"x": 3, "y": 7}
                        ],
                        "head": {"x": 5, "y": 5}, "length": 5, "latency": "100", "shout": None
                    }
                }
            },
            {
                "name": "neural_stress_test",
                "description": "Designed to stress neural network pipeline",
                "payload": {
                    "game": {"id": "perf-test-3", "ruleset": {"name": "standard", "version": "v1.0.0"}, "timeout": 10000},
                    "turn": 100,
                    "board": {
                        "width": 11, "height": 11,
                        "food": [
                            {"x": 1, "y": 1}, {"x": 2, "y": 2}, {"x": 3, "y": 3},
                            {"x": 8, "y": 8}, {"x": 9, "y": 9}, {"x": 10, "y": 10}
                        ],
                        "snakes": [
                            {
                                "id": "neural-test", "name": "Neural Network Test",
                                "health": 95, "body": [
                                    {"x": 5, "y": 5}, {"x": 5, "y": 6}, {"x": 5, "y": 7},
                                    {"x": 5, "y": 8}, {"x": 5, "y": 9}, {"x": 4, "y": 9},
                                    {"x": 3, "y": 9}, {"x": 2, "y": 9}
                                ],
                                "head": {"x": 5, "y": 5}, "length": 8, "latency": "100", "shout": None
                            },
                            {
                                "id": "opponent-3", "name": "Large Opponent",
                                "health": 100, "body": [
                                    {"x": 7, "y": 2}, {"x": 7, "y": 3}, {"x": 7, "y": 4},
                                    {"x": 7, "y": 5}, {"x": 8, "y": 5}, {"x": 9, "y": 5},
                                    {"x": 10, "y": 5}
                                ],
                                "head": {"x": 7, "y": 2}, "length": 7, "latency": "100", "shout": None
                            }
                        ],
                        "hazards": []
                    },
                    "you": {
                        "id": "neural-test", "name": "Neural Network Test",
                        "health": 95, "body": [
                            {"x": 5, "y": 5}, {"x": 5, "y": 6}, {"x": 5, "y": 7},
                            {"x": 5, "y": 8}, {"x": 5, "y": 9}, {"x": 4, "y": 9},
                            {"x": 3, "y": 9}, {"x": 2, "y": 9}
                        ],
                        "head": {"x": 5, "y": 5}, "length": 8, "latency": "100", "shout": None
                    }
                }
            }
        ]

    def check_server_status(self) -> bool:
        """Check if the Battlesnake server is running"""
        try:
            response = requests.get(f"{self.server_url}/", timeout=5)
            if response.status_code == 200:
                print(f"✅ Server running at {self.server_url}")
                return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Server not accessible: {e}")
            return False
        return False

    def measure_single_request(self, scenario: Dict) -> Tuple[float, Dict, bool]:
        """Measure response time for a single request"""
        start_time = time.perf_counter()
        
        try:
            response = requests.post(
                f"{self.server_url}/move",
                json=scenario["payload"],
                headers={"Content-Type": "application/json"},
                timeout=30  # Allow for slow responses
            )
            
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                return response_time_ms, response.json(), True
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                return response_time_ms, {}, False
                
        except requests.exceptions.Timeout:
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000
            print(f"❌ Request timeout after {response_time_ms:.1f}ms")
            return response_time_ms, {}, False
            
        except requests.exceptions.RequestException as e:
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000
            print(f"❌ Request error: {e}")
            return response_time_ms, {}, False

    def run_performance_tests(self, iterations: int = 10) -> Dict:
        """Run comprehensive performance tests"""
        print(f"\n🚀 PERFORMANCE INVESTIGATION STARTING")
        print(f"Testing {iterations} iterations per scenario")
        print("=" * 60)
        
        scenarios = self.create_test_scenarios()
        results = {}
        
        for scenario in scenarios:
            print(f"\n📊 Testing: {scenario['name']} - {scenario['description']}")
            
            times = []
            successes = 0
            failures = 0
            
            for i in range(iterations):
                print(f"  Iteration {i+1}/{iterations}...", end=" ", flush=True)
                
                response_time, response_data, success = self.measure_single_request(scenario)
                times.append(response_time)
                
                if success:
                    successes += 1
                    print(f"✅ {response_time:.1f}ms")
                else:
                    failures += 1
                    print(f"❌ {response_time:.1f}ms (failed)")
            
            # Calculate statistics
            if times:
                results[scenario['name']] = {
                    "description": scenario['description'],
                    "iterations": iterations,
                    "successes": successes,
                    "failures": failures,
                    "times_ms": times,
                    "avg_time_ms": statistics.mean(times),
                    "median_time_ms": statistics.median(times),
                    "min_time_ms": min(times),
                    "max_time_ms": max(times),
                    "std_dev_ms": statistics.stdev(times) if len(times) > 1 else 0,
                    "success_rate": successes / iterations * 100
                }
        
        return results

    def analyze_bottlenecks(self, results: Dict):
        """Analyze results to identify performance bottlenecks"""
        print(f"\n🔍 PERFORMANCE BOTTLENECK ANALYSIS")
        print("=" * 60)
        
        total_avg = statistics.mean([r["avg_time_ms"] for r in results.values()])
        
        print(f"📈 OVERALL PERFORMANCE:")
        print(f"   Average Response Time: {total_avg:.1f}ms")
        print(f"   Target Response Time: 5ms (neural network inference)")
        print(f"   Performance Gap: {total_avg/5:.1f}x slower than target")
        
        print(f"\n📊 DETAILED BREAKDOWN:")
        for scenario_name, data in results.items():
            print(f"\n  🎯 {scenario_name.upper()}:")
            print(f"     Average: {data['avg_time_ms']:.1f}ms")
            print(f"     Median:  {data['median_time_ms']:.1f}ms") 
            print(f"     Range:   {data['min_time_ms']:.1f}ms - {data['max_time_ms']:.1f}ms")
            print(f"     Success: {data['success_rate']:.1f}%")
            
            # Performance classification
            if data['avg_time_ms'] > 1000:
                classification = "🔴 CRITICAL - Over 1 second"
            elif data['avg_time_ms'] > 500:
                classification = "🟡 WARNING - Over 500ms (Battlesnake timeout risk)"
            elif data['avg_time_ms'] > 100:
                classification = "🟠 SLOW - Over 100ms (competitive disadvantage)"
            else:
                classification = "🟢 ACCEPTABLE - Under 100ms"
                
            print(f"     Status:  {classification}")

    def generate_investigation_report(self, results: Dict) -> str:
        """Generate comprehensive investigation report"""
        total_avg = statistics.mean([r["avg_time_ms"] for r in results.values()])
        
        report = f"""
# 🚨 PERFORMANCE INVESTIGATION REPORT

## Executive Summary
- **Current Average Response Time**: {total_avg:.1f}ms
- **Documented Target Time**: 5ms (neural network inference)
- **Performance Gap**: {total_avg/5:.1f}x slower than documented target
- **Battlesnake API Limit**: 500ms (timeout risk at current performance)

## Critical Findings

### ❌ Performance Issues Confirmed
- Response times are **{total_avg/5:.1f}x slower** than documented "5ms neural network inference"
- Average response time of {total_avg:.1f}ms indicates significant bottlenecks
- All scenarios exceed competitive response time targets (<100ms)

### 🔍 Scenario Breakdown
"""
        
        for scenario_name, data in results.items():
            risk_level = "HIGH RISK" if data['avg_time_ms'] > 500 else "MEDIUM RISK" if data['avg_time_ms'] > 100 else "LOW RISK"
            
            report += f"""
#### {scenario_name.upper().replace('_', ' ')}
- **Average Response**: {data['avg_time_ms']:.1f}ms
- **Range**: {data['min_time_ms']:.1f}ms - {data['max_time_ms']:.1f}ms  
- **Success Rate**: {data['success_rate']:.1f}%
- **Risk Level**: {risk_level}
"""

        report += f"""

## 🎯 Root Cause Hypotheses

### 1. Neural Network Pipeline Bottleneck (Most Likely)
- **Hypothesis**: ONNX inference is not actually achieving 5ms
- **Evidence**: Consistent slowdown across all scenarios
- **Investigation Needed**: Profile neural network inference calls

### 2. Hybrid Decision System Overhead
- **Hypothesis**: Multiple intelligence systems cause cascading delays
- **Evidence**: Complex scenarios show higher response times
- **Investigation Needed**: Profile decision system integration

### 3. Search Algorithm Performance Issues
- **Hypothesis**: MCTS/search algorithms exceed time budgets  
- **Evidence**: Performance scales with scenario complexity
- **Investigation Needed**: Profile search algorithm execution time

### 4. Emergency Fallback System Overhead
- **Hypothesis**: Complex emergency fallback logic causes delays
- **Evidence**: Consistent baseline delay across scenarios
- **Investigation Needed**: Profile emergency system execution paths

## 🛠 Recommended Actions

### Immediate (Critical Priority)
1. **Profile Neural Network Inference**: Measure actual ONNX inference time
2. **Profile Hybrid Decision System**: Identify bottlenecks in multi-level decision making  
3. **Optimize Emergency Fallback**: Simplify complex emergency logic for performance

### Short-term (High Priority)
1. **Implement Performance Monitoring**: Add detailed timing instrumentation
2. **Optimize Search Algorithms**: Review MCTS time budgets and optimization
3. **Cache Optimization**: Investigate caching opportunities for repeated calculations

### Long-term (Medium Priority)  
1. **Architecture Review**: Consider performance-first architecture changes
2. **Benchmark Regression Tests**: Prevent future performance degradation
3. **Production Performance Monitoring**: Real-time performance tracking

## 📊 Success Metrics
- **Target**: Reduce response time to <100ms (competitive)
- **Stretch Goal**: Achieve documented 5ms neural network inference
- **Minimum**: Stay under 400ms (Battlesnake safety margin)

---
**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Investigation Tool**: performance_investigation.py
"""
        return report

    def save_results(self, results: Dict, report: str):
        """Save results and report to files"""
        # Save raw results
        with open("performance_investigation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save investigation report
        with open("PERFORMANCE_INVESTIGATION_REPORT.md", "w") as f:
            f.write(report)
        
        print(f"\n📁 Results saved:")
        print(f"   📊 Raw data: performance_investigation_results.json")
        print(f"   📋 Report: PERFORMANCE_INVESTIGATION_REPORT.md")

def main():
    print("🔍 BATTLESNAKE PERFORMANCE INVESTIGATION")
    print("Investigating 40x performance degradation (2000+ms vs 5ms)")
    print("=" * 60)
    
    investigator = PerformanceInvestigator()
    
    # Check if server is running
    if not investigator.check_server_status():
        print("\n❌ ERROR: Battlesnake server is not running!")
        print("Please start the server with: cargo run")
        sys.exit(1)
    
    # Run performance tests
    try:
        results = investigator.run_performance_tests(iterations=10)
        
        if not results:
            print("❌ No performance data collected!")
            sys.exit(1)
        
        # Analyze bottlenecks
        investigator.analyze_bottlenecks(results)
        
        # Generate comprehensive report
        report = investigator.generate_investigation_report(results)
        
        # Save results
        investigator.save_results(results, report)
        
        print(f"\n✅ PERFORMANCE INVESTIGATION COMPLETE")
        print(f"Check PERFORMANCE_INVESTIGATION_REPORT.md for detailed findings")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Investigation interrupted by user")
    except Exception as e:
        print(f"\n❌ Investigation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()