#!/usr/bin/env python3
"""
Component-Level Performance Profiling for 12-Channel Encoding Pipeline
Measures performance of individual components in advanced spatial analysis
"""

import sys
import time
import json
import numpy as np
import cProfile
import pstats
import io
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import gc
import tracemalloc

# Add neural_networks to path
sys.path.append('neural_networks')
from advanced_board_encoding import (
    Advanced12ChannelBoardEncoder, 
    GameState,
    VoronoiTerritoryAnalyzer,
    DangerZonePredictor,
    MovementHistoryTracker,
    StrategicPositionAnalyzer
)

class ComponentProfiler:
    """Profiles individual components of the 12-channel encoding pipeline"""
    
    def __init__(self):
        self.encoder = Advanced12ChannelBoardEncoder()
        self.test_game_states = self._create_test_scenarios()
        self.profiling_results = {}
        
    def _create_test_scenarios(self) -> List[GameState]:
        """Create various test scenarios for comprehensive profiling"""
        scenarios = []
        
        # Simple scenario - minimal complexity
        scenarios.append(GameState(
            board_width=11,
            board_height=11,
            our_snake={
                'body': [{'x': 5, 'y': 5}, {'x': 5, 'y': 6}],
                'health': 100,
                'id': 'test-us'
            },
            opponent_snakes=[],
            food=[{'x': 8, 'y': 8}],
            turn=1,
            game_id='simple_test'
        ))
        
        # Medium complexity - multiple opponents and food
        scenarios.append(GameState(
            board_width=11,
            board_height=11,
            our_snake={
                'body': [{'x': 5, 'y': 5}, {'x': 5, 'y': 4}, {'x': 6, 'y': 4}, {'x': 7, 'y': 4}],
                'health': 85,
                'id': 'test-us'
            },
            opponent_snakes=[
                {
                    'body': [{'x': 9, 'y': 8}, {'x': 9, 'y': 7}, {'x': 10, 'y': 7}],
                    'health': 75,
                    'id': 'opponent-1'
                },
                {
                    'body': [{'x': 2, 'y': 2}, {'x': 1, 'y': 2}],
                    'health': 60,
                    'id': 'opponent-2'
                }
            ],
            food=[{'x': 3, 'y': 3}, {'x': 8, 'y': 7}, {'x': 1, 'y': 9}],
            turn=25,
            game_id='medium_test'
        ))
        
        # Complex scenario - multiple long snakes, lots of food
        scenarios.append(GameState(
            board_width=11,
            board_height=11,
            our_snake={
                'body': [{'x': 5, 'y': 5}, {'x': 5, 'y': 4}, {'x': 5, 'y': 3}, {'x': 6, 'y': 3}, 
                        {'x': 7, 'y': 3}, {'x': 8, 'y': 3}, {'x': 8, 'y': 4}, {'x': 8, 'y': 5}],
                'health': 90,
                'id': 'test-us'
            },
            opponent_snakes=[
                {
                    'body': [{'x': 1, 'y': 1}, {'x': 1, 'y': 2}, {'x': 1, 'y': 3}, {'x': 2, 'y': 3}, 
                            {'x': 3, 'y': 3}, {'x': 3, 'y': 4}, {'x': 3, 'y': 5}],
                    'health': 80,
                    'id': 'opponent-1'
                },
                {
                    'body': [{'x': 9, 'y': 9}, {'x': 9, 'y': 8}, {'x': 8, 'y': 8}, {'x': 7, 'y': 8}, 
                            {'x': 6, 'y': 8}, {'x': 6, 'y': 9}],
                    'health': 70,
                    'id': 'opponent-2'
                },
                {
                    'body': [{'x': 0, 'y': 8}, {'x': 0, 'y': 7}, {'x': 0, 'y': 6}, {'x': 1, 'y': 6}],
                    'health': 65,
                    'id': 'opponent-3'
                }
            ],
            food=[{'x': 4, 'y': 1}, {'x': 7, 'y': 2}, {'x': 10, 'y': 4}, {'x': 2, 'y': 7}, 
                  {'x': 6, 'y': 10}, {'x': 9, 'y': 0}],
            turn=150,
            game_id='complex_test'
        ))
        
        return scenarios
    
    def profile_component_timing(self, component_name: str, func, *args) -> Dict:
        """Profile timing for a specific component"""
        times = []
        memory_usage = []
        
        # Warm up
        for _ in range(3):
            try:
                func(*args)
            except Exception:
                pass
        
        # Profile multiple runs
        for i in range(20):
            gc.collect()  # Clean up before measurement
            
            # Memory tracking
            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            
            # Time measurement
            start_time = time.perf_counter()
            try:
                result = func(*args)
                success = True
            except Exception as e:
                success = False
                result = None
            
            end_time = time.perf_counter()
            
            # Memory measurement
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            execution_time = (end_time - start_time) * 1000  # Convert to ms
            memory_delta = (current_memory - start_memory) / 1024  # Convert to KB
            
            times.append(execution_time)
            memory_usage.append(memory_delta)
            
            if not success:
                break
        
        # Calculate statistics
        if times:
            return {
                "component": component_name,
                "success": success,
                "timing": {
                    "average_ms": np.mean(times),
                    "median_ms": np.median(times),
                    "min_ms": np.min(times),
                    "max_ms": np.max(times),
                    "std_ms": np.std(times),
                    "p95_ms": np.percentile(times, 95),
                    "runs": len(times)
                },
                "memory": {
                    "average_kb": np.mean(memory_usage),
                    "peak_kb": np.max(memory_usage),
                    "min_kb": np.min(memory_usage)
                }
            }
        else:
            return {
                "component": component_name,
                "success": False,
                "error": "No successful runs"
            }
    
    def profile_voronoi_analyzer(self, game_state: GameState) -> Dict:
        """Profile Voronoi territory analysis component"""
        analyzer = VoronoiTerritoryAnalyzer(11, 11)
        
        # Extract positions
        our_head = (game_state.our_snake['body'][0]['x'], game_state.our_snake['body'][0]['y'])
        opponent_heads = [
            (snake['body'][0]['x'], snake['body'][0]['y']) 
            for snake in game_state.opponent_snakes
        ]
        
        return self.profile_component_timing(
            "VoronoiTerritoryAnalyzer",
            analyzer.calculate_territory_control,
            our_head, opponent_heads
        )
    
    def profile_danger_predictor(self, game_state: GameState) -> Dict:
        """Profile danger zone prediction component"""
        predictor = DangerZonePredictor(11, 11)
        
        # Extract snake bodies
        our_body = [(seg['x'], seg['y']) for seg in game_state.our_snake['body']]
        opponent_bodies = [
            [(seg['x'], seg['y']) for seg in snake['body']]
            for snake in game_state.opponent_snakes
        ]
        
        return self.profile_component_timing(
            "DangerZonePredictor",
            predictor.calculate_danger_zones,
            our_body, opponent_bodies, game_state.turn
        )
    
    def profile_movement_tracker(self, game_state: GameState) -> Dict:
        """Profile movement history tracking component"""
        tracker = MovementHistoryTracker(11, 11)
        
        # Simulate some history
        our_head = (game_state.our_snake['body'][0]['x'], game_state.our_snake['body'][0]['y'])
        opponent_heads = [
            (snake['body'][0]['x'], snake['body'][0]['y']) 
            for snake in game_state.opponent_snakes
        ]
        
        # Add some history entries
        for turn in range(max(1, game_state.turn - 5), game_state.turn):
            tracker.update_history(our_head, opponent_heads, turn)
        
        return self.profile_component_timing(
            "MovementHistoryTracker",
            tracker.generate_movement_history_channel
        )
    
    def profile_strategic_analyzer(self, game_state: GameState) -> Dict:
        """Profile strategic position analysis component"""
        analyzer = StrategicPositionAnalyzer(11, 11)
        
        our_head = (game_state.our_snake['body'][0]['x'], game_state.our_snake['body'][0]['y'])
        opponent_heads = [
            (snake['body'][0]['x'], snake['body'][0]['y']) 
            for snake in game_state.opponent_snakes
        ]
        food_positions = [(food['x'], food['y']) for food in game_state.food]
        
        return self.profile_component_timing(
            "StrategicPositionAnalyzer",
            analyzer.calculate_strategic_positions,
            our_head, opponent_heads, food_positions
        )
    
    def profile_full_encoding(self, game_state: GameState) -> Dict:
        """Profile the complete 12-channel encoding process"""
        return self.profile_component_timing(
            "Full12ChannelEncoding",
            self.encoder.encode_game_state,
            game_state
        )
    
    def profile_detailed_execution(self, game_state: GameState) -> Dict:
        """Use cProfile for detailed function-level profiling"""
        pr = cProfile.Profile()
        
        # Profile the encoding
        pr.enable()
        try:
            board_tensor, snake_features, game_context = self.encoder.encode_game_state(game_state)
            success = True
        except Exception as e:
            success = False
        pr.disable()
        
        # Capture profiling results
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        profile_output = s.getvalue()
        
        # Extract key metrics
        total_time = ps.total_tt
        function_count = ps.total_calls
        
        return {
            "component": "DetailedProfiling",
            "success": success,
            "total_time_seconds": total_time,
            "total_function_calls": function_count,
            "calls_per_second": function_count / total_time if total_time > 0 else 0,
            "profile_output": profile_output[:2000],  # Truncate for readability
            "top_functions": self._extract_top_functions(ps, 10)
        }
    
    def _extract_top_functions(self, ps: pstats.Stats, count: int) -> List[Dict]:
        """Extract top functions from profiling stats"""
        try:
            stats = ps.stats
            
            # Sort by cumulative time
            sorted_stats = sorted(stats.items(), key=lambda x: x[1][3], reverse=True)
            
            top_functions = []
            for i, (func, (cc, nc, tt, ct, callers)) in enumerate(sorted_stats[:count]):
                filename, line_num, func_name = func
                top_functions.append({
                    "rank": i + 1,
                    "function": f"{filename}:{line_num}({func_name})",
                    "calls": nc,
                    "total_time": tt,
                    "cumulative_time": ct,
                    "time_per_call": tt / nc if nc > 0 else 0
                })
            
            return top_functions
        except Exception as e:
            return [{"error": f"Failed to extract functions: {e}"}]
    
    def run_comprehensive_profiling(self) -> Dict:
        """Run complete component profiling across all scenarios"""
        print("🔍 Starting Component-Level Performance Profiling...")
        print("=" * 70)
        
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "scenarios": {},
            "component_averages": {},
            "summary": {}
        }
        
        # Profile each scenario
        for i, scenario in enumerate(self.test_game_states):
            scenario_name = f"scenario_{i+1}_{scenario.game_id}"
            print(f"\n🎯 Profiling {scenario_name}...")
            print(f"   Turn: {scenario.turn}, Opponents: {len(scenario.opponent_snakes)}, Food: {len(scenario.food)}")
            
            scenario_results = {}
            
            # Profile individual components
            print("   🔧 Voronoi Territory Analysis...")
            scenario_results["voronoi"] = self.profile_voronoi_analyzer(scenario)
            
            print("   🔧 Danger Zone Prediction...")
            scenario_results["danger_zones"] = self.profile_danger_predictor(scenario)
            
            print("   🔧 Movement History Tracking...")
            scenario_results["movement_history"] = self.profile_movement_tracker(scenario)
            
            print("   🔧 Strategic Position Analysis...")
            scenario_results["strategic_analysis"] = self.profile_strategic_analyzer(scenario)
            
            print("   🔧 Full 12-Channel Encoding...")
            scenario_results["full_encoding"] = self.profile_full_encoding(scenario)
            
            print("   🔧 Detailed Function Profiling...")
            scenario_results["detailed_profiling"] = self.profile_detailed_execution(scenario)
            
            results["scenarios"][scenario_name] = scenario_results
            
            # Print scenario summary
            full_encoding = scenario_results["full_encoding"]
            if full_encoding.get("success"):
                print(f"   📊 Full encoding: {full_encoding['timing']['average_ms']:.2f}ms avg")
        
        # Calculate component averages across all scenarios
        self._calculate_component_averages(results)
        
        # Generate summary
        self._generate_performance_summary(results)
        
        # Print summary
        print(f"\n{'='*70}")
        print("📊 Component Performance Profiling Summary:")
        print("=" * 70)
        
        summary = results["summary"]
        print(f"✅ Scenarios Tested: {len(self.test_game_states)}")
        print(f"🔧 Components Profiled: {len(results['component_averages'])}")
        
        if "full_encoding_avg_ms" in summary:
            print(f"⚡ Full 12-Channel Encoding: {summary['full_encoding_avg_ms']:.2f}ms average")
            print(f"🧠 Advanced Components Total: {summary['advanced_components_total_ms']:.2f}ms")
            print(f"📈 Advanced Component Overhead: {summary['advanced_overhead_percent']:.1f}%")
        
        if "bottleneck_component" in summary:
            print(f"🐌 Slowest Component: {summary['bottleneck_component']} ({summary['bottleneck_time_ms']:.2f}ms)")
        
        print(f"🎯 Performance Classification: {summary.get('performance_class', 'Unknown')}")
        
        # Save detailed results
        report_file = "component_performance_report.json"
        with open(report_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        return results
    
    def _calculate_component_averages(self, results: Dict):
        """Calculate average performance metrics across all scenarios"""
        component_names = ["voronoi", "danger_zones", "movement_history", "strategic_analysis", "full_encoding"]
        
        for component in component_names:
            times = []
            memory_usage = []
            
            for scenario_results in results["scenarios"].values():
                if component in scenario_results and scenario_results[component].get("success"):
                    comp_result = scenario_results[component]
                    times.append(comp_result["timing"]["average_ms"])
                    memory_usage.append(comp_result["memory"]["average_kb"])
            
            if times:
                results["component_averages"][component] = {
                    "average_time_ms": np.mean(times),
                    "min_time_ms": np.min(times),
                    "max_time_ms": np.max(times),
                    "std_time_ms": np.std(times),
                    "average_memory_kb": np.mean(memory_usage),
                    "scenarios_tested": len(times)
                }
    
    def _generate_performance_summary(self, results: Dict):
        """Generate high-level performance summary"""
        averages = results["component_averages"]
        
        if "full_encoding" in averages:
            full_encoding_time = averages["full_encoding"]["average_time_ms"]
            
            # Calculate advanced components overhead
            advanced_components = ["voronoi", "danger_zones", "movement_history", "strategic_analysis"]
            advanced_total = sum(
                averages.get(comp, {}).get("average_time_ms", 0) 
                for comp in advanced_components
            )
            
            # Find bottleneck
            bottleneck_component = None
            bottleneck_time = 0
            for comp in advanced_components:
                if comp in averages:
                    comp_time = averages[comp]["average_time_ms"]
                    if comp_time > bottleneck_time:
                        bottleneck_time = comp_time
                        bottleneck_component = comp
            
            # Performance classification
            if full_encoding_time < 10:
                perf_class = "EXCELLENT (< 10ms)"
            elif full_encoding_time < 25:
                perf_class = "GOOD (< 25ms)" 
            elif full_encoding_time < 50:
                perf_class = "ACCEPTABLE (< 50ms)"
            else:
                perf_class = "NEEDS OPTIMIZATION (> 50ms)"
            
            results["summary"] = {
                "full_encoding_avg_ms": full_encoding_time,
                "advanced_components_total_ms": advanced_total,
                "advanced_overhead_percent": (advanced_total / full_encoding_time * 100) if full_encoding_time > 0 else 0,
                "bottleneck_component": bottleneck_component,
                "bottleneck_time_ms": bottleneck_time,
                "performance_class": perf_class,
                "recommendation": "READY FOR PRODUCTION" if full_encoding_time < 50 else "OPTIMIZATION RECOMMENDED"
            }

def main():
    """Main profiling entry point"""
    profiler = ComponentProfiler()
    results = profiler.run_comprehensive_profiling()
    
    # Return success/failure based on performance
    summary = results.get("summary", {})
    if summary.get("recommendation") == "READY FOR PRODUCTION":
        return 0
    else:
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())