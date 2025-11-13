#!/usr/bin/env python3
"""
Enhanced Battlesnake-Specific Performance Validation Metrics
===========================================================

Comprehensive validation framework for measuring neural network performance
in actual Battlesnake gameplay scenarios with domain-specific metrics.

Key Metrics:
- Solo game completion rates
- Survival time analysis 
- Food finding efficiency
- Move quality assessment
- Strategic decision validation
- Neural confidence correlation analysis
"""

import json
import time
import requests
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class GameScenario:
    """Defines a specific game testing scenario"""
    name: str
    description: str
    board_width: int
    board_height: int
    food_count: int
    initial_length: int
    max_turns: int
    opponent_count: int = 0

@dataclass
class GameResult:
    """Results from a single game execution"""
    scenario: str
    survival_turns: int
    max_turns: int
    survival_rate: float
    food_consumed: int
    final_length: int
    length_growth: int
    food_efficiency: float  # food/turn ratio
    move_decisions: List[Dict]
    neural_confidence_avg: float
    neural_confidence_std: float
    decision_sources: Dict[str, int]
    game_completed: bool
    death_cause: Optional[str]
    response_times: List[float]
    avg_response_time: float

class BattlesnakeEnhancedMetrics:
    """Enhanced metrics validator for Battlesnake neural network performance"""
    
    def __init__(self, server_url: str = "http://localhost:8888"):
        self.server_url = server_url
        self.results: List[GameResult] = []
        
        # Define comprehensive test scenarios
        self.scenarios = [
            GameScenario(
                name="solo_basic_survival",
                description="Basic survival in open arena",
                board_width=11, board_height=11,
                food_count=3, initial_length=3, max_turns=200
            ),
            GameScenario(
                name="solo_food_hunting",
                description="Food efficiency optimization",
                board_width=15, board_height=15,
                food_count=8, initial_length=3, max_turns=300
            ),
            GameScenario(
                name="solo_endurance",
                description="Long-term survival test",
                board_width=19, board_height=19,
                food_count=5, initial_length=4, max_turns=500
            ),
            GameScenario(
                name="constrained_space",
                description="Survival in tight spaces",
                board_width=7, board_height=7,
                food_count=2, initial_length=3, max_turns=150
            ),
            GameScenario(
                name="sparse_food",
                description="Food scarcity management",
                board_width=11, board_height=11,
                food_count=1, initial_length=3, max_turns=250
            ),
            GameScenario(
                name="abundant_food",
                description="Growth optimization",
                board_width=15, board_height=15,
                food_count=12, initial_length=3, max_turns=400
            )
        ]
        
    def check_server_health(self) -> bool:
        """Verify server is running and responsive"""
        try:
            response = requests.get(f"{self.server_url}/", timeout=5)
            logger.info(f"✅ Server health check: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"❌ Server health check failed: {e}")
            return False
    
    def create_game_state(self, scenario: GameScenario, turn: int, 
                         snake_body: List[Tuple[int, int]], 
                         food_positions: List[Tuple[int, int]],
                         health: int = 100) -> Dict:
        """Create a standardized game state for testing"""
        return {
            "game": {
                "id": f"enhanced-metrics-{scenario.name}-{int(time.time())}",
                "ruleset": {"name": "standard", "version": "v1.2.3"},
                "timeout": 500,
                "source": "enhanced_metrics"
            },
            "turn": turn,
            "board": {
                "width": scenario.board_width,
                "height": scenario.board_height,
                "food": [{"x": x, "y": y} for x, y in food_positions],
                "snakes": [{
                    "id": "test-snake",
                    "name": "Neural Test Snake",
                    "body": [{"x": x, "y": y} for x, y in snake_body],
                    "head": {"x": snake_body[0][0], "y": snake_body[0][1]},
                    "health": health,
                    "length": len(snake_body),
                    "latency": "0",
                    "shout": ""
                }],
                "hazards": []
            },
            "you": {
                "id": "test-snake",
                "name": "Neural Test Snake", 
                "body": [{"x": x, "y": y} for x, y in snake_body],
                "head": {"x": snake_body[0][0], "y": snake_body[0][1]},
                "health": health,
                "length": len(snake_body),
                "latency": "0",
                "shout": ""
            }
        }
    
    def generate_food_positions(self, scenario: GameScenario, 
                               snake_body: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Generate strategic food positions for testing"""
        occupied = set(snake_body)
        food_positions = []
        
        # Add food at strategic locations
        potential_positions = []
        for x in range(scenario.board_width):
            for y in range(scenario.board_height):
                if (x, y) not in occupied:
                    potential_positions.append((x, y))
        
        # Select food positions with some randomness but avoiding snake
        import random
        random.seed(42)  # Deterministic for reproducibility
        
        if len(potential_positions) >= scenario.food_count:
            food_positions = random.sample(potential_positions, scenario.food_count)
        else:
            food_positions = potential_positions
            
        return food_positions
    
    def execute_move(self, game_state: Dict) -> Tuple[str, Dict]:
        """Execute a single move and return decision + metadata"""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.server_url}/move",
                json=game_state,
                headers={"Content-Type": "application/json"},
                timeout=1.0
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                move_data = response.json()
                return move_data.get("move", "up"), {
                    "response_time": response_time,
                    "confidence": move_data.get("confidence", 0.0),
                    "decision_source": move_data.get("decision_source", "unknown"),
                    "raw_response": move_data
                }
            else:
                logger.warning(f"Move request failed: {response.status_code}")
                return "up", {
                    "response_time": response_time,
                    "confidence": 0.0,
                    "decision_source": "error_fallback",
                    "error": f"HTTP {response.status_code}"
                }
                
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Move execution error: {e}")
            return "up", {
                "response_time": response_time,
                "confidence": 0.0,
                "decision_source": "exception_fallback",
                "error": str(e)
            }
    
    def simulate_move(self, snake_body: List[Tuple[int, int]], 
                     move: str, scenario: GameScenario) -> Tuple[List[Tuple[int, int]], bool]:
        """Simulate snake movement and collision detection"""
        head_x, head_y = snake_body[0]
        
        # Calculate new head position
        if move == "up":
            new_head = (head_x, head_y + 1)
        elif move == "down":
            new_head = (head_x, head_y - 1)
        elif move == "left":
            new_head = (head_x - 1, head_y)
        elif move == "right":
            new_head = (head_x + 1, head_y)
        else:
            new_head = (head_x, head_y + 1)  # Default fallback
        
        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= scenario.board_width or
            new_head[1] < 0 or new_head[1] >= scenario.board_height):
            return snake_body, True  # Wall collision
        
        # Check self collision
        if new_head in snake_body:
            return snake_body, True  # Self collision
        
        # Move snake (without growing)
        new_body = [new_head] + snake_body[:-1]
        return new_body, False
    
    def run_scenario(self, scenario: GameScenario, iterations: int = 5) -> List[GameResult]:
        """Run a specific scenario multiple times"""
        logger.info(f"🎯 Running scenario: {scenario.name}")
        logger.info(f"   Description: {scenario.description}")
        logger.info(f"   Board: {scenario.board_width}x{scenario.board_height}")
        logger.info(f"   Food: {scenario.food_count}, Max turns: {scenario.max_turns}")
        logger.info(f"   Iterations: {iterations}")
        
        scenario_results = []
        
        for iteration in range(iterations):
            logger.info(f"   🔄 Iteration {iteration + 1}/{iterations}")
            
            # Initialize game state
            initial_body = [(scenario.board_width // 2, scenario.board_height // 2)]
            for i in range(scenario.initial_length - 1):
                initial_body.append((initial_body[-1][0], initial_body[-1][1] - 1))
            
            snake_body = initial_body[:]
            food_positions = self.generate_food_positions(scenario, snake_body)
            health = 100
            food_consumed = 0
            move_decisions = []
            response_times = []
            confidences = []
            decision_sources = {}
            
            # Simulate game
            for turn in range(1, scenario.max_turns + 1):
                # Update food positions if consumed
                current_food = [pos for pos in food_positions if pos not in snake_body]
                if len(current_food) < len(food_positions):
                    # Food was consumed
                    food_consumed += len(food_positions) - len(current_food)
                    # Grow snake
                    if len(snake_body) < len(initial_body) + food_consumed:
                        # Add segment to tail
                        tail = snake_body[-1]
                        snake_body.append(tail)
                    
                    # Regenerate food
                    food_positions = self.generate_food_positions(scenario, snake_body)
                    health = min(100, health + 25)  # Health boost from food
                
                # Create game state
                game_state = self.create_game_state(
                    scenario, turn, snake_body, food_positions, health
                )
                
                # Get move decision
                move, metadata = self.execute_move(game_state)
                
                # Record decision data
                move_decisions.append({
                    "turn": turn,
                    "move": move,
                    "confidence": metadata["confidence"],
                    "decision_source": metadata["decision_source"],
                    "response_time": metadata["response_time"]
                })
                
                response_times.append(metadata["response_time"])
                confidences.append(metadata["confidence"])
                
                source = metadata["decision_source"]
                decision_sources[source] = decision_sources.get(source, 0) + 1
                
                # Simulate move
                new_snake_body, collision = self.simulate_move(snake_body, move, scenario)
                
                if collision:
                    # Game over
                    death_cause = "collision"
                    break
                
                snake_body = new_snake_body
                health = max(0, health - 1)  # Hunger
                
                if health <= 0:
                    # Starved
                    death_cause = "starvation"
                    break
                    
                death_cause = None  # Completed successfully
            
            else:
                # Loop completed without break - game finished successfully
                death_cause = None
            
            # Calculate results
            survival_turns = turn if death_cause else scenario.max_turns
            survival_rate = survival_turns / scenario.max_turns
            final_length = len(snake_body)
            length_growth = final_length - scenario.initial_length
            food_efficiency = food_consumed / survival_turns if survival_turns > 0 else 0.0
            
            result = GameResult(
                scenario=scenario.name,
                survival_turns=survival_turns,
                max_turns=scenario.max_turns,
                survival_rate=survival_rate,
                food_consumed=food_consumed,
                final_length=final_length,
                length_growth=length_growth,
                food_efficiency=food_efficiency,
                move_decisions=move_decisions,
                neural_confidence_avg=statistics.mean(confidences) if confidences else 0.0,
                neural_confidence_std=statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
                decision_sources=decision_sources,
                game_completed=(death_cause is None),
                death_cause=death_cause,
                response_times=response_times,
                avg_response_time=statistics.mean(response_times) if response_times else 0.0
            )
            
            scenario_results.append(result)
            
            logger.info(f"      ✅ Survival: {survival_turns}/{scenario.max_turns} turns ({survival_rate:.1%})")
            logger.info(f"      🍎 Food consumed: {food_consumed} (efficiency: {food_efficiency:.3f}/turn)")
            logger.info(f"      📏 Length: {scenario.initial_length} → {final_length} (+{length_growth})")
            logger.info(f"      🧠 Neural confidence: {result.neural_confidence_avg:.3f} ± {result.neural_confidence_std:.3f}")
            logger.info(f"      ⚡ Response time: {result.avg_response_time:.3f}s avg")
        
        return scenario_results
    
    def run_comprehensive_validation(self, iterations_per_scenario: int = 3) -> Dict:
        """Execute comprehensive validation across all scenarios"""
        logger.info("🚀 Starting Enhanced Battlesnake Metrics Validation")
        logger.info("=" * 70)
        
        # Check server health
        if not self.check_server_health():
            raise RuntimeError("Server is not healthy - cannot proceed with validation")
        
        all_results = []
        scenario_summaries = {}
        
        # Run all scenarios
        for scenario in self.scenarios:
            scenario_results = self.run_scenario(scenario, iterations_per_scenario)
            all_results.extend(scenario_results)
            
            # Calculate scenario summary statistics
            survival_rates = [r.survival_rate for r in scenario_results]
            food_efficiencies = [r.food_efficiency for r in scenario_results]
            confidences = [r.neural_confidence_avg for r in scenario_results]
            response_times = [r.avg_response_time for r in scenario_results]
            completion_rate = sum(1 for r in scenario_results if r.game_completed) / len(scenario_results)
            
            scenario_summaries[scenario.name] = {
                "scenario_info": asdict(scenario),
                "iterations": len(scenario_results),
                "completion_rate": completion_rate,
                "survival_rate": {
                    "mean": statistics.mean(survival_rates),
                    "std": statistics.stdev(survival_rates) if len(survival_rates) > 1 else 0.0,
                    "min": min(survival_rates),
                    "max": max(survival_rates)
                },
                "food_efficiency": {
                    "mean": statistics.mean(food_efficiencies),
                    "std": statistics.stdev(food_efficiencies) if len(food_efficiencies) > 1 else 0.0,
                    "min": min(food_efficiencies),
                    "max": max(food_efficiencies)
                },
                "neural_confidence": {
                    "mean": statistics.mean(confidences),
                    "std": statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
                    "min": min(confidences),
                    "max": max(confidences)
                },
                "response_time": {
                    "mean": statistics.mean(response_times),
                    "std": statistics.stdev(response_times) if len(response_times) > 1 else 0.0,
                    "min": min(response_times),
                    "max": max(response_times)
                }
            }
        
        # Calculate overall metrics
        overall_completion_rate = sum(1 for r in all_results if r.game_completed) / len(all_results)
        overall_survival_rate = statistics.mean([r.survival_rate for r in all_results])
        overall_food_efficiency = statistics.mean([r.food_efficiency for r in all_results])
        overall_neural_confidence = statistics.mean([r.neural_confidence_avg for r in all_results])
        overall_response_time = statistics.mean([r.avg_response_time for r in all_results])
        
        # Compile comprehensive report
        validation_report = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "total_scenarios": len(self.scenarios),
                "total_games": len(all_results),
                "iterations_per_scenario": iterations_per_scenario,
                "server_url": self.server_url
            },
            "overall_metrics": {
                "game_completion_rate": overall_completion_rate,
                "average_survival_rate": overall_survival_rate,
                "average_food_efficiency": overall_food_efficiency,
                "average_neural_confidence": overall_neural_confidence,
                "average_response_time": overall_response_time
            },
            "scenario_results": scenario_summaries,
            "detailed_results": [asdict(result) for result in all_results]
        }
        
        # Save results
        with open("battlesnake_enhanced_metrics.json", "w") as f:
            json.dump(validation_report, f, indent=2)
        
        # Print summary
        self.print_validation_summary(validation_report)
        
        return validation_report
    
    def print_validation_summary(self, report: Dict):
        """Print comprehensive validation summary"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 ENHANCED BATTLESNAKE VALIDATION SUMMARY")
        logger.info("=" * 70)
        
        overall = report["overall_metrics"]
        logger.info(f"🎮 Total Games Executed: {report['metadata']['total_games']}")
        logger.info(f"🏁 Game Completion Rate: {overall['game_completion_rate']:.1%}")
        logger.info(f"⏱️ Average Survival Rate: {overall['average_survival_rate']:.1%}")
        logger.info(f"🍎 Average Food Efficiency: {overall['average_food_efficiency']:.3f} food/turn")
        logger.info(f"🧠 Average Neural Confidence: {overall['average_neural_confidence']:.3f}")
        logger.info(f"⚡ Average Response Time: {overall['average_response_time']:.3f}s")
        
        logger.info("\n📈 SCENARIO BREAKDOWN:")
        logger.info("-" * 70)
        
        for scenario_name, results in report["scenario_results"].items():
            logger.info(f"\n🎯 {scenario_name.replace('_', ' ').title()}:")
            logger.info(f"   Completion Rate: {results['completion_rate']:.1%}")
            logger.info(f"   Survival Rate: {results['survival_rate']['mean']:.1%} ± {results['survival_rate']['std']:.1%}")
            logger.info(f"   Food Efficiency: {results['food_efficiency']['mean']:.3f} ± {results['food_efficiency']['std']:.3f}")
            logger.info(f"   Neural Confidence: {results['neural_confidence']['mean']:.3f} ± {results['neural_confidence']['std']:.3f}")
            logger.info(f"   Response Time: {results['response_time']['mean']:.3f}s ± {results['response_time']['std']:.3f}s")
        
        # Performance classification
        if overall['game_completion_rate'] >= 0.9:
            performance_class = "🌟 EXCELLENT"
        elif overall['game_completion_rate'] >= 0.7:
            performance_class = "✅ GOOD"
        elif overall['game_completion_rate'] >= 0.5:
            performance_class = "⚠️ ACCEPTABLE"
        else:
            performance_class = "❌ NEEDS IMPROVEMENT"
        
        logger.info(f"\n🏆 OVERALL PERFORMANCE: {performance_class}")
        logger.info(f"💾 Detailed results saved to: battlesnake_enhanced_metrics.json")
        logger.info("=" * 70)

def main():
    """Main execution function"""
    metrics_validator = BattlesnakeEnhancedMetrics()
    
    try:
        validation_report = metrics_validator.run_comprehensive_validation(
            iterations_per_scenario=3  # Adjust based on time constraints
        )
        
        logger.info("✅ Enhanced validation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced validation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)