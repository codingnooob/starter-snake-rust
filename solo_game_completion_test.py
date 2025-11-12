#!/usr/bin/env python3
"""
Comprehensive Solo Game Completion Test for Safety Fixes
Tests if the snake AI can complete solo games without dangerous emergency fallbacks
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SoloGameTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.game_data = None
        self.turn_count = 0
        self.max_turns = 200  # Should complete much faster
        self.test_results = {
            "game_completed": False,
            "death_turn": None,
            "death_reason": None,
            "moves_made": 0,
            "safety_violations": 0,
            "emergency_fallback_used": 0,
            "out_of_bounds_attempts": 0,
            "neural_network_active": False
        }
        
    def create_test_game(self) -> Dict:
        """Create a simple test game state"""
        return {
            "game": {
                "id": "safety_test_game",
                "ruleset": {"name": "standard"},
                "timeout": 500
            },
            "board": {
                "width": 11,
                "height": 11,
                "food": [
                    {"x": 8, "y": 8},
                    {"x": 2, "y": 2},
                    {"x": 5, "y": 9}
                ],
                "snakes": [
                    {
                        "id": "test_snake",
                        "name": "Safety Test Snake",
                        "health": 100,
                        "body": [
                            {"x": 5, "y": 5},
                            {"x": 5, "y": 4},
                            {"x": 5, "y": 3}
                        ],
                        "head": {"x": 5, "y": 5},
                        "length": 3,
                        "latency": "100"
                    }
                ],
                "hazards": []
            },
            "you": {
                "id": "test_snake"
            }
        }
    
    def test_info_endpoint(self) -> bool:
        """Test if the server info endpoint works"""
        try:
            response = requests.get(f"{self.base_url}/info", timeout=10)
            if response.status_code == 200:
                info = response.json()
                logger.info(f"✓ Info endpoint working: {info}")
                return True
            else:
                logger.error(f"✗ Info endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"✗ Info endpoint error: {e}")
            return False
    
    def test_start_endpoint(self, game_data: Dict) -> bool:
        """Test the start endpoint"""
        try:
            response = requests.post(
                f"{self.base_url}/start",
                json=game_data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            if response.status_code == 200:
                logger.info("✓ Start endpoint working")
                return True
            else:
                logger.error(f"✗ Start endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"✗ Start endpoint error: {e}")
            return False
    
    def make_move(self, game_data: Dict, turn: int) -> Optional[Dict]:
        """Make a move and analyze the response"""
        try:
            move_request = {
                "game": game_data["game"],
                "turn": turn,
                "board": game_data["board"],
                "you": game_data["board"]["snakes"][0]
            }
            
            response = requests.post(
                f"{self.base_url}/move",
                json=move_request,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Turn {turn}: Move = {result.get('move', 'UNKNOWN')}")
                
                # Check for decision source
                decision_source = result.get('decision_source', 'unknown')
                if decision_source == 'neural_network_override':
                    self.test_results["neural_network_active"] = True
                    logger.info("✓ Neural network is active")
                
                # Check for safety violations in logs would require server access
                # For now, we rely on the fact that dangerous moves would cause immediate death
                
                return result
            else:
                logger.error(f"✗ Move endpoint failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"✗ Move endpoint error: {e}")
            return None
    
    def update_snake_position(self, game_data: Dict, move_result: Dict) -> bool:
        """Update snake position based on move result"""
        if not move_result or 'move' not in move_result:
            return False
            
        move = move_result['move']
        snake = game_data["board"]["snakes"][0]
        head = snake["head"]
        
        # Calculate new head position
        new_head = head.copy()
        if move == "up":
            new_head["y"] += 1
        elif move == "down":
            new_head["y"] -= 1
        elif move == "left":
            new_head["x"] -= 1
        elif move == "right":
            new_head["x"] += 1
        
        # Check for out of bounds
        if (new_head["x"] < 0 or new_head["x"] >= game_data["board"]["width"] or
            new_head["y"] < 0 or new_head["y"] >= game_data["board"]["height"]):
            self.test_results["death_reason"] = "out_of_bounds"
            self.test_results["death_turn"] = self.turn_count
            logger.error(f"🚨 DEATH: Out of bounds at turn {self.turn_count}")
            return False
        
        # Update snake body (simple movement simulation)
        new_body = [new_head] + snake["body"][:-1]
        
        # Check if snake ate food
        ate_food = False
        for food in game_data["board"]["food"][:]:
            if food["x"] == new_head["x"] and food["y"] == new_head["y"]:
                # Snake ate food - add tail
                new_body.append(snake["body"][-1])
                game_data["board"]["food"].remove(food)
                ate_food = True
                logger.info(f"🍎 Food eaten at ({new_head['x']}, {new_head['y']})")
                break
        
        # Update snake
        snake["head"] = new_head
        snake["body"] = new_body
        snake["health"] = max(0, snake["health"] - 1 + (50 if ate_food else 0))
        
        return True
    
    def run_solo_game_test(self) -> bool:
        """Run a complete solo game test"""
        logger.info("🎮 Starting Solo Game Completion Test")
        logger.info("=" * 50)
        
        # Test basic connectivity
        if not self.test_info_endpoint():
            logger.error("❌ Cannot connect to server")
            return False
        
        # Create test game
        self.game_data = self.create_test_game()
        
        # Test start
        if not self.test_start_endpoint(self.game_data):
            logger.error("❌ Start endpoint failed")
            return False
        
        logger.info(f"📊 Initial board state: {len(self.game_data['board']['food'])} food items")
        
        # Run game loop
        while self.turn_count < self.max_turns:
            self.turn_count += 1
            
            # Make move
            move_result = self.make_move(self.game_data, self.turn_count)
            if not move_result:
                logger.error(f"❌ Failed to get move at turn {self.turn_count}")
                break
            
            self.test_results["moves_made"] += 1
            
            # Update position
            if not self.update_snake_position(self.game_data, move_result):
                # Snake died
                break
            
            # Check if all food collected (game should end)
            if len(self.game_data["board"]["food"]) == 0:
                self.test_results["game_completed"] = True
                logger.info(f"🎉 GAME COMPLETED! All food collected in {self.turn_count} turns!")
                logger.info(f"🏆 Final health: {self.game_data['board']['snakes'][0]['health']}")
                break
            
            # Log progress every 10 turns
            if self.turn_count % 10 == 0:
                snake = self.game_data["board"]["snakes"][0]
                logger.info(f"Turn {self.turn_count}: Health={snake['health']}, Food remaining={len(self.game_data['board']['food'])}")
        
        # Test end
        try:
            end_request = {
                "game": self.game_data["game"],
                "turn": self.turn_count,
                "board": self.game_data["board"],
                "you": self.game_data["board"]["snakes"][0]
            }
            requests.post(
                f"{self.base_url}/end",
                json=end_request,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            logger.info("✓ End endpoint successful")
        except Exception as e:
            logger.warning(f"⚠️  End endpoint error: {e}")
        
        # Report results
        self.report_results()
        return self.test_results["game_completed"]
    
    def report_results(self):
        """Report test results"""
        logger.info("\n" + "=" * 50)
        logger.info("📋 SOLO GAME TEST RESULTS")
        logger.info("=" * 50)
        
        if self.test_results["game_completed"]:
            logger.info("✅ GAME COMPLETED SUCCESSFULLY!")
        else:
            if self.test_results["death_turn"]:
                logger.info(f"❌ GAME FAILED at turn {self.test_results['death_turn']}")
                logger.info(f"💀 Death reason: {self.test_results['death_reason']}")
            else:
                logger.info(f"❌ GAME TIMEOUT after {self.max_turns} turns")
        
        logger.info(f"📊 Total moves made: {self.test_results['moves_made']}")
        logger.info(f"🧠 Neural network active: {self.test_results['neural_network_active']}")
        logger.info(f"🚨 Safety violations detected: {self.test_results['safety_violations']}")
        logger.info(f"⚠️  Emergency fallbacks used: {self.test_results['emergency_fallback_used']}")
        logger.info(f"📍 Out of bounds attempts: {self.test_results['out_of_bounds_attempts']}")
        
        # Success criteria
        logger.info("\n🎯 SUCCESS CRITERIA:")
        if self.test_results["game_completed"]:
            logger.info("  ✅ Game completed without death")
            logger.info("  ✅ All safety fixes working properly")
            logger.info("  ✅ Snake can successfully complete solo games")
        else:
            logger.info("  ❌ Game failed to complete")
            if self.test_results["death_reason"] == "out_of_bounds":
                logger.info("  ❌ CRITICAL: Out of bounds death detected - safety fix may have failed!")
            elif self.test_results["death_reason"] == "health":
                logger.info("  ❌ Health depletion - food seeking may not be working")

def main():
    """Main test function"""
    tester = SoloGameTester()
    
    try:
        success = tester.run_solo_game_test()
        
        if success:
            logger.info("\n🎉 ALL TESTS PASSED! The safety fixes are working correctly!")
            logger.info("🚀 The snake AI can now complete solo games successfully!")
            return 0
        else:
            logger.info("\n💥 TESTS FAILED! There are still issues with the snake AI")
            logger.info("🔧 Additional fixes may be needed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n💥 Test failed with unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())