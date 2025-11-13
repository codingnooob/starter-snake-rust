#!/usr/bin/env python3
"""
Post-Training Validation Diagnostic Script
==========================================
Enhanced debugging version to identify why games report 0 turns when server logs show they're running
"""

import subprocess
import time
import json
import requests
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'diagnostic_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DiagnosticGameResult:
    game_id: str
    turns: int
    winner: Optional[str]
    raw_cli_output: str
    raw_cli_error: str
    exit_code: int
    execution_time: float
    parsing_errors: List[str]

class DiagnosticValidationSystem:
    """Enhanced diagnostic version of validation system with extensive logging"""
    
    def __init__(self):
        self.port = 8001
        self.server_url = f"http://localhost:{self.port}"
        self.results: List[DiagnosticGameResult] = []
        
    def validate_server_health(self) -> bool:
        """Check server health with detailed diagnostics"""
        logger.info("🔍 DIAGNOSTIC: Checking server health...")
        
        try:
            # Test basic connectivity
            response = requests.get(f"{self.server_url}/", timeout=10)
            logger.debug(f"Server status code: {response.status_code}")
            logger.debug(f"Server response: {response.text}")
            
            if response.status_code == 200:
                logger.info("✅ Server connectivity: GOOD")
            else:
                logger.warning(f"⚠️ Server returned status {response.status_code}")
            
            # Test move endpoint with minimal payload
            test_payload = {
                "game": {"id": "diagnostic-test", "turn": 1, "ruleset": {"name": "standard"}},
                "board": {"height": 11, "width": 11, "food": [], "snakes": []},
                "you": {"id": "diagnostic", "health": 100, "body": [{"x": 5, "y": 5}]}
            }
            
            logger.debug(f"Testing move endpoint with payload: {json.dumps(test_payload)}")
            move_response = requests.post(f"{self.server_url}/move", 
                                        json=test_payload, 
                                        timeout=10)
            
            logger.debug(f"Move endpoint status: {move_response.status_code}")
            logger.debug(f"Move endpoint response: {move_response.text}")
            
            if move_response.status_code == 422:
                logger.error("🚨 CRITICAL: Server rejecting move requests with 422 - JSON format issue!")
                logger.error("This is likely the root cause of game failures")
                return False
            elif move_response.status_code == 200:
                logger.info("✅ Move endpoint: WORKING")
                return True
            else:
                logger.warning(f"⚠️ Move endpoint returned unexpected status: {move_response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Server health check failed: {e}")
            return False
    
    def run_diagnostic_game(self, game_config: Dict[str, Any]) -> DiagnosticGameResult:
        """Run a single game with comprehensive diagnostics"""
        game_id = f"diagnostic_{game_config['name']}"
        logger.info(f"🎮 DIAGNOSTIC: Running game {game_id}")
        
        # Build battlesnake command
        cmd = [
            "battlesnake", "play",
            "-W", str(game_config['width']),
            "-H", str(game_config['height']),
            "--name", "Post-Training Diagnostic",
            "--url", self.server_url,
            "-g", game_config['mode'],
            "--timeout", "1000"
        ]
        
        if game_config.get('num_snakes', 1) > 1:
            cmd.extend(["--num", str(game_config['num_snakes'])])
        
        logger.debug(f"Command: {' '.join(cmd)}")
        
        # Execute with detailed capture
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minutes max per game
                cwd=Path.cwd()
            )
            
            execution_time = time.time() - start_time
            
            logger.debug(f"Game completed in {execution_time:.2f}s")
            logger.debug(f"Exit code: {result.returncode}")
            logger.debug(f"STDOUT length: {len(result.stdout)} chars")
            logger.debug(f"STDERR length: {len(result.stderr)} chars")
            
            # Log raw output for analysis
            if result.stdout:
                logger.debug("=== RAW STDOUT START ===")
                for i, line in enumerate(result.stdout.split('\n')[:20], 1):  # First 20 lines
                    logger.debug(f"STDOUT[{i:2d}]: {line}")
                if len(result.stdout.split('\n')) > 20:
                    logger.debug(f"... (truncated, total lines: {len(result.stdout.split('\n'))})")
                logger.debug("=== RAW STDOUT END ===")
            
            if result.stderr:
                logger.debug("=== RAW STDERR START ===")
                for i, line in enumerate(result.stderr.split('\n')[:10], 1):  # First 10 lines
                    logger.debug(f"STDERR[{i:2d}]: {line}")
                logger.debug("=== RAW STDERR END ===")
            
            # Parse game results with diagnostics
            turns, parsing_errors = self.parse_game_output_with_diagnostics(result.stdout, result.stderr)
            
            return DiagnosticGameResult(
                game_id=game_id,
                turns=turns,
                winner=self.extract_winner(result.stdout),
                raw_cli_output=result.stdout,
                raw_cli_error=result.stderr,
                exit_code=result.returncode,
                execution_time=execution_time,
                parsing_errors=parsing_errors
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            logger.error(f"❌ Game {game_id} timed out after {execution_time:.2f}s")
            return DiagnosticGameResult(
                game_id=game_id,
                turns=0,
                winner=None,
                raw_cli_output="",
                raw_cli_error="Game timed out",
                exit_code=-1,
                execution_time=execution_time,
                parsing_errors=["Timeout"]
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Game {game_id} failed with exception: {e}")
            return DiagnosticGameResult(
                game_id=game_id,
                turns=0,
                winner=None,
                raw_cli_output="",
                raw_cli_error=str(e),
                exit_code=-2,
                execution_time=execution_time,
                parsing_errors=[f"Exception: {e}"]
            )
    
    def parse_game_output_with_diagnostics(self, stdout: str, stderr: str) -> tuple[int, List[str]]:
        """Parse game output with comprehensive error tracking"""
        parsing_errors = []
        turns = 0
        
        logger.debug("🔍 DIAGNOSTIC: Parsing game output...")
        
        # Strategy 1: Look for turn indicators in output
        turn_patterns = [
            r"Turn (\d+)",
            r"turn[:\s]+(\d+)",
            r"MOVE (\d+):",
            r"Game lasted (\d+) turns",
            r"Finished after (\d+) turns"
        ]
        
        import re
        for pattern in turn_patterns:
            matches = re.findall(pattern, stdout, re.IGNORECASE)
            if matches:
                try:
                    max_turn = max(int(m) for m in matches)
                    logger.debug(f"Pattern '{pattern}' found max turn: {max_turn}")
                    turns = max(turns, max_turn)
                except ValueError as e:
                    parsing_errors.append(f"Turn parsing error for pattern '{pattern}': {e}")
        
        # Strategy 2: Count game state changes
        game_events = stdout.count("GAME START") + stdout.count("Game Started")
        move_events = stdout.count("POST /move") + stdout.count("move:")
        end_events = stdout.count("GAME OVER") + stdout.count("Game Over")
        
        logger.debug(f"Event counts: START={game_events}, MOVES={move_events}, END={end_events}")
        
        if move_events > turns:
            logger.debug(f"Using move count {move_events} as turn estimate")
            turns = max(turns, move_events)
        
        # Strategy 3: Look for JSON game state
        try:
            # Find JSON objects that might contain turn information
            json_objects = re.findall(r'\{[^{}]*"turn"[^{}]*\}', stdout)
            for json_str in json_objects:
                try:
                    data = json.loads(json_str)
                    if 'turn' in data:
                        turn_num = int(data['turn'])
                        turns = max(turns, turn_num)
                        logger.debug(f"JSON turn found: {turn_num}")
                except json.JSONDecodeError:
                    parsing_errors.append(f"Invalid JSON: {json_str}")
                except (KeyError, ValueError) as e:
                    parsing_errors.append(f"JSON parsing error: {e}")
        except Exception as e:
            parsing_errors.append(f"JSON search error: {e}")
        
        # Check for error indicators
        error_indicators = [
            "connection refused",
            "timeout",
            "failed to connect",
            "422",
            "400",
            "500",
            "missing field",
            "parse error"
        ]
        
        combined_output = f"{stdout} {stderr}".lower()
        for indicator in error_indicators:
            if indicator in combined_output:
                parsing_errors.append(f"Error indicator found: {indicator}")
        
        logger.debug(f"Final parsed turns: {turns}")
        if parsing_errors:
            logger.warning(f"Parsing errors: {parsing_errors}")
        
        return turns, parsing_errors
    
    def extract_winner(self, output: str) -> Optional[str]:
        """Extract winner information with diagnostics"""
        winner_patterns = [
            r"Winner:\s*(.+)",
            r"Victory:\s*(.+)",
            r"won by\s*(.+)",
            r"POST-TRAINING.*?(\w+)"
        ]
        
        import re
        for pattern in winner_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                winner = match.group(1).strip()
                logger.debug(f"Winner found: {winner}")
                return winner
        
        return None
    
    def run_diagnostic_suite(self) -> Dict[str, Any]:
        """Run comprehensive diagnostic suite"""
        logger.info("🚀 STARTING POST-TRAINING VALIDATION DIAGNOSTICS")
        logger.info("=" * 60)
        
        # Step 1: Server health check
        if not self.validate_server_health():
            logger.error("❌ Server health check failed - aborting diagnostics")
            return {"error": "Server health check failed"}
        
        # Step 2: Run small set of diagnostic games
        test_games = [
            {"name": "solo_11x11_diagnostic", "width": 11, "height": 11, "mode": "solo", "num_snakes": 1},
            {"name": "multi_2snakes_diagnostic", "width": 11, "height": 11, "mode": "standard", "num_snakes": 2}
        ]
        
        logger.info(f"🎮 Running {len(test_games)} diagnostic games...")
        
        for game_config in test_games:
            result = self.run_diagnostic_game(game_config)
            self.results.append(result)
            
            # Log detailed result
            logger.info(f"📊 GAME RESULT: {result.game_id}")
            logger.info(f"   Turns: {result.turns}")
            logger.info(f"   Exit Code: {result.exit_code}")
            logger.info(f"   Execution Time: {result.execution_time:.2f}s")
            logger.info(f"   Parsing Errors: {len(result.parsing_errors)}")
            
            if result.parsing_errors:
                for error in result.parsing_errors:
                    logger.warning(f"   ⚠️ {error}")
        
        # Step 3: Generate diagnostic summary
        return self.generate_diagnostic_summary()
    
    def generate_diagnostic_summary(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic summary"""
        total_games = len(self.results)
        successful_games = [r for r in self.results if r.turns > 0]
        failed_games = [r for r in self.results if r.turns == 0]
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_games": total_games,
            "successful_games": len(successful_games),
            "failed_games": len(failed_games),
            "success_rate": len(successful_games) / total_games * 100 if total_games > 0 else 0,
            "average_turns": sum(r.turns for r in self.results) / total_games if total_games > 0 else 0,
            "average_execution_time": sum(r.execution_time for r in self.results) / total_games if total_games > 0 else 0,
            "common_parsing_errors": {},
            "exit_code_distribution": {},
            "detailed_results": []
        }
        
        # Analyze common issues
        all_errors = []
        exit_codes = []
        
        for result in self.results:
            all_errors.extend(result.parsing_errors)
            exit_codes.append(result.exit_code)
            
            summary["detailed_results"].append({
                "game_id": result.game_id,
                "turns": result.turns,
                "winner": result.winner,
                "exit_code": result.exit_code,
                "execution_time": result.execution_time,
                "parsing_errors": result.parsing_errors,
                "output_length": len(result.raw_cli_output),
                "error_length": len(result.raw_cli_error)
            })
        
        # Count error frequencies
        from collections import Counter
        summary["common_parsing_errors"] = dict(Counter(all_errors))
        summary["exit_code_distribution"] = dict(Counter(exit_codes))
        
        return summary

def main():
    """Run diagnostic validation"""
    print("🔍 POST-TRAINING VALIDATION DIAGNOSTICS")
    print("=" * 50)
    print("This diagnostic script will identify why games report 0 turns")
    print("when server logs show they're actually running properly.")
    print()
    
    diagnostic_system = DiagnosticValidationSystem()
    
    try:
        results = diagnostic_system.run_diagnostic_suite()
        
        # Save diagnostic results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"diagnostic_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Diagnostic results saved to: {results_file}")
        
        # Print summary
        print("\n🎯 DIAGNOSTIC SUMMARY")
        print("=" * 30)
        print(f"Total Games: {results['total_games']}")
        print(f"Successful Games: {results['successful_games']}")
        print(f"Failed Games: {results['failed_games']}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print(f"Average Turns: {results['average_turns']:.1f}")
        print(f"Average Execution Time: {results['average_execution_time']:.2f}s")
        
        if results['common_parsing_errors']:
            print("\n⚠️ COMMON PARSING ERRORS:")
            for error, count in results['common_parsing_errors'].items():
                print(f"   {count}x: {error}")
        
        if results['exit_code_distribution']:
            print("\n📊 EXIT CODE DISTRIBUTION:")
            for code, count in results['exit_code_distribution'].items():
                print(f"   Code {code}: {count} games")
        
        # Determine root cause
        print("\n🎯 ROOT CAUSE ANALYSIS:")
        if results['failed_games'] > 0:
            if any("422" in str(errors) for errors in results['common_parsing_errors']):
                print("   🚨 PRIMARY ISSUE: Server rejecting requests (HTTP 422)")
                print("   🔧 SOLUTION: Fix JSON format compatibility between CLI and server")
            elif any("timeout" in str(errors).lower() for errors in results['common_parsing_errors']):
                print("   ⏱️ PRIMARY ISSUE: Games timing out")
                print("   🔧 SOLUTION: Increase timeout or optimize game execution")
            elif results['average_execution_time'] < 5:
                print("   ⚡ PRIMARY ISSUE: Games ending too quickly (likely immediate failure)")
                print("   🔧 SOLUTION: Check server startup and game initialization")
            else:
                print("   📊 PRIMARY ISSUE: Data collection/parsing problem")
                print("   🔧 SOLUTION: Fix game result extraction logic")
        else:
            print("   ✅ All games successful - issue is likely in main validation script")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Diagnostic failed: {e}")
        raise

if __name__ == "__main__":
    main()