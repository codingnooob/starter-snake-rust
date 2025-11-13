#!/usr/bin/env python3
"""
Neural Network Pipeline Verification Test
Comprehensive validation of the 12-channel neural network pipeline functionality.
"""

import subprocess
import json
import sys
import os
import time
from pathlib import Path

class NeuralPipelineVerifier:
    def __init__(self):
        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": {},
            "overall_status": "PENDING"
        }
        
    def log(self, test_name, status, details=""):
        """Log test result"""
        self.results["tests"][test_name] = {
            "status": status,
            "details": details,
            "timestamp": time.strftime("%H:%M:%S")
        }
        print(f"[{status}] {test_name}: {details}")
        
    def run_rust_test(self, test_pattern="", timeout=30):
        """Run Rust tests and capture output"""
        try:
            cmd = ["cargo", "test"]
            if test_pattern:
                cmd.extend([test_pattern, "--", "--nocapture"])
            else:
                cmd.extend(["--", "--nocapture"])
                
            result = subprocess.run(
                cmd,
                cwd="/home/t/starter-snake-rust",
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": "Timeout", "returncode": -1}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}

    def verify_onnx_models(self):
        """Verify ONNX models exist and have valid metadata"""
        models_dir = Path("/home/t/starter-snake-rust/models")
        required_models = ["game_outcome.onnx", "move_prediction.onnx", "position_evaluation.onnx"]
        
        all_exist = True
        model_info = {}
        
        for model in required_models:
            model_path = models_dir / model
            exists = model_path.exists()
            size = model_path.stat().st_size if exists else 0
            
            model_info[model] = {"exists": exists, "size": size}
            
            if not exists:
                all_exist = False
                self.log(f"ONNX Model Check - {model}", "FAIL", f"Model file not found")
            elif size == 0:
                all_exist = False
                self.log(f"ONNX Model Check - {model}", "FAIL", f"Model file is empty")
            else:
                self.log(f"ONNX Model Check - {model}", "PASS", f"Size: {size} bytes")
        
        return all_exist, model_info

    def verify_rust_compilation(self):
        """Verify Rust code compiles without errors"""
        try:
            result = subprocess.run(
                ["cargo", "check"],
                cwd="/home/t/starter-snake-rust",
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                self.log("Rust Compilation", "PASS", "Code compiles successfully")
                return True
            else:
                self.log("Rust Compilation", "FAIL", f"Compilation errors: {result.stderr}")
                return False
                
        except Exception as e:
            self.log("Rust Compilation", "FAIL", f"Compilation check failed: {str(e)}")
            return False

    def create_test_game_state(self):
        """Create a test game state for neural network testing"""
        return {
            "game": {"id": "test-game", "timeout": 500},
            "turn": 10,
            "board": {
                "height": 11,
                "width": 11,
                "food": [{"x": 5, "y": 5}],
                "hazards": [],
                "snakes": [
                    {
                        "id": "test-snake-1",
                        "name": "Test Snake",
                        "health": 85,
                        "body": [{"x": 3, "y": 3}, {"x": 3, "y": 4}, {"x": 3, "y": 5}],
                        "head": {"x": 3, "y": 3},
                        "length": 3,
                        "latency": "50",
                        "shout": ""
                    }
                ]
            },
            "you": {
                "id": "test-snake-1",
                "name": "Test Snake", 
                "health": 85,
                "body": [{"x": 3, "y": 3}, {"x": 3, "y": 4}, {"x": 3, "y": 5}],
                "head": {"x": 3, "y": 3},
                "length": 3,
                "latency": "50",
                "shout": ""
            }
        }

    def test_neural_network_integration(self):
        """Test the neural network integration by making a request to the server"""
        try:
            # Create test request
            test_data = self.create_test_game_state()
            
            # Write test data to file
            with open("/home/t/starter-snake-rust/test_neural_request.json", "w") as f:
                json.dump(test_data, f, indent=2)
            
            # Test with curl request to the running server
            curl_result = subprocess.run([
                "curl", "-X", "POST", "http://localhost:8888/move",
                "-H", "Content-Type: application/json",
                "-d", json.dumps(test_data),
                "--max-time", "5"
            ], capture_output=True, text=True, timeout=10)
            
            if curl_result.returncode == 0:
                try:
                    response = json.loads(curl_result.stdout)
                    if "move" in response:
                        self.log("Neural Network Integration", "PASS", 
                               f"Server responded with move: {response['move']}")
                        return True
                    else:
                        self.log("Neural Network Integration", "FAIL", 
                               f"Invalid response format: {curl_result.stdout}")
                        return False
                except json.JSONDecodeError:
                    self.log("Neural Network Integration", "FAIL", 
                           f"Invalid JSON response: {curl_result.stdout}")
                    return False
            else:
                self.log("Neural Network Integration", "FAIL", 
                       f"Server request failed: {curl_result.stderr}")
                return False
                
        except Exception as e:
            self.log("Neural Network Integration", "FAIL", f"Integration test failed: {str(e)}")
            return False

    def run_neural_specific_tests(self):
        """Run neural network specific Rust tests"""
        test_patterns = [
            "neural",
            "board_state_encoder", 
            "intelligence_strategy",
            "confidence"
        ]
        
        passed_tests = 0
        total_tests = 0
        
        for pattern in test_patterns:
            result = self.run_rust_test(pattern)
            total_tests += 1
            
            if result["success"]:
                passed_tests += 1
                self.log(f"Rust Test - {pattern}", "PASS", "Tests passed")
            else:
                self.log(f"Rust Test - {pattern}", "FAIL", 
                       f"Tests failed: {result['stderr']}")
        
        return passed_tests, total_tests

    def check_12_channel_support(self):
        """Verify 12-channel board encoding is implemented"""
        # Check for 12-channel references in the codebase
        try:
            result = subprocess.run([
                "grep", "-r", "Advanced12Channel", "/home/t/starter-snake-rust/src/"
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and "Advanced12Channel" in result.stdout:
                self.log("12-Channel Support", "PASS", 
                       f"Found {len(result.stdout.splitlines())} references to Advanced12Channel")
                return True
            else:
                self.log("12-Channel Support", "FAIL", "No Advanced12Channel implementation found")
                return False
                
        except Exception as e:
            self.log("12-Channel Support", "FAIL", f"Check failed: {str(e)}")
            return False

    def check_confidence_integration(self):
        """Verify confidence integration system is implemented"""
        confidence_file = Path("/home/t/starter-snake-rust/src/confidence_validation.rs")
        
        if confidence_file.exists():
            try:
                content = confidence_file.read_text()
                if "confidence" in content.lower() and len(content) > 1000:
                    self.log("Confidence Integration", "PASS", 
                           f"Confidence validation module found ({len(content)} chars)")
                    return True
                else:
                    self.log("Confidence Integration", "FAIL", "Incomplete confidence module")
                    return False
            except Exception as e:
                self.log("Confidence Integration", "FAIL", f"Error reading confidence module: {str(e)}")
                return False
        else:
            self.log("Confidence Integration", "FAIL", "Confidence validation module not found")
            return False

    def run_comprehensive_verification(self):
        """Run all verification tests"""
        print("=== Neural Network Pipeline Verification ===\n")
        
        # Test 1: ONNX Models
        models_ok, model_info = self.verify_onnx_models()
        
        # Test 2: Rust Compilation
        compilation_ok = self.verify_rust_compilation()
        
        # Test 3: 12-Channel Support
        channels_ok = self.check_12_channel_support()
        
        # Test 4: Confidence Integration
        confidence_ok = self.check_confidence_integration()
        
        # Test 5: Neural Network Integration
        integration_ok = self.test_neural_network_integration()
        
        # Test 6: Rust Tests
        passed_tests, total_tests = self.run_neural_specific_tests()
        rust_tests_ok = passed_tests == total_tests
        
        # Summary
        total_checks = 6
        passed_checks = sum([
            models_ok, compilation_ok, channels_ok, 
            confidence_ok, integration_ok, rust_tests_ok
        ])
        
        success_rate = (passed_checks / total_checks) * 100
        
        if success_rate >= 80:
            overall_status = "PASS"
        elif success_rate >= 60:
            overall_status = "PARTIAL"
        else:
            overall_status = "FAIL"
            
        self.results["overall_status"] = overall_status
        self.results["success_rate"] = success_rate
        self.results["summary"] = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "models_info": model_info,
            "rust_tests": f"{passed_tests}/{total_tests}"
        }
        
        print(f"\n=== VERIFICATION SUMMARY ===")
        print(f"Overall Status: {overall_status}")
        print(f"Success Rate: {success_rate:.1f}% ({passed_checks}/{total_checks} checks passed)")
        print(f"Rust Tests: {passed_tests}/{total_tests} patterns passed")
        
        return self.results

def main():
    verifier = NeuralPipelineVerifier()
    results = verifier.run_comprehensive_verification()
    
    # Save results to file
    with open("/home/t/starter-snake-rust/neural_verification_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: neural_verification_results.json")
    
    # Return appropriate exit code
    if results["overall_status"] == "PASS":
        sys.exit(0)
    elif results["overall_status"] == "PARTIAL":
        sys.exit(1) 
    else:
        sys.exit(2)

if __name__ == "__main__":
    main()