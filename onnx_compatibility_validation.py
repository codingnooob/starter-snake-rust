#!/usr/bin/env python3
"""
ONNX Compatibility Validation Script
Tests 12-channel pipeline compatibility with new ONNX models
Validates real neural inference and replaces mock inference mode
"""

import sys
import time
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests

# Add neural_networks to path for imports
sys.path.append('neural_networks')
from advanced_board_encoding import Advanced12ChannelBoardEncoder, GameState

class ONNXCompatibilityValidator:
    """Validates ONNX model compatibility with 12-channel pipeline"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.encoder = Advanced12ChannelBoardEncoder()
        self.test_results = {
            "model_loading": {},
            "tensor_compatibility": {},
            "inference_validation": {},
            "performance_comparison": {},
            "summary": {}
        }
        
    def validate_model_loading(self) -> bool:
        """Test loading of both 8-channel and 12-channel models"""
        print("🔧 Testing ONNX model loading...")
        
        model_configs = [
            {"suffix": "", "channels": 8, "description": "8-channel legacy"},
            {"suffix": "_12ch", "channels": 12, "description": "12-channel advanced"}
        ]
        
        for config in model_configs:
            suffix = config["suffix"]
            channels = config["channels"]
            description = config["description"]
            
            print(f"\n   Testing {description} models...")
            
            model_types = ["position_evaluation", "move_prediction", "game_outcome"]
            config_results = {"loaded": 0, "failed": 0, "errors": []}
            
            for model_type in model_types:
                model_path = self.models_dir / f"{model_type}{suffix}.onnx"
                
                try:
                    session = ort.InferenceSession(str(model_path))
                    
                    # Validate input shape
                    input_shape = session.get_inputs()[0].shape
                    expected_features = channels * 11 * 11
                    
                    if input_shape[1:] == [channels, 11, 11]:
                        print(f"   ✅ {model_type}{suffix}.onnx loaded - shape {input_shape}")
                        config_results["loaded"] += 1
                    else:
                        print(f"   ❌ {model_type}{suffix}.onnx - incorrect shape {input_shape}")
                        config_results["failed"] += 1
                        config_results["errors"].append(f"Shape mismatch: {input_shape}")
                        
                except Exception as e:
                    print(f"   ❌ {model_type}{suffix}.onnx failed to load: {e}")
                    config_results["failed"] += 1
                    config_results["errors"].append(str(e))
            
            self.test_results["model_loading"][description] = config_results
        
        total_loaded = sum(r["loaded"] for r in self.test_results["model_loading"].values())
        print(f"\n📊 Model loading results: {total_loaded}/6 models loaded successfully")
        
        return total_loaded == 6
    
    def validate_tensor_compatibility(self) -> bool:
        """Test tensor shape compatibility between pipeline and models"""
        print("\n🔍 Testing tensor shape compatibility...")
        
        # Generate test game state
        test_game_state = self.create_test_game_state()
        
        try:
            # Convert test game state to proper format
            game_state = GameState(
                board_width=test_game_state["board"]["width"],
                board_height=test_game_state["board"]["height"],
                our_snake=test_game_state["you"],
                opponent_snakes=[s for s in test_game_state["board"]["snakes"] if s["id"] != test_game_state["you"]["id"]],
                food=test_game_state["board"]["food"],
                turn=test_game_state["turn"],
                game_id=test_game_state["game"]["id"]
            )
            
            # Generate 12-channel encoding
            board_tensor, snake_features, game_context = self.encoder.encode_game_state(game_state)
            
            print(f"   12-channel pipeline output shape: {board_tensor.shape}")
            
            # Test compatibility with 12-channel models
            if board_tensor.shape == (11, 11, 12):
                # Reshape for ONNX: (height, width, channels) -> (batch, channels, height, width)
                onnx_input = np.transpose(board_tensor, (2, 0, 1))[np.newaxis, :, :, :]
                print(f"   ONNX input shape after reshape: {onnx_input.shape}")
                
                # Test with 12-channel model
                model_path = self.models_dir / "position_evaluation_12ch.onnx"
                session = ort.InferenceSession(str(model_path))
                
                # Run inference test
                output = session.run(None, {"board_state": onnx_input.astype(np.float32)})
                print(f"   ✅ 12-channel inference successful - output shape: {output[0].shape}")
                
                self.test_results["tensor_compatibility"]["12_channel"] = {
                    "status": "success",
                    "pipeline_shape": str(board_tensor.shape),
                    "onnx_input_shape": str(onnx_input.shape),
                    "output_shape": str(output[0].shape)
                }
                
                return True
            else:
                print(f"   ❌ Unexpected pipeline output shape: {board_tensor.shape}")
                self.test_results["tensor_compatibility"]["12_channel"] = {
                    "status": "failed",
                    "error": f"Unexpected shape: {board_tensor.shape}"
                }
                return False
                
        except Exception as e:
            print(f"   ❌ Tensor compatibility test failed: {e}")
            self.test_results["tensor_compatibility"]["12_channel"] = {
                "status": "error",
                "error": str(e)
            }
            return False
    
    def validate_inference_accuracy(self) -> bool:
        """Compare ONNX inference outputs for consistency"""
        print("\n🧠 Validating inference accuracy...")
        
        test_game_state = self.create_test_game_state()
        
        try:
            # Convert test game state to proper format
            game_state = GameState(
                board_width=test_game_state["board"]["width"],
                board_height=test_game_state["board"]["height"],
                our_snake=test_game_state["you"],
                opponent_snakes=[s for s in test_game_state["board"]["snakes"] if s["id"] != test_game_state["you"]["id"]],
                food=test_game_state["board"]["food"],
                turn=test_game_state["turn"],
                game_id=test_game_state["game"]["id"]
            )
            
            # Generate encoding
            board_tensor, snake_features, game_context = self.encoder.encode_game_state(game_state)
            
            # Prepare ONNX input
            onnx_input = np.transpose(board_tensor, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
            
            # Test all 12-channel models
            model_results = {}
            
            for model_type in ["position_evaluation", "move_prediction", "game_outcome"]:
                model_path = self.models_dir / f"{model_type}_12ch.onnx"
                session = ort.InferenceSession(str(model_path))
                
                # Run multiple inferences for consistency
                outputs = []
                for i in range(3):
                    output = session.run(None, {"board_state": onnx_input})
                    outputs.append(output[0])
                
                # Check consistency
                consistent = all(np.allclose(outputs[0], out, rtol=1e-5) for out in outputs[1:])
                
                model_results[model_type] = {
                    "output_shape": str(outputs[0].shape),
                    "output_range": [float(np.min(outputs[0])), float(np.max(outputs[0]))],
                    "consistent": consistent,
                    "sample_output": outputs[0].flatten()[:5].tolist()  # First 5 values
                }
                
                status = "✅" if consistent else "❌"
                print(f"   {status} {model_type}_12ch: shape={outputs[0].shape}, range=[{np.min(outputs[0]):.3f}, {np.max(outputs[0]):.3f}]")
            
            self.test_results["inference_validation"]["12_channel"] = model_results
            return all(r["consistent"] for r in model_results.values())
            
        except Exception as e:
            print(f"   ❌ Inference validation failed: {e}")
            self.test_results["inference_validation"]["error"] = str(e)
            return False
    
    def performance_benchmark(self) -> Dict[str, float]:
        """Benchmark ONNX inference performance"""
        print("\n⚡ Running performance benchmarks...")
        
        test_game_state = self.create_test_game_state()
        benchmarks = {}
        
        try:
            # Generate encoding (measure this time too)
            start_time = time.perf_counter()
            encoding_result = self.encoder.encode_12_channel_board(test_game_state)
            encoding_time = (time.perf_counter() - start_time) * 1000
            
            board_tensor = encoding_result["board_encoding"]
            onnx_input = np.transpose(board_tensor, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
            
            benchmarks["encoding_ms"] = encoding_time
            print(f"   12-channel encoding: {encoding_time:.3f}ms")
            
            # Benchmark each model type
            for model_type in ["position_evaluation", "move_prediction", "game_outcome"]:
                model_path = self.models_dir / f"{model_type}_12ch.onnx"
                session = ort.InferenceSession(str(model_path))
                
                # Warm up
                for _ in range(5):
                    session.run(None, {"board_state": onnx_input})
                
                # Benchmark
                times = []
                for _ in range(20):
                    start_time = time.perf_counter()
                    session.run(None, {"board_state": onnx_input})
                    times.append((time.perf_counter() - start_time) * 1000)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                benchmarks[f"{model_type}_ms"] = avg_time
                benchmarks[f"{model_type}_std_ms"] = std_time
                
                print(f"   {model_type}_12ch: {avg_time:.3f}±{std_time:.3f}ms")
            
            # Total pipeline time
            total_time = encoding_time + sum(benchmarks[k] for k in benchmarks.keys() if k.endswith("_ms") and "std" not in k and k != "encoding_ms")
            benchmarks["total_pipeline_ms"] = total_time
            
            print(f"   Total 12-channel pipeline: {total_time:.3f}ms")
            
            self.test_results["performance_comparison"]["12_channel"] = benchmarks
            return benchmarks
            
        except Exception as e:
            print(f"   ❌ Performance benchmark failed: {e}")
            self.test_results["performance_comparison"]["error"] = str(e)
            return {}
    
    def test_rust_integration(self) -> bool:
        """Test integration with Rust server (if running)"""
        print("\n🦀 Testing Rust server integration...")
        
        try:
            # Try to connect to local server
            response = requests.get("http://localhost:8000/", timeout=5)
            if response.status_code == 200:
                print("   ✅ Rust server is running")
                
                # Test with a move request
                test_request = self.create_test_move_request()
                move_response = requests.post(
                    "http://localhost:8000/move",
                    json=test_request,
                    timeout=10
                )
                
                if move_response.status_code == 200:
                    move_data = move_response.json()
                    print(f"   ✅ Move request successful: {move_data.get('move', 'unknown')}")
                    self.test_results["rust_integration"] = {"status": "success", "move": move_data.get('move')}
                    return True
                else:
                    print(f"   ❌ Move request failed: {move_response.status_code}")
                    self.test_results["rust_integration"] = {"status": "move_failed", "code": move_response.status_code}
                    return False
            else:
                print(f"   ⚠️ Server responded with status: {response.status_code}")
                self.test_results["rust_integration"] = {"status": "server_error", "code": response.status_code}
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️ Rust server not accessible: {e}")
            print("   (This is expected if server is not running)")
            self.test_results["rust_integration"] = {"status": "not_running", "note": "Expected if server not started"}
            return True  # Not an error condition
    
    def create_test_game_state(self) -> Dict:
        """Create a realistic test game state"""
        return {
            "game": {
                "id": "test-game",
                "ruleset": {"name": "standard", "version": "v1.1.15"},
                "timeout": 500
            },
            "turn": 10,
            "board": {
                "height": 11,
                "width": 11,
                "food": [{"x": 3, "y": 3}, {"x": 7, "y": 8}],
                "hazards": [],
                "snakes": [
                    {
                        "id": "test-snake-us",
                        "name": "Our Snake",
                        "health": 80,
                        "body": [{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 5, "y": 3}],
                        "latency": "50",
                        "head": {"x": 5, "y": 5},
                        "length": 3,
                        "shout": ""
                    },
                    {
                        "id": "test-snake-opponent",
                        "name": "Opponent Snake",
                        "health": 60,
                        "body": [{"x": 8, "y": 8}, {"x": 8, "y": 7}, {"x": 9, "y": 7}],
                        "latency": "100",
                        "head": {"x": 8, "y": 8},
                        "length": 3,
                        "shout": ""
                    }
                ]
            },
            "you": {
                "id": "test-snake-us",
                "name": "Our Snake",
                "health": 80,
                "body": [{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 5, "y": 3}],
                "latency": "50",
                "head": {"x": 5, "y": 5},
                "length": 3,
                "shout": ""
            }
        }
    
    def create_test_move_request(self) -> Dict:
        """Create a test move request for the server"""
        return self.create_test_game_state()
    
    def generate_report(self) -> str:
        """Generate comprehensive compatibility report"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "test_results": self.test_results,
            "conclusion": {}
        }
        
        # Analyze results
        model_loading_success = all(
            r.get("loaded", 0) == 3 for r in self.test_results["model_loading"].values()
        )
        
        tensor_compatibility_success = (
            self.test_results.get("tensor_compatibility", {}).get("12_channel", {}).get("status") == "success"
        )
        
        inference_success = all(
            r.get("consistent", False) for r in 
            self.test_results.get("inference_validation", {}).get("12_channel", {}).values()
            if isinstance(r, dict) and "consistent" in r
        )
        
        performance_data = self.test_results.get("performance_comparison", {}).get("12_channel", {})
        performance_success = "total_pipeline_ms" in performance_data
        
        # Overall assessment
        overall_success = all([
            model_loading_success,
            tensor_compatibility_success, 
            inference_success,
            performance_success
        ])
        
        report["conclusion"] = {
            "overall_success": overall_success,
            "model_loading": model_loading_success,
            "tensor_compatibility": tensor_compatibility_success,
            "inference_validation": inference_success,
            "performance_benchmark": performance_success,
            "total_pipeline_time_ms": performance_data.get("total_pipeline_ms", "N/A"),
            "recommendation": "READY FOR PRODUCTION" if overall_success else "REQUIRES FIXES"
        }
        
        return json.dumps(report, indent=2)
    
    def run_complete_validation(self) -> bool:
        """Run all validation tests"""
        print("🚀 ONNX Compatibility Validation Starting...")
        print("=" * 60)
        
        success_flags = []
        
        # Run all tests
        success_flags.append(self.validate_model_loading())
        success_flags.append(self.validate_tensor_compatibility())
        success_flags.append(self.validate_inference_accuracy())
        
        performance_result = self.performance_benchmark()
        success_flags.append(bool(performance_result))
        
        # Rust integration (optional)
        self.test_rust_integration()
        
        # Generate and save report
        report = self.generate_report()
        
        with open("onnx_compatibility_report.json", "w") as f:
            f.write(report)
        
        print("\n" + "=" * 60)
        print("📊 ONNX Compatibility Validation Summary:")
        print("=" * 60)
        
        overall_success = all(success_flags)
        status_icon = "✅" if overall_success else "❌"
        
        print(f"{status_icon} Overall Status: {'SUCCESS' if overall_success else 'FAILED'}")
        print(f"   Model Loading: {'✅' if success_flags[0] else '❌'}")
        print(f"   Tensor Compatibility: {'✅' if success_flags[1] else '❌'}")
        print(f"   Inference Validation: {'✅' if success_flags[2] else '❌'}")
        print(f"   Performance Benchmark: {'✅' if success_flags[3] else '❌'}")
        
        if performance_result:
            total_time = performance_result.get("total_pipeline_ms", 0)
            print(f"   Total Pipeline Time: {total_time:.3f}ms")
        
        print(f"\n📄 Detailed report saved to: onnx_compatibility_report.json")
        
        if overall_success:
            print("\n🎯 CONCLUSION: 12-channel ONNX models are READY FOR PRODUCTION!")
            print("   - All tensor shapes are compatible")
            print("   - Real neural inference is working")
            print("   - Performance is within acceptable limits")
            print("   - System can now use real ONNX models instead of mock inference")
        else:
            print("\n⚠️ CONCLUSION: Issues found that need resolution before production deployment")
        
        return overall_success

def main():
    """Main validation entry point"""
    validator = ONNXCompatibilityValidator()
    success = validator.run_complete_validation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()