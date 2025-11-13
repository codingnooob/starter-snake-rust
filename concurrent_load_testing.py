#!/usr/bin/env python3
"""
Concurrent Load Testing Infrastructure with Memory Monitoring
Tests system performance under multiple concurrent requests with resource tracking
"""

import asyncio
import aiohttp
import psutil
import time
import json
import threading
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import requests

@dataclass
class PerformanceMetrics:
    """Container for performance test metrics"""
    response_times: List[float]
    success_count: int
    failure_count: int
    memory_usage: List[float]
    cpu_usage: List[float]
    error_details: List[str]
    start_time: float
    end_time: float

@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios"""
    concurrent_requests: int
    total_requests: int
    request_interval_ms: float
    timeout_seconds: float
    warmup_requests: int
    scenario_name: str

class ResourceMonitor:
    """Monitors system resources during load testing"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = {
            "memory": [],
            "cpu": [],
            "timestamps": []
        }
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        self.monitoring = True
        self.metrics = {"memory": [], "cpu": [], "timestamps": []}
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> Dict:
        """Stop monitoring and return collected metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        return self.metrics.copy()
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                memory_mb = psutil.virtual_memory().used / (1024 * 1024)
                cpu_percent = psutil.cpu_percent(interval=0.1)
                timestamp = time.time()
                
                self.metrics["memory"].append(memory_mb)
                self.metrics["cpu"].append(cpu_percent)
                self.metrics["timestamps"].append(timestamp)
                
                time.sleep(0.5)  # Sample every 500ms
            except Exception as e:
                print(f"   Warning: Resource monitoring error: {e}")

class ConcurrentLoadTester:
    """Comprehensive concurrent load testing system"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.resource_monitor = ResourceMonitor()
        self.test_data = self._create_test_request()
        
    def _create_test_request(self) -> Dict:
        """Create a realistic Battlesnake move request"""
        return {
            "game": {
                "id": "load-test-game",
                "ruleset": {"name": "standard", "version": "v1.1.15"},
                "timeout": 500
            },
            "turn": 25,
            "board": {
                "height": 11,
                "width": 11,
                "food": [{"x": 3, "y": 3}, {"x": 8, "y": 7}, {"x": 1, "y": 9}],
                "hazards": [],
                "snakes": [
                    {
                        "id": "load-test-us",
                        "name": "Our Snake",
                        "health": 85,
                        "body": [{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 6, "y": 4}, {"x": 7, "y": 4}],
                        "latency": "45",
                        "head": {"x": 5, "y": 5},
                        "length": 4,
                        "shout": ""
                    },
                    {
                        "id": "load-test-opponent1",
                        "name": "Opponent 1",
                        "health": 70,
                        "body": [{"x": 9, "y": 8}, {"x": 9, "y": 7}, {"x": 10, "y": 7}],
                        "latency": "80",
                        "head": {"x": 9, "y": 8},
                        "length": 3,
                        "shout": ""
                    },
                    {
                        "id": "load-test-opponent2", 
                        "name": "Opponent 2",
                        "health": 60,
                        "body": [{"x": 2, "y": 2}, {"x": 1, "y": 2}, {"x": 0, "y": 2}],
                        "latency": "120",
                        "head": {"x": 2, "y": 2},
                        "length": 3,
                        "shout": ""
                    }
                ]
            },
            "you": {
                "id": "load-test-us",
                "name": "Our Snake",
                "health": 85,
                "body": [{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 6, "y": 4}, {"x": 7, "y": 4}],
                "latency": "45",
                "head": {"x": 5, "y": 5},
                "length": 4,
                "shout": ""
            }
        }
    
    def _single_request(self, session_id: int) -> Tuple[float, bool, Optional[str]]:
        """Execute a single request and return timing/status"""
        start_time = time.perf_counter()
        
        try:
            response = requests.post(
                f"{self.server_url}/move",
                json=self.test_data,
                timeout=10.0,
                headers={"Content-Type": "application/json"}
            )
            
            end_time = time.perf_counter()
            response_time = (end_time - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                return response_time, True, None
            else:
                return response_time, False, f"HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            end_time = time.perf_counter()
            response_time = (end_time - start_time) * 1000
            return response_time, False, "Timeout"
            
        except Exception as e:
            end_time = time.perf_counter()
            response_time = (end_time - start_time) * 1000
            return response_time, False, str(e)
    
    def _concurrent_batch(self, config: LoadTestConfig) -> PerformanceMetrics:
        """Execute a batch of concurrent requests"""
        print(f"   Executing {config.total_requests} requests with {config.concurrent_requests} concurrent...")
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        start_time = time.perf_counter()
        response_times = []
        success_count = 0
        failure_count = 0
        error_details = []
        
        # Use ThreadPoolExecutor for concurrent requests
        with ThreadPoolExecutor(max_workers=config.concurrent_requests) as executor:
            futures = []
            
            # Submit all requests
            for i in range(config.total_requests):
                future = executor.submit(self._single_request, i)
                futures.append(future)
                
                # Add interval delay between request submissions
                if config.request_interval_ms > 0 and i < config.total_requests - 1:
                    time.sleep(config.request_interval_ms / 1000.0)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    response_time, success, error = future.result(timeout=config.timeout_seconds)
                    response_times.append(response_time)
                    
                    if success:
                        success_count += 1
                    else:
                        failure_count += 1
                        if error:
                            error_details.append(error)
                            
                except Exception as e:
                    failure_count += 1
                    error_details.append(f"Future error: {e}")
        
        end_time = time.perf_counter()
        
        # Stop resource monitoring and collect metrics
        resource_metrics = self.resource_monitor.stop_monitoring()
        
        return PerformanceMetrics(
            response_times=response_times,
            success_count=success_count,
            failure_count=failure_count,
            memory_usage=resource_metrics.get("memory", []),
            cpu_usage=resource_metrics.get("cpu", []),
            error_details=error_details,
            start_time=start_time,
            end_time=end_time
        )
    
    def _warmup(self, config: LoadTestConfig):
        """Warm up the server with initial requests"""
        print(f"   Warming up with {config.warmup_requests} requests...")
        
        for i in range(config.warmup_requests):
            try:
                self._single_request(i)
                time.sleep(0.1)  # Small delay between warmup requests
            except Exception:
                pass  # Ignore warmup failures
    
    def test_concurrent_load(self, config: LoadTestConfig) -> Dict:
        """Execute comprehensive concurrent load testing"""
        print(f"\n🔥 Testing {config.scenario_name}...")
        print(f"   Configuration: {config.concurrent_requests} concurrent, {config.total_requests} total")
        
        # Warmup phase
        if config.warmup_requests > 0:
            self._warmup(config)
        
        # Execute main test
        metrics = self._concurrent_batch(config)
        
        # Calculate statistics
        if metrics.response_times:
            avg_response = np.mean(metrics.response_times)
            p50_response = np.percentile(metrics.response_times, 50)
            p95_response = np.percentile(metrics.response_times, 95)
            p99_response = np.percentile(metrics.response_times, 99)
            max_response = np.max(metrics.response_times)
            min_response = np.min(metrics.response_times)
        else:
            avg_response = p50_response = p95_response = p99_response = max_response = min_response = 0
        
        total_duration = metrics.end_time - metrics.start_time
        requests_per_second = config.total_requests / total_duration if total_duration > 0 else 0
        success_rate = metrics.success_count / config.total_requests if config.total_requests > 0 else 0
        
        # Resource usage statistics
        avg_memory = np.mean(metrics.memory_usage) if metrics.memory_usage else 0
        max_memory = np.max(metrics.memory_usage) if metrics.memory_usage else 0
        avg_cpu = np.mean(metrics.cpu_usage) if metrics.cpu_usage else 0
        max_cpu = np.max(metrics.cpu_usage) if metrics.cpu_usage else 0
        
        # Print results
        print(f"   📈 Results:")
        print(f"      Success Rate: {success_rate:.1%} ({metrics.success_count}/{config.total_requests})")
        print(f"      Requests/sec: {requests_per_second:.1f}")
        print(f"      Response Times: avg={avg_response:.1f}ms, p50={p50_response:.1f}ms, p95={p95_response:.1f}ms")
        print(f"      Memory Usage: avg={avg_memory:.1f}MB, peak={max_memory:.1f}MB")
        print(f"      CPU Usage: avg={avg_cpu:.1f}%, peak={max_cpu:.1f}%")
        
        if metrics.failure_count > 0:
            error_summary = {}
            for error in metrics.error_details:
                error_summary[error] = error_summary.get(error, 0) + 1
            print(f"      Errors: {dict(list(error_summary.items())[:3])}")  # Show top 3 errors
        
        # Return comprehensive results
        results = {
            "scenario": config.scenario_name,
            "config": {
                "concurrent_requests": config.concurrent_requests,
                "total_requests": config.total_requests,
                "request_interval_ms": config.request_interval_ms,
                "timeout_seconds": config.timeout_seconds
            },
            "performance": {
                "success_rate": success_rate,
                "requests_per_second": requests_per_second,
                "total_duration_seconds": total_duration,
                "response_times": {
                    "average_ms": avg_response,
                    "p50_ms": p50_response,
                    "p95_ms": p95_response,
                    "p99_ms": p99_response,
                    "min_ms": min_response,
                    "max_ms": max_response
                }
            },
            "resources": {
                "memory": {
                    "average_mb": avg_memory,
                    "peak_mb": max_memory
                },
                "cpu": {
                    "average_percent": avg_cpu,
                    "peak_percent": max_cpu
                }
            },
            "errors": {
                "total_failures": metrics.failure_count,
                "error_types": dict(
                    (error, metrics.error_details.count(error)) 
                    for error in set(metrics.error_details[:10])  # Top 10 unique errors
                )
            }
        }
        
        return results
    
    def run_comprehensive_load_tests(self) -> Dict:
        """Run a complete suite of load tests"""
        print("🚀 Starting Comprehensive Concurrent Load Testing...")
        print("=" * 70)
        
        # Test server availability first
        try:
            response = requests.get(f"{self.server_url}/", timeout=5)
            if response.status_code != 200:
                print(f"❌ Server not available at {self.server_url} (status: {response.status_code})")
                return {"error": "Server not available", "server_url": self.server_url}
        except Exception as e:
            print(f"❌ Cannot connect to server at {self.server_url}: {e}")
            return {"error": f"Connection failed: {e}", "server_url": self.server_url}
        
        print(f"✅ Server available at {self.server_url}")
        
        # Define test scenarios
        test_scenarios = [
            LoadTestConfig(
                concurrent_requests=1,
                total_requests=10,
                request_interval_ms=100,
                timeout_seconds=30,
                warmup_requests=3,
                scenario_name="Baseline Sequential"
            ),
            LoadTestConfig(
                concurrent_requests=2,
                total_requests=20,
                request_interval_ms=50,
                timeout_seconds=30,
                warmup_requests=5,
                scenario_name="Light Concurrent Load"
            ),
            LoadTestConfig(
                concurrent_requests=5,
                total_requests=50,
                request_interval_ms=20,
                timeout_seconds=45,
                warmup_requests=10,
                scenario_name="Moderate Concurrent Load"
            ),
            LoadTestConfig(
                concurrent_requests=10,
                total_requests=100,
                request_interval_ms=10,
                timeout_seconds=60,
                warmup_requests=15,
                scenario_name="Heavy Concurrent Load"
            ),
            LoadTestConfig(
                concurrent_requests=20,
                total_requests=200,
                request_interval_ms=5,
                timeout_seconds=90,
                warmup_requests=20,
                scenario_name="Stress Test"
            )
        ]
        
        # Execute all test scenarios
        all_results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
        
        for config in test_scenarios:
            try:
                result = self.test_concurrent_load(config)
                all_results[config.scenario_name] = result
                
                # Brief pause between tests for system recovery
                time.sleep(2)
                gc.collect()  # Force garbage collection
                
            except Exception as e:
                print(f"   ❌ Test failed: {e}")
                all_results[config.scenario_name] = {"error": str(e)}
        
        # Generate summary
        successful_tests = [
            test for test in all_results.values() 
            if isinstance(test, dict) and "error" not in test and "performance" in test
        ]
        
        if successful_tests:
            avg_success_rate = np.mean([test["performance"]["success_rate"] for test in successful_tests])
            avg_response_time = np.mean([test["performance"]["response_times"]["average_ms"] for test in successful_tests])
            max_rps = max([test["performance"]["requests_per_second"] for test in successful_tests])
            
            all_results["summary"] = {
                "total_tests": len(test_scenarios),
                "successful_tests": len(successful_tests),
                "average_success_rate": avg_success_rate,
                "average_response_time_ms": avg_response_time,
                "max_requests_per_second": max_rps,
                "recommendation": "READY FOR PRODUCTION" if avg_success_rate > 0.95 and avg_response_time < 100 else "NEEDS OPTIMIZATION"
            }
        else:
            all_results["summary"] = {
                "total_tests": len(test_scenarios),
                "successful_tests": 0,
                "recommendation": "SYSTEM ISSUES DETECTED"
            }
        
        print(f"\n{'='*70}")
        print("📊 Concurrent Load Testing Summary:")
        print("=" * 70)
        
        summary = all_results["summary"]
        print(f"✅ Tests Completed: {summary['successful_tests']}/{summary['total_tests']}")
        
        if "average_success_rate" in summary:
            print(f"📈 Average Success Rate: {summary['average_success_rate']:.1%}")
            print(f"⚡ Average Response Time: {summary['average_response_time_ms']:.1f}ms")
            print(f"🔥 Peak Requests/Second: {summary['max_requests_per_second']:.1f}")
            print(f"🎯 Recommendation: {summary['recommendation']}")
        
        # Save detailed results
        report_file = "concurrent_load_test_report.json"
        with open(report_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        return all_results

def main():
    """Main testing entry point"""
    tester = ConcurrentLoadTester()
    results = tester.run_comprehensive_load_tests()
    
    # Exit with appropriate code
    if "error" in results:
        return 1
    elif results.get("summary", {}).get("successful_tests", 0) == 0:
        return 1
    else:
        return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())