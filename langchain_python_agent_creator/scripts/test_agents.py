#!/usr/bin/env python3
"""
Agent Testing and Validation Script

This script provides comprehensive testing and validation utilities for LangChain agents,
including performance testing, error handling validation, and integration testing.
"""

import time
import json
import statistics
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.tools import BaseTool

class AgentTester:
    """Comprehensive agent testing and validation utilities."""
    
    def __init__(self, timeout: int = 30):
        """
        Initialize agent tester.
        
        Args:
            timeout: Default timeout for agent operations (seconds)
        """
        self.timeout = timeout
        self.test_results = []
    
    def create_test_tools(self) -> List[BaseTool]:
        """Create a set of test tools for validation."""
        
        @tool
        def fast_operation(input_data: str) -> str:
            """Fast operation for performance testing."""
            return f"Processed: {input_data.upper()}"
        
        @tool
        def slow_operation(delay: float = 1.0) -> str:
            """Slow operation for timeout testing."""
            time.sleep(delay)
            return f"Completed after {delay}s delay"
        
        @tool
        def failing_operation(should_fail: bool = False) -> str:
            """Operation that can fail for error testing."""
            if should_fail:
                raise ValueError("Intentional failure for testing")
            return "Operation succeeded"
        
        @tool
        def validation_operation(data: str) -> str:
            """Operation with input validation."""
            if not data or len(data.strip()) == 0:
                return "Error: Empty input not allowed"
            
            if len(data) > 100:
                return "Error: Input too long (max 100 characters)"
            
            return f"Valid input processed: {data}"
        
        return [fast_operation, slow_operation, failing_operation, validation_operation]
    
    def test_agent_creation(self, agent: Any, test_queries: List[str]) -> Dict[str, Any]:
        """
        Test basic agent creation and functionality.
        
        Args:
            agent: Agent instance to test
            test_queries: List of test queries
            
        Returns:
            Test results dictionary
        """
        results = {
            "test_name": "basic_functionality",
            "total_queries": len(test_queries),
            "successful_queries": 0,
            "failed_queries": 0,
            "errors": [],
            "response_times": []
        }
        
        for query in test_queries:
            try:
                start_time = time.time()
                
                response = agent.invoke({
                    "messages": [{"role": "user", "content": query}]
                })
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # Validate response structure
                if self._validate_response(response):
                    results["successful_queries"] += 1
                    results["response_times"].append(response_time)
                else:
                    results["failed_queries"] += 1
                    results["errors"].append(f"Invalid response structure for query: {query}")
                
            except Exception as e:
                results["failed_queries"] += 1
                results["errors"].append(f"Query '{query}' failed: {str(e)}")
        
        # Calculate statistics
        if results["response_times"]:
            results["avg_response_time"] = statistics.mean(results["response_times"])
            results["max_response_time"] = max(results["response_times"])
            results["min_response_time"] = min(results["response_times"])
        
        self.test_results.append(results)
        return results
    
    def test_performance(self, agent: Any, test_query: str, iterations: int = 10) -> Dict[str, Any]:
        """
        Test agent performance with multiple iterations.
        
        Args:
            agent: Agent instance to test
            test_query: Query to repeat
            iterations: Number of iterations
            
        Returns:
            Performance test results
        """
        results = {
            "test_name": "performance",
            "iterations": iterations,
            "query": test_query,
            "response_times": [],
            "errors": []
        }
        
        for i in range(iterations):
            try:
                start_time = time.time()
                
                response = agent.invoke({
                    "messages": [{"role": "user", "content": test_query}]
                })
                
                end_time = time.time()
                response_time = end_time - start_time
                
                if self._validate_response(response):
                    results["response_times"].append(response_time)
                else:
                    results["errors"].append(f"Invalid response on iteration {i+1}")
                
            except Exception as e:
                results["errors"].append(f"Iteration {i+1} failed: {str(e)}")
        
        # Calculate statistics
        if results["response_times"]:
            results["avg_response_time"] = statistics.mean(results["response_times"])
            results["median_response_time"] = statistics.median(results["response_times"])
            results["stdev_response_time"] = statistics.stdev(results["response_times"]) if len(results["response_times"]) > 1 else 0
            results["max_response_time"] = max(results["response_times"])
            results["min_response_time"] = min(results["response_times"])
        
        self.test_results.append(results)
        return results
    
    def test_error_handling(self, agent: Any, error_test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test agent error handling capabilities.
        
        Args:
            agent: Agent instance to test
            error_test_cases: List of test cases that should trigger errors
            
        Returns:
            Error handling test results
        """
        results = {
            "test_name": "error_handling",
            "total_cases": len(error_test_cases),
            "handled_errors": 0,
            "unhandled_errors": 0,
            "error_details": []
        }
        
        for test_case in error_test_cases:
            try:
                response = agent.invoke({
                    "messages": [{"role": "user", "content": test_case["query"]}]
                })
                
                # Check if error was handled gracefully
                if self._is_error_handled(response, test_case.get("expected_error_type")):
                    results["handled_errors"] += 1
                else:
                    results["unhandled_errors"] += 1
                    results["error_details"].append({
                        "query": test_case["query"],
                        "issue": "Error not handled properly",
                        "response": str(response)
                    })
                
            except Exception as e:
                results["unhandled_errors"] += 1
                results["error_details"].append({
                    "query": test_case["query"],
                    "issue": f"Exception thrown: {str(e)}",
                    "exception_type": type(e).__name__
                })
        
        self.test_results.append(results)
        return results
    
    def test_timeout_handling(self, agent: Any, timeout_queries: List[str], timeout_seconds: int = 5) -> Dict[str, Any]:
        """
        Test agent timeout handling.
        
        Args:
            agent: Agent instance to test
            timeout_queries: Queries that might cause timeouts
            timeout_seconds: Timeout threshold
            
        Returns:
            Timeout test results
        """
        results = {
            "test_name": "timeout_handling",
            "timeout_threshold": timeout_seconds,
            "total_queries": len(timeout_queries),
            "completed_in_time": 0,
            "timed_out": 0,
            "timeout_details": []
        }
        
        def run_with_timeout(query: str):
            return agent.invoke({
                "messages": [{"role": "user", "content": query}]
            })
        
        for query in timeout_queries:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_with_timeout, query)
                    response = future.result(timeout=timeout_seconds)
                    
                    results["completed_in_time"] += 1
                    
            except TimeoutError:
                results["timed_out"] += 1
                results["timeout_details"].append({
                    "query": query,
                    "timeout": timeout_seconds
                })
            
            except Exception as e:
                results["completed_in_time"] += 1  # Completed but with error
                
        self.test_results.append(results)
        return results
    
    def test_tool_usage(self, agent: Any, tool_test_queries: List[str]) -> Dict[str, Any]:
        """
        Test agent's ability to use tools correctly.
        
        Args:
            agent: Agent instance to test
            tool_test_queries: Queries that require tool usage
            
        Returns:
            Tool usage test results
        """
        results = {
            "test_name": "tool_usage",
            "total_queries": len(tool_test_queries),
            "correct_tool_usage": 0,
            "incorrect_tool_usage": 0,
            "tool_usage_details": []
        }
        
        for query in tool_test_queries:
            try:
                response = agent.invoke({
                    "messages": [{"role": "user", "content": query}]
                })
                
                # Analyze if tools were used appropriately
                tool_usage_analysis = self._analyze_tool_usage(response, query)
                
                if tool_usage_analysis["appropriate"]:
                    results["correct_tool_usage"] += 1
                else:
                    results["incorrect_tool_usage"] += 1
                    results["tool_usage_details"].append({
                        "query": query,
                        "issue": tool_usage_analysis["issue"],
                        "tools_used": tool_usage_analysis.get("tools_used", [])
                    })
                
            except Exception as e:
                results["incorrect_tool_usage"] += 1
                results["tool_usage_details"].append({
                    "query": query,
                    "issue": f"Exception during tool usage: {str(e)}"
                })
        
        self.test_results.append(results)
        return results
    
    def generate_test_report(self) -> str:
        """
        Generate comprehensive test report.
        
        Returns:
            Formatted test report
        """
        if not self.test_results:
            return "No tests were run."
        
        report = []
        report.append("=" * 60)
        report.append("LANGCHAIN AGENT TEST REPORT")
        report.append("=" * 60)
        
        total_tests = len(self.test_results)
        total_passed = 0
        total_failed = 0
        
        for result in self.test_results:
            report.append(f"\n{result['test_name'].upper()} TEST RESULTS:")
            report.append("-" * 40)
            
            if result["test_name"] == "basic_functionality":
                success_rate = (result["successful_queries"] / result["total_queries"]) * 100
                report.append(f"Success Rate: {success_rate:.1f}%")
                report.append(f"Successful Queries: {result['successful_queries']}/{result['total_queries']}")
                if result["response_times"]:
                    report.append(f"Average Response Time: {statistics.mean(result['response_times']):.2f}s")
                if result["errors"]:
                    report.append(f"Errors: {len(result['errors'])}")
                
                total_passed += result["successful_queries"]
                total_failed += result["failed_queries"]
            
            elif result["test_name"] == "performance":
                if "avg_response_time" in result:
                    report.append(f"Average Response Time: {result['avg_response_time']:.2f}s")
                    report.append(f"Min/Max Response Time: {result['min_response_time']:.2f}s / {result['max_response_time']:.2f}s")
                    report.append(f"Standard Deviation: {result['stdev_response_time']:.2f}s")
                if result["errors"]:
                    report.append(f"Errors: {len(result['errors'])}")
            
            elif result["test_name"] == "error_handling":
                error_handling_rate = (result["handled_errors"] / result["total_cases"]) * 100
                report.append(f"Error Handling Rate: {error_handling_rate:.1f}%")
                report.append(f"Handled Errors: {result['handled_errors']}/{result['total_cases']}")
                if result["error_details"]:
                    report.append(f"Unhandled Error Details: {len(result['error_details'])}")
            
            elif result["test_name"] == "timeout_handling":
                timeout_rate = (result["timed_out"] / result["total_queries"]) * 100
                report.append(f"Timeout Rate: {timeout_rate:.1f}%")
                report.append(f"Completed in Time: {result['completed_in_time']}/{result['total_queries']}")
                report.append(f"Timeout Threshold: {result['timeout_threshold']}s")
            
            elif result["test_name"] == "tool_usage":
                tool_usage_rate = (result["correct_tool_usage"] / result["total_queries"]) * 100
                report.append(f"Tool Usage Accuracy: {tool_usage_rate:.1f}%")
                report.append(f"Correct Tool Usage: {result['correct_tool_usage']}/{result['total_queries']}")
                if result["tool_usage_details"]:
                    report.append(f"Tool Usage Issues: {len(result['tool_usage_details'])}")
        
        # Summary
        report.append("\n" + "=" * 60)
        report.append("OVERALL SUMMARY")
        report.append("=" * 60)
        report.append(f"Total Tests Run: {total_tests}")
        
        if total_passed + total_failed > 0:
            overall_success = (total_passed / (total_passed + total_failed)) * 100
            report.append(f"Overall Success Rate: {overall_success:.1f}%")
            report.append(f"Total Passed: {total_passed}")
            report.append(f"Total Failed: {total_failed}")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def _validate_response(self, response: Any) -> bool:
        """Validate that agent response has expected structure."""
        try:
            if not isinstance(response, dict):
                return False
            
            if "messages" not in response:
                return False
            
            messages = response["messages"]
            if not isinstance(messages, list) or len(messages) == 0:
                return False
            
            last_message = messages[-1]
            if not isinstance(last_message, dict):
                return False
            
            if "content" not in last_message:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _is_error_handled(self, response: Any, expected_error_type: Optional[str] = None) -> bool:
        """Check if error was handled gracefully."""
        try:
            if not self._validate_response(response):
                return False
            
            last_message = response["messages"][-1]
            content = last_message["content"].lower()
            
            # Check for error indicators
            error_indicators = ["error", "failed", "unable", "cannot", "invalid"]
            has_error_indicator = any(indicator in content for indicator in error_indicators)
            
            # If we expected a specific error type, check for it
            if expected_error_type:
                return has_error_indicator and expected_error_type.lower() in content
            
            return has_error_indicator
            
        except Exception:
            return False
    
    def _analyze_tool_usage(self, response: Any, query: str) -> Dict[str, Any]:
        """Analyze if tools were used appropriately for the query."""
        # This is a simplified analysis - in practice, you'd want more sophisticated logic
        analysis = {
            "appropriate": True,
            "issue": "",
            "tools_used": []
        }
        
        try:
            # Look for tool usage indicators in the response
            content = str(response).lower()
            
            # Simple keyword-based analysis
            if "calculate" in query.lower() and "calculat" not in content:
                analysis["appropriate"] = False
                analysis["issue"] = "Calculation query but no calculation performed"
            
            if "process" in query.lower() and "process" not in content:
                analysis["appropriate"] = False
                analysis["issue"] = "Processing requested but no processing indicated"
            
            return analysis
            
        except Exception as e:
            return {
                "appropriate": False,
                "issue": f"Analysis failed: {str(e)}",
                "tools_used": []
            }

def run_comprehensive_tests(model: str = "claude-3-haiku-20240307") -> str:
    """
    Run comprehensive agent testing suite.
    
    Args:
        model: Model to use for testing
        
    Returns:
        Test report
    """
    tester = AgentTester()
    
    # Create test agent
    tools = tester.create_test_tools()
    
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="You are a helpful assistant with access to calculation, processing, and validation tools."
    )
    
    # Run tests
    print("Running basic functionality tests...")
    basic_queries = [
        "Calculate 15 + 27",
        "Process the text 'hello world' with uppercase",
        "Validate the data 'test123'",
        "What is 2 * 8?"
    ]
    tester.test_agent_creation(agent, basic_queries)
    
    print("Running performance tests...")
    tester.test_performance(agent, "Calculate 2 + 2", iterations=5)
    
    print("Running error handling tests...")
    error_cases = [
        {"query": "Calculate abc + def", "expected_error_type": "invalid"},
        {"query": "Process empty string with uppercase", "expected_error_type": "empty"},
        {"query": "Validate very long data " + "x" * 200, "expected_error_type": "long"}
    ]
    tester.test_error_handling(agent, error_cases)
    
    print("Running timeout tests...")
    timeout_queries = [
        "Run slow operation with 2 second delay",
        "Calculate complex expression: 1 + 2 + 3 + 4 + 5"
    ]
    tester.test_timeout_handling(agent, timeout_queries, timeout_seconds=3)
    
    print("Running tool usage tests...")
    tool_queries = [
        "Calculate 25 * 4",
        "Process 'test' with reverse operation",
        "Validate the string 'valid_data'"
    ]
    tester.test_tool_usage(agent, tool_queries)
    
    # Generate and return report
    return tester.generate_test_report()

if __name__ == "__main__":
    print("Running LangChain Agent Testing Suite...")
    print("=" * 60)
    
    # Run comprehensive tests
    report = run_comprehensive_tests()
    
    print(report)
    
    # Save report to file
    with open("agent_test_report.txt", "w") as f:
        f.write(report)
    
    print("\nTest report saved to 'agent_test_report.txt'")