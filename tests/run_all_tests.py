#!/usr/bin/env python3
"""
Comprehensive test runner for the Protein Design System
Runs all unit tests and provides detailed reporting
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def run_test_file(test_file):
    """Run a single test file and return results"""
    print(f"\n{'='*60}")
    print(f"Running tests in: {test_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Set PYTHONPATH to include the project root
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
        
        # Run pytest on the specific file
        result = subprocess.run([
            sys.executable, '-m', 'pytest', test_file, '-v', '--tb=short'
        ], env=env, capture_output=True, text=True, cwd=os.getcwd())
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"PASS: {test_file} ({duration:.2f}s)")
            return True, result.stdout, result.stderr, duration
        else:
            print(f"FAIL: {test_file} ({duration:.2f}s)")
            return False, result.stdout, result.stderr, duration
            
    except Exception as e:
        print(f"ERROR: {test_file} - {e}")
        return False, "", str(e), 0

def run_all_tests():
    """Run all test files and provide summary"""
    print("Protein Design System - Test Suite")
    print("=" * 60)
    
    # Find all test files
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob("test_*.py"))
    
    if not test_files:
        print("No test files found!")
        return
    
    print(f"Found {len(test_files)} test files:")
    for tf in test_files:
        print(f"  - {tf.name}")
    
    # Run tests
    results = []
    total_tests = 0
    passed_tests = 0
    total_duration = 0
    
    for test_file in test_files:
        success, stdout, stderr, duration = run_test_file(test_file)
        
        # Parse test count from output
        test_count = 0
        if success:
            # Count dots in pytest output (each dot is a test)
            test_count = stdout.count('.')
            passed_tests += test_count
        
        total_tests += test_count
        total_duration += duration
        
        results.append({
            'file': test_file.name,
            'success': success,
            'test_count': test_count,
            'duration': duration,
            'stdout': stdout,
            'stderr': stderr
        })
        
        # Show output for failed tests
        if not success:
            if stdout:
                print("\nSTDOUT:")
                print(stdout)
            if stderr:
                print("\nSTDERR:")
                print(stderr)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    successful_files = sum(1 for r in results if r['success'])
    failed_files = len(results) - successful_files
    
    print(f"Files tested: {len(results)}")
    print(f"Files passed: {successful_files}")
    print(f"Files failed: {failed_files}")
    print(f"Total tests: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Total duration: {total_duration:.2f}s")
    
    if failed_files > 0:
        print(f"\nFAILED TEST FILES:")
        for r in results:
            if not r['success']:
                print(f"  - {r['file']}")
    
    print(f"\n{'='*60}")
    if failed_files == 0:
        print("ALL TESTS PASSED! The Protein Design System is working correctly.")
    else:
        print(f"{failed_files} test file(s) failed. Please check the output above.")
    print(f"{'='*60}")
    
    return failed_files == 0

def run_specific_tests():
    """Run specific test categories"""
    print("Running Specific Test Categories")
    print("=" * 60)
    
    test_categories = {
        "Core Components": ["test_fsa_occurrences.py", "test_transformer_model.py"],
        "Data Pipeline": ["test_data_contract.py"],
        "Controller": ["test_controller.py"],
        "Safety & Ledger": ["test_safety.py"],
        "Retrieval": ["test_retrieval.py"],
        "Generation": ["test_generation.py"],
        "Utilities": ["test_utils.py"],
        "Geometry": ["test_geometry.py", "test_tmalign_mock.py"]
    }
    
    for category, test_files in test_categories.items():
        print(f"\n{category}")
        print("-" * 40)
        
        for test_file in test_files:
            test_path = Path(__file__).parent / test_file
            if test_path.exists():
                success, stdout, stderr, duration = run_test_file(test_path)
                if not success:
                    print(f"  FAIL: {test_file}")
                else:
                    print(f"  PASS: {test_file}")
            else:
                print(f"  NOT FOUND: {test_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--specific":
        run_specific_tests()
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)
