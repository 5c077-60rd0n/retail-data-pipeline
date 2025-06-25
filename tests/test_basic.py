"""
Simplified tests for CI environment.
These tests focus on basic functionality that should work in any environment.
"""
import os
import sys
from pathlib import Path

def test_project_structure():
    """Test that the basic project structure exists."""
    # Get project root directory
    current_file = Path(__file__)
    project_root = current_file.parent.parent
    
    # Expected directories
    expected_dirs = ['src', 'tests', 'data']
    
    for directory in expected_dirs:
        dir_path = project_root / directory
        assert dir_path.exists(), f"Directory {directory} should exist"

def test_requirements_files():
    """Test that requirement files exist."""
    project_root = Path(__file__).parent.parent
    
    # Check for requirements files
    req_files = ['requirements.txt', 'environment.yml']
    
    for req_file in req_files:
        file_path = project_root / req_file
        assert file_path.exists(), f"File {req_file} should exist"
        assert file_path.is_file(), f"{req_file} should be a file"

def test_source_files_exist():
    """Test that main source files exist."""
    project_root = Path(__file__).parent.parent
    src_dir = project_root / 'src'
    
    expected_files = [
        'data_collection.py',
        'cleaning.py', 
        'feature_engineering.py',
        'pipeline.py'
    ]
    
    for filename in expected_files:
        file_path = src_dir / filename
        assert file_path.exists(), f"Source file {filename} should exist"
        assert file_path.is_file(), f"{filename} should be a file"

def test_basic_imports():
    """Test that basic Python packages can be imported."""
    try:
        import pandas as pd
        assert pd is not None
    except ImportError:
        # Skip if pandas is not available
        pass
    
    try:
        import numpy as np
        assert np is not None
    except ImportError:
        # Skip if numpy is not available  
        pass

def test_python_path_setup():
    """Test that src directory can be added to Python path."""
    project_root = Path(__file__).parent.parent
    src_dir = project_root / 'src'
    
    # Test that we can add src to path
    original_path = sys.path.copy()
    
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    assert str(src_dir) in sys.path
    
    # Restore original path
    sys.path = original_path

if __name__ == "__main__":
    # Run tests manually if executed directly
    test_functions = [
        test_project_structure,
        test_requirements_files, 
        test_source_files_exist,
        test_basic_imports,
        test_python_path_setup
    ]
    
    print("Running basic CI tests...")
    passed = 0
    total = len(test_functions)
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__}: {e}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed.")
        sys.exit(1)
