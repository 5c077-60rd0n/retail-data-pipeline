#!/usr/bin/env python3
"""
Test runner for the retail data pipeline.
This script validates that the pipeline components work correctly.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all required modules can be imported."""
    try:
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        print("✅ All required packages are available")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("\n💡 Setup Instructions:")
        print("   For pip users: pip install -r requirements.txt")
        print("   For conda users: conda env create -f environment.yml")
        print("                   conda activate retail-data-pipeline")
        return False

def test_data_files():
    """Test that required data files exist."""
    required_files = [
        'data/raw/customers.csv',
        'data/raw/products.json',
        'data/raw/product_descriptions.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing data files: {missing_files}")
        return False
    else:
        print("✅ All required data files exist")
        return True

def test_pipeline_modules():
    """Test that pipeline modules can be imported."""
    try:
        from data_collection import load_customers, load_products, load_descriptions, merge_data
        from cleaning import clean_data
        from feature_engineering import feature_engineering
        print("✅ All pipeline modules can be imported")
        return True
    except ImportError as e:
        print(f"❌ Cannot import pipeline modules: {e}")
        return False

def test_pipeline_execution():
    """Test basic pipeline execution (dry run)."""
    try:
        os.chdir(src_path)
        
        # Test if we can create the basic structure
        sample_df = pd.DataFrame({
            'price': [10.0, 15.0, 20.0],
            'stock_level': [100, 200, 150],
            'signup_date': ['2023-01-01', '2023-02-01', '2023-03-01'],
            'country': ['USA', 'Canada', 'USA'],
            'category': ['Electronics', 'Books', 'Electronics'],
            'description': ['Good product', 'Great book', 'Amazing gadget']
        })
        
        # Test data cleaning
        from cleaning import clean_data
        cleaned_df = clean_data(sample_df.copy())
        
        # Test feature engineering
        from feature_engineering import feature_engineering
        transformed_df = feature_engineering(cleaned_df.copy())
        
        print("✅ Pipeline components execute successfully with sample data")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        return False
    finally:
        os.chdir(project_root)

def run_all_tests():
    """Run all validation tests."""
    print("🧪 Running Retail Data Pipeline Tests\n")
    
    tests = [
        ("Package Dependencies", test_imports),
        ("Data Files", test_data_files),
        ("Pipeline Modules", test_pipeline_modules),
        ("Pipeline Execution", test_pipeline_execution)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        result = test_func()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("="*50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please address the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
