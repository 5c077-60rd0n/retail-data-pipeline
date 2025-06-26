#!/usr/bin/env python3
"""
Comprehensive test runner for all AI/ML features in the retail data pipeline.
This script validates the complete portfolio showcase for AI Solutions Architect role.
"""

import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
import importlib.util

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

def print_status(message, success=True):
    """Print a status message with emoji."""
    emoji = "✅" if success else "❌"
    print(f"{emoji} {message}")

def test_environment_setup():
    """Test that the environment is properly set up."""
    print_section("ENVIRONMENT SETUP TEST")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 
        'seaborn', 'joblib', 'jupyter'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print_status(f"{package} installed")
        except ImportError:
            missing_packages.append(package)
            print_status(f"{package} missing", success=False)
    
    if missing_packages:
        print(f"\n💡 Install missing packages: pip install {' '.join(missing_packages)}")
        return False
    
    print_status("All required packages installed!")
    return True

def test_data_pipeline():
    """Test the basic data pipeline."""
    print_section("DATA PIPELINE TEST")
    
    try:
        # Run the data pipeline
        print("🔄 Running data pipeline...")
        result = subprocess.run([sys.executable, "src/pipeline.py"], 
                              capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print_status("Data pipeline executed successfully")
            
            # Check outputs
            expected_files = [
                "data/raw/combined_data.csv",
                "data/processed/cleaned_data.csv", 
                "data/processed/transformed_features.csv"
            ]
            
            for file_path in expected_files:
                if Path(file_path).exists():
                    print_status(f"Output file created: {file_path}")
                else:
                    print_status(f"Missing output file: {file_path}", success=False)
                    return False
            
            return True
        else:
            print_status(f"Data pipeline failed: {result.stderr}", success=False)
            return False
            
    except Exception as e:
        print_status(f"Error running data pipeline: {e}", success=False)
        return False

def test_ml_pipeline():
    """Test the ML pipeline."""
    print_section("ML PIPELINE TEST")
    
    try:
        # Check if transformed features exist
        features_path = Path("data/processed/transformed_features.csv")
        if not features_path.exists():
            print_status("Transformed features not found, running data pipeline first...", success=False)
            if not test_data_pipeline():
                return False
        
        print("🤖 Running ML pipeline...")
        result = subprocess.run([sys.executable, "src/ml_pipeline.py"], 
                              capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print_status("ML pipeline executed successfully")
            
            # Check for model outputs
            models_dir = Path("models")
            results_dir = Path("outputs/ml_results")
            
            if models_dir.exists() and any(models_dir.glob("*.joblib")):
                print_status("ML models saved successfully")
            else:
                print_status("ML models not found", success=False)
            
            if results_dir.exists() and any(results_dir.glob("*.json")):
                print_status("ML results saved successfully")
            else:
                print_status("ML results not found", success=False)
            
            plots_dir = Path("outputs/ml_results/plots")
            if plots_dir.exists() and any(plots_dir.glob("*.png")):
                print_status("ML visualizations created successfully")
            else:
                print_status("ML visualizations not found", success=False)
            
            return True
        else:
            print_status(f"ML pipeline failed: {result.stderr}", success=False)
            return False
            
    except Exception as e:
        print_status(f"Error running ML pipeline: {e}", success=False)
        return False

def test_jupyter_notebook():
    """Test that Jupyter notebook can be accessed."""
    print_section("JUPYTER NOTEBOOK TEST")
    
    try:
        # Check if notebook file exists
        notebook_path = Path("notebooks/AI_Solutions_Architect_Data_Analysis.ipynb")
        if notebook_path.exists():
            print_status("Analysis notebook found")
            
            # Try to validate notebook structure
            import json
            with open(notebook_path, 'r') as f:
                notebook_data = json.load(f)
            
            if 'cells' in notebook_data and len(notebook_data['cells']) > 0:
                print_status(f"Notebook has {len(notebook_data['cells'])} cells")
                return True
            else:
                print_status("Notebook appears empty", success=False)
                return False
        else:
            print_status("Analysis notebook not found", success=False)
            return False
            
    except Exception as e:
        print_status(f"Error checking notebook: {e}", success=False)
        return False

def test_feature_validation():
    """Test specific AI/ML features."""
    print_section("FEATURE VALIDATION TEST")
    
    try:
        # Test data loading
        features_path = Path("data/processed/transformed_features.csv")
        if features_path.exists():
            df = pd.read_csv(features_path)
            print_status(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Check for expected columns
            expected_columns = ['price', 'total_spend', 'country_encoded']
            found_columns = [col for col in expected_columns if col in df.columns]
            print_status(f"Key features found: {len(found_columns)}/{len(expected_columns)}")
            
            # Check for categorical features
            cat_columns = [col for col in df.columns if col.startswith('cat_')]
            print_status(f"Category features found: {len(cat_columns)}")
            
            return len(found_columns) == len(expected_columns)
        else:
            print_status("Transformed features not found", success=False)
            return False
            
    except Exception as e:
        print_status(f"Error validating features: {e}", success=False)
        return False

def test_imports():
    """Test that all custom modules can be imported."""
    print_section("MODULE IMPORT TEST")
    
    try:
        sys.path.insert(0, 'src')
        
        # Test data pipeline imports
        from data_collection import load_customers, merge_data
        print_status("Data collection module imported")
        
        from cleaning import clean_data
        print_status("Data cleaning module imported")
        
        from feature_engineering import feature_engineering
        print_status("Feature engineering module imported")
        
        # Test ML pipeline import
        try:
            from ml_pipeline import RetailMLPipeline
            print_status("ML pipeline module imported")
        except ImportError as e:
            print_status(f"ML pipeline import failed: {e}", success=False)
            return False
        
        return True
        
    except Exception as e:
        print_status(f"Import error: {e}", success=False)
        return False

def run_comprehensive_tests():
    """Run all tests and provide summary."""
    print("🚀 COMPREHENSIVE AI/ML PORTFOLIO TEST")
    print("Testing all features for AI Solutions Architect showcase")
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Module Imports", test_imports),
        ("Data Pipeline", test_data_pipeline),
        ("Feature Validation", test_feature_validation),
        ("ML Pipeline", test_ml_pipeline),
        ("Jupyter Notebook", test_jupyter_notebook)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_status(f"Test {test_name} crashed: {e}", success=False)
            results.append((test_name, False))
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        print_status(f"{test_name}: {'PASSED' if result else 'FAILED'}", success=result)
    
    print(f"\n🏆 OVERALL SCORE: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 EXCELLENT! All AI/ML features working perfectly!")
        print("📈 Your portfolio demonstrates:")
        print("   ✅ End-to-end ML pipeline capability")
        print("   ✅ Production-ready data engineering")
        print("   ✅ Comprehensive analysis and visualization")
        print("   ✅ Professional software development practices")
    elif passed >= total * 0.8:
        print("\n🎯 GOOD! Most features working with minor issues.")
        print("💡 Address failing tests for complete portfolio showcase.")
    else:
        print("\n⚠️ NEEDS WORK! Several critical features need attention.")
        print("🔧 Focus on fixing failed tests before showcasing to employers.")
    
    print(f"\n📋 NEXT STEPS:")
    if passed == total:
        print("   • Run 'jupyter notebook' to explore the analysis")
        print("   • Review generated models in models/ directory")
        print("   • Check visualizations in outputs/ml_results/plots/")
        print("   • Ready to showcase to potential employers!")
    else:
        print("   • Fix failing tests using troubleshooting guide in README")
        print("   • Ensure all dependencies are installed")
        print("   • Re-run this test script until all tests pass")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
