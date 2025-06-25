"""
Pytest configuration and fixtures for the retail data pipeline tests.
"""
import pytest
import os
import sys
from pathlib import Path

# Add src directory to Python path for all tests
@pytest.fixture(scope="session", autouse=True)
def setup_python_path():
    """Automatically add src directory to Python path for all tests."""
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    src_dir = project_root / "src"
    
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    return str(src_dir)

@pytest.fixture
def sample_dataframe():
    """Provide a sample DataFrame for testing."""
    import pandas as pd
    return pd.DataFrame({
        'price': [10.0, 15.0, 20.0],
        'stock_level': [100, 200, 150],
        'signup_date': ['2023-01-01', '2023-02-01', '2023-03-01'],
        'country': ['USA', 'Canada', 'USA'],
        'category': ['Electronics', 'Books', 'Electronics'],
        'description': ['Good product', 'Great book', 'Amazing gadget']
    })

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory structure for testing."""
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    
    return data_dir
