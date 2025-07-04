# retail-data-pipeline

---

### ✅ Step 1: Data Collection & Integration

This step simulates the ingestion of multi-format data sources common in AI pipelines.

**📄 Input Datasets:**
- `customers.csv` — structured (tabular) customer data  
- `products.json` — semi-structured product catalog  
- `product_descriptions.txt` — unstructured descriptions in text format

**⚙️ What this step does:**
- Loads and parses CSV, JSON, and TXT files
- Normalizes and merges datasets on `product_id` and `customer_id`
- Outputs a unified dataset: `combined_data.csv`

**🧰 Tools used:** `pandas`, `json`, standard Python file I/O

> 🧠 *Purpose:* Prepare heterogeneous data for cleaning and transformation by integrating into a single DataFrame.

---

### ✅ Step 2: Data Cleaning

This step applies classic data preprocessing techniques to ready the dataset for modeling.

**📄 Input Dataset:**  
- `combined_data.csv` from Step 1

**⚙️ What this step does:**
- Handles missing values with imputation/fallbacks
- Removes duplicate entries
- Standardizes date formats and currency precision
- Tags outliers using z-score detection

**🧰 Tools used:** `pandas`, `numpy`, `datetime`

> 🧠 *Purpose:* Improve data quality, consistency, and reliability for downstream transformation and analysis.

---


### ✅ Step 3: Feature Engineering

This step prepares your cleaned dataset for modeling by transforming raw inputs into meaningful features.

**📄 Input Dataset:**  
- `cleaned_data.csv` from Step 2

**⚙️ What this step does:**
- Encodes categorical variables (One-Hot, Label Encoding)  
- Scales numerical variables (Standardization, MinMax)  
- Creates new features (e.g., `total_spend`, `days_since_signup`)  
- Flags temporal fields and applies aggregation (optional)

**🧰 Tools used:** `pandas`, `scikit-learn.preprocessing`

> 🧠 *Purpose:* Enhance model input quality, reduce noise, and inject signal through domain-specific transformations.


---

### ✅ Step 4: Pipeline Integration (Optional or Final Assembly)

This step connects all prior components into a cohesive, reusable script that performs end-to-end preprocessing.

**📄 Input Datasets:**  
- Raw data files (`customers.csv`, `products.json`, `product_descriptions.txt`)  
- Or intermediate files (`cleaned_data.csv`)

**⚙️ What this step does:**
- Runs all pipeline steps via CLI or main function  
- Accepts runtime arguments for custom inputs/outputs  
- Organizes outputs into `data/processed/` and `outputs/plots/` folders

**🧰 Tools used:** `argparse`, `os`, `pandas`, custom modules from `src/`

> 🧠 *Purpose:* Enable seamless automation of your data pipeline — reusable for demos, experimentation, or integration into larger ML systems.

---

## 🧪 Testing the Pipeline

### Environment Setup

**Option 1: Using pip**
```bash
pip install -r requirements.txt
```

**Option 2: Using conda**
```bash
conda env create -f environment.yml
conda activate retail-data-pipeline
```

### Quick Start Testing

**1. Set up your environment** (choose option 1 or 2 above)

**2. Run Automated Test Suite**
```bash
# Run the comprehensive test runner
python test_pipeline.py
```

This will validate:
- ✅ All required packages are installed
- ✅ Required data files exist
- ✅ Pipeline modules can be imported
- ✅ Pipeline components execute correctly

**3. Manual Testing (Alternative)**

**Verify Data Files** - Ensure these exist in `data/raw/`:
- `customers.csv`
- `products.json` 
- `product_descriptions.txt`

**Run the Complete Pipeline:**
```bash
# From project root
python src/pipeline.py

# Or from src directory
cd src
python pipeline.py
```

**Run Individual Components:**
```bash
# From project root
python src/data_collection.py
python src/cleaning.py
python src/feature_engineering.py

# Or from src directory
cd src
python data_collection.py
python cleaning.py
python feature_engineering.py
```

### Expected Outputs

After successful execution, you should see:
- `data/raw/combined_data.csv` - Merged dataset
- `data/processed/cleaned_data.csv` - Cleaned dataset
- `data/processed/transformed_features.csv` - Feature-engineered dataset

### Unit Testing (Optional)

Run the test suite to validate pipeline components:
```bash
# Install testing dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

### Troubleshooting

**Common Issues:**
- **Missing data files**: Ensure sample data exists in `data/raw/`
- **Import errors**: Verify you're running from the correct directory
- **Permission errors**: Check write permissions for `data/processed/` and `outputs/`
- **Environment issues**: 
  - If using conda: `conda activate retail-data-pipeline`
  - If using pip: Ensure virtual environment is activated
  - Check Python version compatibility (requires Python 3.9+)

**Validation Checks:**
- Pipeline should complete without errors
- Output files should contain expected columns
- No empty DataFrames in intermediate steps

**Environment Management:**
```bash
# Conda users
conda env list                    # List all environments
conda activate retail-data-pipeline  # Activate environment
conda deactivate                 # Deactivate environment
conda env remove -n retail-data-pipeline  # Remove environment

# Pip users (with virtual environment)
python -m venv venv              # Create virtual environment
source venv/bin/activate         # Activate (Linux/Mac)
venv\Scripts\activate            # Activate (Windows)
deactivate                       # Deactivate
```

---

## 🏗️ AI Solutions Architecture Overview

### Enterprise AI/ML Capabilities Demonstrated

**🎯 Business Intelligence & Analytics**
- Customer segmentation and behavior analysis
- Predictive modeling for high-value customer identification
- Data-driven insights for business strategy optimization
- Automated reporting and visualization dashboards

**🔧 Technical Architecture**
- **Scalable Data Pipeline**: Multi-format data ingestion (CSV, JSON, TXT)
- **Feature Engineering Framework**: Automated preprocessing and transformation
- **ML Model Training**: Multi-algorithm comparison and hyperparameter tuning
- **Model Deployment Ready**: Serialized models with versioning and metadata
- **Quality Assurance**: Comprehensive testing and validation framework

**🚀 Production-Ready Components**
- CI/CD pipeline with automated testing
- Environment management (conda + pip)
- Error handling and logging
- Modular, maintainable codebase
- Documentation and reproducibility

### AI/ML Use Cases Supported

1. **Customer Lifetime Value Prediction** 📈
   - Predict high-value customers using behavioral patterns
   - ROI: Improved marketing spend efficiency

2. **Dynamic Pricing Optimization** 💰
   - Price elasticity analysis by customer segment
   - ROI: Revenue optimization through data-driven pricing

3. **Personalized Recommendations** 🎯
   - Product recommendation engine based on preferences
   - ROI: Increased cross-sell and customer satisfaction

4. **Churn Prevention** 🛡️
   - Early warning system for at-risk customers
   - ROI: Reduced customer acquisition costs

### Technology Stack

**Data Engineering**: pandas, numpy, scikit-learn  
**Machine Learning**: Random Forest, Gradient Boosting, Logistic Regression  
**Visualization**: matplotlib, seaborn, plotly  
**Testing**: pytest, automated CI/CD  
**Deployment**: Docker-ready, cloud-native architecture  

---

## 🧪 Testing the AI/ML Features

### Quick Start: Test Everything

**1. Install Enhanced Dependencies**
```bash
# Install all packages including ML libraries
pip install -r requirements.txt

# Or with conda
conda env create -f environment.yml
conda activate retail-data-pipeline
```

**2. Run Complete AI/ML Pipeline**
```bash
# First run the data pipeline to generate features
python src/pipeline.py

# Then run the ML pipeline
python src/ml_pipeline.py
```

**3. Explore the Jupyter Analysis**
```bash
# Start Jupyter and open the analysis notebook
jupyter notebook notebooks/AI_Solutions_Architect_Data_Analysis.ipynb
```

### Step-by-Step Testing Guide

#### 🔸 **Test 1: Data Pipeline (Existing)**
```bash
# Test basic data processing
python test_pipeline.py

# Expected output: ✅ All tests passed!
```

#### 🔸 **Test 2: ML Pipeline (New)**
```bash
# Run the complete ML pipeline
python src/ml_pipeline.py

# Expected outputs:
# - models/ directory with saved models
# - outputs/ml_results/ with performance metrics
# - outputs/ml_results/plots/ with visualizations
```

#### 🔸 **Test 3: Jupyter Analysis Notebook (New)**
```bash
# Launch Jupyter
jupyter notebook

# Navigate to: notebooks/AI_Solutions_Architect_Data_Analysis.ipynb
# Run all cells to see comprehensive data analysis
```

#### 🔸 **Test 4: Individual ML Components**
```bash
# Test feature engineering validation
python -c "
import pandas as pd
from pathlib import Path
df = pd.read_csv('data/processed/transformed_features.csv')
print('✅ Features loaded successfully')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
"

# Test ML pipeline imports
python -c "
import sys
sys.path.append('src')
from ml_pipeline import RetailMLPipeline
print('✅ ML pipeline imports successfully')
"
```

### Expected Test Results

#### 📊 **After Running ML Pipeline**
You should see these new directories and files:

```
retail-data-pipeline/
├── models/
│   ├── best_model_YYYYMMDD_HHMMSS.joblib
│   └── scaler_YYYYMMDD_HHMMSS.joblib
├── outputs/
│   └── ml_results/
│       ├── model_results_YYYYMMDD_HHMMSS.json
│       └── plots/
│           └── ml_performance_analysis.png
```

#### 🎯 **Expected Console Output**
```
🚀 Starting Complete ML Pipeline...
==================================================
🔄 Loading transformed data...
🤖 Training multiple ML models...
  Training Logistic Regression...
    AUC Score: 0.XXX
    CV Score: 0.XXX (+/- 0.XXX)
  Training Random Forest...
    AUC Score: 0.XXX
    CV Score: 0.XXX (+/- 0.XXX)
🏆 Best model: Random Forest (AUC: 0.XXX)
🔧 Performing hyperparameter tuning...
📊 Generating business insights...
📈 Creating visualizations...
💾 Saving model and results...
==================================================
🎉 ML Pipeline Complete!
📊 Business Insights: {...}
```

#### 📓 **Jupyter Notebook Results**
The analysis notebook will show:
- Data quality assessment (✅ 90%+ score expected)
- Feature distribution plots
- Correlation analysis
- Model readiness evaluation
- Business insight recommendations

### Troubleshooting New Features

**Issue: Import errors for ML libraries**
```bash
# Solution: Install missing packages
pip install scikit-learn matplotlib seaborn joblib

# Or reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**Issue: Jupyter notebook not opening**
```bash
# Solution: Install and configure Jupyter
pip install jupyter notebook ipykernel
python -m ipykernel install --user --name=retail-pipeline
jupyter notebook --ip=0.0.0.0 --port=8888
```

**Issue: ML pipeline fails on small dataset**
```bash
# This is expected with only 5 sample records
# The pipeline will still run and demonstrate capabilities
# For production, you'd need larger datasets
```

**Issue: Plots not displaying**
```bash
# Solution: Install plotting backend
pip install matplotlib seaborn
# For Jupyter: pip install ipympl
```

### Performance Validation

**✅ What Good Results Look Like:**
- Data pipeline: Completes without errors, generates 3 CSV files
- ML pipeline: AUC scores > 0.5 (random baseline), models saved successfully
- Jupyter analysis: All cells execute, visualizations render properly
- File outputs: All expected directories and files created

**⚠️ Expected Limitations (This is Normal):**
- Small dataset (5 records) limits ML performance
- Cross-validation may show high variance
- Some advanced features need more data to be meaningful

**🎯 Portfolio Value:**
- Demonstrates end-to-end ML capability
- Shows production-ready code structure  
- Exhibits data science methodology
- Proves business acumen and technical depth

---
