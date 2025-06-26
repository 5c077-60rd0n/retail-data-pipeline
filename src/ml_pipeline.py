"""
ML Model Training and Evaluation Module for Retail Data Pipeline

This module demonstrates AI/ML capabilities for an AI Solutions Architect portfolio.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import json
from datetime import datetime

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Paths
TRANSFORMED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "transformed_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "ml_results"

class RetailMLPipeline:
    """
    Comprehensive ML pipeline for retail data analysis and prediction.
    
    This class demonstrates enterprise-level ML capabilities including:
    - Multi-model comparison
    - Hyperparameter tuning
    - Model evaluation and validation
    - Business metric calculation
    - Model persistence and versioning
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()
        
        # Ensure output directories exist
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load and prepare the transformed features for ML."""
        print("🔄 Loading transformed data...")
        df = pd.read_csv(TRANSFORMED_DATA_PATH)
        
        # Create target variable: high-value customer classification
        # (This is a business-relevant ML problem)
        df['high_value_customer'] = (df['total_spend'] > df['total_spend'].median()).astype(int)
        
        # Prepare features (exclude non-predictive columns)
        feature_columns = [col for col in df.columns if col not in 
                          ['product_id', 'customer_id', 'name', 'email', 'description', 
                           'signup_date', 'high_value_customer']]
        
        X = df[feature_columns]
        y = df['high_value_customer']
        
        return X, y, df
    
    def train_models(self, X, y):
        """Train multiple ML models and compare performance."""
        print("🤖 Training multiple ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"  Training {name}...")
            
            # Use scaled data for logistic regression, original for tree-based
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_test, y_pred_proba)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            # Store results
            self.results[name] = {
                'model': model,
                'auc_score': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"    AUC Score: {auc_score:.3f}")
            print(f"    CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Select best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc_score'])
        self.best_model = self.results[best_model_name]['model']
        
        print(f"🏆 Best model: {best_model_name} (AUC: {self.results[best_model_name]['auc_score']:.3f})")
        
        return X_test, y_test
    
    def hyperparameter_tuning(self, X, y):
        """Perform hyperparameter tuning on the best model."""
        print("🔧 Performing hyperparameter tuning...")
        
        # Example: Tune Random Forest (assuming it's often the best)
        rf = RandomForestClassifier(random_state=42)
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        
        # Update best model
        self.best_model = grid_search.best_estimator_
        
        return grid_search
    
    def generate_business_insights(self, X, y, df):
        """Generate business-relevant insights from the ML model."""
        print("📊 Generating business insights...")
        
        insights = {}
        
        # Feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            insights['top_features'] = feature_importance.head(10).to_dict('records')
        
        # Customer segmentation insights
        predictions = self.best_model.predict_proba(X)[:, 1]
        df['predicted_value_score'] = predictions
        
        # Business metrics
        high_value_threshold = 0.7
        predicted_high_value = (predictions > high_value_threshold).sum()
        total_customers = len(df)
        
        insights['business_metrics'] = {
            'predicted_high_value_customers': int(predicted_high_value),
            'total_customers': int(total_customers),
            'high_value_percentage': round(predicted_high_value / total_customers * 100, 2),
            'model_confidence_threshold': high_value_threshold
        }
        
        return insights
    
    def save_model_and_results(self, insights):
        """Save trained model and results for production use."""
        print("💾 Saving model and results...")
        
        # Save model
        model_path = MODELS_DIR / f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(self.best_model, model_path)
        
        # Save scaler
        scaler_path = MODELS_DIR / f"scaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(self.scaler, scaler_path)
        
        # Save results
        results_path = RESULTS_DIR / f"model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Prepare results for JSON serialization
        json_results = {}
        for model_name, result in self.results.items():
            json_results[model_name] = {
                'auc_score': float(result['auc_score']),
                'cv_mean': float(result['cv_mean']),
                'cv_std': float(result['cv_std']),
                'classification_report': result['classification_report']
            }
        
        json_results['business_insights'] = insights
        json_results['model_path'] = str(model_path)
        json_results['scaler_path'] = str(scaler_path)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✅ Model saved to: {model_path}")
        print(f"✅ Results saved to: {results_path}")
        
        return model_path, results_path
    
    def create_visualizations(self, X_test, y_test):
        """Create ML performance visualizations."""
        print("📈 Creating visualizations...")
        
        # Create plots directory
        plots_dir = RESULTS_DIR / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # ROC Curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
            plt.plot(fpr, tpr, label=f"{name} (AUC: {result['auc_score']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Feature Importance
        plt.subplot(1, 3, 2)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            plt.barh(range(len(feature_importance)), feature_importance['importance'])
            plt.yticks(range(len(feature_importance)), feature_importance['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 10 Feature Importance')
            plt.gca().invert_yaxis()
        
        # Model Comparison
        plt.subplot(1, 3, 3)
        model_names = list(self.results.keys())
        auc_scores = [self.results[name]['auc_score'] for name in model_names]
        
        plt.bar(model_names, auc_scores)
        plt.ylabel('AUC Score')
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        for i, score in enumerate(auc_scores):
            plt.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'ml_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizations saved to: {plots_dir}")
    
    def run_complete_pipeline(self):
        """Execute the complete ML pipeline."""
        print("🚀 Starting Complete ML Pipeline...")
        print("=" * 50)
        
        # Load data
        X, y, df = self.load_data()
        
        # Train models
        X_test, y_test = self.train_models(X, y)
        
        # Hyperparameter tuning
        grid_search = self.hyperparameter_tuning(X, y)
        
        # Generate insights
        insights = self.generate_business_insights(X, y, df)
        
        # Create visualizations
        self.create_visualizations(X_test, y_test)
        
        # Save everything
        model_path, results_path = self.save_model_and_results(insights)
        
        print("=" * 50)
        print("🎉 ML Pipeline Complete!")
        print(f"📊 Business Insights: {insights['business_metrics']}")
        print("=" * 50)
        
        return insights, model_path, results_path


def main():
    """Main execution function."""
    pipeline = RetailMLPipeline()
    insights, model_path, results_path = pipeline.run_complete_pipeline()
    
    # Print summary
    print("\n📋 EXECUTIVE SUMMARY")
    print("-" * 30)
    metrics = insights['business_metrics']
    print(f"• Total Customers Analyzed: {metrics['total_customers']}")
    print(f"• Predicted High-Value Customers: {metrics['predicted_high_value_customers']}")
    print(f"• High-Value Customer Rate: {metrics['high_value_percentage']}%")
    print(f"• Model Confidence Threshold: {metrics['model_confidence_threshold']}")
    print(f"• Model Saved: {model_path}")
    print(f"• Results Saved: {results_path}")


if __name__ == "__main__":
    main()
