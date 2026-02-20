"""
March Madness Model Evaluation and Analysis

This script provides tools for evaluating the model's performance
and analyzing feature importance.

Usage:
1. Train a model first using march_madness_predictor.py
2. Run this script: python model_evaluation.py

Requirements:
- Trained model file (march_madness_model.pkl)
- NCAA data files in the data/ directory
- matplotlib, seaborn for visualizations
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, roc_curve, auc, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

# Import the main predictor module
sys.path.append(".")
from march_madness_predictor import (
    load_data, load_model, prepare_tournament_data, train_model
)

# Configuration
DATA_DIR = "data/"
MODEL_PATH = "models/march_madness_model.pkl"
RESULTS_DIR = "results/"
TEST_SEASONS = [2023, 2024]  # Use the most recent tournaments for testing

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_model_performance(model_dict, features_df):
    """
    Evaluate model performance on historical tournament data.
    """
    # Extract model components
    model = model_dict['model']
    scaler = model_dict['scaler']
    feature_names = model_dict['feature_names']
    
    # Split data into features and target
    X = features_df[feature_names]
    y = features_df['result']
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate log loss
    loss = log_loss(y, y_pred_proba)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Generate confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Generate classification report
    cr = classification_report(y, y_pred, output_dict=True)
    
    # Print results
    print(f"Log Loss: {loss:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Create ROC curve plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    
    return {
        'log_loss': loss,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': cr,
        'y_true': y,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def analyze_feature_importance(model_dict, features_df):
    """
    Analyze and visualize feature importance.
    """
    # Extract model components
    model = model_dict['model']
    scaler = model_dict['scaler']
    feature_names = model_dict['feature_names']
    
    # Get feature importance from model
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Print top features
    print("\nTop 20 Most Important Features:")
    for i, (feature, importance) in enumerate(
        zip(feature_importance['feature'].head(20), feature_importance['importance'].head(20)), 1
    ):
        print(f"{i:2d}. {feature:<30} {importance:.4f}")
    
    # Create feature importance plot
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    
    # Calculate permutation importance (more robust measure)
    X = features_df[feature_names]
    y = features_df['result']
    X_scaled = scaler.transform(X)
    
    result = permutation_importance(
        model, X_scaled, y, n_repeats=10, random_state=42, n_jobs=-1
    )
    
    # Create permutation importance DataFrame
    perm_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': result.importances_mean,
        'std': result.importances_std
    }).sort_values('importance', ascending=False)
    
    # Print permutation importance
    print("\nTop 20 Features (Permutation Importance):")
    for i, (feature, importance, std) in enumerate(
        zip(perm_importance['feature'].head(20), 
            perm_importance['importance'].head(20),
            perm_importance['std'].head(20)), 1
    ):
        print(f"{i:2d}. {feature:<30} {importance:.4f} ± {std:.4f}")
    
    # Create permutation importance plot
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=perm_importance.head(20))
    plt.title('Permutation Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'permutation_importance.png'), dpi=300, bbox_inches='tight')
    
    return {
        'feature_importance': feature_importance,
        'permutation_importance': perm_importance
    }

def analyze_seed_performance(features_df):
    """
    Analyze how well seeds predict tournament outcomes.
    """
    # Check if seed_diff is in the features
    if 'seed_diff' not in features_df.columns:
        print("Seed difference feature not found in dataset.")
        return None
    
    # Group by seed difference and calculate win rate
    seed_performance = features_df.groupby('seed_diff').agg(
        games=('result', 'count'),
        wins=('result', 'sum'),
        win_rate=('result', 'mean')
    ).reset_index()
    
    # Print seed performance
    print("\nSeed Difference Performance:")
    print("(Negative seed_diff means higher seed vs. lower seed)")
    print(seed_performance)
    
    # Create seed performance plot
    plt.figure(figsize=(12, 8))
    sns.lineplot(x='seed_diff', y='win_rate', data=seed_performance, marker='o')
    plt.axhline(y=0.5, color='red', linestyle='--')
    plt.title('Win Rate by Seed Difference')
    plt.xlabel('Seed Difference (Team A - Team B)')
    plt.ylabel('Win Rate for Team A')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, 'seed_performance.png'), dpi=300, bbox_inches='tight')
    
    return seed_performance

def analyze_upsets(features_df):
    """
    Analyze upset probabilities based on seed differences.
    """
    # Create upset feature
    features_df['upset'] = ((features_df['seed_diff'] > 0) & (features_df['result'] == 1)) | \
                         ((features_df['seed_diff'] < 0) & (features_df['result'] == 0))
    
    # Calculate upset rate by absolute seed difference
    features_df['abs_seed_diff'] = abs(features_df['seed_diff'])
    
    upset_analysis = features_df.groupby('abs_seed_diff').agg(
        games=('result', 'count'),
        upsets=('upset', 'sum'),
        upset_rate=('upset', 'mean')
    ).reset_index()
    
    # Print upset analysis
    print("\nUpset Analysis by Seed Difference:")
    print(upset_analysis)
    
    # Create upset rate plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='abs_seed_diff', y='upset_rate', data=upset_analysis)
    plt.title('Upset Rate by Absolute Seed Difference')
    plt.xlabel('Absolute Seed Difference')
    plt.ylabel('Upset Rate')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'upset_analysis.png'), dpi=300, bbox_inches='tight')
    
    # Analyze upsets by round
    if 'round' in features_df.columns:
        round_upset = features_df.groupby('round').agg(
            games=('result', 'count'),
            upsets=('upset', 'sum'),
            upset_rate=('upset', 'mean')
        ).reset_index()
        
        print("\nUpset Analysis by Tournament Round:")
        print(round_upset)
        
        # Create round upset plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='round', y='upset_rate', data=round_upset)
        plt.title('Upset Rate by Tournament Round')
        plt.xlabel('Tournament Round')
        plt.ylabel('Upset Rate')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'round_upset_analysis.png'), dpi=300, bbox_inches='tight')
    
    return upset_analysis

def analyze_model_calibration(eval_results):
    """
    Analyze how well the model's probabilities are calibrated.
    """
    y_true = eval_results['y_true']
    y_pred_proba = eval_results['y_pred_proba']
    
    # Create bins for probability ranges
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    
    # Calculate actual win rate in each bin
    bin_counts = np.bincount(bin_indices, minlength=len(bins) - 1)
    bin_wins = np.bincount(bin_indices, weights=y_true, minlength=len(bins) - 1)
    bin_win_rates = np.divide(bin_wins, bin_counts, out=np.zeros_like(bin_wins, dtype=float), where=bin_counts > 0)
    
    # Create DataFrame for analysis
    calibration_df = pd.DataFrame({
        'bin_center': bin_centers,
        'predicted_prob': bin_centers,
        'actual_prob': bin_win_rates,
        'games': bin_counts
    })
    
    # Print calibration analysis
    print("\nModel Calibration Analysis:")
    print("(Predicted probability vs. actual win rate)")
    print(calibration_df)
    
    # Create calibration plot
    plt.figure(figsize=(10, 8))
    
    # Plot bins with data points
    sns.scatterplot(
        x='predicted_prob', 
        y='actual_prob', 
        size='games',
        sizes=(20, 200),
        alpha=0.7,
        data=calibration_df[calibration_df['games'] > 0]
    )
    
    # Add perfect calibration line
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Win Rate')
    plt.title('Model Calibration Analysis')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_calibration.png'), dpi=300, bbox_inches='tight')
    
    return calibration_df

def analyze_test_seasons(model_dict, data, test_seasons):
    """
    Analyze model performance specifically on test seasons.
    """
    # Prepare data for test seasons
    test_features = prepare_tournament_data(data, start_season=min(test_seasons), end_season=max(test_seasons))
    
    # Evaluate model on test seasons
    print(f"\nModel Performance on {test_seasons} Tournaments:")
    return evaluate_model_performance(model_dict, test_features)

def main():
    """
    Main function to run the evaluation workflow.
    """
    print("=" * 80)
    print("MARCH MADNESS MODEL EVALUATION AND ANALYSIS")
    print("=" * 80)
    
    try:
        # Load NCAA data
        data = load_data(DATA_DIR)
        
        # Load trained model
        model_dict = load_model(MODEL_PATH)
        
        # Prepare tournament data for evaluation (all available seasons)
        print("Preparing tournament data for analysis...")
        tournament_features = prepare_tournament_data(data)
        
        # Evaluate model performance
        print("\nEvaluating overall model performance:")
        eval_results = evaluate_model_performance(model_dict, tournament_features)
        
        # Analyze feature importance
        print("\nAnalyzing feature importance:")
        importance_results = analyze_feature_importance(model_dict, tournament_features)
        
        # Analyze seed performance
        print("\nAnalyzing seed performance:")
        seed_results = analyze_seed_performance(tournament_features)
        
        # Analyze upsets
        print("\nAnalyzing upset patterns:")
        upset_results = analyze_upsets(tournament_features)
        
        # Analyze model calibration
        print("\nAnalyzing model calibration:")
        calibration_results = analyze_model_calibration(eval_results)
        
        # Analyze recent tournaments
        print("\nAnalyzing performance on recent tournaments:")
        test_results = analyze_test_seasons(model_dict, data, TEST_SEASONS)
        
        print("\nAll analysis results have been saved to the results directory.")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
