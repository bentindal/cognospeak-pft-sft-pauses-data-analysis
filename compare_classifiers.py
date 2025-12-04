#!/usr/bin/env python3
"""Compare Logistic Regression vs Random Forest with demographics using 5-fold CV."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from binary_classifier import build_feature_dataset, get_classifier


def run_detailed_cv_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    k_folds: int = 5,
) -> None:
    """
    Run detailed 5-fold CV comparison between logistic regression and random forest.
    
    Args:
        X: Feature matrix
        y: Target labels
        k_folds: Number of folds for cross-validation
    """
    # Use StratifiedKFold with same random state for both classifiers
    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Get both classifiers
    lr_classifier = get_classifier('logistic')
    rf_classifier = get_classifier('random_forest')
    
    # Store results for each fold
    lr_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
    }
    rf_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
    }
    
    print("=" * 100)
    print("5-FOLD CROSS-VALIDATION COMPARISON")
    print("=" * 100)
    print(f"\nRunning {k_folds}-fold CV for both classifiers...\n")
    
    # Suppress warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train and evaluate Logistic Regression
            lr_classifier.fit(X_train, y_train)
            lr_pred = lr_classifier.predict(X_test)
            lr_pred_proba = lr_classifier.predict_proba(X_test)[:, 1]
            
            lr_acc = accuracy_score(y_test, lr_pred)
            lr_prec = precision_score(y_test, lr_pred, zero_division=0)
            lr_rec = recall_score(y_test, lr_pred, zero_division=0)
            lr_f1 = f1_score(y_test, lr_pred, zero_division=0)
            lr_auc = roc_auc_score(y_test, lr_pred_proba)
            
            lr_results['accuracy'].append(lr_acc)
            lr_results['precision'].append(lr_prec)
            lr_results['recall'].append(lr_rec)
            lr_results['f1'].append(lr_f1)
            lr_results['roc_auc'].append(lr_auc)
            
            # Train and evaluate Random Forest
            rf_classifier.fit(X_train, y_train)
            rf_pred = rf_classifier.predict(X_test)
            rf_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]
            
            rf_acc = accuracy_score(y_test, rf_pred)
            rf_prec = precision_score(y_test, rf_pred, zero_division=0)
            rf_rec = recall_score(y_test, rf_pred, zero_division=0)
            rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
            rf_auc = roc_auc_score(y_test, rf_pred_proba)
            
            rf_results['accuracy'].append(rf_acc)
            rf_results['precision'].append(rf_prec)
            rf_results['recall'].append(rf_rec)
            rf_results['f1'].append(rf_f1)
            rf_results['roc_auc'].append(rf_auc)
            
            # Print fold results
            print(f"Fold {fold_idx}:")
            print(f"  Logistic Regression - Acc: {lr_acc:.4f}, Prec: {lr_prec:.4f}, "
                  f"Rec: {lr_rec:.4f}, F1: {lr_f1:.4f}, AUC: {lr_auc:.4f}")
            print(f"  Random Forest        - Acc: {rf_acc:.4f}, Prec: {rf_prec:.4f}, "
                  f"Rec: {rf_rec:.4f}, F1: {rf_f1:.4f}, AUC: {rf_auc:.4f}")
            print()
    
    # Calculate means and stds
    print("=" * 100)
    print("AGGREGATED RESULTS (Mean ± Std across 5 folds)")
    print("=" * 100)
    print()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    comparison_data = []
    for metric in metrics:
        lr_mean = np.mean(lr_results[metric])
        lr_std = np.std(lr_results[metric])
        rf_mean = np.mean(rf_results[metric])
        rf_std = np.std(rf_results[metric])
        
        diff = lr_mean - rf_mean
        diff_pct = (diff / rf_mean * 100) if rf_mean > 0 else 0
        
        comparison_data.append({
            'Metric': metric.capitalize(),
            'Logistic Regression': f'{lr_mean:.4f} ± {lr_std:.4f}',
            'Random Forest': f'{rf_mean:.4f} ± {rf_std:.4f}',
            'Difference': f'{diff:+.4f} ({diff_pct:+.2f}%)',
            'Winner': 'Logistic Regression' if diff > 0 else 'Random Forest',
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print()
    
    # Summary
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    lr_wins = sum(1 for row in comparison_data if row['Winner'] == 'Logistic Regression')
    rf_wins = sum(1 for row in comparison_data if row['Winner'] == 'Random Forest')
    
    print(f"\nLogistic Regression wins on {lr_wins} out of {len(metrics)} metrics")
    print(f"Random Forest wins on {rf_wins} out of {len(metrics)} metrics")
    
    # Overall winner based on weighted importance (accuracy, F1, ROC AUC)
    lr_weighted = (
        np.mean(lr_results['accuracy']) * 0.3 +
        np.mean(lr_results['f1']) * 0.3 +
        np.mean(lr_results['roc_auc']) * 0.2 +
        np.mean(lr_results['precision']) * 0.1 +
        np.mean(lr_results['recall']) * 0.1
    )
    rf_weighted = (
        np.mean(rf_results['accuracy']) * 0.3 +
        np.mean(rf_results['f1']) * 0.3 +
        np.mean(rf_results['roc_auc']) * 0.2 +
        np.mean(rf_results['precision']) * 0.1 +
        np.mean(rf_results['recall']) * 0.1
    )
    
    print(f"\nWeighted Score (Accuracy*0.3 + F1*0.3 + AUC*0.2 + Prec*0.1 + Rec*0.1):")
    print(f"  Logistic Regression: {lr_weighted:.4f}")
    print(f"  Random Forest:        {rf_weighted:.4f}")
    
    if lr_weighted > rf_weighted:
        print(f"\n✓ Logistic Regression outperforms Random Forest overall")
    else:
        print(f"\n✓ Random Forest outperforms Logistic Regression overall")
    
    print("=" * 100)


def main() -> None:
    """Main entry point for the comparison script."""
    data_dir = Path('data')
    metadata_path = data_dir / 'speechpause_mata.csv' if (data_dir / 'speechpause_mata.csv').exists() else None
    
    print("Building feature dataset with demographics...")
    features_df = build_feature_dataset(
        data_dir=data_dir,
        metadata_path=metadata_path,
        include_pft=False,
        include_demographics=True,
    )
    
    print(f"\nLoaded {len(features_df)} recordings")
    print(f"  - Dementia: {sum(features_df['label'] == 1)}")
    print(f"  - Healthy (HC): {sum(features_df['label'] == 0)}")
    
    # Prepare features and labels
    feature_cols = [col for col in features_df.columns 
                   if col.startswith('sft_') or col.startswith('pft_')]
    
    # Add demographic features
    if 'age' in features_df.columns:
        feature_cols.append('age')
    if 'gender_encoded' in features_df.columns:
        feature_cols.append('gender_encoded')
    
    X = features_df[feature_cols]
    y = features_df['label']
    
    print(f"\nUsing {len(feature_cols)} features:")
    for col in feature_cols:
        print(f"  - {col}")
    
    # Handle any missing values
    if X.isna().any().any():
        print("\nWarning: Found missing values in features.")
        if 'age' in X.columns:
            age_median = X['age'].median()
            if pd.notna(age_median):
                X['age'] = X['age'].fillna(age_median)
                print(f"  - Filled missing age values with median: {age_median:.1f}")
            else:
                X['age'] = X['age'].fillna(0)
        X = X.fillna(0)
    
    # Run detailed comparison
    run_detailed_cv_comparison(X, y, k_folds=5)


if __name__ == '__main__':
    main()

