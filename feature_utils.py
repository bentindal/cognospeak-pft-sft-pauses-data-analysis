#!/usr/bin/env python3
"""Utilities for feature analysis and comparison."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_feature_weights(file_path: Path) -> pd.DataFrame:
    """
    Load and format feature weights from CSV file.
    
    Args:
        file_path: Path to feature weights CSV file
        
    Returns:
        DataFrame with formatted feature weights, sorted by absolute coefficient
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Feature weights file not found at {file_path}")
    
    feature_weights_df = pd.read_csv(file_path)
    
    # Calculate absolute coefficient for sorting BEFORE converting to string
    feature_weights_df['abs_coefficient'] = feature_weights_df['coefficient'].abs()
    feature_weights_df = feature_weights_df.sort_values('abs_coefficient', ascending=False)
    
    # Format the coefficients for better readability AFTER sorting
    feature_weights_df['coefficient'] = feature_weights_df['coefficient'].apply(
        lambda x: f'{x:.6f}'
    )
    feature_weights_df = feature_weights_df.drop('abs_coefficient', axis=1)
    
    return feature_weights_df


def display_feature_importance(
    feature_importance_df: pd.DataFrame,
    feature_names: list[str],
    title: str = "FEATURE IMPORTANCE",
) -> None:
    """
    Display feature importance information.
    
    Args:
        feature_importance_df: DataFrame with feature importance (must have 'feature' and 'coefficient' columns)
        feature_names: List of feature names to highlight
        title: Title for the display
    """
    print("=" * 80)
    print(title)
    print("=" * 80)
    
    for feature_name in feature_names:
        feature_row = feature_importance_df[
            feature_importance_df['feature'] == feature_name
        ]
        
        if len(feature_row) > 0:
            row = feature_row.iloc[0]
            print(f"\n  Feature: {row['feature']}")
            print(f"  Coefficient: {row['coefficient']:.6f}")
            print(f"  Absolute Coefficient: {abs(row['coefficient']):.6f}")
            
            # Rank among all features
            rank = (
                feature_importance_df['coefficient'].abs()
                .rank(ascending=False, method='min')
                [feature_importance_df['feature'] == feature_name]
            ).iloc[0]
            print(f"  Rank (by absolute coefficient): {int(rank)} out of {len(feature_importance_df)}")
            
            # Interpretation
            if row['coefficient'] > 0:
                print(f"  Interpretation: Higher {feature_name} is associated with Dementia")
            else:
                print(f"  Interpretation: Higher {feature_name} is associated with Healthy controls")
        else:
            print(f"\n  {feature_name} feature not found in feature list")
    
    print("\nTop 10 Features (by absolute coefficient):")
    print("-" * 80)
    print(feature_importance_df.head(10).to_string(index=False))
    print("\n" + "=" * 80)


def compare_classifier_performance(
    metrics_with: dict[str, float],
    metrics_without: dict[str, float],
    comparison_name: str = "With vs Without",
) -> pd.DataFrame:
    """
    Compare classifier performance metrics.
    
    Args:
        metrics_with: Dictionary with metrics (with feature/condition)
        metrics_without: Dictionary with metrics (without feature/condition)
        comparison_name: Name for the comparison
        
    Returns:
        DataFrame with comparison results
    """
    comparison_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    comparison_data = []
    
    for metric in comparison_metrics:
        with_mean = metrics_with.get(f'{metric}_mean', 0)
        with_std = metrics_with.get(f'{metric}_std', 0)
        without_mean = metrics_without.get(f'{metric}_mean', 0)
        without_std = metrics_without.get(f'{metric}_std', 0)
        
        improvement = with_mean - without_mean
        improvement_pct = (improvement / without_mean * 100) if without_mean > 0 else 0
        
        comparison_data.append({
            'Metric': metric.capitalize(),
            f'Without {comparison_name}': f'{without_mean:.4f} ± {without_std:.4f}',
            f'With {comparison_name}': f'{with_mean:.4f} ± {with_std:.4f}',
            'Improvement': f'{improvement:+.4f} ({improvement_pct:+.2f}%)'
        })
    
    return pd.DataFrame(comparison_data)

