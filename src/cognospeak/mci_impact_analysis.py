#!/usr/bin/env python3
"""
Analysis script to evaluate the impact of including MCI data in classification.

This script explores different scenarios:
1. Current: Binary classification (Dementia vs HC, MCI excluded)
2. Scenario A: Binary classification with MCI assigned to Dementia class
3. Scenario B: Binary classification with MCI assigned to HC class
4. Scenario C: Binary classification (Non-Healthy vs Healthy: MCI+Dementia vs HC)
5. Scenario D: 3-class classification (Dementia vs MCI vs HC)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .binary_classifier import (
    build_feature_dataset,
    compute_pause_metrics_for_recording,
    get_classifier,
)
from .dataloader import TranscriptLoader
from .pause_analyzer import compute_pauses, load_transcript, summarize


def build_feature_dataset_with_mci(
    data_dir: Path,
    metadata_path: Optional[Path],
    include_pft: bool = False,
    mci_strategy: str = 'exclude',
) -> pd.DataFrame:
    """
    Build a feature dataset with pause metrics per recording.
    
    Args:
        data_dir: Directory containing transcript files
        metadata_path: Path to metadata CSV file
        include_pft: Whether to include PFT features in addition to SFT
        mci_strategy: How to handle MCI data:
            - 'exclude': Exclude MCI (current default)
            - 'dementia': Assign MCI to Dementia class
            - 'hc': Assign MCI to HC class
            - 'non_healthy': MCI + Dementia vs HC
            - 'include_all': Include all three classes (for 3-class classification)
        
    Returns:
        DataFrame with one row per recording, columns: recording, diagnosis, 
        and pause feature columns (prefixed with 'sft_' and optionally 'pft_')
    """
    loader = TranscriptLoader(data_dir=data_dir, metadata_path=metadata_path)
    metadata = loader.load_metadata()
    
    # Get all transcript files
    transcript_files = loader.discover_transcript_files()
    
    # Group transcripts by recording and task
    recordings_data = {}
    
    for transcript_path in transcript_files:
        recording_name = transcript_path.parent.name
        task = loader.infer_task_from_filename(transcript_path.name)
        
        if task not in ['SFT', 'PFT']:
            continue
        
        if recording_name not in recordings_data:
            recordings_data[recording_name] = {}
        
        # Compute pause metrics
        metrics = compute_pause_metrics_for_recording(transcript_path)
        if metrics is not None:
            recordings_data[recording_name][task] = metrics
    
    # Build feature rows
    rows = []
    for recording_name, tasks_data in recordings_data.items():
        row = {'recording': recording_name}
        
        # Add SFT features (required)
        if 'SFT' in tasks_data:
            sft_metrics = tasks_data['SFT']
            for key, value in sft_metrics.items():
                if key.startswith('pause_'):
                    row[f'sft_{key}'] = value
        else:
            # Skip recordings without SFT
            continue
        
        # Optionally add PFT features
        if include_pft:
            if 'PFT' in tasks_data:
                pft_metrics = tasks_data['PFT']
                for key, value in pft_metrics.items():
                    if key.startswith('pause_'):
                        row[f'pft_{key}'] = value
            else:
                # Fill missing PFT with zeros
                for key in sft_metrics.keys():
                    if key.startswith('pause_'):
                        row[f'pft_{key}'] = 0.0
        
        rows.append(row)
    
    features_df = pd.DataFrame(rows)
    
    # Merge with metadata to get diagnosis
    features_df = features_df.merge(
        metadata[['recording', 'diagnosis']],
        on='recording',
        how='left',
    )
    
    # Handle MCI based on strategy
    if mci_strategy == 'exclude':
        # Filter to only Dementia and HC (exclude MCI and missing)
        features_df = features_df[
            features_df['diagnosis'].isin(['DEMENTIA', 'HC'])
        ].copy()
        features_df['label'] = (features_df['diagnosis'] == 'DEMENTIA').astype(int)
        
    elif mci_strategy == 'dementia':
        # Assign MCI to Dementia class
        features_df = features_df[
            features_df['diagnosis'].isin(['DEMENTIA', 'HC', 'MCI'])
        ].copy()
        features_df['label'] = (
            features_df['diagnosis'].isin(['DEMENTIA', 'MCI'])
        ).astype(int)
        
    elif mci_strategy == 'hc':
        # Assign MCI to HC class
        features_df = features_df[
            features_df['diagnosis'].isin(['DEMENTIA', 'HC', 'MCI'])
        ].copy()
        features_df['label'] = (features_df['diagnosis'] == 'DEMENTIA').astype(int)
        
    elif mci_strategy == 'non_healthy':
        # MCI + Dementia vs HC
        features_df = features_df[
            features_df['diagnosis'].isin(['DEMENTIA', 'HC', 'MCI'])
        ].copy()
        features_df['label'] = (
            features_df['diagnosis'].isin(['DEMENTIA', 'MCI'])
        ).astype(int)
        
    elif mci_strategy == 'include_all':
        # Keep all three classes for 3-class classification
        features_df = features_df[
            features_df['diagnosis'].isin(['DEMENTIA', 'HC', 'MCI'])
        ].copy()
        # Create 3-class labels: 0=HC, 1=MCI, 2=DEMENTIA
        diagnosis_map = {'HC': 0, 'MCI': 1, 'DEMENTIA': 2}
        features_df['label'] = features_df['diagnosis'].map(diagnosis_map)
        
    else:
        raise ValueError(f"Unknown mci_strategy: {mci_strategy}")
    
    return features_df


def evaluate_with_cv(
    X: pd.DataFrame,
    y: pd.Series,
    classifier: Pipeline,
    k_folds: int = 5,
    is_multiclass: bool = False,
) -> dict[str, float]:
    """
    Perform k-fold cross-validation and return aggregated metrics.
    
    Args:
        X: Feature matrix
        y: Target labels
        classifier: sklearn Pipeline with classifier
        k_folds: Number of folds for cross-validation
        is_multiclass: Whether this is a multi-class problem (affects ROC-AUC)
        
    Returns:
        Dictionary with mean and std of metrics across folds
    """
    # Use StratifiedKFold to ensure balanced class distribution in each fold
    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_macro' if is_multiclass else 'precision',
        'recall': 'recall_macro' if is_multiclass else 'recall',
        'f1': 'f1_macro' if is_multiclass else 'f1',
    }
    
    # Add ROC-AUC only for binary classification
    if not is_multiclass:
        scoring['roc_auc'] = 'roc_auc'
    
    # Suppress warnings for edge cases in small datasets
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        cv_results = cross_validate(
            classifier,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
        )
    
    results = {}
    for metric_name in scoring.keys():
        test_scores = cv_results[f'test_{metric_name}']
        results[f'{metric_name}_mean'] = float(np.mean(test_scores))
        results[f'{metric_name}_std'] = float(np.std(test_scores))
    
    return results


def print_comparison_results(results_dict: dict[str, dict[str, float]]) -> None:
    """Print comparison of results across different scenarios."""
    print("\n" + "=" * 100)
    print("COMPARISON: Impact of Including MCI Data")
    print("=" * 100)
    
    scenarios = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Print header
    header = f"{'Scenario':<25}"
    for metric in metrics:
        if any('roc_auc' in results_dict[s] for s in scenarios):
            header += f"{metric.capitalize():>15}"
        elif metric == 'roc_auc':
            continue
        else:
            header += f"{metric.capitalize():>15}"
    print(header)
    print("-" * 100)
    
    # Print results for each scenario
    for scenario in scenarios:
        results = results_dict[scenario]
        row = f"{scenario:<25}"
        for metric in metrics:
            if metric == 'roc_auc' and 'roc_auc_mean' not in results:
                continue
            if f'{metric}_mean' in results:
                mean = results[f'{metric}_mean']
                std = results[f'{metric}_std']
                row += f"{mean:>7.4f} ± {std:>6.4f}"
            else:
                row += f"{'N/A':>15}"
        print(row)
    
    print("=" * 100 + "\n")


def analyze_mci_impact(
    data_dir: Path,
    metadata_path: Optional[Path],
    include_pft: bool = False,
    k_folds: int = 5,
) -> None:
    """
    Analyze the impact of including MCI data in different scenarios.
    """
    print("=" * 100)
    print("MCI IMPACT ANALYSIS")
    print("=" * 100)
    print("\nThis analysis compares different strategies for handling MCI data:")
    print("  1. Current: Exclude MCI (Dementia vs HC only)")
    print("  2. MCI → Dementia: Assign MCI to Dementia class")
    print("  3. MCI → HC: Assign MCI to HC class")
    print("  4. Non-Healthy vs Healthy: MCI + Dementia vs HC")
    print("  5. 3-Class: Dementia vs MCI vs HC (multiclass)")
    print("\n" + "=" * 100 + "\n")
    
    strategies = [
        ('exclude', 'Current (Exclude MCI)'),
        ('dementia', 'MCI → Dementia'),
        ('hc', 'MCI → HC'),
        ('non_healthy', 'Non-Healthy vs Healthy'),
        ('include_all', '3-Class (Dementia/MCI/HC)'),
    ]
    
    results_dict = {}
    dataset_info = {}
    
    for strategy, strategy_name in strategies:
        print(f"\n{'='*100}")
        print(f"Analyzing: {strategy_name}")
        print(f"{'='*100}\n")
        
        # Build dataset
        features_df = build_feature_dataset_with_mci(
            data_dir=data_dir,
            metadata_path=metadata_path,
            include_pft=include_pft,
            mci_strategy=strategy,
        )
        
        # Store dataset info
        label_counts = features_df['label'].value_counts().sort_index()
        dataset_info[strategy_name] = {
            'total': len(features_df),
            'label_counts': label_counts.to_dict(),
            'diagnosis_dist': features_df['diagnosis'].value_counts().to_dict(),
        }
        
        print(f"Dataset size: {len(features_df)} recordings")
        print("Label distribution:")
        for label_val, count in label_counts.items():
            diagnosis_names = features_df[features_df['label'] == label_val]['diagnosis'].unique()
            print(f"  Label {label_val}: {count} ({', '.join(diagnosis_names)})")
        
        # Prepare features and labels
        feature_cols = [col for col in features_df.columns 
                       if col.startswith('sft_') or col.startswith('pft_')]
        X = features_df[feature_cols].fillna(0)
        y = features_df['label']
        
        # Handle any missing values
        if X.isna().any().any():
            print("Warning: Found missing values in features. Filling with 0.")
            X = X.fillna(0)
        
        # Use logistic regression for binary, random forest for multiclass
        is_multiclass = strategy == 'include_all'
        classifier_name = 'random_forest' if is_multiclass else 'logistic'
        classifier = get_classifier(classifier_name)
        
        print(f"\nUsing classifier: {classifier_name}")
        print(f"Performing {k_folds}-fold cross-validation...")
        
        # Perform cross-validation
        cv_results = evaluate_with_cv(
            X, y, classifier, k_folds=k_folds, is_multiclass=is_multiclass
        )
        
        results_dict[strategy_name] = cv_results
        
        # Print results
        print("\nCross-validation results:")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if f'{metric}_mean' in cv_results:
                mean = cv_results[f'{metric}_mean']
                std = cv_results[f'{metric}_std']
                print(f"  {metric.capitalize():12s}: {mean:.4f} ± {std:.4f}")
        if 'roc_auc_mean' in cv_results:
            mean = cv_results['roc_auc_mean']
            std = cv_results['roc_auc_std']
            print(f"  ROC-AUC      : {mean:.4f} ± {std:.4f}")
    
    # Print comparison
    print_comparison_results(results_dict)
    
    # Print dataset summary
    print("\n" + "=" * 100)
    print("DATASET SUMMARY")
    print("=" * 100)
    for strategy_name, info in dataset_info.items():
        print(f"\n{strategy_name}:")
        print(f"  Total recordings: {info['total']}")
        print(f"  Diagnosis distribution: {info['diagnosis_dist']}")
        print(f"  Label distribution: {info['label_counts']}")
    
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)
    print("""
1. MCI data characteristics:
   - MCI pause metrics are intermediate between HC and Dementia
   - Statistical tests show MCI is not significantly different from either HC or Dementia
   - This makes MCI a "confusing" class for binary classification

2. Expected impacts:
   - Including MCI with either class will likely reduce accuracy due to:
     * Increased class overlap
     * More ambiguous decision boundaries
     * MCI samples may be misclassified more often
   
   - 3-class classification may be more appropriate if:
     * You want to distinguish all three groups
     * You have sufficient data for each class
     * Clinical application requires MCI identification

3. Recommendations:
   - For binary Dementia vs HC: Keep MCI excluded (current approach)
   - For screening applications: Consider Non-Healthy vs Healthy (MCI+Dementia vs HC)
   - For diagnostic applications: Use 3-class classification if data allows
   - Consider ensemble methods or separate models for different use cases
    """)


def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze the impact of including MCI data in classification.'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='Directory containing transcript files (default: data)',
    )
    parser.add_argument(
        '--metadata',
        type=Path,
        default=None,
        help='Path to metadata CSV file (default: data/speechpause_mata.csv)',
    )
    parser.add_argument(
        '--k-folds',
        type=int,
        default=5,
        help='Number of folds for cross-validation (default: 5)',
    )
    parser.add_argument(
        '--include-pft',
        action='store_true',
        help='Include PFT pause features in addition to SFT',
    )
    
    args = parser.parse_args()
    
    analyze_mci_impact(
        data_dir=args.data_dir,
        metadata_path=args.metadata,
        include_pft=args.include_pft,
        k_folds=args.k_folds,
    )


if __name__ == '__main__':
    main()
