#!/usr/bin/env python3
"""Binary classifier for Dementia vs Healthy using SFT pause metrics."""

from __future__ import annotations

import argparse
import warnings
from collections import Counter
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

from dataloader import TranscriptLoader
from pause_analyzer import compute_pauses, load_transcript, summarize


def compute_repeated_words_metrics(transcript_path: Path) -> dict[str, float]:
    """
    Compute repeated words metrics for a transcript.
    
    A word is considered repeated if it appears more than once in the transcript.
    This function computes:
    - repeated_words_count: Total number of word occurrences that are repetitions
      (i.e., if a word appears 3 times, it contributes 2 to the count)
    - repeated_words_std: Standard deviation of word frequencies, measuring
      variability in word repetition patterns
    
    Args:
        transcript_path: Path to the transcript .txt file
        
    Returns:
        Dictionary with 'repeated_words_count' and 'repeated_words_std'
    """
    try:
        utterances = load_transcript(transcript_path)
        # Extract all tokens (words) and normalize to lowercase
        words = [utterance.token.lower().strip() for utterance in utterances if utterance.token.strip()]
        
        if not words:
            return {'repeated_words_count': 0, 'repeated_words_std': 0.0}
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Count total occurrences of words that appear more than once
        repeated_count = sum(count - 1 for count in word_counts.values() if count > 1)
        
        # Compute standard deviation of word frequencies
        # This measures variability in repetition patterns
        import statistics
        frequencies = list(word_counts.values())
        if len(frequencies) > 1:
            freq_std = statistics.stdev(frequencies)
        else:
            freq_std = 0.0
        
        return {
            'repeated_words_count': repeated_count,
            'repeated_words_std': freq_std
        }
    except Exception as e:
        print(f"Warning: Could not compute repeated words for {transcript_path}: {e}")
        return {'repeated_words_count': 0, 'repeated_words_std': 0.0}


def get_classifier(classifier_name: str) -> Pipeline:
    """
    Factory function to create a configured sklearn classifier pipeline.
    
    Args:
        classifier_name: Name of the classifier ('logistic' or 'random_forest')
        
    Returns:
        sklearn Pipeline with StandardScaler and the selected classifier
        
    Raises:
        ValueError: If classifier_name is not recognized
    """
    scaler = StandardScaler()
    
    if classifier_name == 'logistic':
        clf = LogisticRegression(random_state=42, max_iter=1000)
    elif classifier_name == 'random_forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    else:
        raise ValueError(
            f"Unknown classifier: {classifier_name}. "
            "Supported classifiers: 'logistic', 'random_forest'"
        )
    
    return Pipeline([('scaler', scaler), ('classifier', clf)])


def compute_pause_metrics_for_recording(
    transcript_path: Path,
) -> Optional[dict[str, float | int]]:
    """
    Compute pause metrics for a single transcript file.
    
    Args:
        transcript_path: Path to the transcript .txt file
        
    Returns:
        Dictionary with pause metrics, or None if file cannot be loaded
    """
    try:
        utterances = load_transcript(transcript_path)
        pauses = compute_pauses(utterances)
        summary = summarize(pauses)
        return summary
    except Exception as e:
        print(f"Warning: Could not process {transcript_path}: {e}")
        return None


def build_feature_dataset(
    data_dir: Path,
    metadata_path: Optional[Path],
    include_pft: bool = False,
    include_demographics: bool = False,
) -> pd.DataFrame:
    """
    Build a feature dataset with pause metrics per recording.
    
    Args:
        data_dir: Directory containing transcript files
        metadata_path: Path to metadata CSV file
        include_pft: Whether to include PFT features in addition to SFT
        include_demographics: Whether to include age and gender as features
        
    Returns:
        DataFrame with one row per recording, columns: recording, diagnosis, 
        and pause feature columns (prefixed with 'sft_' and optionally 'pft_'),
        and optionally 'age' and 'gender_encoded' if include_demographics is True
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
            # Add repeated words metrics as features
            repeated_words_metrics = compute_repeated_words_metrics(transcript_path)
            metrics.update(repeated_words_metrics)
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
                elif key == 'repeated_words_count':
                    row['sft_repeated_words_count'] = value
                elif key == 'repeated_words_std':
                    row['sft_repeated_words_std'] = value
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
                    elif key == 'repeated_words_count':
                        row['pft_repeated_words_count'] = value
                    elif key == 'repeated_words_std':
                        row['pft_repeated_words_std'] = value
            else:
                # Fill missing PFT with zeros
                for key in sft_metrics.keys():
                    if key.startswith('pause_'):
                        row[f'pft_{key}'] = 0.0
                row['pft_repeated_words_count'] = 0
                row['pft_repeated_words_std'] = 0.0
        
        rows.append(row)
    
    features_df = pd.DataFrame(rows)
    
    # Merge with metadata to get diagnosis and optionally demographics
    merge_cols = ['recording', 'diagnosis']
    if include_demographics:
        merge_cols.extend(['age', 'gender'])
    
    features_df = features_df.merge(
        metadata[merge_cols],
        on='recording',
        how='left',
    )
    
    # Filter to only Dementia and HC (exclude MCI and missing)
    features_df = features_df[
        features_df['diagnosis'].isin(['DEMENTIA', 'HC'])
    ].copy()
    
    # Create binary labels: 0 = HC, 1 = Dementia
    features_df['label'] = (features_df['diagnosis'] == 'DEMENTIA').astype(int)
    
    # Encode gender if demographics are included
    if include_demographics:
        # Normalize gender values (handle case variations)
        features_df['gender'] = features_df['gender'].str.lower().str.strip()
        # Encode gender as binary: 0 for 'male', 1 for 'female' (or other)
        # Map common gender values
        gender_mapping = {
            'male': 0,
            'm': 0,
            'female': 1,
            'f': 1,
        }
        features_df['gender_encoded'] = features_df['gender'].map(gender_mapping)
        # Fill any unmapped values with the mode or 0.5 (neutral)
        if features_df['gender_encoded'].isna().any():
            mode_value = features_df['gender_encoded'].mode()
            fill_value = mode_value[0] if len(mode_value) > 0 else 0.5
            features_df['gender_encoded'] = features_df['gender_encoded'].fillna(fill_value)
    
    return features_df


def evaluate_with_cv(
    X: pd.DataFrame,
    y: pd.Series,
    classifier: Pipeline,
    k_folds: int = 5,
) -> dict[str, float]:
    """
    Perform k-fold cross-validation and return aggregated metrics.
    
    Args:
        X: Feature matrix
        y: Target labels
        classifier: sklearn Pipeline with classifier
        k_folds: Number of folds for cross-validation
        
    Returns:
        Dictionary with mean and std of metrics across folds
    """
    # Use StratifiedKFold to ensure balanced class distribution in each fold
    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
    }
    
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


def print_cv_results(results: dict[str, float]) -> None:
    """Print cross-validation results in a formatted way."""
    print("\n" + "=" * 80)
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 80)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    for metric in metrics:
        mean = results[f'{metric}_mean']
        std = results[f'{metric}_std']
        print(f"{metric.capitalize():12s}: {mean:.4f} ± {std:.4f}")
    
    print("=" * 80 + "\n")


def main() -> None:
    """Main entry point for the binary classifier script."""
    parser = argparse.ArgumentParser(
        description='Binary classifier for Dementia vs Healthy using SFT pause metrics.'
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
        '--classifier',
        type=str,
        default='logistic',
        choices=['logistic', 'random_forest'],
        help='Classifier to use (default: logistic)',
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
    parser.add_argument(
        '--include-demographics',
        action='store_true',
        help='Include age and gender as features',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output'),
        help='Directory to save output files (default: output)',
    )
    
    args = parser.parse_args()
    
    print("Building feature dataset...")
    features_df = build_feature_dataset(
        data_dir=args.data_dir,
        metadata_path=args.metadata,
        include_pft=args.include_pft,
        include_demographics=args.include_demographics,
    )
    
    print(f"Loaded {len(features_df)} recordings")
    print(f"  - Dementia: {sum(features_df['label'] == 1)}")
    print(f"  - Healthy (HC): {sum(features_df['label'] == 0)}")
    
    # Prepare features and labels
    feature_cols = [col for col in features_df.columns 
                   if col.startswith('sft_') or col.startswith('pft_')]
    
    # Add demographic features if enabled
    if args.include_demographics:
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
        # For age, fill with median; for other features, fill with 0
        if 'age' in X.columns:
            age_median = X['age'].median()
            if pd.notna(age_median):
                X['age'] = X['age'].fillna(age_median)
                print(f"  - Filled missing age values with median: {age_median:.1f}")
            else:
                X['age'] = X['age'].fillna(0)
                print("  - Filled missing age values with 0 (no median available)")
        # Fill remaining missing values with 0
        X = X.fillna(0)
    
    # Get classifier
    print(f"\nUsing classifier: {args.classifier}")
    classifier = get_classifier(args.classifier)
    
    # Perform cross-validation
    print(f"\nPerforming {args.k_folds}-fold cross-validation...")
    cv_results = evaluate_with_cv(X, y, classifier, k_folds=args.k_folds)
    print_cv_results(cv_results)
    
    # Train final model on all data for feature importance/coefficients
    print("Training final model on all data...")
    classifier.fit(X, y)
    
    # Print feature importance or coefficients
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE / COEFFICIENTS")
    print("=" * 80)
    
    if args.classifier == 'logistic':
        coefficients = classifier.named_steps['classifier'].coef_[0]
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': coefficients,
        }).sort_values('coefficient', key=abs, ascending=False)
        print(feature_importance.to_string(index=False))
    elif args.classifier == 'random_forest':
        importances = classifier.named_steps['classifier'].feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances,
        }).sort_values('importance', ascending=False)
        print(feature_importance.to_string(index=False))
    
    print("=" * 80 + "\n")
    
    # Save feature importance to file
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f'feature_importance_{args.classifier}.csv'
    feature_importance.to_csv(output_file, index=False)
    print(f"Feature importance saved to: {output_file}")


if __name__ == '__main__':
    main()

