#!/usr/bin/env python3
"""Multiclass classifier for Dementia vs MCI vs HC using SFT pause metrics."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dataloader import TranscriptLoader
from pause_analyzer import compute_pauses, load_transcript, summarize


def get_classifier(classifier_name: str) -> Pipeline:
    """
    Factory function to create a configured sklearn classifier pipeline.
    
    Args:
        classifier_name: Name of the classifier ('logistic' or 'random_forest')
        
    Returns:
        sklearn Pipeline with StandardScaler and the selected classifier
    """
    scaler = StandardScaler()
    
    if classifier_name == 'logistic':
        # Use multinomial logistic regression for multiclass
        clf = LogisticRegression(
            random_state=42, 
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs'
        )
    elif classifier_name == 'random_forest':
        clf = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            max_depth=10
        )
    else:
        raise ValueError(
            f"Unknown classifier: {classifier_name}. "
            "Supported classifiers: 'logistic', 'random_forest'"
        )
    
    return Pipeline([('scaler', scaler), ('classifier', clf)])


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    classifier_name: str,
    cv_folds: int = 5,
) -> tuple[Pipeline, dict]:
    """
    Tune hyperparameters using GridSearchCV for multiclass classification.
    
    Args:
        X: Feature matrix
        y: Target labels
        classifier_name: Name of the classifier ('logistic' or 'random_forest')
        cv_folds: Number of folds for cross-validation in GridSearchCV
        
    Returns:
        Tuple of (best_estimator, best_params)
    """
    scaler = StandardScaler()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    if classifier_name == 'logistic':
        base_clf = LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs'
        )
        param_grid_list = [{
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        }]
    elif classifier_name == 'random_forest':
        base_clf = RandomForestClassifier(random_state=42)
        param_grid_list = [{
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [5, 10, 20, None],
            'classifier__min_samples_split': [2, 5, 10],
        }]
    else:
        raise ValueError(
            f"Unknown classifier: {classifier_name}. "
            "Supported classifiers: 'logistic', 'random_forest'"
        )
    
    pipeline = Pipeline([('scaler', scaler), ('classifier', base_clf)])
    
    # Use balanced_accuracy as the scoring metric for GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid_list,
        cv=cv,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=1,
    )
    
    print(f"\nTuning hyperparameters for {classifier_name}...")
    print(f"GridSearchCV with {cv_folds}-fold cross-validation...")
    grid_search.fit(X, y)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"\nBest cross-validation score (balanced_accuracy): {best_score:.4f}")
    print("Best hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    return grid_search.best_estimator_, best_params


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
    Build a feature dataset with pause metrics per recording for multiclass classification.
    
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
    
    # Merge with metadata to get diagnosis and optionally demographics
    merge_cols = ['recording', 'diagnosis']
    if include_demographics:
        merge_cols.extend(['age', 'gender'])
    
    features_df = features_df.merge(
        metadata[merge_cols],
        on='recording',
        how='left',
    )
    
    # Filter to only Dementia, MCI, and HC (exclude missing)
    features_df = features_df[
        features_df['diagnosis'].isin(['DEMENTIA', 'HC', 'MCI'])
    ].copy()
    
    # Create multiclass labels: 0 = HC, 1 = MCI, 2 = DEMENTIA
    diagnosis_map = {'HC': 0, 'MCI': 1, 'DEMENTIA': 2}
    features_df['label'] = features_df['diagnosis'].map(diagnosis_map)
    
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
    Perform k-fold cross-validation and return aggregated metrics for multiclass.
    
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
    
    # Create custom scorer for multiclass ROC-AUC with macro averaging
    roc_auc_macro_scorer = make_scorer(
        roc_auc_score,
        multi_class='ovr',
        average='macro',
        needs_proba=True,
    )
    
    scoring = {
        'accuracy': 'accuracy',
        'balanced_accuracy': 'balanced_accuracy',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1': 'f1_macro',
        'roc_auc': roc_auc_macro_scorer,
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
    print("K-FOLD CROSS-VALIDATION RESULTS (Multiclass: HC vs MCI vs Dementia)")
    print("=" * 80)
    
    metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    for metric in metrics:
        mean = results[f'{metric}_mean']
        std = results[f'{metric}_std']
        print(f"{metric.replace('_', ' ').title():20s}: {mean:.4f} ± {std:.4f}")
    
    print("=" * 80 + "\n")


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot confusion matrix for multiclass classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Optional path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages as percentage of each row (true class)
    # This ensures each row sums to 100%
    row_sums = cm.sum(axis=1)
    # Avoid division by zero (shouldn't happen, but safety check)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    
    # Calculate percentages and normalize to ensure each row sums to exactly 100%
    cm_percent = np.zeros_like(cm, dtype=float)
    for i in range(len(cm)):
        if row_sums[i] > 0:
            # Calculate percentages for this row
            cm_percent[i, :] = cm[i, :].astype('float') / row_sums[i] * 100
            # Ensure it sums to exactly 100% by adjusting for any rounding errors
            row_total = cm_percent[i, :].sum()
            if row_total > 0:
                # Normalize to exactly 100%
                cm_percent[i, :] = cm_percent[i, :] / row_total * 100
    
    # Create annotation labels
    annot_labels = []
    class_names = ['HC', 'MCI', 'Dementia']
    for i in range(len(cm)):
        row = []
        for j in range(len(cm)):
            count = cm[i, j]
            percent = cm_percent[i, j]
            row.append(f'{count}\n({percent:.1f}%)')
        annot_labels.append(row)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=annot_labels, 
        fmt='', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(
        'Confusion Matrix: 3-Class Classification\n(HC vs MCI vs Dementia)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Add accuracy text
    total = cm.sum()
    correct = np.trace(cm)
    accuracy = correct / total
    ax.text(
        0.5, -0.15,
        f'Overall Accuracy: {accuracy*100:.2f}% ({correct}/{total})',
        transform=ax.transAxes,
        ha='center',
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {output_path}")
    
    plt.show()


def main() -> None:
    """Main entry point for the multiclass classifier script."""
    parser = argparse.ArgumentParser(
        description='Multiclass classifier for Dementia vs MCI vs HC using SFT pause metrics.'
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
        default='random_forest',
        choices=['logistic', 'random_forest'],
        help='Classifier to use (default: random_forest)',
    )
    parser.add_argument(
        '--k-folds',
        type=int,
        default=5,
        help='Number of folds for cross-validation (default: 5)',
    )
    parser.add_argument(
        '--tune-hyperparameters',
        action='store_true',
        help='Enable GridSearchCV for hyperparameter tuning',
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of folds for GridSearchCV (default: 5)',
    )
    parser.add_argument(
        '--include-pft',
        action='store_true',
        help='Include PFT pause features in addition to SFT',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output'),
        help='Directory to save output files (default: output)',
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MULTICLASS CLASSIFIER: Dementia vs MCI vs HC")
    print("=" * 80)
    print("\nBuilding feature dataset...")
    
    features_df = build_feature_dataset(
        data_dir=args.data_dir,
        metadata_path=args.metadata,
        include_pft=args.include_pft,
    )
    
    print(f"\nLoaded {len(features_df)} recordings")
    print("Class distribution:")
    for label_val, count in features_df['label'].value_counts().sort_index().items():
        diagnosis_name = features_df[features_df['label'] == label_val]['diagnosis'].iloc[0]
        print(f"  {diagnosis_name}: {count}")
    
    # Prepare features and labels
    feature_cols = [col for col in features_df.columns 
                   if col.startswith('sft_') or col.startswith('pft_')]
    X = features_df[feature_cols]
    y = features_df['label']
    
    print(f"\nUsing {len(feature_cols)} features:")
    for col in feature_cols:
        print(f"  - {col}")
    
    # Handle any missing values
    if X.isna().any().any():
        print("\nWarning: Found missing values in features. Filling with 0.")
        X = X.fillna(0)
    
    # Get classifier
    print(f"\nUsing classifier: {args.classifier}")
    
    # Tune hyperparameters if requested
    best_params = None
    if args.tune_hyperparameters:
        classifier, best_params = tune_hyperparameters(
            X, y, args.classifier, cv_folds=args.cv_folds
        )
        
        # Save best hyperparameters to JSON
        if best_params:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            hyperparams_file = args.output_dir / f'best_hyperparameters_multiclass_{args.classifier}.json'
            with open(hyperparams_file, 'w') as f:
                json.dump(best_params, f, indent=2)
            print(f"\nBest hyperparameters saved to: {hyperparams_file}")
    else:
        classifier = get_classifier(args.classifier)
    
    # Perform cross-validation
    print(f"\nPerforming {args.k_folds}-fold cross-validation...")
    cv_results = evaluate_with_cv(X, y, classifier, k_folds=args.k_folds)
    print_cv_results(cv_results)
    
    # Collect predictions for confusion matrix
    print("Collecting predictions for confusion matrix...")
    cv = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    all_y_true = []
    all_y_pred = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        all_y_true.extend(y_test.values)
        all_y_pred.extend(y_pred)
    
    # Print detailed classification report
    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 80)
    print(classification_report(
        all_y_true,
        all_y_pred,
        target_names=['HC', 'MCI', 'Dementia'],
        digits=4
    ))
    print("=" * 80 + "\n")
    
    # Plot confusion matrix
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / f'confusion_matrix_multiclass_{args.classifier}.png'
    plot_confusion_matrix(all_y_true, all_y_pred, output_path=output_file)
    
    # Print feature importance or coefficients
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE / COEFFICIENTS")
    print("=" * 80)
    
    if args.classifier == 'logistic':
        coefficients = classifier.named_steps['classifier'].coef_
        feature_importance = pd.DataFrame(
            coefficients,
            columns=feature_cols,
            index=['HC', 'MCI', 'Dementia']
        ).T
        print(feature_importance.to_string())
    elif args.classifier == 'random_forest':
        importances = classifier.named_steps['classifier'].feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances,
        }).sort_values('importance', ascending=False)
        print(feature_importance.to_string(index=False))
    
    print("=" * 80 + "\n")
    
    # Save feature importance to file
    output_file = args.output_dir / f'feature_importance_multiclass_{args.classifier}.csv'
    feature_importance.to_csv(output_file, index=False)
    print(f"Feature importance saved to: {output_file}")


if __name__ == '__main__':
    main()

