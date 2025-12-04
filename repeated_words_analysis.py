#!/usr/bin/env python3
"""Analysis functions for repeated words features."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from binary_classifier import compute_repeated_words_metrics
from dataloader import TranscriptLoader


def compute_repeated_words_dataset(
    loader: TranscriptLoader,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute repeated words metrics for all transcripts and merge with metadata.
    
    Args:
        loader: TranscriptLoader instance
        metadata: Metadata DataFrame
        
    Returns:
        DataFrame with repeated words metrics merged with diagnosis
    """
    # Get all transcript files
    transcript_paths = loader.discover_transcript_files()
    
    # Compute repeated words for each transcript
    repeated_words_data = []
    for path in transcript_paths:
        recording_name = path.parent.name
        task = loader.infer_task_from_filename(path.name)
        
        if task not in ['PFT', 'SFT']:
            continue
        
        repeated_metrics = compute_repeated_words_metrics(path)
        
        repeated_words_data.append({
            'recording': recording_name,
            'task': task,
            'repeated_words_count': repeated_metrics['repeated_words_count'],
            'repeated_words_std': repeated_metrics['repeated_words_std'],
            'transcript_path': str(path)
        })
    
    repeated_words_df = pd.DataFrame(repeated_words_data)
    
    # Merge with metadata to get diagnosis
    metadata_for_merge = metadata.copy()
    if 'anyon_IDs' in metadata_for_merge.columns:
        metadata_for_merge = metadata_for_merge.rename(columns={'anyon_IDs': 'recording'})
    
    repeated_words_with_diag = repeated_words_df.merge(
        metadata_for_merge[['recording', 'diagnosis']],
        on='recording',
        how='left'
    )
    
    # Filter to only records with diagnosis
    repeated_words_with_diag = repeated_words_with_diag.dropna(subset=['diagnosis'])
    
    # Convert diagnosis to categorical
    repeated_words_with_diag['diagnosis'] = pd.Categorical(
        repeated_words_with_diag['diagnosis'],
        categories=['DEMENTIA', 'HC', 'MCI'],
        ordered=True
    )
    
    return repeated_words_with_diag


def print_repeated_words_statistics(repeated_words_df: pd.DataFrame) -> None:
    """
    Print summary statistics for repeated words by diagnosis and task.
    
    Args:
        repeated_words_df: DataFrame with repeated words metrics and diagnosis
    """
    print("=" * 80)
    print("REPEATED WORDS STATISTICS BY DIAGNOSIS")
    print("=" * 80)
    
    # Summary statistics by diagnosis and task
    for task in ['PFT', 'SFT']:
        task_data = repeated_words_df[repeated_words_df['task'] == task]
        print(f"\n{task} Task:")
        print("-" * 80)
        
        summary = task_data.groupby('diagnosis')['repeated_words_count'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(2)
        
        print(summary)
        
        # Statistical comparison
        print(f"\n{task} - Statistical Summary:")
        for diag in ['DEMENTIA', 'HC', 'MCI']:
            diag_data = task_data[task_data['diagnosis'] == diag]['repeated_words_count']
            if len(diag_data) > 0:
                print(f"  {diag:10s}: Mean = {diag_data.mean():.2f}, Median = {diag_data.median():.2f}, "
                      f"Std = {diag_data.std():.2f}, N = {len(diag_data)}")
    
    print("\n" + "=" * 80)

