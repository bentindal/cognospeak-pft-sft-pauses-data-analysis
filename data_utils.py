#!/usr/bin/env python3
"""Data preparation and utility functions for transcript analysis."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from dataloader import TranscriptLoader
from pause_analyzer import analyze_pauses


def prepare_pause_data_with_metadata(
    loader: TranscriptLoader,
    metadata: pd.DataFrame,
    pause_summary_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Prepare pause data by merging with metadata.
    
    Args:
        loader: TranscriptLoader instance
        metadata: Metadata DataFrame
        pause_summary_path: Optional path to pause_summary.csv. If None, will analyze pauses.
        
    Returns:
        DataFrame with pause data merged with metadata
    """
    # Load or analyze pause data
    if pause_summary_path and pause_summary_path.exists():
        pause_summary = pd.read_csv(pause_summary_path)
    else:
        # Analyze pauses if summary doesn't exist
        pause_summary = pd.DataFrame(analyze_pauses(loader.data_dir, Path('pause_summary.csv')))
    
    # Merge with metadata
    metadata_for_merge = metadata.copy()
    if 'anyon_IDs' in metadata_for_merge.columns:
        metadata_for_merge = metadata_for_merge.rename(columns={'anyon_IDs': 'recording'})
    
    pause_with_diag = pause_summary.merge(
        metadata_for_merge[['recording', 'diagnosis']],
        on='recording',
        how='left'
    )
    
    # Extract task (PFT/SFT) from transcript path
    pause_with_diag['task'] = pause_with_diag['transcript'].str.extract(
        r'([PS]FT)', flags=re.IGNORECASE
    )[0].str.upper()
    
    # Convert diagnosis to categorical
    pause_with_diag['diagnosis'] = pd.Categorical(
        pause_with_diag['diagnosis'],
        categories=['DEMENTIA', 'HC', 'MCI'],
        ordered=True
    )
    
    # Filter to only PFT and SFT tasks
    pause_with_diag_filtered = pause_with_diag[
        pause_with_diag['task'].isin(['PFT', 'SFT'])
    ].copy()
    
    return pause_with_diag_filtered


def prepare_demographic_data(
    pause_data: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge demographic data (age and gender) with pause data.
    
    Args:
        pause_data: DataFrame with pause data
        metadata: Metadata DataFrame with age and gender columns
        
    Returns:
        Tuple of (pft_data_demo, sft_data_demo) DataFrames
    """
    metadata_for_merge = metadata.copy()
    if 'anyon_IDs' in metadata_for_merge.columns:
        metadata_for_merge = metadata_for_merge.rename(columns={'anyon_IDs': 'recording'})
    
    pause_data_demo = pause_data.merge(
        metadata_for_merge[['recording', 'diagnosis', 'age', 'gender']],
        on=['recording', 'diagnosis'],
        how='left'
    )
    
    # Split by task
    pft_data_demo = pause_data_demo[pause_data_demo['task'] == 'PFT'].copy()
    sft_data_demo = pause_data_demo[pause_data_demo['task'] == 'SFT'].copy()
    
    return pft_data_demo, sft_data_demo


def get_original_data_for_comparison(
    pause_with_diag_filtered: Optional[pd.DataFrame] = None,
    pft_data: Optional[pd.DataFrame] = None,
    sft_data: Optional[pd.DataFrame] = None,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Get original data (without demographics) for comparison.
    
    Args:
        pause_with_diag_filtered: Filtered pause data with diagnosis
        pft_data: PFT data (alternative source)
        sft_data: SFT data (alternative source)
        
    Returns:
        Tuple of (pft_data_orig, sft_data_orig)
    """
    if pft_data is not None and sft_data is not None:
        return pft_data.copy(), sft_data.copy()
    
    if pause_with_diag_filtered is not None:
        # Create health_status if not already present
        if 'health_status' not in pause_with_diag_filtered.columns:
            pause_with_diag_filtered['health_status'] = (
                pause_with_diag_filtered['diagnosis']
                .apply(lambda x: 'Non-Healthy' if x in ['DEMENTIA', 'MCI'] else 'Healthy')
            )
        pft_data_orig = pause_with_diag_filtered[pause_with_diag_filtered['task'] == 'PFT']
        sft_data_orig = pause_with_diag_filtered[pause_with_diag_filtered['task'] == 'SFT']
        return pft_data_orig, sft_data_orig
    
    return None, None

