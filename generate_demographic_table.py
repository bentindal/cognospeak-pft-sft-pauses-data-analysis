#!/usr/bin/env python3
"""
Generate a demographic summary table similar to the reference table.
Includes: Group, Diagnosis, Number of Subjects, Age, Gender (M/F)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from dataloader import TranscriptLoader
from pause_analyzer import load_transcript, compute_pauses, summarize


def format_age(mean: float, std: float) -> str:
    """Format age as mean (std) with 2 decimal places."""
    return f"{mean:.2f} ({std:.2f})"


def compute_pause_count_for_recording(transcript_path: Path) -> int:
    """Compute total pause count for a single transcript file."""
    try:
        utterances = load_transcript(transcript_path)
        pauses = compute_pauses(utterances)
        return len(pauses)
    except Exception:
        return 0


def get_pause_stats_by_recording(
    loader: TranscriptLoader,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute pause counts for each recording.
    
    Returns:
        DataFrame with columns: recording, total_pause_count
    """
    transcript_files = loader.discover_transcript_files()
    
    # Group by recording and compute total pauses across all tasks
    recording_pauses = {}
    
    for transcript_path in transcript_files:
        recording_name = transcript_path.parent.name
        task = loader.infer_task_from_filename(transcript_path.name)
        
        # Only process SFT and PFT tasks
        if task not in ['SFT', 'PFT']:
            continue
        
        pause_count = compute_pause_count_for_recording(transcript_path)
        
        if recording_name not in recording_pauses:
            recording_pauses[recording_name] = 0
        
        # Sum pauses across all tasks for this recording
        recording_pauses[recording_name] += pause_count
    
    # Create DataFrame
    pause_df = pd.DataFrame([
        {'recording': recording, 'total_pause_count': count}
        for recording, count in recording_pauses.items()
    ])
    
    # Merge with metadata to get diagnosis
    pause_df = pause_df.merge(
        metadata[['recording', 'diagnosis']],
        on='recording',
        how='left'
    )
    
    return pause_df


def generate_demographic_table(
    data_dir: Path = None,
    metadata_path: Path = None,
    output_path: Path = None,
) -> pd.DataFrame:
    """
    Generate a demographic summary table by diagnosis group.
    
    Args:
        data_dir: Directory containing transcript files (optional)
        metadata_path: Path to metadata CSV file (optional)
        output_path: Path to save the table as CSV (optional)
        
    Returns:
        DataFrame with demographic summary
    """
    # Load metadata
    if data_dir is None and metadata_path is None:
        loader = TranscriptLoader()
    elif metadata_path is None:
        loader = TranscriptLoader(data_dir=data_dir)
    elif data_dir is None:
        loader = TranscriptLoader(metadata_path=metadata_path)
    else:
        loader = TranscriptLoader(data_dir=data_dir, metadata_path=metadata_path)
    metadata = loader.load_metadata()
    
    # Filter to only valid diagnoses
    valid_diagnoses = ['DEMENTIA', 'MCI', 'HC']
    metadata = metadata[metadata['diagnosis'].isin(valid_diagnoses)].copy()
    
    # Exclude specific recordings that should not be included
    exclude_recordings = ['Process-test-011']
    metadata = metadata[~metadata['recording'].isin(exclude_recordings)].copy()
    
    if len(exclude_recordings) > 0:
        print(f"Excluded recordings: {exclude_recordings}")
        print(f"Remaining subjects: {len(metadata)}")
    
    # Normalize gender column (handle case variations)
    if 'gender' in metadata.columns:
        metadata['gender'] = metadata['gender'].str.upper().str.strip()
    
    # Get pause statistics by recording
    pause_stats = get_pause_stats_by_recording(loader, metadata)
    
    # Prepare rows for the table
    rows = []
    
    # Individual diagnosis groups
    for diagnosis in ['DEMENTIA', 'MCI', 'HC']:
        subset = metadata[metadata['diagnosis'] == diagnosis]
        
        if len(subset) == 0:
            continue
        
        # Determine group
        if diagnosis in ['DEMENTIA', 'MCI']:
            group = 'Case'
        else:
            group = 'Control'
        
        # Number of subjects
        n_subjects = len(subset)
        
        # Age statistics
        age_mean = subset['age'].mean()
        age_std = subset['age'].std()
        age_formatted = format_age(age_mean, age_std) if pd.notna(age_mean) else "N/A"
        
        # Gender counts (handle various formats: male/M/MALE, female/F/FEMALE)
        if 'gender' in subset.columns:
            gender_upper = subset['gender'].str.upper().str.strip()
            gender_m = len(subset[gender_upper.isin(['M', 'MALE'])])
            gender_f = len(subset[gender_upper.isin(['F', 'FEMALE'])])
        else:
            gender_m = 0
            gender_f = 0
        
        # Pause statistics for this diagnosis group
        diagnosis_recordings = subset['recording'].tolist()
        diagnosis_pauses = pause_stats[pause_stats['recording'].isin(diagnosis_recordings)]
        
        if len(diagnosis_pauses) > 0:
            total_pauses = diagnosis_pauses['total_pause_count'].sum()
            avg_pauses = diagnosis_pauses['total_pause_count'].mean()
            pause_formatted = f"{int(total_pauses)} ({avg_pauses:.2f})"
        else:
            pause_formatted = "0 (0.00)"
        
        rows.append({
            'Group': group,
            'Diagnosis': diagnosis,
            'Number of Subjects': n_subjects,
            'Age': age_formatted,
            'Gender M': gender_m,
            'Gender F': gender_f,
            'Pauses (Total, Avg)': pause_formatted,
        })
    
    # Total row
    total_n = len(metadata)
    total_age_mean = metadata['age'].mean()
    total_age_std = metadata['age'].std()
    total_age_formatted = format_age(total_age_mean, total_age_std) if pd.notna(total_age_mean) else "N/A"
    
    if 'gender' in metadata.columns:
        gender_upper = metadata['gender'].str.upper().str.strip()
        total_gender_m = len(metadata[gender_upper.isin(['M', 'MALE'])])
        total_gender_f = len(metadata[gender_upper.isin(['F', 'FEMALE'])])
    else:
        total_gender_m = 0
        total_gender_f = 0
    
    # Total pause statistics
    total_recordings = metadata['recording'].tolist()
    total_pause_stats = pause_stats[pause_stats['recording'].isin(total_recordings)]
    
    if len(total_pause_stats) > 0:
        total_pauses = total_pause_stats['total_pause_count'].sum()
        total_avg_pauses = total_pause_stats['total_pause_count'].mean()
        total_pause_formatted = f"{int(total_pauses)} ({total_avg_pauses:.2f})"
    else:
        total_pause_formatted = "0 (0.00)"
    
    rows.append({
        'Group': 'Total',
        'Diagnosis': 'All',
        'Number of Subjects': total_n,
        'Age': total_age_formatted,
        'Gender M': total_gender_m,
        'Gender F': total_gender_f,
        'Pauses (Total, Avg)': total_pause_formatted,
    })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder rows to match reference table format
    # Case groups first, then Control, then Total
    order = ['Case', 'Control', 'Total']
    df['Group_order'] = df['Group'].map({g: i for i, g in enumerate(order)})
    df = df.sort_values(['Group_order', 'Diagnosis']).drop('Group_order', axis=1)
    
    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Demographic table saved to: {output_path}")
    
    return df


def print_table(df: pd.DataFrame) -> None:
    """Print the table in a nicely formatted way."""
    print("\n" + "="*80)
    print("DEMOGRAPHIC SUMMARY TABLE")
    print("="*80 + "\n")
    
    # Print header
    header = f"{'Group':<10} {'Diagnosis':<12} {'N':<6} {'Age (Mean ± SD)':<20} {'M':<6} {'F':<6} {'Pauses':<25}"
    print(header)
    print("-" * 110)
    
    # Print rows
    for _, row in df.iterrows():
        line = f"{row['Group']:<10} {row['Diagnosis']:<12} {row['Number of Subjects']:<6} {row['Age']:<20} {row['Gender M']:<6} {row['Gender F']:<6} {row['Pauses (Total, Avg)']:<25}"
        print(line)
    
    print("="*110 + "\n")


def main():
    """Main function to generate and display the table."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate demographic summary table'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=None,
        help='Directory containing transcript files (optional)'
    )
    parser.add_argument(
        '--metadata',
        type=Path,
        default=None,
        help='Path to metadata CSV file (optional)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('demographic_table.csv'),
        help='Path to save the table as CSV (default: demographic_table.csv)'
    )
    
    args = parser.parse_args()
    
    # Generate table
    df = generate_demographic_table(
        data_dir=args.data_dir,
        metadata_path=args.metadata,
        output_path=args.output,
    )
    
    # Print table
    print_table(df)
    
    return df


if __name__ == '__main__':
    main()
