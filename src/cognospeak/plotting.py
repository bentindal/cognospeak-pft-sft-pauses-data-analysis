"""Plotting functions for transcript analysis."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Color palettes
DIAGNOSIS_PALETTE = {
    'DEMENTIA': '#d62728',
    'MCI': '#ff7f0e',
    'HC': '#2ca02c',
}

HEALTH_PALETTE = {
    'Non-Healthy': '#d62728',
    'Healthy': '#2ca02c',
}


def plot_3class_comparison(pft_data: pd.DataFrame, sft_data: pd.DataFrame) -> None:
    """
    Create combined figure with PFT and SFT box plots for 3-class diagnosis comparison.
    
    Args:
        pft_data: DataFrame with PFT data including 'diagnosis' and 'pause_mean' columns
        sft_data: DataFrame with SFT data including 'diagnosis' and 'pause_mean' columns
    """
    # Create combined figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Pause Mean Distribution by Diagnosis', fontsize=16, fontweight='bold', y=1.02)
    
    # Plot PFT on the left
    sns.boxplot(
        data=pft_data,
        x='diagnosis',
        y='pause_mean',
        hue='diagnosis',
        palette=DIAGNOSIS_PALETTE,
        legend=False,
        ax=ax1
    )
    ax1.set_title('PFT', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Diagnosis')
    ax1.set_ylabel('Mean Pause Duration (s)')
    ax1.set_ylim(0, 10)
    
    # Plot SFT on the right
    sns.boxplot(
        data=sft_data,
        x='diagnosis',
        y='pause_mean',
        hue='diagnosis',
        palette=DIAGNOSIS_PALETTE,
        legend=False,
        ax=ax2
    )
    ax2.set_title('SFT', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Diagnosis')
    ax2.set_ylabel('Mean Pause Duration (s)')
    ax2.set_ylim(0, 10)
    
    plt.tight_layout()
    plt.show()


def plot_binary_comparison(pft_data: pd.DataFrame, sft_data: pd.DataFrame) -> None:
    """
    Create combined figure with PFT and SFT box plots for binary (Healthy vs Non-Healthy) comparison.
    
    Args:
        pft_data: DataFrame with PFT data including 'health_status' and 'pause_mean' columns
        sft_data: DataFrame with SFT data including 'health_status' and 'pause_mean' columns
    """
    # Create combined figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Pause Mean Distribution: Healthy vs Non-Healthy', fontsize=16, fontweight='bold', y=1.02)
    
    # Plot PFT on the left
    sns.boxplot(
        data=pft_data,
        x='health_status',
        y='pause_mean',
        hue='health_status',
        palette=HEALTH_PALETTE,
        legend=False,
        ax=ax1
    )
    ax1.set_title('PFT', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Health Status')
    ax1.set_ylabel('Mean Pause Duration (s)')
    ax1.set_ylim(0, 10)
    
    # Plot SFT on the right
    sns.boxplot(
        data=sft_data,
        x='health_status',
        y='pause_mean',
        hue='health_status',
        palette=HEALTH_PALETTE,
        legend=False,
        ax=ax2
    )
    ax2.set_title('SFT', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Health Status')
    ax2.set_ylabel('Mean Pause Duration (s)')
    ax2.set_ylim(0, 10)
    
    plt.tight_layout()
    plt.show()


def print_descriptive_stats(data: pd.DataFrame, group_col: str, value_col: str = 'pause_mean') -> None:
    """
    Print descriptive statistics grouped by a column.
    
    Args:
        data: DataFrame containing the data
        group_col: Column name to group by
        value_col: Column name for the values to analyze
    """
    stats = data.groupby(group_col, observed=True)[value_col].describe()[['count', 'mean', 'std', '25%', '50%', '75%']]
    print(f"\n{group_col} Statistics:")
    print(stats)

