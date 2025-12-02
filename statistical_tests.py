"""Statistical analysis functions for transcript data."""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

from plotting import DIAGNOSIS_PALETTE, HEALTH_PALETTE


def perform_3class_ttests(pft_data: pd.DataFrame, sft_data: pd.DataFrame) -> dict:
    """
    Perform two-tailed t-tests for 3-class diagnosis comparison (DEMENTIA, MCI, HC).
    
    Args:
        pft_data: DataFrame with PFT data including 'diagnosis' and 'pause_mean' columns
        sft_data: DataFrame with SFT data including 'diagnosis' and 'pause_mean' columns
        
    Returns:
        Dictionary with test results for PFT and SFT
    """
    # Prepare data for t-tests
    pft_dementia = pft_data[pft_data['diagnosis'] == 'DEMENTIA']['pause_mean'].dropna()
    pft_mci = pft_data[pft_data['diagnosis'] == 'MCI']['pause_mean'].dropna()
    pft_hc = pft_data[pft_data['diagnosis'] == 'HC']['pause_mean'].dropna()
    
    sft_dementia = sft_data[sft_data['diagnosis'] == 'DEMENTIA']['pause_mean'].dropna()
    sft_mci = sft_data[sft_data['diagnosis'] == 'MCI']['pause_mean'].dropna()
    sft_hc = sft_data[sft_data['diagnosis'] == 'HC']['pause_mean'].dropna()
    
    # Perform t-tests and store results
    test_results = {
        'PFT': {},
        'SFT': {}
    }
    
    comparisons = [
        ('DEMENTIA', 'HC', pft_dementia, pft_hc, sft_dementia, sft_hc),
        ('DEMENTIA', 'MCI', pft_dementia, pft_mci, sft_dementia, sft_mci),
        ('MCI', 'HC', pft_mci, pft_hc, sft_mci, sft_hc)
    ]
    
    for group1, group2, pft1, pft2, sft1, sft2 in comparisons:
        # PFT tests
        t_stat, p_value = ttest_ind(pft1, pft2)
        test_results['PFT'][f'{group1} vs {group2}'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # SFT tests
        t_stat, p_value = ttest_ind(sft1, sft2)
        test_results['SFT'][f'{group1} vs {group2}'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    return test_results


def perform_binary_ttests(pft_data: pd.DataFrame, sft_data: pd.DataFrame) -> dict:
    """
    Perform two-tailed t-tests for binary comparison (Healthy vs Non-Healthy).
    
    Args:
        pft_data: DataFrame with PFT data including 'health_status' and 'pause_mean' columns
        sft_data: DataFrame with SFT data including 'health_status' and 'pause_mean' columns
        
    Returns:
        Dictionary with test results for PFT and SFT
    """
    # Prepare data for t-tests
    pft_healthy = pft_data[pft_data['health_status'] == 'Healthy']['pause_mean'].dropna()
    pft_non_healthy = pft_data[pft_data['health_status'] == 'Non-Healthy']['pause_mean'].dropna()
    
    sft_healthy = sft_data[sft_data['health_status'] == 'Healthy']['pause_mean'].dropna()
    sft_non_healthy = sft_data[sft_data['health_status'] == 'Non-Healthy']['pause_mean'].dropna()
    
    # Perform t-tests and store results
    test_results = {
        'PFT': {},
        'SFT': {}
    }
    
    # PFT: Healthy vs Non-Healthy
    t_stat, p_value = ttest_ind(pft_healthy, pft_non_healthy)
    test_results['PFT']['Healthy vs Non-Healthy'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    # SFT: Healthy vs Non-Healthy
    t_stat, p_value = ttest_ind(sft_healthy, sft_non_healthy)
    test_results['SFT']['Healthy vs Non-Healthy'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    return test_results


def print_ttest_results(test_results: dict, task_type: str = 'both') -> None:
    """
    Print t-test results in a formatted way.
    
    Args:
        test_results: Dictionary returned from perform_3class_ttests() or perform_binary_ttests()
        task_type: 'PFT', 'SFT', or 'both' (default: 'both')
    """
    tasks = ['PFT', 'SFT'] if task_type == 'both' else [task_type]
    
    for task in tasks:
        if task not in test_results:
            continue
            
        print("="*80)
        print(f"TWO-TAILED T-TESTS: {task}")
        print("="*80)
        for comparison, results in test_results[task].items():
            sig_marker = '*' if results['significant'] else ''
            stat_key = 't_statistic' if 't_statistic' in results else 'f_statistic'
            print(f"\n{comparison}:")
            print(f"  {stat_key.replace('_', '-')}: {results[stat_key]:.4f}")
            print(f"  p-value: {results['p_value']:.4f}")
            print(f"  Significant (p < 0.05): {'Yes' if results['significant'] else 'No'} {sig_marker}")


def plot_3class_statistics(pft_data: pd.DataFrame, sft_data: pd.DataFrame, test_results: dict) -> None:
    """
    Create comprehensive visualization of 3-class statistical test results.
    
    Args:
        pft_data: DataFrame with PFT data
        sft_data: DataFrame with SFT data
        test_results: Dictionary returned from perform_3class_ttests()
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Bar plots with error bars and significance annotations
    ax1 = plt.subplot(2, 3, 1)
    pft_means = pft_data.groupby('diagnosis', observed=True)['pause_mean'].mean()
    pft_sems = pft_data.groupby('diagnosis', observed=True)['pause_mean'].sem()
    bars = ax1.bar(pft_means.index, pft_means.values, yerr=pft_sems.values, 
                   color=[DIAGNOSIS_PALETTE[d] for d in pft_means.index], capsize=5, alpha=0.7)
    ax1.set_title('PFT: Mean Pause Duration by Diagnosis', fontweight='bold')
    ax1.set_ylabel('Mean Pause Duration (s)')
    ax1.set_xlabel('Diagnosis')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add significance brackets
    y_max = pft_means.max() + pft_sems.max() + 1
    for comparison, results in test_results['PFT'].items():
        if results['significant']:
            groups = comparison.split(' vs ')
            x1, x2 = pft_means.index.get_loc(groups[0]), pft_means.index.get_loc(groups[1])
            y_pos = y_max + 0.5
            ax1.plot([x1, x1, x2, x2], [y_pos-0.2, y_pos, y_pos, y_pos-0.2], 'k-', linewidth=1)
            ax1.text((x1+x2)/2, y_pos+0.1, '*', ha='center', fontsize=14, fontweight='bold')
            y_max += 1
    
    ax2 = plt.subplot(2, 3, 2)
    sft_means = sft_data.groupby('diagnosis', observed=True)['pause_mean'].mean()
    sft_sems = sft_data.groupby('diagnosis', observed=True)['pause_mean'].sem()
    bars = ax2.bar(sft_means.index, sft_means.values, yerr=sft_sems.values,
                   color=[DIAGNOSIS_PALETTE[d] for d in sft_means.index], capsize=5, alpha=0.7)
    ax2.set_title('SFT: Mean Pause Duration by Diagnosis', fontweight='bold')
    ax2.set_ylabel('Mean Pause Duration (s)')
    ax2.set_xlabel('Diagnosis')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add significance brackets
    y_max = sft_means.max() + sft_sems.max() + 1
    for comparison, results in test_results['SFT'].items():
        if results['significant']:
            groups = comparison.split(' vs ')
            x1, x2 = sft_means.index.get_loc(groups[0]), sft_means.index.get_loc(groups[1])
            y_pos = y_max + 0.2
            ax2.plot([x1, x1, x2, x2], [y_pos-0.1, y_pos, y_pos, y_pos-0.1], 'k-', linewidth=1)
            ax2.text((x1+x2)/2, y_pos+0.05, '*', ha='center', fontsize=14, fontweight='bold')
            y_max += 0.5
    
    # 2. P-value heatmap
    ax3 = plt.subplot(2, 3, 3)
    pft_pvalues = [test_results['PFT'][comp]['p_value'] for comp in ['DEMENTIA vs HC', 'DEMENTIA vs MCI', 'MCI vs HC']]
    sft_pvalues = [test_results['SFT'][comp]['p_value'] for comp in ['DEMENTIA vs HC', 'DEMENTIA vs MCI', 'MCI vs HC']]
    pvalue_matrix = np.array([pft_pvalues, sft_pvalues])
    im = ax3.imshow(pvalue_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.1)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(['DEMENTIA\nvs HC', 'DEMENTIA\nvs MCI', 'MCI\nvs HC'], fontsize=9)
    ax3.set_yticks(range(2))
    ax3.set_yticklabels(['PFT', 'SFT'])
    ax3.set_title('P-Values Heatmap', fontweight='bold')
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('p-value', rotation=270, labelpad=15)
    # Add text annotations
    for i in range(2):
        for j in range(3):
            pval = pvalue_matrix[i, j]
            color = 'white' if pval < 0.05 else 'black'
            ax3.text(j, i, f'{pval:.3f}', ha='center', va='center', color=color, fontweight='bold')
    
    # 3. T-statistic comparison
    ax4 = plt.subplot(2, 3, 4)
    comparisons_list = ['DEMENTIA\nvs HC', 'DEMENTIA\nvs MCI', 'MCI\nvs HC']
    stat_key = 't_statistic' if 't_statistic' in test_results['PFT']['DEMENTIA vs HC'] else 'f_statistic'
    pft_tstats = [test_results['PFT'][comp][stat_key] for comp in ['DEMENTIA vs HC', 'DEMENTIA vs MCI', 'MCI vs HC']]
    sft_tstats = [test_results['SFT'][comp][stat_key] for comp in ['DEMENTIA vs HC', 'DEMENTIA vs MCI', 'MCI vs HC']]
    x = np.arange(len(comparisons_list))
    width = 0.35
    ax4.bar(x - width/2, pft_tstats, width, label='PFT', alpha=0.7)
    ax4.bar(x + width/2, sft_tstats, width, label='SFT', alpha=0.7)
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Comparison')
    ax4.set_ylabel(stat_key.replace('_', '-'))
    ax4.set_title(f'{stat_key.replace("_", "-").title()}s by Comparison', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(comparisons_list)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 4. Box plots with significance markers
    ax5 = plt.subplot(2, 3, 5)
    sns.boxplot(data=pft_data, x='diagnosis', y='pause_mean', hue='diagnosis', palette=DIAGNOSIS_PALETTE, legend=False, ax=ax5)
    ax5.set_title('PFT: Distribution with Significance', fontweight='bold')
    ax5.set_ylabel('Mean Pause Duration (s)')
    ax5.set_xlabel('Diagnosis')
    # Add significance markers
    y_max = pft_data['pause_mean'].max() + 1
    for comparison, results in test_results['PFT'].items():
        if results['significant']:
            groups = comparison.split(' vs ')
            x1, x2 = pft_means.index.get_loc(groups[0]), pft_means.index.get_loc(groups[1])
            ax5.plot([x1, x2], [y_max, y_max], 'k-', linewidth=1.5)
            ax5.text((x1+x2)/2, y_max+0.3, '*', ha='center', fontsize=16, fontweight='bold')
            y_max += 1.5
    
    ax6 = plt.subplot(2, 3, 6)
    sns.boxplot(data=sft_data, x='diagnosis', y='pause_mean', hue='diagnosis', palette=DIAGNOSIS_PALETTE, legend=False, ax=ax6)
    ax6.set_title('SFT: Distribution with Significance', fontweight='bold')
    ax6.set_ylabel('Mean Pause Duration (s)')
    ax6.set_xlabel('Diagnosis')
    # Add significance markers
    y_max = sft_data['pause_mean'].max() + 0.5
    for comparison, results in test_results['SFT'].items():
        if results['significant']:
            groups = comparison.split(' vs ')
            x1, x2 = sft_means.index.get_loc(groups[0]), sft_means.index.get_loc(groups[1])
            ax6.plot([x1, x2], [y_max, y_max], 'k-', linewidth=1.5)
            ax6.text((x1+x2)/2, y_max+0.2, '*', ha='center', fontsize=16, fontweight='bold')
            y_max += 0.8
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*80)
    print("SUMMARY: Significant differences (p < 0.05) are marked with *")
    print("="*80)


def plot_binary_statistics(pft_data: pd.DataFrame, sft_data: pd.DataFrame, test_results: dict) -> None:
    """
    Create comprehensive visualization of binary statistical test results.
    
    Args:
        pft_data: DataFrame with PFT data
        sft_data: DataFrame with SFT data
        test_results: Dictionary returned from perform_binary_ttests()
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Bar plots with error bars and significance annotations
    ax1 = plt.subplot(2, 3, 1)
    pft_means = pft_data.groupby('health_status', observed=True)['pause_mean'].mean()
    pft_sems = pft_data.groupby('health_status', observed=True)['pause_mean'].sem()
    bars = ax1.bar(pft_means.index, pft_means.values, yerr=pft_sems.values,
                   color=[HEALTH_PALETTE[h] for h in pft_means.index], capsize=5, alpha=0.7)
    ax1.set_title('PFT: Mean Pause Duration by Health Status', fontweight='bold')
    ax1.set_ylabel('Mean Pause Duration (s)')
    ax1.set_xlabel('Health Status')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add significance bracket if significant
    if test_results['PFT']['Healthy vs Non-Healthy']['significant']:
        y_max = pft_means.max() + pft_sems.max() + 1
        x1, x2 = 0, 1
        ax1.plot([x1, x1, x2, x2], [y_max-0.2, y_max, y_max, y_max-0.2], 'k-', linewidth=1.5)
        ax1.text(0.5, y_max+0.2, '*', ha='center', fontsize=16, fontweight='bold')
    
    ax2 = plt.subplot(2, 3, 2)
    sft_means = sft_data.groupby('health_status', observed=True)['pause_mean'].mean()
    sft_sems = sft_data.groupby('health_status', observed=True)['pause_mean'].sem()
    bars = ax2.bar(sft_means.index, sft_means.values, yerr=sft_sems.values,
                   color=[HEALTH_PALETTE[h] for h in sft_means.index], capsize=5, alpha=0.7)
    ax2.set_title('SFT: Mean Pause Duration by Health Status', fontweight='bold')
    ax2.set_ylabel('Mean Pause Duration (s)')
    ax2.set_xlabel('Health Status')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add significance bracket if significant
    if test_results['SFT']['Healthy vs Non-Healthy']['significant']:
        y_max = sft_means.max() + sft_sems.max() + 0.5
        x1, x2 = 0, 1
        ax2.plot([x1, x1, x2, x2], [y_max-0.1, y_max, y_max, y_max-0.1], 'k-', linewidth=1.5)
        ax2.text(0.5, y_max+0.15, '*', ha='center', fontsize=16, fontweight='bold')
    
    # 2. P-value comparison
    ax3 = plt.subplot(2, 3, 3)
    pvalues = [test_results['PFT']['Healthy vs Non-Healthy']['p_value'],
               test_results['SFT']['Healthy vs Non-Healthy']['p_value']]
    tasks = ['PFT', 'SFT']
    colors = ['#d62728' if p < 0.05 else '#2ca02c' for p in pvalues]
    bars = ax3.bar(tasks, pvalues, color=colors, alpha=0.7)
    ax3.axhline(y=0.05, color='r', linestyle='--', linewidth=1, label='p = 0.05')
    ax3.set_ylabel('p-value')
    ax3.set_title('P-Values: Healthy vs Non-Healthy', fontweight='bold')
    ax3.set_ylim(0, max(pvalues) * 1.2)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    # Add text annotations
    for i, (task, pval) in enumerate(zip(tasks, pvalues)):
        ax3.text(i, pval + max(pvalues) * 0.05, f'{pval:.4f}', ha='center', fontweight='bold')
        if pval < 0.05:
            ax3.text(i, pval + max(pvalues) * 0.12, '*', ha='center', fontsize=16, fontweight='bold')
    
    # 3. T-statistic comparison
    ax4 = plt.subplot(2, 3, 4)
    stat_key = 't_statistic' if 't_statistic' in test_results['PFT']['Healthy vs Non-Healthy'] else 'f_statistic'
    tstats = [test_results['PFT']['Healthy vs Non-Healthy'][stat_key],
              test_results['SFT']['Healthy vs Non-Healthy'][stat_key]]
    bars = ax4.bar(tasks, tstats, color=['#1f77b4', '#ff7f0e'], alpha=0.7)
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax4.set_ylabel(stat_key.replace('_', '-'))
    ax4.set_title(f'{stat_key.replace("_", "-").title()}s: Healthy vs Non-Healthy', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    # Add text annotations
    for i, (task, tstat) in enumerate(zip(tasks, tstats)):
        y_pos = tstat + 0.1 if tstat > 0 else tstat - 0.3
        ax4.text(i, y_pos, f'{tstat:.4f}', ha='center', fontweight='bold')
    
    # 4. Box plots with significance markers
    ax5 = plt.subplot(2, 3, 5)
    sns.boxplot(data=pft_data, x='health_status', y='pause_mean', hue='health_status', palette=HEALTH_PALETTE, legend=False, ax=ax5)
    ax5.set_title('PFT: Distribution with Significance', fontweight='bold')
    ax5.set_ylabel('Mean Pause Duration (s)')
    ax5.set_xlabel('Health Status')
    # Add significance marker if significant
    if test_results['PFT']['Healthy vs Non-Healthy']['significant']:
        y_max = pft_data['pause_mean'].max() + 1
        ax5.plot([0, 1], [y_max, y_max], 'k-', linewidth=1.5)
        ax5.text(0.5, y_max+0.3, '*', ha='center', fontsize=16, fontweight='bold')
    
    ax6 = plt.subplot(2, 3, 6)
    sns.boxplot(data=sft_data, x='health_status', y='pause_mean', hue='health_status', palette=HEALTH_PALETTE, legend=False, ax=ax6)
    ax6.set_title('SFT: Distribution with Significance', fontweight='bold')
    ax6.set_ylabel('Mean Pause Duration (s)')
    ax6.set_xlabel('Health Status')
    # Add significance marker if significant
    if test_results['SFT']['Healthy vs Non-Healthy']['significant']:
        y_max = sft_data['pause_mean'].max() + 0.5
        ax6.plot([0, 1], [y_max, y_max], 'k-', linewidth=1.5)
        ax6.text(0.5, y_max+0.2, '*', ha='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*80)
    print("SUMMARY: Significant differences (p < 0.05) are marked with *")
    print("="*80)


def perform_3class_ancova(pft_data: pd.DataFrame, sft_data: pd.DataFrame) -> dict:
    """
    Perform ANCOVA (Analysis of Covariance) for 3-class diagnosis comparison 
    controlling for age and gender.
    
    Args:
        pft_data: DataFrame with PFT data including 'diagnosis', 'pause_mean', 'age', 'gender' columns
        sft_data: DataFrame with SFT data including 'diagnosis', 'pause_mean', 'age', 'gender' columns
        
    Returns:
        Dictionary with test results for PFT and SFT
    """
    test_results = {
        'PFT': {},
        'SFT': {}
    }
    
    comparisons = [
        ('DEMENTIA', 'HC'),
        ('DEMENTIA', 'MCI'),
        ('MCI', 'HC')
    ]
    
    for group1, group2 in comparisons:
        # PFT ANCOVA
        pft_subset = pft_data[pft_data['diagnosis'].isin([group1, group2])].copy()
        pft_subset = pft_subset.dropna(subset=['pause_mean', 'age', 'gender'])
        
        if len(pft_subset) > 0 and 'gender' in pft_subset.columns:
            pft_subset['gender_encoded'] = (pft_subset['gender'] == 'M').astype(int)
            pft_subset['group'] = (pft_subset['diagnosis'] == group1).astype(int)
            
            try:
                model = ols('pause_mean ~ group + age + gender_encoded', data=pft_subset).fit()
                anova_table = anova_lm(model, typ=2)
                p_value = anova_table.loc['group', 'PR(>F)']
                f_stat = anova_table.loc['group', 'F']
                
                test_results['PFT'][f'{group1} vs {group2}'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            except Exception as e:
                # Fallback to t-test if ANCOVA fails
                pft1 = pft_subset[pft_subset['diagnosis'] == group1]['pause_mean'].dropna()
                pft2 = pft_subset[pft_subset['diagnosis'] == group2]['pause_mean'].dropna()
                t_stat, p_value = ttest_ind(pft1, pft2)
                test_results['PFT'][f'{group1} vs {group2}'] = {
                    'f_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        else:
            # Fallback to t-test if demographics not available
            pft1 = pft_data[pft_data['diagnosis'] == group1]['pause_mean'].dropna()
            pft2 = pft_data[pft_data['diagnosis'] == group2]['pause_mean'].dropna()
            t_stat, p_value = ttest_ind(pft1, pft2)
            test_results['PFT'][f'{group1} vs {group2}'] = {
                'f_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        # SFT ANCOVA
        sft_subset = sft_data[sft_data['diagnosis'].isin([group1, group2])].copy()
        sft_subset = sft_subset.dropna(subset=['pause_mean', 'age', 'gender'])
        
        if len(sft_subset) > 0 and 'gender' in sft_subset.columns:
            sft_subset['gender_encoded'] = (sft_subset['gender'] == 'M').astype(int)
            sft_subset['group'] = (sft_subset['diagnosis'] == group1).astype(int)
            
            try:
                model = ols('pause_mean ~ group + age + gender_encoded', data=sft_subset).fit()
                anova_table = anova_lm(model, typ=2)
                p_value = anova_table.loc['group', 'PR(>F)']
                f_stat = anova_table.loc['group', 'F']
                
                test_results['SFT'][f'{group1} vs {group2}'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            except Exception as e:
                # Fallback to t-test if ANCOVA fails
                sft1 = sft_subset[sft_subset['diagnosis'] == group1]['pause_mean'].dropna()
                sft2 = sft_subset[sft_subset['diagnosis'] == group2]['pause_mean'].dropna()
                t_stat, p_value = ttest_ind(sft1, sft2)
                test_results['SFT'][f'{group1} vs {group2}'] = {
                    'f_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        else:
            # Fallback to t-test if demographics not available
            sft1 = sft_data[sft_data['diagnosis'] == group1]['pause_mean'].dropna()
            sft2 = sft_data[sft_data['diagnosis'] == group2]['pause_mean'].dropna()
            t_stat, p_value = ttest_ind(sft1, sft2)
            test_results['SFT'][f'{group1} vs {group2}'] = {
                'f_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    
    return test_results


def perform_binary_ancova(pft_data: pd.DataFrame, sft_data: pd.DataFrame) -> dict:
    """
    Perform ANCOVA (Analysis of Covariance) for binary comparison (Healthy vs Non-Healthy)
    controlling for age and gender.
    
    Args:
        pft_data: DataFrame with PFT data including 'health_status', 'pause_mean', 'age', 'gender' columns
        sft_data: DataFrame with SFT data including 'health_status', 'pause_mean', 'age', 'gender' columns
        
    Returns:
        Dictionary with test results for PFT and SFT
    """
    test_results = {
        'PFT': {},
        'SFT': {}
    }
    
    # PFT ANCOVA
    pft_subset = pft_data.dropna(subset=['pause_mean', 'age', 'gender']).copy()
    
    if len(pft_subset) > 0 and 'gender' in pft_subset.columns:
        pft_subset['gender_encoded'] = (pft_subset['gender'] == 'M').astype(int)
        pft_subset['group'] = (pft_subset['health_status'] == 'Non-Healthy').astype(int)
        
        try:
            model = ols('pause_mean ~ group + age + gender_encoded', data=pft_subset).fit()
            anova_table = anova_lm(model, typ=2)
            p_value = anova_table.loc['group', 'PR(>F)']
            f_stat = anova_table.loc['group', 'F']
            
            test_results['PFT']['Healthy vs Non-Healthy'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        except Exception as e:
            # Fallback to t-test if ANCOVA fails
            pft_healthy = pft_data[pft_data['health_status'] == 'Healthy']['pause_mean'].dropna()
            pft_non_healthy = pft_data[pft_data['health_status'] == 'Non-Healthy']['pause_mean'].dropna()
            t_stat, p_value = ttest_ind(pft_healthy, pft_non_healthy)
            test_results['PFT']['Healthy vs Non-Healthy'] = {
                'f_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    else:
        # Fallback to t-test if demographics not available
        pft_healthy = pft_data[pft_data['health_status'] == 'Healthy']['pause_mean'].dropna()
        pft_non_healthy = pft_data[pft_data['health_status'] == 'Non-Healthy']['pause_mean'].dropna()
        t_stat, p_value = ttest_ind(pft_healthy, pft_non_healthy)
        test_results['PFT']['Healthy vs Non-Healthy'] = {
            'f_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    # SFT ANCOVA
    sft_subset = sft_data.dropna(subset=['pause_mean', 'age', 'gender']).copy()
    
    if len(sft_subset) > 0 and 'gender' in sft_subset.columns:
        sft_subset['gender_encoded'] = (sft_subset['gender'] == 'M').astype(int)
        sft_subset['group'] = (sft_subset['health_status'] == 'Non-Healthy').astype(int)
        
        try:
            model = ols('pause_mean ~ group + age + gender_encoded', data=sft_subset).fit()
            anova_table = anova_lm(model, typ=2)
            p_value = anova_table.loc['group', 'PR(>F)']
            f_stat = anova_table.loc['group', 'F']
            
            test_results['SFT']['Healthy vs Non-Healthy'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        except Exception as e:
            # Fallback to t-test if ANCOVA fails
            sft_healthy = sft_data[sft_data['health_status'] == 'Healthy']['pause_mean'].dropna()
            sft_non_healthy = sft_data[sft_data['health_status'] == 'Non-Healthy']['pause_mean'].dropna()
            t_stat, p_value = ttest_ind(sft_healthy, sft_non_healthy)
            test_results['SFT']['Healthy vs Non-Healthy'] = {
                'f_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    else:
        # Fallback to t-test if demographics not available
        sft_healthy = sft_data[sft_data['health_status'] == 'Healthy']['pause_mean'].dropna()
        sft_non_healthy = sft_data[sft_data['health_status'] == 'Non-Healthy']['pause_mean'].dropna()
        t_stat, p_value = ttest_ind(sft_healthy, sft_non_healthy)
        test_results['SFT']['Healthy vs Non-Healthy'] = {
            'f_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    return test_results
