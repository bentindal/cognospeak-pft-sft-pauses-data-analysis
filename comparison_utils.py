#!/usr/bin/env python3
"""Utilities for comparing statistical test results."""

from __future__ import annotations

from typing import Optional


def compare_test_results(
    test_results_orig: dict,
    test_results_demo: dict,
    comparisons: list[str],
    tasks: list[str] = ['PFT', 'SFT'],
) -> None:
    """
    Compare original test results with demographic-adjusted results.
    
    Args:
        test_results_orig: Original test results dictionary
        test_results_demo: Demographic-adjusted test results dictionary
        comparisons: List of comparison names (e.g., ['DEMENTIA vs HC', 'DEMENTIA vs MCI'])
        tasks: List of tasks to compare (default: ['PFT', 'SFT'])
    """
    print("\n" + "=" * 80)
    print("COMPARISON: Original t-tests vs ANCOVA (with demographics)")
    print("=" * 80)
    
    for task in tasks:
        print(f"\n{task}:")
        for comp in comparisons:
            # Check if both results exist
            if comp not in test_results_orig.get(task, {}):
                print(f"  {comp}:")
                print(f"    Original t-test: NOT AVAILABLE")
                if test_results_demo and comp in test_results_demo.get(task, {}):
                    demo_p = test_results_demo[task][comp]['p_value']
                    demo_sig = test_results_demo[task][comp]['significant']
                    print(f"    ANCOVA p-value:   {demo_p:.4f} {'*' if demo_sig else ''}")
                else:
                    print(f"    ANCOVA: NOT AVAILABLE (test failed or not computed)")
                continue
            
            if not test_results_demo or comp not in test_results_demo.get(task, {}):
                orig_p = test_results_orig[task][comp]['p_value']
                orig_sig = test_results_orig[task][comp]['significant']
                print(f"  {comp}:")
                print(f"    Original p-value: {orig_p:.4f} {'*' if orig_sig else ''}")
                print(f"    ANCOVA: NOT AVAILABLE (test failed or not computed)")
                continue
            
            # Both results exist, compare them
            orig_p = test_results_orig[task][comp]['p_value']
            demo_p = test_results_demo[task][comp]['p_value']
            orig_sig = test_results_orig[task][comp]['significant']
            demo_sig = test_results_demo[task][comp]['significant']
            
            change = ""
            if not orig_sig and demo_sig:
                change = " → NOW SIGNIFICANT!"
            elif orig_sig and not demo_sig:
                change = " → NO LONGER SIGNIFICANT"
            
            print(f"  {comp}:")
            print(f"    Original p-value: {orig_p:.4f} {'*' if orig_sig else ''}")
            print(f"    ANCOVA p-value:   {demo_p:.4f} {'*' if demo_sig else ''}{change}")


def compare_binary_test_results(
    binary_test_results_orig: dict,
    binary_test_results_demo: dict,
    tasks: list[str] = ['PFT', 'SFT'],
    comparison_name: str = 'Healthy vs Non-Healthy',
) -> None:
    """
    Compare binary test results (original vs demographic-adjusted).
    
    Args:
        binary_test_results_orig: Original binary test results
        binary_test_results_demo: Demographic-adjusted binary test results
        tasks: List of tasks to compare
        comparison_name: Name of the comparison (default: 'Healthy vs Non-Healthy')
    """
    print("\n" + "=" * 80)
    print("COMPARISON: Original t-tests vs ANCOVA (with demographics)")
    print("=" * 80)
    
    for task in tasks:
        if task not in binary_test_results_orig or task not in binary_test_results_demo:
            continue
        
        orig_p = binary_test_results_orig[task][comparison_name]['p_value']
        demo_p = binary_test_results_demo[task][comparison_name]['p_value']
        orig_sig = binary_test_results_orig[task][comparison_name]['significant']
        demo_sig = binary_test_results_demo[task][comparison_name]['significant']
        
        change = ""
        if not orig_sig and demo_sig:
            change = " → NOW SIGNIFICANT!"
        elif orig_sig and not demo_sig:
            change = " → NO LONGER SIGNIFICANT"
        
        print(f"\n{task}:")
        print(f"  Original p-value: {orig_p:.4f} {'*' if orig_sig else ''}")
        print(f"  ANCOVA p-value:   {demo_p:.4f} {'*' if demo_sig else ''}{change}")

