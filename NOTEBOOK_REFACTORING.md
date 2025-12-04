# Notebook Refactoring Guide

This document describes the utility modules created to extract common code from the notebook.

## New Modules

### 1. `data_utils.py`
Functions for data preparation and merging:
- `prepare_pause_data_with_metadata()` - Merge pause data with metadata
- `prepare_demographic_data()` - Add demographic data (age, gender) to pause data
- `get_original_data_for_comparison()` - Get original data for statistical comparisons

### 2. `repeated_words_analysis.py`
Functions for repeated words feature analysis:
- `compute_repeated_words_dataset()` - Compute repeated words metrics for all transcripts
- `print_repeated_words_statistics()` - Print summary statistics by diagnosis and task

### 3. `feature_utils.py`
Utilities for feature analysis:
- `load_feature_weights()` - Load and format feature weights from CSV
- `display_feature_importance()` - Display feature importance information
- `compare_classifier_performance()` - Compare metrics between two classifier configurations

### 4. `comparison_utils.py`
Utilities for comparing statistical test results:
- `compare_test_results()` - Compare original vs demographic-adjusted test results
- `compare_binary_test_results()` - Compare binary test results

## Usage Examples

### Data Preparation
```python
from data_utils import prepare_pause_data_with_metadata, prepare_demographic_data

# Prepare pause data with metadata
pause_with_diag_filtered = prepare_pause_data_with_metadata(loader, metadata)

# Prepare demographic data
pft_data_demo, sft_data_demo = prepare_demographic_data(
    pause_with_diag_filtered, 
    metadata
)
```

### Repeated Words Analysis
```python
from repeated_words_analysis import (
    compute_repeated_words_dataset,
    print_repeated_words_statistics
)

# Compute repeated words dataset
repeated_words_with_diag = compute_repeated_words_dataset(loader, metadata)

# Print statistics
print_repeated_words_statistics(repeated_words_with_diag)
```

### Feature Importance
```python
from feature_utils import load_feature_weights, display_feature_importance
from pathlib import Path

# Load feature weights
feature_weights_df = load_feature_weights(Path('output/feature_importance_logistic.csv'))

# Display importance for specific features
display_feature_importance(
    feature_weights_df,
    feature_names=['sft_repeated_words_count', 'sft_repeated_words_std']
)
```

### Statistical Comparisons
```python
from comparison_utils import compare_test_results, compare_binary_test_results

# Compare 3-class test results
compare_test_results(
    test_results,
    test_results_demo,
    comparisons=['DEMENTIA vs HC', 'DEMENTIA vs MCI', 'MCI vs HC']
)

# Compare binary test results
compare_binary_test_results(
    binary_test_results_orig,
    binary_test_results_demo
)
```

## Benefits

1. **Code Reusability** - Common patterns extracted into reusable functions
2. **Maintainability** - Changes to logic only need to be made in one place
3. **Testability** - Functions can be unit tested independently
4. **Readability** - Notebook cells become shorter and more focused
5. **Consistency** - Same logic used across different analyses

## Migration Notes

When updating notebook cells to use these modules:
1. Import the necessary functions at the top of the cell
2. Replace inline code with function calls
3. Keep visualization and display code in the notebook (that's the notebook's strength)
4. Keep exploratory analysis in the notebook (one-off analyses don't need to be extracted)

