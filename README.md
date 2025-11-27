# CognoSpeak Transcript Analysis

## Quick Start

1. **Install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Set up your data folder structure:**
   ```
   data/
   ├── Process-rec-XXX/
   │   ├── Process-rec-XXX__PFT.txt  (or Process-rec-XXX_PFT.txt)
   │   └── Process-rec-XXX__SFT.txt  (or Process-rec-XXX_SFT.txt)
   ├── Process-test-XXX/
   │   ├── Process-test-XXX__PFT.txt
   │   └── Process-test-XXX__SFT.txt
   └── speechpause_mata.csv
   ```

3. **Export your metadata CSV file** with these columns:
   - `anyon_IDs` (recording identifier, e.g., `Process-rec-006`)
   - `age`, `gender`, `diagnosis` (Dementia/MCI/HC), `MMSE`, `Assigned`, `Notes`

4. **Place the CSV in `data/` as `speechpause_mata.csv`**

5. **Run the analysis `transcript_analysis.ipynb`**

**Note:** The `data/` directory is excluded from git. Never commit private data files.

## Binary Classifier: Dementia vs Healthy

The `binary_classifier.py` script implements a supervised binary classification model to distinguish Dementia (1) from Healthy (0) using SFT pause metrics as primary features.

### Usage

**Basic usage (SFT features only, Logistic Regression):**
```bash
python binary_classifier.py
```

**With PFT features included:**
```bash
python binary_classifier.py --include-pft
```

**Using Random Forest classifier:**
```bash
python binary_classifier.py --classifier random_forest
```

**Custom k-fold cross-validation:**
```bash
python binary_classifier.py --k-folds 10
```

**Full example with all options:**
```bash
python binary_classifier.py \
    --data-dir data \
    --metadata data/speechpause_mata.csv \
    --classifier random_forest \
    --k-folds 5 \
    --include-pft \
    --output-dir output
```

### Command-line Arguments

- `--data-dir`: Directory containing transcript files (default: `data`)
- `--metadata`: Path to metadata CSV file (default: `data/speechpause_mata.csv`)
- `--classifier`: Classifier to use - `logistic` or `random_forest` (default: `logistic`)
- `--k-folds`: Number of folds for cross-validation (default: `5`)
- `--include-pft`: Include PFT pause features in addition to SFT
- `--output-dir`: Directory to save output files (default: `output`)

### Features

The classifier uses the following pause metrics from SFT (and optionally PFT) transcripts:
- `pause_mean`: Mean pause duration
- `pause_median`: Median pause duration
- `pause_min`: Minimum pause duration
- `pause_max`: Maximum pause duration
- `pause_std`: Standard deviation of pause durations
- `pause_total`: Total pause time
- `pause_count`: Number of pauses
- `pause_q1`: First quartile (25th percentile)
- `pause_q2`: Second quartile (median, 50th percentile)
- `pause_q3`: Third quartile (75th percentile)

### Output

The script prints:
- Dataset summary (number of recordings, class distribution)
- List of features used
- K-fold cross-validation results (mean ± std for accuracy, precision, recall, F1-score, ROC-AUC)
- Feature importance/coefficients (classifier-dependent)

It also saves:
- Feature importance/coefficients CSV file to `output/feature_importance_{classifier}.csv`

### Classifier Selection

The code is designed for easy classifier swapping. Supported classifiers:
- **logistic**: Logistic Regression (default, interpretable, good baseline)
- **random_forest**: Random Forest (handles non-linear relationships)

To add a new classifier, extend the `get_classifier()` function in `binary_classifier.py`.
