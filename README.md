# CognoSpeak: Speech-Based Cognitive Assessment Through Fluency and Pause Analysis

A research toolkit for analysing **phonemic fluency (PFT)** and **semantic fluency (SFT)** recordings using speech transcripts, temporal pause features, statistical analysis, classical machine learning, and deep learning.

CognoSpeak provides an end-to-end research pipeline for investigating speech-derived markers of cognitive decline across Dementia, Mild Cognitive Impairment (MCI), and Healthy Control (HC) cohorts.

The repository includes reproducible statistical analyses, baseline classifiers, lexical embedding pipelines, temporal deep-learning models, and cross-attention fusion experiments.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Module Reference](#module-reference)
  - [Data Loading and Validation](#1-data-loading-and-validation)
  - [Pause Analysis and Feature Engineering](#2-pause-analysis-and-feature-engineering)
  - [Statistical Analysis](#3-statistical-analysis)
  - [Classical Classification](#4-classical-classification)
  - [Word and Sentence Embeddings](#5-word-and-sentence-embeddings)
  - [Deep Learning Models](#6-deep-learning-models)
  - [Experiment Runners](#7-experiment-runners)
- [Notebooks](#notebooks)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results Summary](#results-summary)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Data Governance and Responsible Use](#data-governance-and-responsible-use)
- [Citation and Contact](#citation-and-contact)

---

## Overview

CognoSpeak is a speech-based cognitive assessment research platform that investigates verbal fluency and speech timing as potential markers of cognitive decline.

Participants complete two short verbal-fluency tasks.

### Phonemic Fluency Test (PFT)

Participants are asked to produce as many words as possible beginning with a specified letter, typically **P**, within one minute.

This task probes phonemic retrieval, executive function, and lexical search.

### Semantic Fluency Test (SFT)

Participants are asked to name as many animals as possible within one minute.

This task probes semantic memory and category fluency.

Both tasks are widely used in cognitive assessment and can provide useful information when investigating cognitive impairment. However, the results of this research pipeline should not be interpreted as a standalone clinical diagnosis.

### Analysis Pipeline

For each recording, the pipeline supports the following workflow:

1. **Transcript loading:** Load token-level transcripts containing `start_time`, `end_time`, and `token`.
2. **Pause analysis:** Calculate inter-word pauses and aggregate them into recording-level statistics, including mean, median, standard deviation, and quantiles.
3. **Statistical analysis:** Compare Dementia, MCI, and Healthy Control cohorts using statistical tests and ANCOVA.
4. **Feature extraction:** Generate pause-based, lexical, and temporal representations from the recordings.
5. **Model training and evaluation:** Evaluate classical machine-learning classifiers, TF-IDF representations, Sentence-Transformer embeddings, Word2Vec embeddings, and deep sequence models.
6. **Comparative experiments:** Evaluate four model families using stratified cross-validation:
   - Temporal Logistic Regression
   - Word2Vec + Logistic Regression
   - Autoencoder + Temporal Convolutional Network (AE + TCN) Fusion
   - AE + TCN + Word2Vec Cross-Attention Fusion

The project is designed to support reproducible experimentation and comparative analysis of speech-derived features for cognitive assessment research.

---

## Key Features

- **End-to-end research pipeline:** Transcript ingestion, validation, feature extraction, statistical analysis, classification, and visualisation.
- **Pause-based analysis:** Recording-level pause statistics and cohort comparisons.
- **Statistical inference:** Pairwise t-tests and ANCOVA with age and gender covariates.
- **Classical machine learning:** Logistic Regression and Random Forest baselines.
- **Explicit MCI analysis:** Three-class classification and alternative strategies for handling MCI.
- **Lexical representations:** TF-IDF, Word2Vec, and Sentence-Transformer embeddings.
- **Temporal modelling:** Convolutional autoencoders and Temporal Convolutional Networks.
- **Cross-modal fusion:** Multi-head cross-attention between temporal and lexical representations.
- **Cross-validation:** Stratified k-fold evaluation, with fold-specific embedding training where implemented.
- **Reproducible experiments:** Configurable command-line interfaces and automated result generation.
- **Visual reporting:** Statistical plots, confusion matrices, classification reports, and model-comparison figures.

---

## Repository Structure

The repository is organised into a Python package, analysis notebooks, experiment outputs, and a dataset directory.

```text
.
├── data/                              # Dataset root
│   ├── Process-rec-004/
│   │   ├── Process-rec-004__PFT.txt
│   │   ├── Process-rec-004__PFT.wav
│   │   ├── Process-rec-004__SFT.txt
│   │   └── Process-rec-004__SFT.wav
│   ├── Process-rec-006/
│   │   ├── Process-rec-006_PFT.txt
│   │   ├── Process-rec-006_PFT.wav
│   │   ├── Process-rec-006_SFT.txt
│   │   └── Process-rec-006_SFT.wav
│   └── speechpause_mata.csv           # Cohort metadata
│
├── notebooks/                         # Exploratory analysis
│   ├── transcript_analysis_baseline.ipynb
│   ├── transcript_analysis.ipynb
│   ├── tfidf-analysis.ipynb
│   └── LLM_analysis.ipynb
│
├── output/                            # Generated results and figures
│
├── src/
│   └── cognospeak/                    # Importable Python package
│       ├── dataloader.py
│       ├── file_validator.py
│       ├── pause_analyzer.py
│       ├── feature_extraction.py
│       ├── data_utils.py
│       ├── statistical_tests.py
│       ├── plotting.py
│       ├── binary_classifier.py
│       ├── multiclass_classifier.py
│       ├── mci_impact_analysis.py
│       ├── word2vec.py
│       ├── word2vecv2_stratified.py
│       ├── clustering_and_classification.py
│       ├── ae.py
│       ├── tcn_model.py
│       ├── cross_attention_fusion.py
│       └── run_temporal_experiment_cross_attention.py
│
├── pyproject.toml                     # Package configuration
├── requirements.txt                   # Python dependencies
├── reset_experiment.sh                # Reset generated artefacts
└── README.md
```

---

## Module Reference

The following sections describe the primary Python modules and their responsibilities.

### 1. Data Loading and Validation

| Module | Description |
|---|---|
| `dataloader.py` | Implements `TranscriptLoader`, which discovers transcript files recursively, loads token-level data into pandas DataFrames, infers PFT/SFT tasks from filenames, and merges transcripts with diagnostic and demographic metadata from `speechpause_mata.csv`. Supports both `_PFT` and `__PFT` filename conventions. |
| `file_validator.py` | Provides `validate_file_naming()` and `print_validation_report()` to check transcript naming conventions and identify missing or malformed entries. |

### 2. Pause Analysis and Feature Engineering

| Module | Description |
|---|---|
| `pause_analyzer.py` | Implements transcript parsing, inter-word pause computation, and recording-level summaries. The `load_transcript()`, `compute_pauses()`, and `summarize()` functions support pause counts, mean, median, minimum, maximum, standard deviation, total pause duration, and quartiles. The `analyze_pauses` CLI writes `pause_summary.csv`. |
| `feature_extraction.py` | Extracts temporal and static features for machine-learning models. `bin_features()` divides transcripts into fixed three-second windows and generates six features per bin: token count, unique-token count, speech duration, pause sum, mean pause, and maximum pause. `transcript_summary_features()` produces an 11-dimensional static feature vector. `dict_to_feature_tensors()` creates tensors with shapes `(N, bins, 6)` and `(N, 11)`. `base_recording_from_key()` maps transcript keys to their parent recording IDs. |
| `data_utils.py` | Implements `TemporalDataModule` and `TaskData` for loading PFT/SFT features, retaining recordings with valid metadata, standardising sequence and static features within each fold, and preparing PyTorch tensors. Also provides `make_fold_split()` for cross-validation. |

### 3. Statistical Analysis

| Module | Description |
|---|---|
| `statistical_tests.py` | Provides `perform_3class_ttests()` and `perform_binary_ttests()` for two-tailed comparisons of pause means across diagnostic groups. Also implements `perform_3class_ancova()` and `perform_binary_ancova()` to control for age and gender, with fallback to t-tests when covariates are unavailable. The plotting functions generate multi-panel visualisations, p-value heatmaps, boxplots, and significance annotations. |
| `plotting.py` | Contains shared visualisation utilities and colour palettes, including `DIAGNOSIS_PALETTE` and `HEALTH_PALETTE`. Provides `plot_3class_comparison()`, `plot_binary_comparison()`, and `print_descriptive_stats()`. |

### 4. Classical Classification

| Module | Description |
|---|---|
| `binary_classifier.py` | Implements binary Dementia-versus-Healthy classification using SFT pause features, with optional PFT features and demographic covariates. Supports Logistic Regression and Random Forest classifiers with stratified k-fold cross-validation. Reports accuracy, precision, recall, F1, and ROC-AUC, and saves feature importances or coefficients. |
| `multiclass_classifier.py` | Implements three-class classification across Healthy Controls, MCI, and Dementia using pause-based features. Produces per-class classification reports, macro-averaged metrics, confusion matrices, and feature-importance outputs. |
| `mci_impact_analysis.py` | Compares alternative MCI-handling strategies using identical cross-validation folds: exclude MCI, assign MCI to Dementia, assign MCI to Healthy Controls, classify Non-Healthy versus Healthy, or retain all three classes. Produces a unified comparison table and a summary of trade-offs. |

### 5. Word and Sentence Embeddings

| Module | Description |
|---|---|
| `word2vec.py` | Implements a standalone skip-gram Word2Vec training pipeline. Builds a vocabulary for each task, generates negative-sampled training pairs, trains a Keras embedding model, and provides `most_similar()` for nearest-neighbour inspection. Includes PCA and t-SNE visualisations. |
| `word2vecv2_stratified.py` | Implements `SemanticEmbeddingPipeline`, which supports task-specific vocabulary construction, fold-safe Word2Vec training, embedding persistence in `.npz` files, nearest-neighbour queries, and document-level mean pooling. Also provides `run_word2vec_stratified_cv()` for cross-validation experiments. Embeddings are retrained using each training split to reduce data leakage. |
| `clustering_and_classification.py` | Provides Sentence-Transformer-based semantic analysis using `all-MiniLM-L6-v2`. Supports diagnosis-coloured PCA projections through `visualise_by_diagnosis()` and classification through `train_lr_classifier()`, which combines StandardScaler, PCA, and Logistic Regression. Includes k-fold evaluation and automatic fusion of SFT and PFT embeddings. |

### 6. Deep Learning Models

| Module | Description |
|---|---|
| `ae.py` | Implements a convolutional autoencoder for binned temporal features. The encoder maps `(B, 6, T)` inputs to latent representations shaped approximately `(B, latent_ch, T/2)`, and the decoder reconstructs the input sequence. `ConvolutionalAutoencoder` provides training, encoding, and reconstruction functionality. |
| `tcn_model.py` | Implements a Temporal Convolutional Network using dilated causal convolutions and residual connections. It consumes autoencoder latent sequences and generates timestep-level temporal representations. |
| `cross_attention_fusion.py` | Implements multi-head attention for aligning temporal representations from the AE + TCN pipeline with lexical Word2Vec embeddings. Supports causal masking, attention-head splitting, and dropout. |

### 7. Experiment Runners

| Module | Description |
|---|---|
| `run_temporal_experiment_cross_attention.py` | Runs the main deep-learning benchmark across PFT/SFT tasks and binary/three-class classification settings. Evaluates four model families: Temporal Logistic Regression, Word2Vec + Logistic Regression, AE + TCN Fusion, and AE + TCN + Word2Vec Cross-Attention. The autoencoder is trained separately within each fold using the training split. Outputs include fold-accuracy plots, confusion matrices, classification-report heatmaps, and mean-accuracy summaries. |

---

## Notebooks

The notebooks provide interactive workflows for exploratory analysis, statistical evaluation, and baseline modelling.

| Notebook | Purpose |
|---|---|
| `transcript_analysis_baseline.ipynb` | End-to-end baseline workflow covering data loading, filename validation, pause summaries, descriptive statistics by diagnosis, t-tests, ANCOVA, binary and multiclass classification, and a demographics-augmented ablation. Recommended starting point for reproducing the statistical analyses. |
| `transcript_analysis.ipynb` | A more compact analysis focused on descriptive plots, t-tests, and multi-panel statistical summaries. Useful for exploratory analysis and presentation preparation. |
| `tfidf-analysis.ipynb` | Analyses TF-IDF representations of tokenised transcripts. Includes lemmatised tokenisation, per-class term rankings, and scikit-learn classification pipelines using Logistic Regression and Random Forest, with Optuna tuning. Reports confusion matrices and per-class metrics for Healthy versus Non-Healthy classification. |
| `LLM_analysis.ipynb` | Explores semantic representations using Sentence-Transformer embeddings. Includes diagnosis-coloured PCA projections, individual-recording analysis, and a StandardScaler → PCA → Logistic Regression pipeline evaluated using k-fold cross-validation. |

---

## Installation

### Requirements

- Python 3.10 or later
- Git
- Access to the dataset and required cohort metadata
- Dependencies listed in `requirements.txt`

### Set Up the Environment

Clone the repository and navigate to its root directory.

```bash
git clone <repository-url>
cd <repository-directory>
```

Create and activate a virtual environment.

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Upgrade pip and install the dependencies.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The editable installation (`-e .`) makes the `cognospeak` package available to notebooks and command-line scripts.

### Key Dependencies

The project uses the following libraries and frameworks:

- **Data processing:** pandas, NumPy
- **Classical machine learning:** scikit-learn, SciPy
- **Statistical modelling:** statsmodels
- **Visualisation:** Matplotlib, Seaborn
- **Deep learning:** TensorFlow, PyTorch
- **Language representations:** Sentence-Transformers, NLTK
- **Additional analysis and optimisation:** HDBSCAN, Optuna

The exact dependency versions are defined in the repository's dependency configuration files.

---

## Data Setup

### Directory Layout

Place the transcript files under `data/`, organised by recording. Each recording folder should contain one PFT transcript and one SFT transcript. Matching `.wav` files may also be present.

```text
data/
├── Process-rec-004/
│   ├── Process-rec-004__PFT.txt
│   ├── Process-rec-004__PFT.wav
│   ├── Process-rec-004__SFT.txt
│   └── Process-rec-004__SFT.wav
│
├── Process-rec-006/
│   ├── Process-rec-006_PFT.txt
│   ├── Process-rec-006_PFT.wav
│   ├── Process-rec-006_SFT.txt
│   └── Process-rec-006_SFT.wav
│
├── Process-test-008/
│   ├── Process-test-008__PFT.txt
│   └── Process-test-008__SFT.txt
│
└── speechpause_mata.csv
```

The loader supports both single-underscore and double-underscore PFT/SFT filename conventions.

### Transcript Format

Each transcript is a tab-delimited text file with three columns:

| Column | Description |
|---|---|
| `start_time` | Token start time in seconds |
| `end_time` | Token end time in seconds |
| `token` | Recognised word or token |

Whisper-style ASR output is supported directly.

### Example: Phonemic Fluency Test (PFT)

The following example contains words beginning with the target letter P, recorded during a one-minute task.

File: `Process-rec-004__PFT.txt`

```text
0.284000    0.708000    plank
1.948000    2.532000    pool
4.090000    5.000000    pool
8.092000    8.900000    plepthora
9.820000    10.532000   puzzle
11.004000   11.620000   place
12.284000   12.772000   point
13.756000   14.404000   paper
14.876000   15.364000   paint
16.156000   16.964000   pencil
18.460000   19.076000   power
20.732000   21.412000   please
22.076000   22.756000   people
24.668000   25.412000   plastic
27.470000   27.970000   plate
32.732000   33.380000   prevent
34.012000   34.724000   provide
36.316000   37.156000   perhaps
44.060000   45.092000   papyrus
45.724000   46.692000   perplexed
48.476000   49.220000   promote
50.012000   50.916000   promotion
54.812000   55.652000   pandemic
59.516000   60.068000   panic
```

### Example: Semantic Fluency Test (SFT)

The following example contains animal names produced during a one-minute task.

File: `Process-rec-004__SFT.txt`

```text
4.444000    5.188000    cheetah
5.820000    6.692000    lion
7.292000    8.388000    panda
9.596000    11.076000   panda
14.588000   15.448000   elephant
15.448000   16.036000   giraffe
17.596000   18.340000   springbok
18.844000   19.620000   zebra
20.284000   21.124000   rhinoceros
24.764000   25.796000   armadillo
27.516000   28.004000   fox
40.668000   41.476000   eagle
42.460000   43.172000   peacock
44.604000   45.476000   hummingbird
46.748000   48.388000   robin
54.844000   55.972000   hippopotamus
```

### Cohort Metadata

The metadata file is `speechpause_mata.csv`. It must contain at least the following fields:

| Field | Description |
|---|---|
| `anyon_IDs` | Identifier matching the recording folder name |
| `diagnosis` | Diagnostic group |
| `age` | Participant age |
| `gender` | Participant gender |

The fields `MMSE`, `Assigned`, and `Notes` are optional.

Ensure that participant identifiers and transcript folder names match correctly so that transcript data can be joined to the appropriate diagnostic and demographic records.

---

## Usage

Run the following commands from the repository root after installing the dependencies and configuring the dataset.

### 1. Reproduce the Statistical Baselines

Start with the baseline notebook for data validation, pause analysis, statistical comparisons, and classical classification.

```bash
jupyter notebook notebooks/transcript_analysis_baseline.ipynb
```

### 2. Binary Dementia-versus-Healthy Classification

Run Logistic Regression:

```bash
python -m cognospeak.binary_classifier \
    --classifier logistic \
    --k-folds 5
```

Run Random Forest with optional PFT features and demographic covariates:

```bash
python -m cognospeak.binary_classifier \
    --classifier random_forest \
    --include-pft \
    --include-demographics
```

### 3. Three-Class Classification

Evaluate Healthy Controls, MCI, and Dementia.

```bash
python -m cognospeak.multiclass_classifier \
    --k-folds 5 \
    --include-pft
```

### 4. Investigate the Impact of MCI Handling

Compare alternative strategies for assigning or excluding MCI participants.

```bash
python -m cognospeak.mci_impact_analysis \
    --k-folds 5
```

### 5. Run the Main Deep-Learning Benchmark

Evaluate the four model families across both tasks and both classification settings.

```bash
python -m cognospeak.run_temporal_experiment_cross_attention \
    --task both \
    --classification-mode both \
    --k-folds 5 \
    --epochs 30 \
    --batch-size 8 \
    --output-dir results_graphs
```

The experiment runner generates accuracy plots, confusion matrices, classification-report heatmaps, and model-comparison summaries.

### 6. Reset Generated Artefacts

To clear generated experiment files and cached Python artefacts:

```bash
./reset_experiment.sh
```

The script prompts for confirmation before deleting the following:

- `output/`
- `pause_summary.csv`
- `combined_transcripts.parquet`
- `__pycache__/`

Review the script before running it to confirm that its deletion behaviour is appropriate for your environment.

---

## Methodology

This section summarises the feature representations, statistical analyses, and model architectures used in the current implementation.

### Pause Representation

Inter-word silences are computed for each recording and summarised using the following ten statistics:

- Pause count
- Mean pause duration
- Median pause duration
- Minimum pause duration
- Maximum pause duration
- Standard deviation of pause duration
- Total pause duration
- First quartile (Q1)
- Second quartile (Q2)
- Third quartile (Q3)

These features support descriptive analysis, statistical comparisons, and classical classification.

### Statistical Testing

The statistical analysis includes:

- Two-tailed independent t-tests for pairwise comparisons between diagnostic groups.
- ANCOVA with age and gender as covariates.
- Descriptive statistics and visualisations of differences between cohorts.
- P-value heatmaps, boxplots, and significance annotations.

The implementation can fall back to t-tests when the required covariates are unavailable.

### Classical Machine Learning

The classical models include Logistic Regression and Random Forest.

The Logistic Regression pipeline uses standardised features and balanced class weights. Random Forest is configured with 100 trees and a maximum depth of 10 in the documented experimental setup.

Cross-validation uses stratified folds with shuffling and `random_state=42` where specified.

### Word2Vec

The lexical embedding pipeline uses skip-gram Word2Vec with the following documented configuration:

- Context window: 1–2
- Negative samples: 3–5
- Embedding dimension: 16–32

The stratified implementation retrains embeddings using only the training split of each fold to reduce information leakage during evaluation.

### Autoencoder and Temporal Convolutional Network Fusion

The temporal deep-learning architecture combines:

1. A one-dimensional convolutional autoencoder operating on six-dimensional binned temporal features.
2. A latent representation with eight channels in the documented configuration.
3. A dilated Temporal Convolutional Network operating on the latent sequence.
4. A small multilayer perceptron for the 11-dimensional static feature vector.
5. Concatenated representations for downstream classification.

The autoencoder acts as a feature extractor in the downstream fusion models.

### Cross-Attention Fusion

The cross-attention architecture integrates temporal and lexical information.

1. The AE + TCN pipeline produces temporal representations.
2. Word2Vec generates lexical sequence embeddings.
3. Both streams are projected into a shared representation space.
4. A multi-head attention block allows the temporal stream to attend to the lexical stream.
5. The fused representation is pooled and passed to the classifier.

This architecture is designed to investigate whether combining speech timing and lexical information improves classification performance.

### Evaluation Protocol

The documented experiments use stratified five-fold cross-validation.

Reported evaluation outputs include:

- Accuracy
- Macro-F1
- Per-class precision, recall, and F1
- Confusion matrices
- Classification reports
- Accuracy comparisons across model families

Where implemented, fold-specific preprocessing and embedding training are used to reduce data leakage.

---

## Results Summary

The following figures are the headline results reported for the current dataset release. They should be treated as **dataset- and experiment-specific research findings**, rather than guaranteed performance on new participants or independent clinical populations.

### Statistical Findings

- SFT mean pause duration is reported to be significantly higher in Dementia than in Healthy Controls.
- The reported significance levels are PFT: `p < 0.05` and SFT: `p < 0.001`.
- Within SFT, the reported comparisons between Dementia and MCI, and between MCI and Healthy Controls, are also statistically significant.

### Classification Performance

| Task | Reported accuracy | Reported macro-F1 |
|---|---:|---:|
| Dementia vs Healthy Controls, using SFT pause features | ~75% | ~0.72 |
| Healthy vs Non-Healthy, using SFT features | ~68% | ~0.68 |
| Three-class HC / MCI / Dementia, using SFT features | ~56–58% | Not specified |
| TF-IDF on SFT for Non-Healthy vs HC | ~64% | Not specified |

The reported three-class performance reflects substantial overlap between MCI and the other diagnostic groups.

### MCI Handling

The current analysis suggests that two formulations may be worth investigating:

- Excluding MCI when the research objective is specifically Dementia-versus-Healthy classification.
- Using a Non-Healthy-versus-Healthy formulation when the research objective includes participants with MCI.

The reported analysis indicates that assigning MCI participants directly to either the Dementia or Healthy class can reduce macro-F1.

### Deep-Learning Comparison

The reported cross-attention fusion model matches or slightly outperforms classical and single-stream baselines on SFT, but not on PFT.

These findings motivate further investigation of how temporal speech characteristics and lexical representations interact across verbal-fluency tasks.

### Interpretation and Limitations

The results above summarise the current experimental release. Performance may change as the dataset grows or the experimental configuration changes.

The reported metrics should be interpreted in the context of the dataset, cohort composition, preprocessing pipeline, and cross-validation protocol. Independent external validation is needed before drawing conclusions about generalisation to other populations or clinical use.

---

## Reproducing the Experiments

To reproduce the analyses:

1. Install the dependencies and the package in editable mode, as described in [Installation](#installation).
2. Populate `data/` with the required transcripts and `speechpause_mata.csv`.
3. Check that recording identifiers, transcript filenames, and metadata fields match.
4. Run `notebooks/transcript_analysis_baseline.ipynb` from beginning to end.
5. Confirm that the expected pause summaries and analysis outputs are generated.
6. Optionally run the binary classifier, multiclass classifier, MCI impact analysis, and deep-learning benchmark.
7. Review the generated results in the configured output directories.
8. Use `reset_experiment.sh` to clear generated artefacts between runs, after checking which files will be deleted.

For the deep-learning benchmark, the main experiment runner is:

```bash
python -m cognospeak.run_temporal_experiment_cross_attention \
    --task both \
    --classification-mode both \
    --k-folds 5 \
    --epochs 30 \
    --batch-size 8 \
    --output-dir results_graphs
```

For the statistical baseline, begin with:

```bash
jupyter notebook notebooks/transcript_analysis_baseline.ipynb
```

Reproducibility depends on using compatible dependency versions, the correct dataset and metadata, and the intended experimental configuration.

---

## Data Governance and Responsible Use

This repository supports research involving speech recordings, transcripts, diagnostic labels, and demographic information. Such data may contain personal or sensitive information.

Before using or distributing the dataset:

- Ensure that you are authorised to access and process the data.
- Follow applicable consent, ethics, data-protection, and institutional requirements.
- Review access controls and `.gitignore` rules before publishing recordings, transcripts, metadata, or derived outputs.
- Avoid committing identifiable or restricted data to a public repository.
- Check generated artefacts before sharing them, as reports and intermediate files may contain sensitive information.
- Interpret model outputs as research findings, not as a substitute for clinical assessment.

Researchers should also evaluate the generalisability, potential bias, and limitations of speech-based models before considering any real-world application.

---

## Citation and Contact

If you use this code or dataset in your research, please cite the associated CognoSpeak publication when its citation details become available.

**Publication:** To be added at release time.

For questions, data-access requests, or bug reports, please open an issue in the repository or contact the CognoSpeak project maintainers.

---

*CognoSpeak is a research toolkit for investigating speech-derived features of cognitive decline. Please review the repository's data-access restrictions and experiment-reset scripts before distributing the dataset or generated artefacts.*