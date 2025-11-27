# CognoSpeak Transcript Analysis

A Python toolkit for analyzing speech transcript data with pause metrics and statistical comparisons across diagnostic groups.

## Setup

1. Clone this repository
2. Install required dependencies (see `requirements.txt` if available)
3. Set up your data directory structure (see below)
4. Export and place your metadata CSV file

## Data Directory Structure

The `data/` folder should be organized as follows:

```
data/
├── Process-rec-XXX/
│   ├── Process-rec-XXX__PFT.txt  (or Process-rec-XXX_PFT.txt)
│   ├── Process-rec-XXX__SFT.txt  (or Process-rec-XXX_SFT.txt)
│   ├── Process-rec-XXX__PFT.wav  (optional)
│   └── Process-rec-XXX__SFT.wav  (optional)
├── Process-test-XXX/
│   ├── Process-test-XXX__PFT.txt
│   ├── Process-test-XXX__SFT.txt
│   └── ...
└── speechpause_mata.csv  (metadata file - see below)
```

### File Naming Conventions

Transcript files can use either naming convention:
- **Double underscore**: `Process-rec-006__PFT.txt`, `Process-rec-006__SFT.txt`
- **Single underscore**: `Process-rec-006_PFT.txt`, `Process-rec-006_SFT.txt`

Each transcript file should be tab-delimited with three columns:
1. `start_time`: Start time of the utterance (float)
2. `end_time`: End time of the utterance (float)
3. `token`: The spoken word/phrase (string)

## Metadata CSV File

The metadata CSV file is **required** for analysis. It should contain participant information and diagnostic labels.

### Exporting the Metadata CSV

1. Export your metadata from your data source (database, spreadsheet, etc.)
2. Save it as a CSV file
3. Place it in the `data/` directory as `speechpause_mata.csv`, OR provide the path when initializing `TranscriptLoader`

### Required CSV Columns

The metadata CSV must include the following columns:

- **anyon_IDs**: Recording identifier that matches the folder names (e.g., `Process-rec-006`, `Process-test-020`)
- **age**: Participant age (numeric, optional)
- **gender**: Participant gender (string, optional)
- **diagnosis**: Diagnosis label - must be one of: `Dementia`, `MCI`, or `HC` (case-insensitive)
- **MMSE**: MMSE score (numeric, optional)
- **Assigned**: Annotator name (string, optional)
- **Notes**: Additional notes (string, optional)

### Example CSV Format

```csv
anyon_IDs,age,gender,diagnosis,MMSE,Assigned,Notes
Process-rec-006,75,female,HC,28,Annotator1,"Good quality recording"
Process-rec-009,68,male,MCI,24,Annotator2,"Some background noise"
Process-test-020,72,female,Dementia,20,Annotator1,"Clear audio"
```

### Using the Metadata File

You can provide the metadata file path in two ways:

**Option 1: Place in data directory**
```python
from dataloader import TranscriptLoader

# Automatically looks for data/speechpause_mata.csv
loader = TranscriptLoader()
metadata = loader.load_metadata()
```

**Option 2: Provide custom path**
```python
from pathlib import Path
from dataloader import TranscriptLoader

# Provide path to your metadata CSV
loader = TranscriptLoader(metadata_path=Path('path/to/your/metadata.csv'))
metadata = loader.load_metadata()
```

If the metadata file is missing, you'll receive a detailed error message with instructions.

## Usage

### Basic Analysis

```python
from dataloader import TranscriptLoader

# Initialize loader (ensure metadata CSV is in place)
loader = TranscriptLoader()

# Discover and load transcripts
transcript_paths = loader.discover_transcript_files()
frames = loader.load_all_transcripts(transcript_paths)
combined = loader.concatenate_transcripts(frames.values())

# Load and merge with metadata
metadata = loader.load_metadata()
merged = loader.merge_with_metadata(combined, metadata)
```

### Pause Analysis

See `transcript_analysis.ipynb` for complete examples of:
- Loading and combining transcripts
- Merging with metadata
- Computing pause metrics
- Statistical comparisons (3-class and binary)
- Visualization

## Privacy Notice

**IMPORTANT**: The `data/` directory contains highly private participant data and is excluded from version control via `.gitignore`. Never commit data files to the repository.

## Files

- `dataloader.py`: Core data loading and processing utilities
- `file_validator.py`: Validates file naming conventions
- `pause_analyzer.py`: Computes pause metrics from transcripts
- `plotting.py`: Visualization functions
- `statistical_tests.py`: Statistical comparison functions
- `transcript_analysis.ipynb`: Interactive analysis notebook

## Requirements

- Python 3.x
- pandas
- matplotlib
- seaborn
- pathlib (standard library)

