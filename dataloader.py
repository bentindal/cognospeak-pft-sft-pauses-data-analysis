from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / 'data'
METADATA_FILE = DATA_DIR / 'speechpause_mata.csv'
COLUMN_NAMES = ['start_time', 'end_time', 'token']


class TranscriptLoader:
    """Utility class for exploring transcript .txt files in notebooks."""

    def __init__(self, data_dir: Path = DATA_DIR, metadata_path: Optional[Path] = None) -> None:
        self.data_dir = data_dir
        if not self.data_dir.exists():
            raise FileNotFoundError(f'Data directory not found: {self.data_dir}')
        self.metadata_path = metadata_path or METADATA_FILE

    def discover_transcript_files(self) -> list[Path]:
        """Return every .txt transcript located under the data directory."""
        return sorted(path for path in self.data_dir.rglob('*.txt') if path.is_file())

    def load_transcript(self, path: Path) -> pd.DataFrame:
        """
        Load a single transcript file into a DataFrame.

        Each file contains tab-delimited rows with start time, end time,
        and the spoken token/phrase. Extra metadata columns describing the
        recording and task are added for downstream processing.
        """
        df = pd.read_csv(
            path,
            sep='\t',
            header=None,
            names=COLUMN_NAMES,
            engine='python',
            dtype={'start_time': 'Float64', 'end_time': 'Float64', 'token': 'string'},
        )
        recording_name = path.parent.name
        df['recording'] = recording_name
        df['task'] = self.infer_task_from_filename(path.name)
        df['recording_number'] = self.extract_recording_number(recording_name)
        df['source_path'] = str(path)
        return df

    @staticmethod
    def infer_task_from_filename(filename: str) -> str:
        """Extract PFT/SFT (or similar) suffixes from filenames.
        
        Handles both single underscore (Process-XXX_PFT.txt) and 
        double underscore (Process-XXX__PFT.txt) naming conventions.
        """
        stem = Path(filename).stem
        # Try double underscore first, then single underscore
        if '__' in stem:
            parts = stem.split('__')
        elif '_' in stem:
            parts = stem.split('_')
        else:
            return 'UNKNOWN'
        
        # Get the last part which should be the task type (PFT, SFT, etc.)
        if parts:
            task = parts[-1].strip()
            if task and task.isalpha():
                return task.upper()
        return 'UNKNOWN'

    def load_all_transcripts(self, paths: Iterable[Path] | None = None) -> Dict[str, pd.DataFrame]:
        """Load every transcript path into a dict keyed by filename stem."""
        paths = paths or self.discover_transcript_files()
        return {path.stem: self.load_transcript(path) for path in paths}

    def load_metadata(self, path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load the CSV metadata file containing cohort metadata.

        The CSV stores diagnosis labels (Dementia, MCI, HC) alongside
        demographic fields. Column names are normalized for easier merging.
        
        Args:
            path: Optional path to the metadata CSV file. If not provided,
                  uses the path specified in TranscriptLoader.__init__().
        
        Raises:
            FileNotFoundError: If the metadata file is not found, with
                              instructions on how to export it.
        """
        metadata_path = path or self.metadata_path
        if not metadata_path.exists():
            error_msg = f"""
================================================================================
ERROR: Metadata file not found: {metadata_path}
================================================================================

The CSV metadata file is required to run this analysis. Please:

1. Export your metadata CSV file from your data source
2. Place it in the data directory or provide the path when initializing TranscriptLoader

Example usage:
    from pathlib import Path
    from dataloader import TranscriptLoader
    
    # Option 1: Place CSV in data/ directory as 'speechpause_mata.csv'
    loader = TranscriptLoader()
    
    # Option 2: Provide custom path to CSV file
    loader = TranscriptLoader(metadata_path=Path('path/to/your/metadata.csv'))

The CSV file should contain the following columns:
    - anyon_IDs: Recording identifier (e.g., 'Process-rec-006')
    - age: Participant age
    - gender: Participant gender
    - diagnosis: Diagnosis label (Dementia, MCI, HC)
    - MMSE: MMSE score (optional)
    - Assigned: Annotator name (optional)
    - Notes: Additional notes (optional)
================================================================================
"""
            raise FileNotFoundError(error_msg)

        metadata = pd.read_csv(
            metadata_path,
            dtype={
                'anyon_IDs': 'string',
                'age': 'Float64',
                'gender': 'string',
                'diagnosis': 'string',
                'MMSE': 'Float64',
                'Assigned': 'string',
                'Notes': 'string',
            },
        )
        metadata = metadata.rename(columns={'anyon_IDs': 'recording', 'Assigned': 'annotator', 'Notes': 'notes'})
        metadata['recording_number'] = metadata['recording'].apply(self.extract_recording_number)
        metadata['diagnosis'] = metadata['diagnosis'].str.upper().str.strip()
        return metadata

    def merge_with_metadata(
        self,
        transcripts: pd.DataFrame,
        metadata: Optional[pd.DataFrame] = None,
        how: str = 'left',
    ) -> pd.DataFrame:
        """
        Join transcript rows with cohort metadata on recording name.

        By default, only diagnosis and demographic columns are appended to the
        transcript DataFrame to avoid overwriting transcript-specific fields.
        """
        metadata = metadata if metadata is not None else self.load_metadata()
        columns_to_keep = [
            'recording',
            'diagnosis',
            'age',
            'gender',
            'MMSE',
            'annotator',
            'notes',
            'recording_number',
        ]
        subset = metadata[columns_to_keep]
        subset = subset.rename(columns={'recording_number': 'metadata_recording_number'})
        merged = transcripts.merge(subset, on='recording', how=how)
        merged['metadata_recording_number'] = merged['metadata_recording_number'].astype('Int64')
        return merged

    @staticmethod
    def extract_recording_number(recording_name: str) -> int | None:
        """Return the numeric suffix (e.g., 6 for Process-rec-006)."""
        match = re.search(r'(\d+)$', recording_name)
        return int(match.group(1)) if match else None

    @staticmethod
    def concatenate_transcripts(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
        """Return a single DataFrame containing every transcript row."""
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def summarize_structure(df: pd.DataFrame) -> pd.Series:
        """
        Provide a quick structural summary that can be printed to stdout.

        Highlights column completeness and basic duration stats to verify that
        all transcripts follow the same three-column schema.
        """
        return pd.Series(
            {
                'rows': len(df),
                'recordings': df['recording'].nunique(),
                'tasks': ', '.join(sorted(df['task'].dropna().unique())),
                'start_time_range': (df['start_time'].min(), df['start_time'].max()),
                'end_time_range': (df['end_time'].min(), df['end_time'].max()),
                'missing_values': int(df.isna().sum().sum()),
            }
        )
