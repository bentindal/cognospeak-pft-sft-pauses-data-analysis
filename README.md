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

4. **Place the CSV in `data/` as `speechpause_mata.csv`**, or provide the path:
   ```python
   from pathlib import Path
   from dataloader import TranscriptLoader
   
   loader = TranscriptLoader(metadata_path=Path('path/to/your/metadata.csv'))
   ```

5. **Run the analysis:**
   ```python
   from dataloader import TranscriptLoader
   
   loader = TranscriptLoader()
   transcript_paths = loader.discover_transcript_files()
   frames = loader.load_all_transcripts(transcript_paths)
   metadata = loader.load_metadata()
   ```

See `transcript_analysis.ipynb` for complete examples.

**Note:** The `data/` directory is excluded from git. Never commit private data files.
