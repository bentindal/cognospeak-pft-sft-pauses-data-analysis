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
