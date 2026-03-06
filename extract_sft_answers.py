"""Extract all unique SFT answers from transcript files to a CSV."""

from pathlib import Path

import pandas as pd

from dataloader import TranscriptLoader


def main() -> None:
    loader = TranscriptLoader()
    transcript_paths = loader.discover_transcript_files()

    sft_paths = [p for p in transcript_paths if loader.infer_task_from_filename(p.name) == 'SFT']
    print(f"Found {len(sft_paths)} SFT transcripts")

    frames = [loader.load_transcript(p) for p in sft_paths]
    combined = pd.concat(frames, ignore_index=True)

    unique_answers = (
        combined['token']
        .str.strip()
        .str.lower()
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    output_path = Path('unique_sft_answers.csv')
    unique_answers.to_csv(output_path, index=False, header=['answer'])
    print(f"Extracted {len(unique_answers)} unique SFT answers to {output_path}")


if __name__ == '__main__':
    main()
