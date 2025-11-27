#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    token: str

    @classmethod
    def from_line(cls, line: str) -> 'Utterance | None':
        parts = line.strip().split()
        if len(parts) < 3:
            return None
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError:
            return None
        token = ' '.join(parts[2:])
        return cls(start=start, end=end, token=token)


def load_transcript(path: Path) -> List[Utterance]:
    utterances: List[Utterance] = []
    with path.open('r', encoding='utf-8') as handle:
        for raw in handle:
            utterance = Utterance.from_line(raw)
            if utterance:
                utterances.append(utterance)
    return utterances


def compute_pauses(utterances: Sequence[Utterance]) -> List[float]:
    pauses: List[float] = []
    if not utterances:
        return pauses
    previous_end = utterances[0].end
    for utterance in utterances[1:]:
        pause = max(0.0, utterance.start - previous_end)
        pauses.append(pause)
        previous_end = utterance.end
    return pauses


def summarize(pauses: Sequence[float]) -> dict[str, float | int]:
    if not pauses:
        return {
            'pause_count': 0,
            'pause_mean': 0.0,
            'pause_median': 0.0,
            'pause_min': 0.0,
            'pause_max': 0.0,
            'pause_std': 0.0,
            'pause_total': 0.0,
        }
    pause_count = len(pauses)
    pause_total = float(sum(pauses))
    pause_mean = statistics.fmean(pauses)
    pause_median = statistics.median(pauses)
    pause_min = min(pauses)
    pause_max = max(pauses)
    pause_std = statistics.stdev(pauses) if pause_count > 1 else 0.0
    return {
        'pause_count': pause_count,
        'pause_mean': pause_mean,
        'pause_median': pause_median,
        'pause_min': pause_min,
        'pause_max': pause_max,
        'pause_std': pause_std,
        'pause_total': pause_total,
    }


def iter_transcripts(root: Path) -> Iterable[Path]:
    return sorted(root.rglob('*.txt'))


def write_summary(rows: List[dict], destination: Path) -> None:
    if not rows:
        print('No transcripts found, nothing to write.')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with destination.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_pauses(input_dir: Path, output_csv: Path) -> List[dict]:
    rows: List[dict] = []
    for transcript_path in iter_transcripts(input_dir):
        utterances = load_transcript(transcript_path)
        pauses = compute_pauses(utterances)
        summary = summarize(pauses)
        row = {
            'transcript': str(transcript_path.relative_to(input_dir)),
            'token_count': len(utterances),
            **summary,
        }
        rows.append(row)
    write_summary(rows, output_csv)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Analyze pauses between words for transcript text files.'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data'),
        help='Root directory containing transcript .txt files (default: data)',
    )
    parser.add_argument(
        '--output-csv',
        type=Path,
        default=Path('pause_summary.csv'),
        help='Path to write the pause summary CSV (default: pause_summary.csv)',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = analyze_pauses(args.input_dir, args.output_csv)
    if rows:
        print(f'Analyzed {len(rows)} transcript files. Summary saved to {args.output_csv}.')
    else:
        print('No transcripts analyzed.')


if __name__ == '__main__':
    main()


