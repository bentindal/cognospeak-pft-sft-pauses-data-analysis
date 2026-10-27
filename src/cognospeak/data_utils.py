from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import torch

from .dataloader import TranscriptLoader
from .feature_extraction import base_recording_from_key, dict_to_feature_tensors

DIAG_TO_INT: Dict[str, int] = {"HC": 0, "MCI": 1, "DEMENTIA": 2}


@dataclass
class TaskData:
    keys: list[str]
    x_seq: torch.Tensor      # (N, 6, 20)
    x_static: torch.Tensor   # (N, D)
    y: np.ndarray


class TemporalDataModule:
    def __init__(self, loader: TranscriptLoader | None = None, bin_sec: float = 3.0, max_time: float = 60.0):
        self.loader = loader or TranscriptLoader()
        self.bin_sec = bin_sec
        self.max_time = max_time
        meta = self.loader.load_metadata()
        self.diagnosis_map = dict(zip(meta["recording"].astype(str), meta["diagnosis"].astype(str)))

    def _discover_task_frames(self, task: str) -> dict[str, pd.DataFrame]:
        task = task.upper()
        transcript_paths = self.loader.discover_transcript_files()
        task_paths = [p for p in transcript_paths if self.loader.infer_task_from_filename(p.name) == task]
        return {p.stem: self.loader.load_transcript(p) for p in task_paths}

    def _labels_for_keys(self, keys: list[str]):
        y_str = []
        missing = []
        for k in keys:
            rec = base_recording_from_key(k)
            d = self.diagnosis_map.get(rec)
            if d is None:
                missing.append((k, rec))
                d = "UNKNOWN"
            y_str.append(d)

        keep_idx = np.array([d in DIAG_TO_INT for d in y_str], dtype=bool)
        y = np.array([DIAG_TO_INT[y_str[i]] for i in range(len(y_str)) if keep_idx[i]], dtype=int)
        return keep_idx, y, missing

    @staticmethod
    def _standardize_train_test(train: torch.Tensor, test: torch.Tensor, dims) -> tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-8
        mu = train.mean(dim=dims, keepdim=True)
        sd = train.std(dim=dims, keepdim=True)
        return (train - mu) / (sd + eps), (test - mu) / (sd + eps)

    def make_fold_split(
        self,
        x_seq: torch.Tensor,
        x_static: torch.Tensor,
        y: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ):
        x_seq_train = x_seq[train_idx]
        x_seq_test = x_seq[test_idx]
        x_static_train = x_static[train_idx]
        x_static_test = x_static[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        x_seq_train, x_seq_test = self._standardize_train_test(x_seq_train, x_seq_test, dims=(0, 2))
        x_static_train, x_static_test = self._standardize_train_test(x_static_train, x_static_test, dims=0)

        return x_seq_train, x_seq_test, x_static_train, x_static_test, y_train, y_test

    def load_task(self, task: str) -> TaskData:
        dfs = self._discover_task_frames(task)
        keys, x_seq_np, x_sum_np = dict_to_feature_tensors(dfs, self.bin_sec, self.max_time)
        keep_idx, y, missing = self._labels_for_keys(keys)

        if missing:
            print(f"WARNING: missing metadata for {len(missing)} {task} items (showing up to 10)")
            for item in missing[:10]:
                print("  key:", item[0], "-> recording:", item[1])

        keys = [k for i, k in enumerate(keys) if keep_idx[i]]
        x_seq_np = x_seq_np[keep_idx]
        x_sum_np = x_sum_np[keep_idx]

        x_seq = torch.from_numpy(x_seq_np).float().permute(0, 2, 1)  # (N, 6, 20)
        x_static = torch.from_numpy(x_sum_np).float()

        return TaskData(
            keys=keys,
            x_seq=x_seq,
            x_static=x_static,
            y=y,
        )