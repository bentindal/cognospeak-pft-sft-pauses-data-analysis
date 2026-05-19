from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import hdbscan

from dataloader import TranscriptLoader

keras = tf.keras
layers = tf.keras.layers

Task = Literal["PFT", "SFT"]


@dataclass
class VocabPack:
    vocab: dict[str, int]
    inv_vocab: dict[int, str]
    vocab_size: int
    seqs: dict[str, list[int]]  # fname -> list of token ids


class Word2Vec(keras.Model):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.target_emb = layers.Embedding(vocab_size, embedding_dim, name="target_embedding")
        self.context_emb = layers.Embedding(vocab_size, embedding_dim, name="context_embedding")

    def call(self, inputs):
        target, context = inputs  # target: (B,), context: (B, 1+K)
        v = self.target_emb(target)      # (B, D)
        u = self.context_emb(context)    # (B, 1+K, D)
        logits = tf.einsum("bd,bkd->bk", v, u)  # (B, 1+K)
        return logits


class SemanticEmbeddingPipeline:
    """
    Pipeline for:
      - loading transcripts via TranscriptLoader
      - building vocab + sequences (per task / per fold)
      - training Word2Vec skip-gram with negative sampling (TF)
      - exposing embeddings + similarity helpers
      - optional TF-IDF and plotting
      - saving/loading embeddings for use in other files
    """

    def __init__(self, data_root: str | Path = "data", seed: int = 42):
        self.data_root = Path(data_root)
        self.seed = seed
        self.loader = TranscriptLoader()

        self.dataframes: dict[Task, dict[str, pd.DataFrame]] = {"PFT": {}, "SFT": {}}
        self.vpacks: dict[Task, Optional[VocabPack]] = {"PFT": None, "SFT": None}
        self.models: dict[Task, Optional[Word2Vec]] = {"PFT": None, "SFT": None}
        self.vectors: dict[Task, Optional[np.ndarray]] = {"PFT": None, "SFT": None}
        self.tfidf: dict[Task, dict] = {"PFT": {}, "SFT": {}}

    # -----------------------------
    # Loading
    # -----------------------------
    def load_transcripts(self) -> None:
        """Discover and load transcripts into self.dataframes['PFT'/'SFT']."""
        transcript_paths = self.loader.discover_transcript_files()

        pft_paths = [p for p in transcript_paths if self.loader.infer_task_from_filename(p.name) == "PFT"]
        sft_paths = [p for p in transcript_paths if self.loader.infer_task_from_filename(p.name) == "SFT"]

        self.dataframes["PFT"] = {p.stem: self.loader.load_transcript(p) for p in pft_paths}
        self.dataframes["SFT"] = {p.stem: self.loader.load_transcript(p) for p in sft_paths}

    # -----------------------------
    # Vocab + sequences
    # -----------------------------
    @staticmethod
    def build_global_vocab_and_sequences(dfs: dict[str, pd.DataFrame], col: str = "token") -> VocabPack:
        all_words = pd.concat((df[col] for df in dfs.values()), ignore_index=True)
        all_words = all_words.dropna().astype("string").str.lower().str.strip()

        unique_tokens = all_words.unique()
        vocab = {"<pad>": 0, "<unk>": 1}
        vocab.update({tok: i for i, tok in enumerate(unique_tokens, start=2)})

        inv_vocab = {i: t for t, i in vocab.items()}
        vocab_size = len(vocab)
        unk_id = vocab["<unk>"]

        seqs = {
            fname: df[col].dropna().astype("string").str.lower().str.strip()
                    .map(lambda w: vocab.get(w, unk_id)).to_list()
            for fname, df in dfs.items()
        }
        return VocabPack(vocab=vocab, inv_vocab=inv_vocab, vocab_size=vocab_size, seqs=seqs)

    def prepare(self, task: Task) -> None:
        """Build vocab + sequences for the full task dataset."""
        if not self.dataframes[task]:
            raise ValueError(f"No transcripts loaded for task={task}. Run load_transcripts() first.")
        self.vpacks[task] = self.build_global_vocab_and_sequences(self.dataframes[task])

    def prepare_from_keys(self, task: Task, keys: list[str]) -> None:
        """Build vocab + sequences using only the provided keys (for fold-safe training)."""
        if not self.dataframes[task]:
            raise ValueError(f"No transcripts loaded for task={task}. Run load_transcripts() first.")

        dfs_subset = {k: self.dataframes[task][k] for k in keys if k in self.dataframes[task]}
        if not dfs_subset:
            raise ValueError(f"No matching transcripts found for task={task} and provided keys.")

        self.vpacks[task] = self.build_global_vocab_and_sequences(dfs_subset)

    # -----------------------------
    # Skip-gram training data
    # -----------------------------
    @staticmethod
    def make_skipgram_training_data(
        seqs: dict[str, list[int]],
        vocab_size: int,
        window_size: int = 2,
        num_negative: int = 5,
        seed: int = 42,
        max_pairs: int | None = None,
    ):
        rng = np.random.default_rng(seed)
        targets_list, contexts_list = [], []

        for _, seq in seqs.items():
            n = len(seq)
            if n < 2:
                continue
            for i, target in enumerate(seq):
                left = max(0, i - window_size)
                right = min(n, i + window_size + 1)
                for j in range(left, right):
                    if j == i:
                        continue
                    targets_list.append(target)
                    contexts_list.append(seq[j])

                    if max_pairs is not None and len(targets_list) >= max_pairs:
                        break
                if max_pairs is not None and len(targets_list) >= max_pairs:
                    break
            if max_pairs is not None and len(targets_list) >= max_pairs:
                break

        if len(targets_list) == 0:
            raise ValueError("No skip-gram pairs were generated. Check sequences/window_size.")

        targets = np.asarray(targets_list, dtype=np.int32)
        true_context = np.asarray(contexts_list, dtype=np.int32).reshape(-1, 1)

        n_pairs = len(targets)
        negative_context = rng.integers(2, vocab_size, size=(n_pairs, num_negative), dtype=np.int32)

        mask = negative_context == true_context
        while mask.any():
            negative_context[mask] = rng.integers(2, vocab_size, size=mask.sum(), dtype=np.int32)
            mask = negative_context == true_context

        context_all = np.concatenate([true_context, negative_context], axis=1)
        labels = np.zeros((n_pairs, 1 + num_negative), dtype=np.float32)
        labels[:, 0] = 1.0
        return targets, context_all, labels

    # -----------------------------
    # Train Word2Vec
    # -----------------------------
    def train_word2vec(
        self,
        task: Task,
        window_size: int = 2,
        num_negative: int = 3,
        embedding_dim: int = 32,
        batch_size: int = 256,
        epochs: int = 10,
        max_pairs: int | None = 300_000,
        lr: float = 1e-3,
    ) -> None:
        if self.vpacks[task] is None:
            self.prepare(task)

        vp = self.vpacks[task]
        assert vp is not None

        tf.keras.utils.set_random_seed(self.seed)

        targets, context_all, labels = self.make_skipgram_training_data(
            seqs=vp.seqs,
            vocab_size=vp.vocab_size,
            window_size=window_size,
            num_negative=num_negative,
            seed=self.seed,
            max_pairs=max_pairs,
        )

        ds = tf.data.Dataset.from_tensor_slices(((targets, context_all), labels))
        ds = ds.shuffle(min(200_000, len(targets)), seed=self.seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        model = Word2Vec(vocab_size=vp.vocab_size, embedding_dim=embedding_dim)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
        )
        model.fit(ds, epochs=epochs, verbose=1)

        self.models[task] = model
        self.vectors[task] = model.get_layer("target_embedding").get_weights()[0]

    # -----------------------------
    # Embedding accessors
    # -----------------------------
    def get_vectors(self, task: Task) -> np.ndarray:
        v = self.vectors.get(task)
        if v is None:
            raise ValueError(f"No vectors cached for {task}. Train or load first.")
        return v

    def get_vocab(self, task: Task) -> dict[str, int]:
        vp = self.vpacks.get(task)
        if vp is None:
            raise ValueError(f"No vocab pack for {task}. Run prepare() or load().")
        return vp.vocab

    def get_inv_vocab(self, task: Task) -> dict[int, str]:
        vp = self.vpacks.get(task)
        if vp is None:
            raise ValueError(f"No vocab pack for {task}. Run prepare() or load().")
        return vp.inv_vocab

    def most_similar(self, task: Task, word: str, top_k: int = 10):
        vp = self.vpacks.get(task)
        vecs = self.vectors.get(task)
        if vp is None or vecs is None:
            raise ValueError(f"{task} not ready. Train or load embeddings first.")

        wid = vp.vocab.get(word, vp.vocab.get("<unk>", 1))
        v = vecs[wid]
        v = v / (np.linalg.norm(v) + 1e-9)

        W = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        sims = W @ v

        best = np.argsort(-sims)[:top_k + 1]
        return [(vp.inv_vocab[i], float(sims[i])) for i in best if i != wid][:top_k]

    def get_sequence_embeddings(self, task: Task, keys: list[str], max_len: int | None = None):
        vp = self.vpacks.get(task)
        vecs = self.vectors.get(task)
        if vp is None or vecs is None:
            raise ValueError(f"{task} not ready. Train or load embeddings first.")

        seq_embs: list[np.ndarray] = []
        lengths: list[int] = []

        for k in keys:
            if k not in vp.seqs:
                raise KeyError(f"{k} not found in sequences for task={task}")

        ids = vp.seqs[k]

        # force at least one token
        if len(ids) == 0:
            ids = [vp.vocab.get("<unk>", 1)]

        emb = vecs[np.asarray(ids, dtype=np.int32)]

        if max_len is not None:
            emb = emb[:max_len]
            L = min(len(ids), max_len)
        else:
            L = len(ids)

        if max_len is None:
            max_len = max((x.shape[0] for x in seq_embs), default=0)

        padded = []
        emb_dim = vecs.shape[1]
        for x in seq_embs:
            if x.shape[0] < max_len:
                pad = np.zeros((max_len - x.shape[0], emb_dim), dtype=np.float32)
                x = np.vstack([x, pad])
            else:
                x = x[:max_len]
            padded.append(x)

        return np.stack(padded), np.asarray(lengths, dtype=np.int64)

    def get_document_embeddings(self, task: Task, keys: list[str], max_len: int | None = None) -> np.ndarray:
        X, lengths = self.get_sequence_embeddings(task, keys, max_len=max_len)
        out = []
        for i, length in enumerate(lengths):
            if length == 0:
                out.append(np.zeros(X.shape[-1], dtype=np.float32))
            else:
                out.append(X[i, :length].mean(axis=0))
        return np.stack(out)

    def get_document_embeddings_from_dataframes(
        self,
        task: Task,
        dfs: dict[str, pd.DataFrame],
        col: str = "token",
    ) -> np.ndarray:
        vp = self.vpacks.get(task)
        vecs = self.vectors.get(task)
        if vp is None or vecs is None:
            raise ValueError(f"{task} not ready. Train or load embeddings first.")

        unk_id = vp.vocab.get("<unk>", 1)
        out = []

        for _, df in dfs.items():
            toks = (
                df[col]
                .dropna()
                .astype("string")
                .str.lower()
                .str.strip()
                .tolist()
            )

            if len(toks) == 0:
                out.append(np.zeros(vecs.shape[1], dtype=np.float32))
                continue

            ids = np.array([vp.vocab.get(tok, unk_id) for tok in toks], dtype=np.int32)
            emb = vecs[ids].mean(axis=0)
            out.append(emb.astype(np.float32))

        return np.stack(out)

    # -----------------------------
    # Fold-safe CV runner
    # -----------------------------
    def run_word2vec_stratified_cv(
        self,
        task: Task,
        labels_by_key: dict[str, int],
        n_splits: int = 5,
        embedding_dim: int = 32,
        window_size: int = 2,
        num_negative: int = 3,
        epochs: int = 10,
        batch_size: int = 256,
        max_pairs: int | None = 300_000,
        lr: float = 1e-3,
        random_state: int = 42,
    ):
        if not self.dataframes[task]:
            raise ValueError(f"No transcripts loaded for task={task}. Run load_transcripts() first.")

        keys = [k for k in self.dataframes[task].keys() if k in labels_by_key]
        if not keys:
            raise ValueError(f"No labelled keys found for task={task}.")

        keys = np.asarray(keys)
        y = np.asarray([labels_by_key[k] for k in keys], dtype=np.int64)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        fold_accuracies: list[float] = []
        all_true: list[int] = []
        all_pred: list[int] = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
            print(f"\n{'=' * 80}")
            print(f"{task} Word2Vec Stratified Fold {fold}/{n_splits}")
            print(f"{'=' * 80}")

            train_keys = keys[train_idx].tolist()
            test_keys = keys[test_idx].tolist()
            y_train = y[train_idx]
            y_test = y[test_idx]

            self.vpacks[task] = None
            self.models[task] = None
            self.vectors[task] = None

            self.prepare_from_keys(task, train_keys)
            self.train_word2vec(
                task=task,
                window_size=window_size,
                num_negative=num_negative,
                embedding_dim=embedding_dim,
                batch_size=batch_size,
                epochs=epochs,
                max_pairs=max_pairs,
                lr=lr,
            )

            train_dfs = {k: self.dataframes[task][k] for k in train_keys}
            test_dfs = {k: self.dataframes[task][k] for k in test_keys}

            X_train = self.get_document_embeddings_from_dataframes(task, train_dfs)
            X_test = self.get_document_embeddings_from_dataframes(task, test_dfs)

            clf = LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=random_state,
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            fold_accuracies.append(float(acc))
            all_true.extend(y_test.tolist())
            all_pred.extend(y_pred.tolist())

            print(f"{task} fold accuracy: {acc:.4f}")

        all_true_arr = np.asarray(all_true)
        all_pred_arr = np.asarray(all_pred)

        print(f"\n{task} mean CV accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_true_arr, all_pred_arr))
        print("\nClassification Report:")
        print(classification_report(all_true_arr, all_pred_arr, zero_division=0))

        return {
            "task": task,
            "fold_accuracies": fold_accuracies,
            "mean_accuracy": float(np.mean(fold_accuracies)),
            "std_accuracy": float(np.std(fold_accuracies)),
            "y_true": all_true_arr,
            "y_pred": all_pred_arr,
        }

    # -----------------------------
    # Save / Load embeddings for reuse
    # -----------------------------
    def save(self, out_path: str | Path, task: Task) -> None:
        vp = self.vpacks.get(task)
        vecs = self.vectors.get(task)
        if vp is None or vecs is None:
            raise ValueError(f"{task} not ready. Train or load before saving.")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        words = np.array(list(vp.vocab.keys()), dtype=object)
        ids = np.array([vp.vocab[w] for w in words], dtype=np.int32)

        np.savez(
            out_path,
            vectors=vecs.astype(np.float32),
            words=words,
            ids=ids,
            task=np.array([task], dtype=object),
        )

    def load(self, npz_path: str | Path) -> Task:
        npz_path = Path(npz_path)
        data = np.load(npz_path, allow_pickle=True)

        vecs = data["vectors"]
        words = data["words"]
        ids = data["ids"]
        task = str(data["task"][0])

        vocab = {str(w): int(i) for w, i in zip(words, ids)}
        inv_vocab = {i: w for w, i in vocab.items()}
        vp = VocabPack(vocab=vocab, inv_vocab=inv_vocab, vocab_size=len(vocab), seqs={})

        self.vpacks[task] = vp
        self.vectors[task] = vecs
        self.models[task] = None
        return task  # type: ignore

    # -----------------------------
    # TF-IDF
    # -----------------------------
    @staticmethod
    def _read_transcript_tokens(path: Path) -> list[str]:
        tokens = []
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                parts = raw.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    float(parts[0])
                    float(parts[1])
                except ValueError:
                    continue
                tok = " ".join(parts[2:]).lower().strip()
                if tok:
                    tokens.append(tok)
        return tokens

    def build_tfidf_global(
        self,
        task: Task,
        pattern: str,
        min_df: int = 1,
        max_df: float = 1.0,
        ngram_range=(1, 2),
        max_features: int | None = None,
    ):
        data_root = self.data_root
        if (data_root / "data").exists():
            data_root = data_root / "data"

        paths = sorted(data_root.rglob(pattern))
        if not paths:
            raise FileNotFoundError(f"No files found under {data_root} with pattern {pattern}")

        docs, meta = [], []
        for p in paths:
            toks = self._read_transcript_tokens(p)
            docs.append(" ".join(toks))
            recording = p.parent.name
            m = re.search(r"__([A-Za-z]+)\.txt$", p.name)
            task_infer = m.group(1).upper() if m else "UNKNOWN"
            meta.append({"path": str(p), "recording": recording, "task": task_infer})

        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
        )
        X = vectorizer.fit_transform(docs)
        terms = vectorizer.get_feature_names_out()
        meta_df = pd.DataFrame(meta)

        self.tfidf[task] = {"X": X, "terms": terms, "vectorizer": vectorizer, "meta": meta_df}
        return X, terms, vectorizer, meta_df
