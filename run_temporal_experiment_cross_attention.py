from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import pandas as pd

from ae import Autoencoder, ConvolutionalAutoencoder, Decoder, Encoder
from cross_attention_fusion import MultiHeadAttention
from data_utils import DIAG_TO_INT, TemporalDataModule
from dataloader import TranscriptLoader
from tcn_model import TCN
from word2vecv2_stratified import SemanticEmbeddingPipeline


class StaticEncoder(nn.Module):
    def __init__(self, in_dim: int = 11, hidden: int = 32, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AETCNFusionClassifier(nn.Module):
    def __init__(
        self,
        ae: Autoencoder,
        tcn: TCN,
        static_dim: int = 11,
        static_latent: int = 16,
        n_classes: int = 3,
        freeze_ae: bool = True,
    ):
        super().__init__()
        self.ae = ae
        self.tcn = tcn
        self.static_enc = StaticEncoder(in_dim=static_dim, out_dim=static_latent)

        if freeze_ae:
            for p in self.ae.parameters():
                p.requires_grad = False

        fused_dim = self.tcn.embed_dim + static_latent
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        _, z = self.ae(x_seq)
        tokens = self.tcn(z)
        pooled = tokens.mean(dim=1)
        s = self.static_enc(x_static)
        fused = torch.cat([pooled, s], dim=1)
        return self.head(fused)


class CrossAttentionFusionClassifier(nn.Module):
    def __init__(
        self,
        ae: Autoencoder,
        tcn: TCN,
        w2v_dim: int,
        static_dim: int = 11,
        static_latent: int = 16,
        d_model: int = 128,
        num_heads: int = 8,
        n_classes: int = 3,
        freeze_ae: bool = True,
    ):
        super().__init__()
        self.ae = ae
        self.tcn = tcn
        self.static_enc = StaticEncoder(in_dim=static_dim, out_dim=static_latent)

        if freeze_ae:
            for p in self.ae.parameters():
                p.requires_grad = False

        self.lex_proj = nn.Linear(w2v_dim, d_model)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm = nn.LayerNorm(d_model)

        fused_dim = d_model + static_latent
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_static: torch.Tensor,
        x_lex: torch.Tensor,
        lex_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, z = self.ae(x_seq)               # [B, 8, T_latent]
        h_time = self.tcn(z)                # [B, T, d_model]
        h_lex = self.lex_proj(x_lex)        # [B, S, d_model]

        h_time_attn, weights = self.cross_attn(h_time, h_lex, h_lex, mask=lex_mask)
        h_time = self.norm(h_time + h_time_attn)

        pooled = h_time.mean(dim=1)
        s = self.static_enc(x_static)
        fused = torch.cat([pooled, s], dim=1)
        logits = self.head(fused)
        return logits, weights


@dataclass
class FusionResult:
    accuracy: float
    y_true: np.ndarray
    y_pred: np.ndarray


CLASS_NAMES = ["HC", "MCI", "DEMENTIA"]


def make_lr_features(x_seq: torch.Tensor, x_static: torch.Tensor) -> np.ndarray:
    x_seq_flat = x_seq.reshape(x_seq.shape[0], -1).cpu().numpy()
    x_static_np = x_static.cpu().numpy()
    return np.concatenate([x_seq_flat, x_static_np], axis=1)


def run_logistic_regression_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, float]:
    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    return y_pred, acc


def build_weighted_ce_from_labels(y_train: np.ndarray, device: torch.device) -> nn.Module:
    class_counts = np.bincount(y_train, minlength=len(DIAG_TO_INT))
    class_weights = class_counts.sum() / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.mean()
    return nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))


def make_sequence_embeddings_from_dataframes(
    pipe: SemanticEmbeddingPipeline,
    task: str,
    dfs: dict[str, object],
    max_len: int | None = None,
    col: str = "token",
) -> tuple[np.ndarray, np.ndarray]:
    vp = pipe.vpacks[task]
    vecs = pipe.vectors[task]
    if vp is None or vecs is None:
        raise ValueError(f"{task} not ready. Train or load embeddings first.")

    unk_id = vp.vocab.get("<unk>", 1)
    seq_embs: list[np.ndarray] = []
    lengths: list[int] = []
    emb_dim = vecs.shape[1]

    for _, df in dfs.items():
        toks = (
            df[col]
            .dropna()
            .astype("string")
            .str.lower()
            .str.strip()
            .tolist()
        )
        ids = np.array([vp.vocab.get(tok, unk_id) for tok in toks], dtype=np.int32)
        emb = vecs[ids] if len(ids) > 0 else np.zeros((0, emb_dim), dtype=np.float32)
        seq_embs.append(np.asarray(emb, dtype=np.float32))
        lengths.append(int(len(ids)))

    if max_len is None:
        max_len = max((len_ for len_ in lengths), default=0)

    padded = []
    clipped_lengths = []
    for emb, length in zip(seq_embs, lengths):
        eff_len = min(length, max_len)
        emb = emb[:max_len]
        if emb.shape[0] < max_len:
            pad = np.zeros((max_len - emb.shape[0], emb_dim), dtype=np.float32)
            emb = np.vstack([emb, pad])
        padded.append(emb)
        clipped_lengths.append(eff_len)

    return np.stack(padded), np.asarray(clipped_lengths, dtype=np.int64)


def lengths_to_attention_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    valid = idx < lengths.unsqueeze(1)
    return valid.unsqueeze(1).unsqueeze(2)


def train_simple_fusion_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
) -> FusionResult:
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    y_train = []
    for _, _, yb in train_loader:
        y_train.extend(yb.numpy())
    criterion = build_weighted_ce_from_labels(np.asarray(y_train), device)

    best_acc = -1.0
    best_y_true = np.array([])
    best_y_pred = np.array([])

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x_seq, x_static, yb in train_loader:
            x_seq = x_seq.to(device)
            x_static = x_static.to(device)
            yb = yb.to(device)

            logits = model(x_seq, x_static)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        mean_train = float(np.mean(train_losses)) if train_losses else 0.0

        model.eval()
        val_losses = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x_seq, x_static, yb in val_loader:
                x_seq = x_seq.to(device)
                x_static = x_static.to(device)
                yb = yb.to(device)

                logits = model(x_seq, x_static)
                val_losses.append(float(criterion(logits, yb).item()))
                preds = torch.argmax(logits, dim=1)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        y_true_np = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)
        acc = float((y_true_np == y_pred_np).mean()) if len(y_true_np) else 0.0
        mean_val = float(np.mean(val_losses)) if val_losses else 0.0
        print(f"CLF Epoch {epoch + 1}/{epochs} | train={mean_train:.4f} | val={mean_val:.4f} | acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_y_true = y_true_np.copy()
            best_y_pred = y_pred_np.copy()

    return FusionResult(best_acc, best_y_true, best_y_pred)


def train_cross_attention_classifier(
    model: CrossAttentionFusionClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
) -> FusionResult:
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    y_train = []
    for _, _, _, _, yb in train_loader:
        y_train.extend(yb.numpy())
    criterion = build_weighted_ce_from_labels(np.asarray(y_train), device)

    best_acc = -1.0
    best_y_true = np.array([])
    best_y_pred = np.array([])

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x_seq, x_static, x_lex, lex_lengths, yb in train_loader:
            x_seq = x_seq.to(device)
            x_static = x_static.to(device)
            x_lex = x_lex.to(device)
            lex_lengths = lex_lengths.to(device)
            yb = yb.to(device)

            lex_mask = lengths_to_attention_mask(lex_lengths, x_lex.shape[1])
            logits, _ = model(x_seq, x_static, x_lex, lex_mask=lex_mask)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        mean_train = float(np.mean(train_losses)) if train_losses else 0.0

        model.eval()
        val_losses = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x_seq, x_static, x_lex, lex_lengths, yb in val_loader:
                x_seq = x_seq.to(device)
                x_static = x_static.to(device)
                x_lex = x_lex.to(device)
                lex_lengths = lex_lengths.to(device)
                yb = yb.to(device)

                lex_mask = lengths_to_attention_mask(lex_lengths, x_lex.shape[1])
                logits, _ = model(x_seq, x_static, x_lex, lex_mask=lex_mask)
                val_losses.append(float(criterion(logits, yb).item()))
                preds = torch.argmax(logits, dim=1)
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        y_true_np = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)
        acc = float((y_true_np == y_pred_np).mean()) if len(y_true_np) else 0.0
        mean_val = float(np.mean(val_losses)) if val_losses else 0.0
        print(f"XATTN Epoch {epoch + 1}/{epochs} | train={mean_train:.4f} | val={mean_val:.4f} | acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_y_true = y_true_np.copy()
            best_y_pred = y_pred_np.copy()

    return FusionResult(best_acc, best_y_true, best_y_pred)


def print_result_block(title: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"\n{title}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nTrue class counts:", np.bincount(y_true, minlength=len(DIAG_TO_INT)))
    print("Pred class counts:", np.bincount(y_pred, minlength=len(DIAG_TO_INT)))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))


def plot_fold_accuracies(
    task: str,
    lr_fold_accs: list[float],
    w2v_fold_accs: list[float],
    fusion_fold_accs: list[float],
    xattn_fold_accs: list[float],
    output_dir: Path,
) -> None:
    folds = np.arange(1, len(lr_fold_accs) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(folds, lr_fold_accs, marker="o", label="Temporal LR")
    plt.plot(folds, w2v_fold_accs, marker="o", label="Word2Vec LR")
    plt.plot(folds, fusion_fold_accs, marker="o", label="AE + TCN Fusion")
    plt.plot(folds, xattn_fold_accs, marker="o", label="AE + TCN + Word2Vec XAttn")
    plt.xticks(folds)
    plt.ylim(0, 1.05)
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title(f"{task} Fold Accuracies")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / f"{task.lower()}_fold_accuracies_cross_attention.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_mean_accuracy_summary(results_dict: dict, output_dir: Path) -> None:
    tasks = list(results_dict.keys())
    lr_means = [results_dict[t]["lr_mean"] for t in tasks]
    w2v_means = [results_dict[t]["w2v_mean"] for t in tasks]
    fusion_means = [results_dict[t]["fusion_mean"] for t in tasks]
    xattn_means = [results_dict[t]["xattn_mean"] for t in tasks]

    x = np.arange(len(tasks))
    width = 0.2
    plt.figure(figsize=(10, 5))
    plt.bar(x - 1.5 * width, lr_means, width, label="Temporal LR")
    plt.bar(x - 0.5 * width, w2v_means, width, label="Word2Vec LR")
    plt.bar(x + 0.5 * width, fusion_means, width, label="AE + TCN Fusion")
    plt.bar(x + 1.5 * width, xattn_means, width, label="AE + TCN + Word2Vec XAttn")
    plt.xticks(x, tasks)
    plt.ylim(0, 1.05)
    plt.ylabel("Mean CV Accuracy")
    plt.title("Mean Cross-Validation Accuracy by Task")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / "mean_accuracy_summary_cross_attention.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


def build_tcn(output_size: int) -> TCN:
    return TCN(
        N_C=1,
        input_size=8,
        output_size=output_size,
        num_channels=[16, 32],
        K_S=[[3], [3]],
        Dil=[[1], [2]],
        Str=[[1], [1]],
        dropout=0.2,
        F_Ns=torch.tensor([32, 16, output_size]),
        dropout_F=0.2,
        Receptive_F=0,
        Norm=0,
        Residual=True,
    )

def plot_full_evaluation(
    task: str,
    y_true_dict: dict,
    y_pred_dict: dict,
    output_dir: Path,
):
    """
    Plots:
    - Accuracy comparison
    - Confusion matrices
    - Classification report heatmaps
    """

    models = list(y_true_dict.keys())

    # Accuracy bar chart

    accs = []
    for m in models:
        acc = accuracy_score(y_true_dict[m], y_pred_dict[m])
        accs.append(acc)

    plt.figure(figsize=(8, 5))
    plt.bar(models, accs)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title(f"{task} Model Accuracy Comparison")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / f"{task.lower()}_accuracy_summary.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Confusion matrices

    for m in models:
        cm = confusion_matrix(y_true_dict[m], y_pred_dict[m])

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["HC", "MCI", "DEM"],
                    yticklabels=["HC", "MCI", "DEM"])
        plt.title(f"{task} - {m} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        out_path = output_dir / f"{task.lower()}_{m}_confusion_matrix.png"
        plt.savefig(out_path, dpi=300)
        plt.show()


    # Classification report heatmaps

    for m in models:
        report = classification_report(
            y_true_dict[m],
            y_pred_dict[m],
            output_dict=True,
            zero_division=0
        )

        df = pd.DataFrame(report).T

        # rename numeric class rows to readable names
        rename_map = {
            "0": "HC",
            "1": "MCI",
            "2": "DEMENTIA",
            0: "HC",
            1: "MCI",
            2: "DEMENTIA",
        }
        df = df.rename(index=rename_map)

        wanted_rows = ["HC", "MCI", "DEMENTIA", "macro avg", "weighted avg"]
        available_rows = [r for r in wanted_rows if r in df.index]
        df = df.loc[available_rows]

        plt.figure(figsize=(8, 4))
        sns.heatmap(
            df[["precision", "recall", "f1-score"]],
            annot=True,
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

        plt.title(f"{task} - {m} Classification Report")
        plt.tight_layout()

        out_path = output_dir / f"{task.lower()}_{m}_classification_report.png"
        plt.savefig(out_path, dpi=300)
        plt.show()

def run_task_cv(
    task: str,
    data_module: TemporalDataModule,
    device: torch.device,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 8,
    n_splits: int = 5,
    w2v_embedding_dim: int = 32,
    w2v_epochs: int = 10,
    w2v_max_seq_len: int = 64,
    num_heads: int = 8,
) -> dict:
    data = data_module.load_task(task)

    print(f"\n{'=' * 80}")
    print(f"{task} STRATIFIED {n_splits}-FOLD CROSS-VALIDATION")
    print(f"{'=' * 80}")
    print(f"{task} label counts:", {name: int((data.y == code).sum()) for name, code in DIAG_TO_INT.items()})
    print(f"All data shapes: seq={data.x_seq.shape} | static={data.x_static.shape}")

    w2v_pipe = SemanticEmbeddingPipeline(data_root="data")
    w2v_pipe.load_transcripts()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_lr_true: list[int] = []
    all_lr_pred: list[int] = []
    lr_fold_accs: list[float] = []

    all_w2v_true: list[int] = []
    all_w2v_pred: list[int] = []
    w2v_fold_accs: list[float] = []

    all_fusion_true: list[int] = []
    all_fusion_pred: list[int] = []
    fusion_fold_accs: list[float] = []

    all_xattn_true: list[int] = []
    all_xattn_pred: list[int] = []
    xattn_fold_accs: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(data.y)), data.y), start=1):
        print(f"\n{'-' * 80}")
        print(f"{task} Fold {fold}/{n_splits}")
        print(f"{'-' * 80}")

        x_seq_train, x_seq_test, x_static_train, x_static_test, y_train, y_test = data_module.make_fold_split(
            data.x_seq, data.x_static, data.y, train_idx, test_idx
        )

        train_keys = [data.keys[i] for i in train_idx]
        test_keys = [data.keys[i] for i in test_idx]

        print(
            f"Train/test: {x_seq_train.shape}, {x_seq_test.shape} | "
            f"static: {x_static_train.shape}, {x_static_test.shape}"
        )

        # Temporal LR baseline
        xlr_train = make_lr_features(x_seq_train, x_static_train)
        xlr_test = make_lr_features(x_seq_test, x_static_test)
        y_lr_pred, lr_acc = run_logistic_regression_fold(xlr_train, y_train, xlr_test, y_test)
        lr_fold_accs.append(lr_acc)
        all_lr_true.extend(y_test.tolist())
        all_lr_pred.extend(y_lr_pred.tolist())
        print(f"{task} temporal LR fold accuracy: {lr_acc:.4f}")

        # Fold-safe Word2Vec training
        w2v_pipe.prepare_from_keys(task, train_keys)
        w2v_pipe.train_word2vec(
            task=task,
            window_size=2,
            num_negative=3,
            embedding_dim=w2v_embedding_dim,
            batch_size=256,
            epochs=w2v_epochs,
            max_pairs=300_000,
            lr=1e-3,
        )

        train_dfs = {k: w2v_pipe.dataframes[task][k] for k in train_keys}
        test_dfs = {k: w2v_pipe.dataframes[task][k] for k in test_keys}

        xw2v_train = w2v_pipe.get_document_embeddings_from_dataframes(task, train_dfs)
        xw2v_test = w2v_pipe.get_document_embeddings_from_dataframes(task, test_dfs)
        y_w2v_pred, w2v_acc = run_logistic_regression_fold(xw2v_train, y_train, xw2v_test, y_test)
        w2v_fold_accs.append(w2v_acc)
        all_w2v_true.extend(y_test.tolist())
        all_w2v_pred.extend(y_w2v_pred.tolist())
        print(f"{task} Word2Vec LR fold accuracy: {w2v_acc:.4f}")

        # AE training
        encoder = Encoder(in_channels=x_seq_train.shape[1], hidden=16, latent_ch=8)
        decoder = Decoder(out_channels=x_seq_train.shape[1], hidden=16, latent_ch=8)
        ae_net = Autoencoder(encoder, decoder)
        ae_trainer = ConvolutionalAutoencoder(ae_net, device=device, lr=1e-3)

        ae_train_ds = TensorDataset(x_seq_train)
        ae_val_ds = TensorDataset(x_seq_test)
        ae_trainer.train_model(
            nn.MSELoss(),
            epochs=epochs,
            batch_size=batch_size,
            training_set=ae_train_ds,
            validation_set=ae_val_ds,
        )

        # AE + TCN fusion baseline
        fusion_model = AETCNFusionClassifier(
            ae=ae_trainer.network,
            tcn=build_tcn(len(DIAG_TO_INT)).to(device),
            static_dim=x_static_train.shape[1],
            static_latent=16,
            n_classes=len(DIAG_TO_INT),
            freeze_ae=True,
        ).to(device)

        train_loader_simple = DataLoader(
            TensorDataset(x_seq_train, x_static_train, torch.from_numpy(y_train).long()),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader_simple = DataLoader(
            TensorDataset(x_seq_test, x_static_test, torch.from_numpy(y_test).long()),
            batch_size=batch_size,
            shuffle=False,
        )

        fusion_result = train_simple_fusion_classifier(
            fusion_model,
            train_loader_simple,
            test_loader_simple,
            device=device,
            epochs=epochs,
            lr=1e-3,
        )
        fusion_fold_accs.append(fusion_result.accuracy)
        all_fusion_true.extend(fusion_result.y_true.tolist())
        all_fusion_pred.extend(fusion_result.y_pred.tolist())
        print(f"{task} AE + TCN fusion fold accuracy: {fusion_result.accuracy:.4f}")

        # Cross-attention lexical sequences
        xlex_train_np, lex_len_train_np = make_sequence_embeddings_from_dataframes(
            w2v_pipe, task, train_dfs, max_len=w2v_max_seq_len
        )
        xlex_test_np, lex_len_test_np = make_sequence_embeddings_from_dataframes(
            w2v_pipe, task, test_dfs, max_len=w2v_max_seq_len
        )

        xlex_train = torch.from_numpy(xlex_train_np).float()
        xlex_test = torch.from_numpy(xlex_test_np).float()
        lex_len_train = torch.from_numpy(lex_len_train_np).long()
        lex_len_test = torch.from_numpy(lex_len_test_np).long()

        xattn_model = CrossAttentionFusionClassifier(
            ae=ae_trainer.network,
            tcn=build_tcn(len(DIAG_TO_INT)).to(device),
            w2v_dim=w2v_embedding_dim,
            static_dim=x_static_train.shape[1],
            static_latent=16,
            d_model=128,
            num_heads=num_heads,
            n_classes=len(DIAG_TO_INT),
            freeze_ae=True,
        ).to(device)

        train_loader_xattn = DataLoader(
            TensorDataset(
                x_seq_train,
                x_static_train,
                xlex_train,
                lex_len_train,
                torch.from_numpy(y_train).long(),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader_xattn = DataLoader(
            TensorDataset(
                x_seq_test,
                x_static_test,
                xlex_test,
                lex_len_test,
                torch.from_numpy(y_test).long(),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        xattn_result = train_cross_attention_classifier(
            xattn_model,
            train_loader_xattn,
            test_loader_xattn,
            device=device,
            epochs=epochs,
            lr=1e-3,
        )
        xattn_fold_accs.append(xattn_result.accuracy)
        all_xattn_true.extend(xattn_result.y_true.tolist())
        all_xattn_pred.extend(xattn_result.y_pred.tolist())
        print(f"{task} AE + TCN + Word2Vec cross-attention fold accuracy: {xattn_result.accuracy:.4f}")

    all_lr_true_np = np.asarray(all_lr_true)
    all_lr_pred_np = np.asarray(all_lr_pred)
    all_w2v_true_np = np.asarray(all_w2v_true)
    all_w2v_pred_np = np.asarray(all_w2v_pred)
    all_fusion_true_np = np.asarray(all_fusion_true)
    all_fusion_pred_np = np.asarray(all_fusion_pred)
    all_xattn_true_np = np.asarray(all_xattn_true)
    all_xattn_pred_np = np.asarray(all_xattn_pred)

    plot_fold_accuracies(task, lr_fold_accs, w2v_fold_accs, fusion_fold_accs, xattn_fold_accs, output_dir)

    plot_confusion_matrix_heatmap(all_lr_true_np, all_lr_pred_np, CLASS_NAMES, f"{task} Temporal LR Confusion Matrix", output_dir / f"{task.lower()}_lr_confusion_matrix.png")
    plot_confusion_matrix_heatmap(all_w2v_true_np, all_w2v_pred_np, CLASS_NAMES, f"{task} Word2Vec LR Confusion Matrix", output_dir / f"{task.lower()}_w2v_confusion_matrix.png")
    plot_confusion_matrix_heatmap(all_fusion_true_np, all_fusion_pred_np, CLASS_NAMES, f"{task} AE + TCN Fusion Confusion Matrix", output_dir / f"{task.lower()}_fusion_confusion_matrix.png")
    plot_confusion_matrix_heatmap(all_xattn_true_np, all_xattn_pred_np, CLASS_NAMES, f"{task} AE + TCN + Word2Vec XAttn Confusion Matrix", output_dir / f"{task.lower()}_xattn_confusion_matrix.png")

    print(f"\n{'=' * 80}")
    print(f"{task} TEMPORAL LOGISTIC REGRESSION CV RESULTS")
    print(f"{'=' * 80}")
    print(f"Mean fold accuracy: {np.mean(lr_fold_accs):.4f} ± {np.std(lr_fold_accs):.4f}")
    print_result_block(f"{task} Temporal LR Aggregated Results", all_lr_true_np, all_lr_pred_np)

    print(f"\n{'=' * 80}")
    print(f"{task} WORD2VEC LOGISTIC REGRESSION CV RESULTS")
    print(f"{'=' * 80}")
    print(f"Mean fold accuracy: {np.mean(w2v_fold_accs):.4f} ± {np.std(w2v_fold_accs):.4f}")
    print_result_block(f"{task} Word2Vec LR Aggregated Results", all_w2v_true_np, all_w2v_pred_np)

    print(f"\n{'=' * 80}")
    print(f"{task} AE + TCN FUSION CV RESULTS")
    print(f"{'=' * 80}")
    print(f"Mean fold accuracy: {np.mean(fusion_fold_accs):.4f} ± {np.std(fusion_fold_accs):.4f}")
    print_result_block(f"{task} Fusion Aggregated Results", all_fusion_true_np, all_fusion_pred_np)

    print(f"\n{'=' * 80}")
    print(f"{task} AE + TCN + WORD2VEC CROSS-ATTENTION CV RESULTS")
    print(f"{'=' * 80}")
    print(f"Mean fold accuracy: {np.mean(xattn_fold_accs):.4f} ± {np.std(xattn_fold_accs):.4f}")
    print_result_block(f"{task} Cross-Attention Aggregated Results", all_xattn_true_np, all_xattn_pred_np)

    # Combined evaluation plots

    y_true_dict = {
        "Temporal LR": all_lr_true,
        "Word2Vec LR": all_w2v_true,
        "AE+TCN": all_fusion_true,
        "Cross-Attn": all_xattn_true,
    }

    y_pred_dict = {
        "Temporal LR": all_lr_pred,
        "Word2Vec LR": all_w2v_pred,
        "AE+TCN": all_fusion_pred,
        "Cross-Attn": all_xattn_pred,
    }

    plot_full_evaluation(task, y_true_dict, y_pred_dict, output_dir)

    return {
        "lr_mean": float(np.mean(lr_fold_accs)),
        "w2v_mean": float(np.mean(w2v_fold_accs)),
        "fusion_mean": float(np.mean(fusion_fold_accs)),
        "xattn_mean": float(np.mean(xattn_fold_accs)),
        "lr_std": float(np.std(lr_fold_accs)),
        "w2v_std": float(np.std(w2v_fold_accs)),
        "fusion_std": float(np.std(fusion_fold_accs)),
        "xattn_std": float(np.std(xattn_fold_accs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run temporal, Word2Vec, AE+TCN, and cross-attention fusion experiments.")
    parser.add_argument("--task", choices=["PFT", "SFT", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results_graphs")
    parser.add_argument("--w2v-embedding-dim", type=int, default=32)
    parser.add_argument("--w2v-epochs", type=int, default=10)
    parser.add_argument("--w2v-max-seq-len", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = TranscriptLoader()
    data_module = TemporalDataModule(loader=loader, bin_sec=3.0, max_time=60.0)

    summary_results: dict[str, dict] = {}

    if args.task in {"PFT", "both"}:
        summary_results["PFT"] = run_task_cv(
            "PFT",
            data_module,
            device=device,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            n_splits=args.k_folds,
            w2v_embedding_dim=args.w2v_embedding_dim,
            w2v_epochs=args.w2v_epochs,
            w2v_max_seq_len=args.w2v_max_seq_len,
            num_heads=args.num_heads,
        )

    if args.task in {"SFT", "both"}:
        summary_results["SFT"] = run_task_cv(
            "SFT",
            data_module,
            device=device,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            n_splits=args.k_folds,
            w2v_embedding_dim=args.w2v_embedding_dim,
            w2v_epochs=args.w2v_epochs,
            w2v_max_seq_len=args.w2v_max_seq_len,
            num_heads=args.num_heads,
        )

    if len(summary_results) >= 2:
        plot_mean_accuracy_summary(summary_results, output_dir)


if __name__ == "__main__":
    main()
