from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt

from ae import Autoencoder, ConvolutionalAutoencoder, Decoder, Encoder
from data_utils import DIAG_TO_INT, TemporalDataModule
from dataloader import TranscriptLoader
from tcn_model import TCN


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
        _, z = self.ae(x_seq)          # z: latent temporal representation
        tokens = self.tcn(z)           # token sequence / temporal embedding
        pooled = tokens.mean(dim=1)    # global average pooling
        s = self.static_enc(x_static)  # static feature branch
        fused = torch.cat([pooled, s], dim=1)
        return self.head(fused)


@dataclass
class FusionResult:
    accuracy: float
    y_true: np.ndarray
    y_pred: np.ndarray


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


def build_weighted_ce(train_loader: DataLoader, device: torch.device) -> nn.Module:
    y_all = []
    for _, _, yb in train_loader:
        y_all.extend(yb.numpy())
    y_all = np.array(y_all)

    class_counts = np.bincount(y_all, minlength=len(DIAG_TO_INT))
    class_weights = class_counts.sum() / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.mean()

    return nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )


def train_fusion_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
) -> FusionResult:
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    criterion = build_weighted_ce(train_loader, device)

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

        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        acc = float((y_true_np == y_pred_np).mean()) if len(y_true_np) else 0.0
        mean_val = float(np.mean(val_losses)) if val_losses else 0.0

        print(f"CLF Epoch {epoch+1}/{epochs} | train={mean_train:.4f} | val={mean_val:.4f} | acc={acc:.4f}")

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
    print(classification_report(
        y_true,
        y_pred,
        target_names=["HC", "MCI", "DEMENTIA"],
        zero_division=0
    ))

def plot_fold_accuracies(
    task: str,
    lr_fold_accs: list[float],
    fusion_fold_accs: list[float],
    output_dir: Path,
) -> None:
    folds = np.arange(1, len(lr_fold_accs) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(folds, lr_fold_accs, marker='o', label='Logistic Regression')
    plt.plot(folds, fusion_fold_accs, marker='o', label='AE + TCN Fusion')
    plt.xticks(folds)
    plt.ylim(0, 1.05)
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title(f"{task} Fold Accuracies")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / f"{task.lower()}_fold_accuracies.png"
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
    plt.imshow(cm, interpolation='nearest')
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


def plot_mean_accuracy_summary(
    results_dict: dict,
    output_dir: Path,
) -> None:
    """
    results_dict format:
    {
        "PFT": {"lr_mean": ..., "fusion_mean": ...},
        "SFT": {"lr_mean": ..., "fusion_mean": ...},
    }
    """
    tasks = list(results_dict.keys())
    lr_means = [results_dict[t]["lr_mean"] for t in tasks]
    fusion_means = [results_dict[t]["fusion_mean"] for t in tasks]

    x = np.arange(len(tasks))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, lr_means, width, label="Logistic Regression")
    plt.bar(x + width/2, fusion_means, width, label="AE + TCN Fusion")

    plt.xticks(x, tasks)
    plt.ylim(0, 1.05)
    plt.ylabel("Mean CV Accuracy")
    plt.title("Mean Cross-Validation Accuracy by Task")
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / "mean_accuracy_summary.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

def run_task_cv(
    task: str,
    data_module: TemporalDataModule,
    device: torch.device,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 8,
    n_splits: int = 5,
) -> dict:
    data = data_module.load_task(task)

    print(f"\n{'='*80}")
    print(f"{task} STRATIFIED {n_splits}-FOLD CROSS-VALIDATION")
    print(f"{'='*80}")
    print(f"{task} label counts:", {name: int((data.y == code).sum()) for name, code in DIAG_TO_INT.items()})
    print(f"All data shapes: seq={data.x_seq.shape} | static={data.x_static.shape}")



    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_lr_true = []
    all_lr_pred = []
    lr_fold_accs = []

    all_fusion_true = []
    all_fusion_pred = []
    fusion_fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(data.y)), data.y), start=1):
        print(f"\n{'-'*80}")
        print(f"{task} Fold {fold}/{n_splits}")
        print(f"{'-'*80}")

        x_seq_train, x_seq_test, x_static_train, x_static_test, y_train, y_test = data_module.make_fold_split(
            data.x_seq, data.x_static, data.y, train_idx, test_idx
        )

        print(
            f"Train/test: {x_seq_train.shape}, {x_seq_test.shape} | "
            f"static: {x_static_train.shape}, {x_static_test.shape}"
        )

        # -----------------------------
        # Logistic regression baseline
        # -----------------------------
        xlr_train = make_lr_features(x_seq_train, x_static_train)
        xlr_test = make_lr_features(x_seq_test, x_static_test)

        y_lr_pred, lr_acc = run_logistic_regression_fold(xlr_train, y_train, xlr_test, y_test)
        lr_fold_accs.append(lr_acc)
        all_lr_true.extend(y_test.tolist())
        all_lr_pred.extend(y_lr_pred.tolist())

        print(f"{task} LR fold accuracy: {lr_acc:.4f}")

        # -----------------------------
        # Autoencoder
        # -----------------------------
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

        # -----------------------------
        # TCN fusion classifier
        # -----------------------------
        tcn = TCN(
            N_C=1,
            input_size=8,
            output_size=len(DIAG_TO_INT),
            num_channels=[16, 32],
            K_S=[[3], [3]],
            Dil=[[1], [2]],
            Str=[[1], [1]],
            dropout=0.2,
            F_Ns=torch.tensor([32, 16, len(DIAG_TO_INT)]),
            dropout_F=0.2,
            Receptive_F=0,
            Norm=0,
            Residual=True,
        ).to(device)

        model = AETCNFusionClassifier(
            ae=ae_trainer.network,
            tcn=tcn,
            static_dim=x_static_train.shape[1],
            static_latent=16,
            n_classes=len(DIAG_TO_INT),
            freeze_ae=True,
        ).to(device)

        train_loader = DataLoader(
            TensorDataset(x_seq_train, x_static_train, torch.from_numpy(y_train).long()),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(x_seq_test, x_static_test, torch.from_numpy(y_test).long()),
            batch_size=batch_size,
            shuffle=False,
        )

        fusion_result = train_fusion_classifier(
            model,
            train_loader,
            test_loader,
            device=device,
            epochs=epochs,
            lr=1e-3,
        )

        fusion_fold_accs.append(fusion_result.accuracy)
        all_fusion_true.extend(fusion_result.y_true.tolist())
        all_fusion_pred.extend(fusion_result.y_pred.tolist())

        print(f"{task} Fusion fold accuracy: {fusion_result.accuracy:.4f}")
    class_names = ["HC", "MCI", "DEMENTIA"]

    plot_fold_accuracies(task, lr_fold_accs, fusion_fold_accs, output_dir)

    plot_confusion_matrix_heatmap(
        all_lr_true,
        all_lr_pred,
        class_names,
        title=f"{task} Logistic Regression Confusion Matrix",
        output_path=output_dir / f"{task.lower()}_lr_confusion_matrix.png",
    )

    plot_confusion_matrix_heatmap(
        all_fusion_true,
        all_fusion_pred,
        class_names,
        title=f"{task} AE + TCN Fusion Confusion Matrix",
        output_path=output_dir / f"{task.lower()}_fusion_confusion_matrix.png",
    )

    # return {
    #     "lr_mean": float(np.mean(lr_fold_accs)),
    #     "fusion_mean": float(np.mean(fusion_fold_accs)),
    #     "lr_std": float(np.std(lr_fold_accs)),
    #     "fusion_std": float(np.std(fusion_fold_accs)),
    # }

    all_lr_true = np.array(all_lr_true)
    all_lr_pred = np.array(all_lr_pred)

    all_fusion_true = np.array(all_fusion_true)
    all_fusion_pred = np.array(all_fusion_pred)

    print(f"\n{'='*80}")
    print(f"{task} LOGISTIC REGRESSION CV RESULTS")
    print(f"{'='*80}")
    print(f"Mean fold accuracy: {np.mean(lr_fold_accs):.4f} ± {np.std(lr_fold_accs):.4f}")
    print_result_block(f"{task} Logistic Regression Aggregated Results", all_lr_true, all_lr_pred)

    print(f"\n{'='*80}")
    print(f"{task} AE + TCN FUSION CV RESULTS")
    print(f"{'='*80}")
    print(f"Mean fold accuracy: {np.mean(fusion_fold_accs):.4f} ± {np.std(fusion_fold_accs):.4f}")
    print_result_block(f"{task} Fusion Aggregated Results", all_fusion_true, all_fusion_pred)

    class_names = ["HC", "MCI", "DEMENTIA"]

    plot_fold_accuracies(task, lr_fold_accs, fusion_fold_accs, output_dir)

    plot_confusion_matrix_heatmap(
        all_lr_true,
        all_lr_pred,
        class_names,
        title=f"{task} Logistic Regression Confusion Matrix",
        output_path=output_dir / f"{task.lower()}_lr_confusion_matrix.png",
    )

    plot_confusion_matrix_heatmap(
        all_fusion_true,
        all_fusion_pred,
        class_names,
        title=f"{task} AE + TCN Fusion Confusion Matrix",
        output_path=output_dir / f"{task.lower()}_fusion_confusion_matrix.png",
    )

    return {
        "lr_mean": float(np.mean(lr_fold_accs)),
        "fusion_mean": float(np.mean(fusion_fold_accs)),
        "lr_std": float(np.std(lr_fold_accs)),
        "fusion_std": float(np.std(fusion_fold_accs)),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Run temporal pause experiments with stratified k-fold CV.")
    parser.add_argument("--task", choices=["PFT", "SFT", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results_graphs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = TranscriptLoader()
    data_module = TemporalDataModule(loader=loader, bin_sec=3.0, max_time=60.0)

    summary_results = {}

    if args.task in {"PFT", "both"}:
        summary_results["PFT"] = run_task_cv(
            "PFT",
            data_module,
            device=device,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            n_splits=args.k_folds,
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
        )

    if len(summary_results) >= 2:
        plot_mean_accuracy_summary(summary_results, output_dir)


if __name__ == "__main__":
    main()