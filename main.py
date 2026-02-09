from __future__ import annotations
import argparse
import csv
import math
import random
from pathlib import Path
from typing import Literal
import torch
from torch.utils.data import DataLoader, TensorDataset
from dataset import make_dataset
from webApp.model.FCNN import FCNN, parse_hidden_dims
from metrics import BinaryClassificationEvaluator


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import pandas as pd

def temporal_purged_split_indices(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    val_frac: float,
    test_frac: float = 0.0,
    horizon_days: int = 14,
):
    if not (0.0 < float(val_frac) < 1.0):
        raise ValueError("val_frac deve essere in (0,1).")
    if not (0.0 <= float(test_frac) < 1.0):
        raise ValueError("test_frac deve essere in [0,1).")
    if float(val_frac) + float(test_frac) >= 1.0:
        raise ValueError("val_frac + test_frac deve essere < 1.")

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values(date_col).reset_index()
    n = len(d)
    if n < 3:
        raise ValueError("Dataset troppo piccolo per fare split temporale.")

    test_size = int(math.floor(n * float(test_frac)))
    val_size = max(1, int(math.floor(n * float(val_frac))))

    if test_size > 0:
        test_df = d.iloc[-test_size:]
        test_start = test_df[date_col].min()
        val_candidates = d[d[date_col] < (test_start - pd.Timedelta(days=int(horizon_days)))]
    else:
        test_df = d.iloc[0:0]
        val_candidates = d

    if len(val_candidates) <= val_size:
        raise ValueError("Val troppo grande (o purge troppo aggressivo): riduci val_frac/test_frac o horizon_days.")

    val_df = val_candidates.iloc[-val_size:]
    val_start = val_df[date_col].min()
    purge_before_val = val_start - pd.Timedelta(days=int(horizon_days))
    train_df = d[d[date_col] < purge_before_val]

    if len(train_df) == 0:
        raise ValueError("Train vuoto dopo purge. Riduci val_frac o horizon_days.")

    train_idx = torch.tensor(train_df["index"].to_numpy(), dtype=torch.long)
    val_idx   = torch.tensor(val_df["index"].to_numpy(), dtype=torch.long)
    test_idx  = torch.tensor(test_df["index"].to_numpy(), dtype=torch.long)
    return train_idx, val_idx, test_idx



@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_probs: list[float] = []
    all_y: list[float] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * int(x.shape[0])
        total += int(x.shape[0])

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).to(dtype=y.dtype)
        correct += int((preds == y).sum().item())

        probs_1d = probs.detach().flatten().to("cpu").tolist()
        y_1d = y.detach().flatten().to("cpu").tolist()
        for prob, yy in zip(probs_1d, y_1d):
            all_probs.append(float(prob))
            all_y.append(float(yy))

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)

    def _roc_auc(probs: list[float], ys: list[float]) -> float:
        # Mann–Whitney U con gestione dei tie (rank medio).
        pairs = list(zip(probs, ys))
        n_pos = sum(1 for _, yy in pairs if float(yy) >= 0.5)
        n_neg = len(pairs) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")

        pairs.sort(key=lambda t: float(t[0]))
        ranks = [0.0] * len(pairs)
        i = 0
        r = 1
        while i < len(pairs):
            j = i
            while j < len(pairs) and float(pairs[j][0]) == float(pairs[i][0]):
                j += 1
            avg_rank = (r + (r + (j - i) - 1)) / 2.0
            for k in range(i, j):
                ranks[k] = avg_rank
            r += (j - i)
            i = j

        sum_ranks_pos = 0.0
        for (_, yy), rk in zip(pairs, ranks):
            if float(yy) >= 0.5:
                sum_ranks_pos += float(rk)

        u = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
        return float(u) / float(n_pos * n_neg)

    def _average_precision(probs: list[float], ys: list[float]) -> float:
        # Average Precision (AUPRC a gradini).
        pairs = sorted(zip(probs, ys), key=lambda t: float(t[0]), reverse=True)
        positives_total = sum(1 for _, yy in pairs if float(yy) >= 0.5)
        if positives_total == 0:
            return float("nan")

        tp = 0
        fp = 0
        precision_sum = 0.0
        for _, yy in pairs:
            if float(yy) >= 0.5:
                tp += 1
                precision_sum += tp / max(1, tp + fp)
            else:
                fp += 1
        return float(precision_sum) / float(positives_total)

    evaluate.last_auc = _roc_auc(all_probs, all_y) if all_probs else float("nan")
    evaluate.last_pr_auc = _average_precision(all_probs, all_y) if all_probs else float("nan")
    return avg_loss, acc


def train_one_epoch(model: torch.nn.Module, loader: DataLoader, criterion: torch.nn.Module, optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * int(x.shape[0])
        total += int(x.shape[0])

    return total_loss / max(1, total)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MLP per predire infortunio entro H giorni (binary classification).")
    p.add_argument(
        "--data",
        default="dataset/processed/objective_rolling_7d_with_injury_next_14_days.csv",
        type=Path,
        help="Path CSV con feature + label injury_next_{H}_days.",
    )
    p.add_argument("--horizon-days", type=int, default=14, help="H in injury_next_{H}_days.")
    p.add_argument("--label-col", type=str, default=None, help="Nome colonna label (override).")

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dims", type=str, default="128,64", help="Es: 256,128,64")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch-norm", action="store_true")

    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None, help="cpu | cuda (default: auto)")

    p.add_argument("--pos-weight", type=str, default="auto", help="BCE pos_weight: 'auto' | numero | 'none'")
    p.add_argument("--balanced-sampler", action="store_true", help="Oversampling per bilanciare classi nel train loader.")

    p.add_argument("--monitor", type=str, default="pr_auc", choices=["val_loss", "roc_auc", "pr_auc"])
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min-delta", type=float, default=1e-4)

    p.add_argument("--threshold", type=float, default=0.3, help="Soglia su sigmoid(probs) per classi 0/1.")

    p.add_argument("--save-path", type=Path, default=Path("outputs/best_model.pt"))
    p.add_argument("--loss-csv", type=Path, default=Path("outputs/loss_curve.csv"))
    p.add_argument("--loss-png", type=Path, default=Path("outputs/loss_curve.png"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    seed_everything(int(args.seed))

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = make_dataset(args.data, horizon_days=int(args.horizon_days), label_col=args.label_col)
    X = ds.X
    y = ds.y


    df = pd.read_csv(args.data)
    train_idx, val_idx, test_idx = temporal_purged_split_indices(
        df,
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        horizon_days=int(args.horizon_days),
    )
    train_pos_rate = y[train_idx].float().mean().item()
    val_pos_rate = y[val_idx].float().mean().item()
    test_pos_rate = y[test_idx].float().mean().item() if len(test_idx) else float("nan")

    print(f"Train size: {len(train_idx)} | Pos rate: {train_pos_rate:.4f}")
    print(f"Val size:   {len(val_idx)} | Pos rate: {val_pos_rate:.4f}")
    if len(test_idx):
        print(f"Test size:  {len(test_idx)} | Pos rate: {test_pos_rate:.4f}")

    dts = pd.to_datetime(df["date"])
    train_max = dts.iloc[train_idx.numpy()].max()
    val_min = dts.iloc[val_idx.numpy()].min()
    val_max = dts.iloc[val_idx.numpy()].max()
    print(f"Train max date: {train_max.date()} | Val date range: {val_min.date()} -> {val_max.date()}")
    if len(test_idx):
        test_min = dts.iloc[test_idx.numpy()].min()
        test_max = dts.iloc[test_idx.numpy()].max()
        print(f"Test date range: {test_min.date()} -> {test_max.date()}")



    mean = X[train_idx].mean(dim=0)
    std = X[train_idx].std(dim=0, unbiased=False)
    std = torch.where(std == 0, torch.ones_like(std), std)

    Xn = (X - mean) / std
    train_set = TensorDataset(Xn[train_idx], y[train_idx])
    val_set = TensorDataset(Xn[val_idx], y[val_idx])
    test_set = TensorDataset(Xn[test_idx], y[test_idx]) if len(test_idx) else None

    if bool(args.balanced_sampler):
        from torch.utils.data import WeightedRandomSampler

        y_train = y[train_idx].flatten().to("cpu")
        n_pos = int((y_train >= 0.5).sum().item())
        n_neg = int((y_train < 0.5).sum().item())
        if n_pos == 0 or n_neg == 0:
            raise ValueError("Impossibile usare --balanced-sampler: train set con una sola classe.")

        w_pos = 0.5 / float(n_pos)
        w_neg = 0.5 / float(n_neg)
        weights = torch.where(y_train >= 0.5, torch.full_like(y_train, w_pos), torch.full_like(y_train, w_neg))
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    test_loader = (
        DataLoader(
            test_set,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
        )
        if test_set is not None
        else None
    )

    hidden_dims = parse_hidden_dims(str(args.hidden_dims))
    print(f"ciao{X.shape}")
    model = FCNN(
        input_dim=int(X.shape[1]),
        hidden_dims=hidden_dims,
        dropout=float(args.dropout),
        batch_norm=bool(args.batch_norm),
    ).to(device)

    pos_weight_arg = str(args.pos_weight).strip().lower()
    if pos_weight_arg in {"none", "off", "false", "0"}:
        pos_weight = None
    elif pos_weight_arg == "auto":
        y_train = y[train_idx].flatten()
        n_pos = float((y_train >= 0.5).sum().item())
        n_neg = float((y_train < 0.5).sum().item())
        if n_pos == 0 or n_neg == 0:
            pos_weight = None
        else:
            pos_weight = n_neg / n_pos
    else:
        pos_weight = float(pos_weight_arg)
        if pos_weight <= 0:
            raise ValueError("--pos-weight deve essere > 0, oppure 'auto'/'none'.")

    if pos_weight is None:
        criterion = torch.nn.BCEWithLogitsLoss()
        print("Loss: BCEWithLogitsLoss (pos_weight: none)")
    else:
        pw = torch.tensor([float(pos_weight)], dtype=torch.float32, device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
        print(f"Loss: BCEWithLogitsLoss (pos_weight={float(pos_weight):.4f})")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    evaluator = BinaryClassificationEvaluator(criterion=criterion, threshold=float(args.threshold))


    monitor: Literal["val_loss", "roc_auc", "pr_auc"] = str(args.monitor)  # type: ignore[assignment]
    if monitor == "val_loss":
        best_metric = float("inf")
        is_better = lambda cur, best: (cur + float(args.min_delta)) < best
    else:
        best_metric = -float("inf")
        is_better = lambda cur, best: (cur - float(args.min_delta)) > best

    bad_epochs = 0
    save_path: Path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluator.evaluate(model, val_loader, device)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics.loss),
                "val_acc": float(val_metrics.acc),
                "val_roc_auc": float(val_metrics.roc_auc),
                "val_pr_auc": float(val_metrics.pr_auc),
            }
        )

        print(
            f"epoch {epoch:03d}/{int(args.epochs)} | "
            f"train_loss={train_loss:.5f} | val_loss={val_metrics.loss:.5f} | val_acc={val_metrics.acc:.4f} | "
            f"ROC-AUC:{val_metrics.roc_auc:.5f} | PR-AUC:{val_metrics.pr_auc:.6f} (baseline~{val_pos_rate:.4f})"
        )

        current_metric = (
            float(val_metrics.loss)
            if monitor == "val_loss"
            else float(val_metrics.roc_auc)
            if monitor == "roc_auc"
            else float(val_metrics.pr_auc)
        )

        if is_better(current_metric, best_metric):
            best_metric = float(current_metric)
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": int(X.shape[1]),
                    "hidden_dims": tuple(int(x) for x in hidden_dims),
                    "dropout": float(args.dropout),
                    "batch_norm": bool(args.batch_norm),
                    "horizon_days": int(args.horizon_days),
                    "label_col": ds.label_col,
                    "feature_names": ds.feature_names,
                    "mean": mean.cpu(),
                    "std": std.cpu(),
                    "monitor": str(monitor),
                    "best_metric": float(best_metric),
                    "threshold": float(args.threshold),
                },
                save_path,
            )
        else:
            bad_epochs += 1

        if int(args.patience) > 0 and bad_epochs >= int(args.patience):
            print(f"Early stopping: nessun miglioramento su {monitor} per {int(args.patience)} epoche.")
            break

    loss_csv: Path = Path(args.loss_csv)
    loss_csv.parent.mkdir(parents=True, exist_ok=True)
    with loss_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "val_roc_auc", "val_pr_auc"],
        )
        w.writeheader()
        for row in history:
            w.writerow(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": row["train_loss"],
                    "val_loss": row["val_loss"],
                    "val_acc": row["val_acc"],
                    "val_roc_auc": row["val_roc_auc"],
                    "val_pr_auc": row["val_pr_auc"],
                }
            )

    try:
        import matplotlib.pyplot as plt  # type: ignore

        epochs = [int(r["epoch"]) for r in history]
        tr = [float(r["train_loss"]) for r in history]
        va = [float(r["val_loss"]) for r in history]

        plt.figure(figsize=(8, 4.5))
        plt.plot(epochs, tr, label="train_loss")
        plt.plot(epochs, va, label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Loss curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        loss_png: Path = Path(args.loss_png)
        loss_png.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(loss_png, dpi=150)
        plt.close()
        print(f"OK: loss curve salvata in {loss_png} (dati in {loss_csv})")
    except Exception as exc:
        print(f"Nota: non ho generato il grafico (matplotlib mancante o errore): {exc}")
        print(f"OK: dati loss salvati in {loss_csv}")

    print(f"OK: best {monitor}={best_metric:.6f} salvato in {save_path}")

    if test_loader is not None:
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = evaluator.evaluate(model, test_loader, device)
        print(
            f"Test | loss={test_metrics.loss:.5f} | acc={test_metrics.acc:.4f} | "
            f"ROC-AUC:{test_metrics.roc_auc:.5f} | PR-AUC:{test_metrics.pr_auc:.6f} (baseline~{test_pos_rate:.4f})"
        )
        cm = test_metrics.cm
        print(f"Confusion matrix (threshold={test_metrics.threshold:.2f}) [[TN, FP], [FN, TP]]: {cm.as_matrix()}")
        print(
            f"Precision={test_metrics.precision:.4f} | Recall={test_metrics.recall:.4f} | "
            f"F1={test_metrics.f1:.4f} | Specificity={test_metrics.specificity:.4f} | "
            f"BalancedAcc={test_metrics.balanced_acc:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
