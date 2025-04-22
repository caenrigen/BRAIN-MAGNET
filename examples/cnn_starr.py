from pathlib import Path
from typing import Literal

import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import mean_squared_error
from scipy import stats


class CNNSTARR(L.LightningModule):
    def __init__(
        self,
        lr: float = 0.01,
        weight_decay: float = 0,
        log_vars_prefix: str = "NSC",
        forward_mode: Literal["main", "revcomp", "both"] = "both",
        loss_fn: nn.Module = nn.MSELoss(),
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.forward_mode = forward_mode
        self.log_vars_prefix = log_vars_prefix
        self.loss_fn = loss_fn

        self.model = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(1, 15), padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.1),
            nn.Conv2d(16, 16, kernel_size=(1, 13), padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.1),
            nn.Conv2d(16, 16, kernel_size=(1, 11), padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        if self.forward_mode == "main":
            res = self.model(x)
        elif self.forward_mode == "revcomp":
            res = self.model(tensor_reverse_complement(x))
        elif self.forward_mode == "both":
            res_fwd = self.model(x)
            res_rc = self.model(tensor_reverse_complement(x))
            res = (res_fwd + res_rc) / 2  # take the average
        else:
            raise ValueError(f"{self.forward_mode = }")
        return res

    def _step(self, batch, batch_idx, suffix: str):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        out = self(inputs)
        loss = self.loss_fn(out, targets)

        # Log the training loss (this shows up in TensorBoard)
        self.log(
            # This var is used by the EarlyStopping
            f"{self.log_vars_prefix}_loss_{suffix}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, suffix="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, suffix="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, suffix="test")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


class EpochCheckpoint(L.Callback):
    def __init__(self, task: str = "ESC"):
        super().__init__()
        self.task = task

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module):
        val_loss = trainer.callback_metrics.get(f"{self.task}_loss_val")
        train_loss = trainer.callback_metrics.get(f"{self.task}_loss_train")
        if val_loss is not None and train_loss is not None:
            checkpoint_dir = Path(trainer.logger.log_dir) / "epoch_checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            fn = f"{self.task}_ep{trainer.current_epoch:02d}_vloss{int(val_loss * 1000):04d}_tloss{int(train_loss * 1000):04d}.pt"
            fp = checkpoint_dir / fn
            torch.save(pl_module.state_dict(), fp)


def load_model(
    fp: Path,
    device,
    forward_mode: Literal["main", "revcomp", "both"] = "main",
):
    model = CNNSTARR(forward_mode=forward_mode, log_vars_prefix="")
    model.to(device)
    model.load_state_dict(torch.load(fp))
    model.to(device)
    return model


def eval_model(model: CNNSTARR, dataloader: DataLoader, device):
    preds = []
    targets = []

    model.eval()
    with torch.no_grad():
        for _batch, data in enumerate(dataloader):
            inputs_, targets_ = data
            inputs_ = inputs_.to(device)
            targets_ = targets_.to(device)
            outputs = model(inputs_)
            targets.append(targets_)
            preds.append(outputs)

    targets = torch.cat(targets, dim=0).squeeze().cpu().numpy()
    preds = torch.cat(preds, dim=0).squeeze().cpu().numpy()
    return targets, preds


def model_stats(targets, preds):
    mse = mean_squared_error(targets, preds)
    pearson = float(stats.pearsonr(targets, preds).statistic)
    spearman = float(stats.spearmanr(targets, preds).statistic)
    return mse, pearson, spearman


def pick_checkpoint(df, fold: int, tolerance: float = 0.10, ax=None):
    df = df[df.fold == fold]
    df.set_index("epoch", inplace=True)
    df.sort_index(inplace=True)
    min_, max_ = df.mse.min(), df.mse.max()
    if ax:
        sns.lineplot(data=df, x="epoch", y="mse", hue="set_name", marker="o", ax=ax)

    df = df.pivot(columns="set_name", values="mse")
    df["diff_train_val"] = (df.train - df.val).abs()
    df["stop"] = (df.train < df.val) & (df.diff_train_val <= tolerance * df.val)
    df = df[df.stop]
    df.sort_values(by=["val", "diff_train_val"], inplace=True)
    epoch = df.index.values[0]
    if ax:
        ax.vlines(epoch, min_, max_, color="red", linestyle="--")
    return epoch
