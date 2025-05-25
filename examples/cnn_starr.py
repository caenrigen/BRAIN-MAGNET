from pathlib import Path
from typing import List, Literal, Optional

import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import mean_squared_error
from scipy import stats

import utils as ut


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

        self.prev_epoch: Optional[int] = None
        self.batches_losses: List[float] = []

        # TODO add notes about layer types to avoid for simplicity of the downstream
        #   SHAP analysis, i.e. motif discovery.
        self.model = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=15, padding="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.1),
            nn.Conv1d(16, 16, kernel_size=13, padding="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.1),
            nn.Conv1d(16, 16, kernel_size=11, padding="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),  # to be able to input into linear layer
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
            res = self.model(ut.tensor_reverse_complement(x))
        elif self.forward_mode == "both":
            res_fwd = self.model(x)
            res_rc = self.model(ut.tensor_reverse_complement(x))
            res = (res_fwd + res_rc) / 2  # take the average
        else:
            raise ValueError(f"{self.forward_mode = }")
        return res

    def _step(self, batch, batch_idx, suffix: str):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        out = self(inputs)
        loss: float = self.loss_fn(out, targets)

        if suffix == "train":
            if self.prev_epoch != self.current_epoch:
                self.prev_epoch = self.current_epoch
                self.batches_losses = []  # new epoch, reset
            self.batches_losses.append(loss)
            # Average across the epoch batches
            # ! This not equivalent to the loss that is obtained by evaluating the
            # ! end-of-epoch model on the entire training set. This is because the
            # ! weights are updated after each batch.
            loss_log = sum(self.batches_losses) / len(self.batches_losses)
        else:
            loss_log = loss

        # Log the training loss (this shows up in TensorBoard)
        self.log(
            # This var is used by the EarlyStopping
            f"{self.log_vars_prefix}_loss_{suffix}",
            loss_log,
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
    device: torch.device,
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

    targets = torch.cat(targets, dim=0).cpu().numpy()
    preds = torch.cat(preds, dim=0).cpu().numpy()
    return targets, preds


def model_stats(targets, preds):
    mse = mean_squared_error(targets, preds)
    pearson = float(stats.pearsonr(targets, preds).statistic)
    spearman = float(stats.spearmanr(targets, preds).statistic)
    return mse, pearson, spearman
