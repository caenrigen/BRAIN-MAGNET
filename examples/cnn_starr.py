from pathlib import Path
from typing import Literal

import lightning as L
import torch
from torch import nn


class CNNSTARR(L.LightningModule):
    def __init__(
        self,
        lr: float = 0.01,
        weight_decay: float = 0,
        log_vars_prefix: str = "NSC",
        forward_mode: Literal["main", "revcomp", "both"] = "both",
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.forward_mode = forward_mode
        self.log_vars_prefix = log_vars_prefix
        self.loss_fn = nn.MSELoss()

        self.model = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(1, 15), padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.1),
            nn.Conv2d(32, 16, kernel_size=(1, 13), padding="same"),
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


class ThresholdCheckpoint(L.Callback):
    def __init__(self, threshold: float = 0.14, task: str = "ESC"):
        super().__init__()
        self.threshold = threshold
        self.task = task

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module):
        val_loss = trainer.callback_metrics.get(f"{self.task}_loss_val")
        if val_loss is not None and val_loss <= self.threshold:
            checkpoint_dir = Path(trainer.logger.log_dir) / "threshold_checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            fn = f"{self.task}_vloss{int(val_loss * 1000):d}_ep{trainer.current_epoch}.pt"
            fp = checkpoint_dir / fn
            torch.save(pl_module.state_dict(), fp)
