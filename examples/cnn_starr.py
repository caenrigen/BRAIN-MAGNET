from pathlib import Path
from typing import List, Literal, Optional

import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from scipy import stats
from torch.utils.tensorboard.writer import SummaryWriter
import utils as ut


class BrainMagnetCNN(L.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        forward_mode: Literal["forward", "reverse_complement", "average"] = "forward",
        loss_fn: nn.Module = nn.MSELoss(),
        **hyper_params,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if forward_mode not in {"forward", "reverse_complement", "average"}:
            raise ValueError(f"{forward_mode = }")
        self.forward_mode = forward_mode

        # Keep the names short to see more columns in the TensorBoard.
        hyper_params["lr"] = learning_rate
        hyper_params["wd"] = weight_decay
        hyper_params["fm"] = forward_mode
        # Save hyperparameters for logging purposes.
        # Calling this method seems to work only from the __init__ method.
        # logger=False is used to avoid logging an initial `hp_metric=-1`.
        self.save_hyperparameters(hyper_params, logger=False)

        self.loss_fn = loss_fn

        self.prev_epoch: Optional[int] = None
        self.losses: List[float] = []

        # ! Avoid Layers like MaxPool1d, AdaptiveMaxPool1d, etc. for simplicity of the
        # ! downstream SHAP analysis, i.e. motif discovery.
        # ! Such layers are tricky to deal with for the SHAP analysis, even if solutions
        # ! exist, there might caveats and performance issues.
        self.model = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=15, padding="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(16, 16, kernel_size=13, padding="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.Conv1d(16, 16, kernel_size=11, padding="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),  # to be able to input into linear layer
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # Did not seem to be needed. The model is already small and the BatchNorm
            # is already taking care of regularization.
            # nn.Dropout(0.1),
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        if self.forward_mode == "forward":
            return self.model(x)
        elif self.forward_mode == "reverse_complement":
            return self.model(ut.tensor_reverse_complement(x))
        elif self.forward_mode == "average":
            res_fwd = self.model(x)
            res_rc = self.model(ut.tensor_reverse_complement(x))
            return (res_fwd + res_rc) / 2  # take the average
        else:
            raise ValueError(f"{self.forward_mode = }")

    def _step(self, batch, batch_idx, suffix: str):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        out = self(inputs)
        loss: float = self.loss_fn(out, targets)

        self.losses.append(loss)
        # Average across the epoch batches
        # ! This not equivalent to the loss that is obtained by evaluating the
        # ! end-of-epoch model on the entire training set. This is because the
        # ! weights are updated after each batch.
        loss_log = sum(self.losses) / len(self.losses)

        # Skip logging if `lightning` is in sanity checking mode.
        if self.trainer.sanity_checking:
            return loss

        # Log the training loss (this shows up in TensorBoard)
        self.log(
            # This var is used by the EarlyStopping
            f"loss/{suffix}",
            loss_log,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def on_train_epoch_start(self):
        self.losses = []

    def on_validation_epoch_start(self):
        self.losses = []

    def on_test_epoch_start(self):
        self.losses = []

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, suffix="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, suffix="val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, suffix="test")

    def configure_optimizers(self):
        # Using AdamW which is supposed to deal more correctly with weight decay,
        # in case we ever need a weight decay.
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


class EpochCheckpoint(L.Callback):
    def __init__(self):
        super().__init__()
        self.min_vloss: Optional[torch.Tensor] = None
        # Corresponds training loss at the best validation loss.
        self.tloss: Optional[torch.Tensor] = None

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: BrainMagnetCNN):
        """Gets called after validation epoch ends"""
        if trainer.sanity_checking:
            return

        assert trainer.logger is not None and trainer.logger.log_dir is not None
        checkpoint_dir = Path(trainer.logger.log_dir) / "epoch_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        fn = f"ep{trainer.current_epoch:03d}.pt"
        fp = checkpoint_dir / fn
        torch.save(pl_module.state_dict(), fp)

        tloss = trainer.callback_metrics.get("loss/train")
        assert tloss is not None
        vloss = trainer.callback_metrics.get("loss/val")
        assert vloss is not None

        # So that we can compare between training runs with different hyperparameters.
        assert trainer.logger is not None
        if self.min_vloss is None:
            # ! This must be called only once. Calling it allows to keep track of
            # ! this metric in the HPARAMS tab of the TensorBoard.
            trainer.logger.log_hyperparams(
                params=pl_module.hparams,  # type: ignore
                metrics={
                    "hp/min_vloss": vloss,
                    "hp/tloss": tloss,
                },
            )

            writer: SummaryWriter = trainer.logger.experiment  # type: ignore
            # This allows to plot both losses in the same chart under the
            # "CUSTOM SCALARS" tab of TensorBoard.
            # This is the most neat way to do it.
            # ! Must be called only once.
            writer.add_custom_scalars_multilinechart(
                ["loss/train", "loss/val"],
                category="main",
                title="losses",
            )

        if self.min_vloss is None or vloss < self.min_vloss:
            self.min_vloss = vloss
            self.tloss = tloss

        assert self.tloss is not None
        # Log both on each epoch end to keep the plots pretty, instead of logging only
        # when there is a new best value.
        pl_module.log(
            "hp/min_vloss",
            self.min_vloss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        pl_module.log(
            "hp/tloss",
            self.tloss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


def load_model(
    fp: Path,
    device: torch.device,
    **kwargs_model,
):
    model = BrainMagnetCNN(**kwargs_model)
    model.to(device)
    model.load_state_dict(torch.load(fp))
    model.to(device)
    return model


def eval_model(model: BrainMagnetCNN, dataloader: DataLoader, device):
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
