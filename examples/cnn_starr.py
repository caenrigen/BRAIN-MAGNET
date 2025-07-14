from functools import partial
from typing import Callable, Literal, Optional, Union, Tuple

import torch
from torch import nn
import lightning as L
from torch.utils.tensorboard.writer import SummaryWriter
import utils as ut


def make_model_starr(
    channels: int = len(ut.DNA_ALPHABET),
    ks: int = 11,
    out0: int = 16,
    out1: int = 32,
    out2: int = 64,
    linear: int = 32,
    dropout_linear: float = 0.1,
    dropout_conv: float = 0.1,
):
    """
    This is a simplified, smaller model that provides the same prediction performance
    as the original model reported in the paper. It is much faster to train/evaluate.

    This models has ~21.2k training parameters.
    """
    # ! Avoid layers like MaxPool1d, AdaptiveMaxPool1d, etc. for simplicity of the
    # ! downstream SHAP analysis, i.e. motif discovery.
    # ! Such layers are tricky to deal within the SHAP analysis, even if solutions
    # ! exist for such layers, there might be caveats and performance issues.
    return nn.Sequential(
        nn.Conv1d(channels, out0, kernel_size=ks, stride=2, padding=ks // 2),
        nn.BatchNorm1d(out0),
        nn.ReLU(),
        nn.Dropout(dropout_conv),
        # ! MaxPool layers are tricky to deal within the SHAP analysis.
        # ! AvgPool1d destroys information, not a good substitute either.
        # ! Instead we can use stride=2 in the conv layer to achieve similar effect.
        # nn.MaxPool1d(2, 2),
        nn.Conv1d(out0, out1, kernel_size=ks - 2, stride=2, padding=(ks - 2) // 2),
        nn.BatchNorm1d(out1),
        nn.ReLU(),
        nn.Dropout(dropout_conv),
        # nn.MaxPool1d(2, 2), # ! see comment above
        nn.Conv1d(out1, out2, kernel_size=ks - 4, stride=2, padding=(ks - 4) // 2),
        nn.BatchNorm1d(out2),
        nn.ReLU(),
        nn.Dropout(dropout_conv),
        # nn.MaxPool1d(2, 2),  # ! see comment above
        # Using AdaptiveAvgPool1d makes the output of the CNN independent of the
        # input length and avoids parameters explosion when flattening.
        # ! However, internally AdaptiveAvgPool1d (aka global average pooling)
        # ! destroys the information about the position of the motifs (in later
        # ! SHAP analysis), despite performing good predictions.
        # nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),  # to be able to input into linear layer
        # figures out the input size automatically, for easy prototyping
        nn.LazyLinear(linear),
        nn.BatchNorm1d(linear),
        nn.ReLU(),
        nn.Dropout(dropout_linear),
        nn.Linear(linear, linear),
        nn.BatchNorm1d(linear),
        nn.ReLU(),
        nn.Dropout(dropout_linear),
        nn.Linear(linear, 1),
    )


class ModelModule(L.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,  # e.g. 1e-6
        forward_mode: Literal["forward", "reverse_complement", "mean"] = "forward",
        loss_fn: nn.Module = nn.MSELoss(),
        make_model: Callable[[], nn.Module] = make_model_starr,
        **hyper_params,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if forward_mode not in {"forward", "reverse_complement", "mean"}:
            raise ValueError(f"{forward_mode = }")
        self.forward_mode = forward_mode

        # Keep the names short to see more columns in the TensorBoard.
        hyper_params["lr"] = learning_rate
        hyper_params["wd"] = weight_decay
        hyper_params["fm"] = forward_mode
        # Save hyperparameters for logging in TensorBoard.
        # The hyperparameters are saved to the “hyper_parameters” key in the checkpoint
        # Calling this method seems to work only from the __init__ method.
        # logger=False is used to avoid logging an initial `hp_metric=-1`.
        self.save_hyperparameters(hyper_params, logger=False)

        self.loss_fn = loss_fn
        # Auxiliary variable to estimate the Pearson correlation.
        # It is handy to observe the Pearson correlation in the TensorBoard, its value
        # is more intuitive than the values of the MSE loss function.
        self.cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)

        self.model = make_model()

    def forward(self, x):
        if self.forward_mode == "forward":
            return self.model(x)
        elif self.forward_mode == "reverse_complement":
            return self.model(ut.one_hot_reverse_complement(x))
        elif self.forward_mode == "mean":
            res_fwd = self.model(x)
            res_rc = self.model(ut.one_hot_reverse_complement(x))
            return (res_fwd + res_rc) / 2  # take the average
        else:
            raise ValueError(f"{self.forward_mode = }")

    def _step(self, batch, batch_idx, suffix: str):
        inputs, targets = batch
        out = self(inputs)

        # Scalar tensors
        loss: torch.Tensor = self.loss_fn(out, targets)
        pearson: torch.Tensor = pearson_correlation_estimate(out, targets, self.cos_sim)
        pearson_neg: torch.Tensor = -pearson

        # Skip logging if `lightning` is in sanity checking mode.
        if self.trainer.sanity_checking:
            return loss

        # Logs the loss and pearson correlation (this shows up in TensorBoard)

        # The logging will average automatically over the batches of the epoch.
        # ! This is not equivalent to the loss that is obtained by evaluating the
        # ! end-of-epoch model on the entire training/validation/test set. This is
        # ! because the weights of the model are updated after each batch.
        log = partial(self.log, on_step=False, on_epoch=True)
        log(f"loss/{suffix}", loss, prog_bar=True)
        log(f"pearson/{suffix}", pearson_neg, prog_bar=False)

        return loss

    def training_step(self, batch, batch_idx: int):
        return self._step(batch, batch_idx, suffix="train")

    def validation_step(self, batch, batch_idx: int):
        return self._step(batch, batch_idx, suffix="val")

    def test_step(self, batch, batch_idx: int):
        return self._step(batch, batch_idx, suffix="test")

    def predict_step(
        self,
        batch: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ):
        """
        This method is required to implement otherwise predict() will not work directly
        with the batches output by our training dataloaders since our batches contain
        both the inputs and the targets.
        """
        # Support both (inputs, targets) and simply inputs
        if isinstance(batch, tuple):
            inputs, targets = batch
            return self(inputs), targets
        else:
            return self(batch)

    def configure_optimizers(self):
        # Using AdamW which is supposed to deal more correctly with weight decay,
        # in case we ever need a weight decay.
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


class LogMetrics(L.Callback):
    def __init__(self):
        super().__init__()
        self.vloss_min: Optional[torch.Tensor] = None
        # Corresponds values at the best validation loss.
        self.tloss: Optional[torch.Tensor] = None
        self.tpearson: Optional[torch.Tensor] = None
        self.vpearson: Optional[torch.Tensor] = None

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: ModelModule):
        """Gets called after validation epoch ends"""
        if trainer.sanity_checking:
            return

        tloss = trainer.callback_metrics.get("loss/train")
        assert tloss is not None
        vloss = trainer.callback_metrics.get("loss/val")
        assert vloss is not None

        tpearson = trainer.callback_metrics.get("pearson/train")
        assert tpearson is not None
        vpearson = trainer.callback_metrics.get("pearson/val")
        assert vpearson is not None

        # So that we can compare between training runs with different hyperparameters.
        assert trainer.logger is not None
        if self.vloss_min is None:
            # ! This must be called only once. Calling it allows to keep track of
            # ! this metric in the HPARAMS tab of the TensorBoard.
            trainer.logger.log_hyperparams(
                params=pl_module.hparams,  # type: ignore
                metrics={
                    "hp/vloss_min": vloss,
                    "hp/tloss": tloss,
                    "hp/tpearson": tpearson,
                    "hp/vpearson": vpearson,
                },
            )

            writer: SummaryWriter = trainer.logger.experiment  # type: ignore
            # This allows to plot both losses in the same chart under the
            # "CUSTOM SCALARS" tab of TensorBoard.
            # This is the most neat way to do it.
            # Structure: {category0: {title0: ["Multiline", tags0]}}
            layout = {
                "main": {
                    "loss": ["Multiline", ["loss/train", "loss/val"]],
                    "pearson": ["Multiline", ["pearson/train", "pearson/val"]],
                }
            }
            # ! Must be called only once.
            # ! Does not work to call it several times to add several layouts.
            writer.add_custom_scalars(layout)

        if self.vloss_min is None or vloss < self.vloss_min:
            self.vloss_min = vloss
            self.tloss = tloss
            self.tpearson = tpearson
            self.vpearson = vpearson

        assert self.tloss is not None
        assert self.vloss_min is not None
        assert self.tpearson is not None
        assert self.vpearson is not None
        log = partial(pl_module.log, on_step=False, on_epoch=True)
        # Log on each epoch end to keep the plots clear, instead of logging only
        # when there is a new best value.
        log("hp/vloss_min", self.vloss_min, prog_bar=True)
        log("hp/tloss", self.tloss, prog_bar=False)
        log("hp/tpearson", self.tpearson, prog_bar=False)
        log("hp/vpearson", self.vpearson, prog_bar=False)


def pearson_correlation_estimate(predictions, targets, cos_sim: nn.CosineSimilarity):
    """Estimate Pearson correlation using cosine similarity."""
    return cos_sim(predictions - predictions.mean(), targets - targets.mean())


def make_model_old():
    """
    This is the original model reported in the paper
    https://www.medrxiv.org/content/10.1101/2024.04.13.24305761v2.

    NB The original model used 2D convolutions (effectively 1xL pixels), which is
    an unnecessary complexity. Nucleotides are encoded as "channels", there is no need
    for the extra dummy dimension.

    This models has ~67.8M training parameters.

    The model above (`make_model`) is a simplified, much smaller, providing the
    same performance.
    """
    return nn.Sequential(
        nn.Conv1d(4, 128, 11, padding="same"),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(128, 256, 9, padding="same"),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Conv1d(256, 512, 7, padding="same"),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.MaxPool1d(2, 2),
        nn.Flatten(),
        nn.Linear(64000, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 1),
    )
