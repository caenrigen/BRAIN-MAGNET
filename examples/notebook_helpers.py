import torch
from pathlib import Path
from typing import Union, Literal, Optional
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import logging
import pandas as pd
import matplotlib.pyplot as plt

import cnn_starr as cnn
import data_module as dm
import utils as ut
import plot_utils as put

logger = logging.getLogger(__name__)


def train(
    save_dir_tensorboard: Path,
    fp_dataset: Path,
    fold: int,
    folds: Union[int, None],
    batch_size: int,
    task: Literal["ESC", "NSC"],
    learning_rate: float,
    max_epochs: int,
    frac_test: float,
    frac_val: float,
    random_state: int,
    device: torch.device,
    version: Optional[str] = None,
    empty_cache: bool = True,
    weight_decay: float = 0.0,
):
    if empty_cache:
        ut.empty_cache(device)
    # We did not use workers, but we keep it here for future reference and reminder.
    L.seed_everything(random_state, workers=True)  # for reproducibility

    model = cnn.BrainMagnetCNN(
        learning_rate=learning_rate,
        # Don't change this for training, reverse complement is handled by the data
        # module as augmentation data.
        forward_mode="forward",
        # The rest are hyperparameters for logging purposes.
        task=task,
        batch_size=batch_size,
        frac_test=frac_test,
        frac_val=frac_val,
        folds=folds,
        fold=fold,
        max_ep=max_epochs,
        weight_decay=weight_decay,
        random_state=random_state,
    )
    # print(model)

    version = ut.make_version(fold=fold if folds else None, version=version)
    logger.info(f"{version = }")
    tb_logger = TensorBoardLogger(
        save_dir=save_dir_tensorboard,
        name=task,
        version=version,
        # avoid inserting a dummy metric with an initial value
        default_hp_metric=False,
    )
    checkpoints_callback = ModelCheckpoint(
        filename="{epoch:03d}",
        every_n_epochs=1,
        save_top_k=-1,
        # Set it to `False` if you intend to, e.g., be able to resume training from a
        # checkpoint and need things like optimizer state, etc. to be saved.
        save_weights_only=True,
    )
    trainer = L.Trainer(
        accelerator=device.type,
        max_epochs=max_epochs,
        logger=tb_logger,
        callbacks=[cnn.LogMetrics(), checkpoints_callback],
        deterministic=True,  # for reproducibility
        enable_checkpointing=True,
    )

    datamodule = dm.DataModule(
        fp_dataset=fp_dataset,
        random_state=random_state,
        folds=folds or None,
        fold=fold,
        frac_test=frac_test,
        frac_val=frac_val,
        # DataLoader kwargs:
        batch_size=batch_size,
        # These might give some speed up if cuda is available
        # pin_memory=True,
        # pin_memory_device="cuda",
    )
    try:
        trainer.fit(model, datamodule=datamodule)
    except (KeyboardInterrupt, NameError):
        logger.info("Training interrupted by user")
        return False

    # Free up memory
    model.cpu()
    if empty_cache:
        ut.empty_cache(device)

    return True


def pick_best_checkpoints(df_ckpts: pd.DataFrame, plot: bool = False):
    best_checkpoints = {}
    if plot:
        fig, axs = plt.subplots(1, 5, figsize=(15, 3), sharex=True, sharey=True)
    for fold in range(5):
        df = df_ckpts[df_ckpts.fold == fold].copy()
        best_epoch = dm.pick_checkpoint(df, ax=axs[fold] if plot else None)
        df_best = df[df.epoch == best_epoch]
        assert len(df_best) == 1
        best_checkpoints[fold] = df_best.fp.iloc[0]
        if plot:
            max_ = df_ckpts.loss_val.quantile(0.95)  # don't plot outliers
            min_ = df_ckpts.loss_train.min()
            axs[fold].set_ylim(min_ - (max_ - min_) * 0.05, max_)

    if plot:
        fig.tight_layout()

    return best_checkpoints, fig, axs


def evaluate_model(
    fp_checkpoint: Path,
    fp_dataset: Path,
    device: torch.device,
    random_state: Optional[int],
    dataloader: str = "test_dataloader",
):
    model = cnn.BrainMagnetCNN.load_from_checkpoint(fp_checkpoint)

    datamodule = dm.DataModule(
        fp_dataset=fp_dataset,
        targets_col=f"{model.hparams_initial.task}_log2_enrichment",
        random_state=model.hparams_initial.get("random_state", random_state),
        folds=model.hparams_initial.get("folds", None),
        fold=model.hparams_initial.fold,
        frac_test=model.hparams_initial.frac_test,
        frac_val=model.hparams_initial.frac_val,
        batch_size=model.hparams_initial.batch_size,
    )
    datamodule.setup()
    dataloader = getattr(datamodule, dataloader)()

    # Reuse our code to evaluate the model on the test set
    trainer = L.Trainer(accelerator=device.type)
    results_list = trainer.predict(model=model, dataloaders=dataloader)

    # Concatenate the results
    preds_list, targets_list = zip(*results_list)

    preds = torch.cat(preds_list).squeeze().numpy()
    targets = torch.cat(targets_list).squeeze().numpy()

    mse, pearson, spearman = ut.model_stats(targets, preds)

    return {
        "model": model,
        "datamodule": datamodule,
        "dataloader": dataloader,
        "mse": mse,
        "pearson": pearson,
        "spearman": spearman,
        "preds": preds,
        "targets": targets,
    }


def plot_corr(
    x,
    y,
    title: str,
    ax=None,
    min_: Optional[float] = None,
    max_: Optional[float] = None,
):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    put.density_scatter(x, y, ax=ax)
    ax.set_aspect("equal")
    mse, pearson, spearman = ut.model_stats(x, y)
    ax.set_title(f"{title}\n{mse=:.3f}, {pearson=:.3f}, {spearman=:.3f}")
    if min_ is not None and max_ is not None:
        diff = max_ - min_
        min_ -= 0.1 * diff
        max_ += 0.1 * diff
        ax.set_xlim(min_, max_)
        ax.set_ylim(min_, max_)
    return ax.get_figure(), ax
