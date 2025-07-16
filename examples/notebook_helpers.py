import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Literal, Optional, Callable
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import logging
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
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
    frac_train_sample: float,
    seed_train: int,
    seed_split: int,
    device: torch.device,
    version: Optional[str] = None,
    empty_cache: bool = True,
    weight_decay: float = 0.0,
    augment_w_rev_comp: bool = True,
    groups_func: Optional[Callable] = partial(dm.bp_dist_groups, threshold=10_000),
    # Pass a model-making function (instead of a model itself) to ensure
    # reproducibility by instantiating the model after seeding everything.
    make_model: Callable[[], nn.Module] = cnn.make_model_starr,
    **hyper_params,
):
    if empty_cache:
        ut.empty_cache(device)

    # NB if you ever use multiple multiprocessing workers, careful with the seeding.
    L.seed_everything(seed_train, workers=True)  # for reproducibility

    model_module = cnn.ModelModule(
        learning_rate=learning_rate,
        # Don't change this for training, reverse complement is handled by the data
        # module as augmentation data.
        forward_mode="forward",
        make_model=make_model,
        # The rest are "hyperparameters" for logging purposes.
        task=task,
        batch_size=batch_size,
        frac_test=frac_test,
        frac_val=frac_val,
        frac_train_sample=frac_train_sample,
        folds=folds,
        fold=fold,
        max_ep=max_epochs,
        weight_decay=weight_decay,
        seed_train=seed_train,
        seed_split=seed_split,
        augment_w_rev_comp=augment_w_rev_comp,
        **hyper_params,
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
        random_state=seed_split,
        augment_w_rev_comp=augment_w_rev_comp,
        targets_col=f"{task}_log2_enrichment",
        x_col="Seq",
        folds=folds or None,
        fold=fold,
        frac_test=frac_test,
        frac_val=frac_val,
        frac_train_sample=frac_train_sample,
        groups_func=groups_func,
        # DataLoader kwargs:
        batch_size=batch_size,
        # These might give some speed up if cuda is available
        # pin_memory=True,
        # pin_memory_device="cuda",
    )
    try:
        trainer.fit(model_module, datamodule=datamodule)
    except (KeyboardInterrupt, NameError):
        logger.info("Training interrupted by user")
        return False

    # Free up memory
    model_module.cpu()
    if empty_cache:
        ut.empty_cache(device)

    return True


def pick_best_checkpoints(
    df_ckpts: pd.DataFrame,
    plot: bool = False,
    col_train: str = "loss_train",
    col_val: str = "loss_val",
):
    best_checkpoints = {}
    folds = len(df_ckpts.fold.unique())
    if plot:
        fig, axs = plt.subplots(
            1, folds, figsize=(folds * 3, 3), sharex=True, sharey=True
        )
        if folds == 1:
            axs = [axs]
    else:
        fig = axs = None
    for fold in range(folds):
        df = df_ckpts[df_ckpts.fold == fold].copy()
        best_epoch = dm.pick_checkpoint(
            df,
            ax=axs[fold] if plot else None,
            col_train=col_train,
            col_val=col_val,
        )
        df_best = df[df.epoch == best_epoch]
        assert len(df_best) == 1
        best_checkpoints[fold] = df_best.fp.iloc[0]
        if axs is not None:
            max_ = df_ckpts[col_val].quantile(0.97)  # don't plot outliers
            min_ = df_ckpts[col_train].min()
            axs[fold].set_ylim(min_ - (max_ - min_) * 0.03, max_)

    if fig is not None:
        fig.tight_layout()

    return best_checkpoints, fig, axs


def evaluate_model(
    fp_checkpoint: Path,
    fp_dataset: Path,
    device: torch.device,
    groups_func: Optional[Callable],
    make_model: Callable[[], nn.Module],
    augment_w_rev_comp: Optional[bool] = None,
    dataloader: str = "test_dataloader",
):
    model = cnn.ModelModule.load_from_checkpoint(fp_checkpoint, make_model=make_model)

    # Allow to override or use the value from training
    if augment_w_rev_comp is None:
        augment_w_rev_comp = model.hparams_initial.augment_w_rev_comp
    assert isinstance(augment_w_rev_comp, bool)

    datamodule = dm.DataModule(
        fp_dataset=fp_dataset,
        targets_col=f"{model.hparams_initial.task}_log2_enrichment",
        random_state=model.hparams_initial.seed_split,
        folds=model.hparams_initial.folds,
        fold=model.hparams_initial.fold,
        frac_test=model.hparams_initial.frac_test,
        frac_val=model.hparams_initial.frac_val,
        frac_train_sample=model.hparams_initial.frac_train_sample,
        batch_size=model.hparams_initial.batch_size,
        augment_w_rev_comp=augment_w_rev_comp,
        groups_func=groups_func,
    )
    datamodule.setup()
    dataloader = getattr(datamodule, dataloader)()

    # Reuse our code to evaluate the model on the test set

    # logger=False to avoid creating a log directory for this run
    trainer = L.Trainer(accelerator=device.type, logger=False)
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
