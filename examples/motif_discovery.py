from functools import partial
from itertools import tee
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
from numpy.lib.format import open_memmap
import warnings
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from dinuc_shuf import shuffle  # a fast implementation of dinucleotide shuffling

# Original repo:
# https://github.com/kundajelab/shap/commit/29d2ffab405619340419fc848de6b53e2ef0f00c
# Mirror over here with a fix to support latest ipython>=9 without import errors
# https://github.com/caenrigen/shap/commit/53bf2b05a04e1a49cb398bc92b4e541860af0276
import shap

import utils as ut
import cnn_starr as cnn

# Specify handler for Flatten used in our model, otherwise `shap_values` will emit
# warnings and (potentially) use an incorrect handler.
dpyt = shap.explainers.deep.deep_pytorch
dpyt.op_handler["Flatten"] = dpyt.passthrough


def tensor_to_onehot(t):
    # Detach is because we won't need the gradients
    return t.detach().transpose(1, 0).cpu().numpy()


def one_hot_to_tensor_shape(one_hot: np.ndarray):
    if one_hot.ndim == 2:
        return one_hot.transpose(1, 0)
    elif one_hot.ndim == 3:
        return one_hot.transpose(0, 2, 1)
    else:
        raise NotImplementedError(f"{one_hot.ndim = }")


def make_shuffled_1hot_seqs(
    inp: List[torch.Tensor],
    device: torch.device,
    num_shufs: int = 10,
    rng_seed: int = 20240413,
):
    # Assuming len(inp) == 1 because this function is designed for models with one
    # input mode (i.e. just sequence as the input mode)
    assert inp is None or len(inp) == 1, inp

    # Internally the `DeepExplainer` performs some checks by using a quick sample and
    # requires this function to accept `None` and return some sample ref data (zeros).
    if inp is None:
        num_bp = 10
        return torch.tensor(np.zeros((1, 4, num_bp), dtype=np.float32), device=device)

    # dinuc_shuffle expects (length x 4) for a one-hot encoded sequence
    seq_1hot = tensor_to_onehot(inp[0])

    # With the implementation of dinuc_shuffle used here, if not unpadded,
    # the shuffled sequences will "move" into the padded region,
    # since we are not experts on the SHAP DeepExplainer code, we stay safe and unpad
    # before shuffling. The shuffled sequences are not passed through the model.
    # These are used only for estimating hypothetical contributions.
    # `np.tile` is used to repeat the sequence `num_shufs` times, to obey the shape
    # expected by `dinuc_shuffle.shuffle`.
    unpadded_tiled = np.tile(ut.unpad_one_hot(seq_1hot), (num_shufs, 1, 1))
    shufs = shuffle(unpadded_tiled, rng=np.random.default_rng(rng_seed), verify=True)
    seqs_shuffled_padded = ut.pad_one_hot(shufs, to=seq_1hot.shape[-2])
    # Moving this data to a device != "cpu" is futile and likely add overhead,
    # but the shap's deep_pytorch module is spaghetti and won't work otherwise
    to_return = torch.tensor(
        one_hot_to_tensor_shape(seqs_shuffled_padded),
        device=device,
        dtype=torch.float32,
    )
    return to_return


def combine_multipliers_and_diff_from_ref(
    mult: List[np.ndarray],  # shape of the (only) element: (num_shufs, 4, N)
    orig_inp: List[np.ndarray],  # shape of the (only) element: (4, N)
    bg_data: List[np.ndarray],  # shape of the (only) element: (num_shufs, 4, N)
    # ! The data output by this function is used only on "cpu" device inside
    # ! the shap module's. There is no need to move it to GPU/MPS, that will create
    # ! errors device-related errors.
    # device: torch.device,
):
    # DeepExaplainer passes in a list of length 1 for our models that have only one
    # input mode (i.e. just sequence as the input mode)
    for arg in mult, orig_inp, bg_data:
        assert len(arg) == 1, arg
        assert isinstance(arg, list), type(arg)
    mult_ = mult[0]
    orig_inp_ = orig_inp[0]
    bg_data_ = bg_data[0]

    # Perform transposing because the code was designed for inputs that are in the
    # format (seq_len x 4)
    # List[(num_shufs, 4, seq_len)] -> List[(num_shufs, seq_len, 4)]
    mult_ = mult_.transpose(0, 2, 1)
    # List[(num_shufs, 4, seq_len)] -> List[(num_shufs, seq_len, 4)]
    bg_data_ = bg_data_.transpose(0, 2, 1)
    # List[(4, seq_len)] -> List[(seq_len, 4)]
    orig_inp_ = orig_inp_.transpose(1, 0)

    len_one_hot, num_bp = 4, orig_inp_.shape[0]

    assert orig_inp_.ndim == 2, orig_inp_.shape
    assert orig_inp_.shape[-1] == len_one_hot, orig_inp_.shape

    # We don't need zeros, these will be overwritten
    projected_hyp_contribs = np.empty_like(bg_data_, dtype=np.float32)
    hyp_contribs = np.empty_like(bg_data_, dtype=np.float32)

    ident = np.eye(len_one_hot, dtype=np.float32)
    # Iterate over 4 hypothetical sequences, each made of the same base,
    # e.g. for idx_col_1hot == 0: "AAAA....AAAA" (but one hot encoded of course)
    for idx_col_1hot in range(len_one_hot):
        # ##########################################################################
        # These two lines allocate extra memory
        # // hyp_seq_1hot = np.zeros_like(orig_inp0, dtype=np.float32)
        # // hyp_seq_1hot[:, idx_col_1hot] = 1.0
        # This trick avoids memory allocation
        hyp_seq_1hot = np.broadcast_to(ident[idx_col_1hot], (num_bp, 4))
        # ##########################################################################

        # `hyp_seq_1hot[None, :, :]` shapes it such that it can match the
        # shape of `bg_data` that has the extra dimension of num_shufs.
        # It is only a view of the underlying memory, so it is efficient.
        np.subtract(hyp_seq_1hot[None, :, :], bg_data_, out=hyp_contribs)
        np.multiply(hyp_contribs, mult_, out=hyp_contribs)

        # Sum on the one-hot axis, save directly to `projected_hyp_contribs`.
        # The sum is to get the total hypothetical contribution (at that bp)
        hyp_contribs.sum(axis=-1, out=projected_hyp_contribs[:, :, idx_col_1hot])

    # Average on the num_shufs axis to arrive to the final hypothetical
    # contribution scores (at each bp).
    p_h_cbs_mean = one_hot_to_tensor_shape(projected_hyp_contribs.mean(axis=0))
    return [torch.tensor(p_h_cbs_mean)]


# Silence an warning that is not relevant for this code
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="torch.nn.modules.module",
    message=".*register_full_backward_hook.*",
)


def calc_contrib_scores_step(
    data_batch: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
    model_trained: cnn.BrainMagnetCNN,
    device: torch.device,
    random_state: int = 20240413,
    num_shufs: int = 10,
    avg_w_revcomp: bool = True,
):
    """
    Calculate the contribution scores for a batch of data.

    num_shufs: number of shuffled sequences to use as reference for the hypothetical
    contributions scores. Suggested values: ~10 if `avg_w_revcomp=True`, else ~30.

    avg_w_revcomp: whether to average the contribution scores obtained from inputting
    both the "forward" and reverse complement strands.
    """
    if isinstance(data_batch, tuple):
        inputs, _targets = data_batch
    elif isinstance(data_batch, torch.Tensor):
        inputs = data_batch
    else:
        raise ValueError(f"Unknown {type(data_batch) = }")
    # the DeepExplainer will evaluate the inputs to get the pytorch gradients,
    # put the inputs on the same device for faster inference when using GPU/MPS
    inputs = inputs.to(device)
    # targets = targets.to(device) # not needed for shap

    # calculate shap
    e = shap.DeepExplainer(
        model=model_trained,
        data=partial(
            make_shuffled_1hot_seqs,
            device=device,
            num_shufs=num_shufs,
            rng_seed=random_state,
        ),
        combine_mult_and_diffref=combine_multipliers_and_diff_from_ref,
    )
    # `inputs` is a tensor of shape (batch_size, 4, num_bp)
    # These will be consumed by TF-MoDISco
    shap_vals = np.array(e.shap_values(inputs))
    if avg_w_revcomp:
        # Padding of odd-length sequences will be flipped, but it is not a problem
        # because we reverse-complement the shap values too before averaging with
        # the forward shap values.
        # np.array() is necessary because `e.shap_values` returns a list.
        shap_vals_revcomp = np.array(
            e.shap_values(ut.one_hot_reverse_complement(inputs))
        )
        # one_hot_reverse_complement handles transposed batches
        shap_vals += ut.one_hot_reverse_complement(shap_vals_revcomp)
        shap_vals /= 2  # average

    return inputs.detach(), shap_vals


def calc_contrib_scores(
    dataloader: DataLoader,
    model_trained: cnn.BrainMagnetCNN,
    device: torch.device,
    fp_out_shap: Optional[Path] = None,
    fp_out_inputs: Optional[Path] = None,
    overwrite: bool = False,
    # sum_inplace and div_after_sum are intended for averaging inplace when evaluating
    # several models on the same dataloader.
    sum_inplace: bool = False,
    div_after_sum: Optional[int] = None,
    seq_len: Optional[int] = None,
    num_seqs: Optional[int] = None,
    random_state: int = 20240413,
    num_shufs: int = 10,
    avg_w_revcomp: bool = True,
    tqdm_bar: bool = True,
):
    """
    A generator that yields the inputs and shap values for each batch in the dataloader.
    Optionally, the inputs and shap values are saved efficiently to disk into numpy
    memory-mapped files. For larger datasets, writing to disk is recommended.
    """
    it = iter(dataloader)
    if fp_out_shap is not None:
        if not overwrite and fp_out_shap.exists():
            raise FileExistsError(f"{fp_out_shap}")
        # `tee` is to efficiently "fork" the iterator into a secondary one, inspect
        # the first element and get its shape, without performance overhead
        it, it_tmp = tee(it)

        if num_seqs is None:
            num_seqs = len(dataloader.sampler)  # type: ignore
        if seq_len is None:
            inputs_tmp, _ = next(it_tmp)
            channels, seq_len = inputs_tmp.shape[-2:]
            assert channels == 4, channels
        else:
            channels = 4

        assert isinstance(seq_len, int), seq_len  # for type checking
        shape_out = (num_seqs, channels, seq_len)

        # Behaves like an array, but writes to disk, saves RAM.
        # Files crated via `open_memmap` are compatible with `np.load(...)`.
        npy_shap = open_memmap(
            fp_out_shap,
            mode="w+",  # create the file if it doesn't exist, otherwise overrides
            shape=shape_out,
            dtype=np.float32,
        )
    if fp_out_inputs is not None:
        if not overwrite and fp_out_inputs.exists():
            raise FileExistsError(f"{fp_out_inputs}")
        npy_inputs = open_memmap(
            fp_out_inputs,
            mode="w+",
            shape=shape_out,
            dtype=np.int8,
        )

    offset = 0
    items = tqdm(it, total=len(dataloader)) if tqdm_bar else it
    for data in items:
        inputs, shap_vals = calc_contrib_scores_step(
            data_batch=data,
            model_trained=model_trained,
            device=device,
            random_state=random_state,
            num_shufs=num_shufs,
            avg_w_revcomp=avg_w_revcomp,
        )

        if fp_out_inputs is not None:
            # ! .cpu() is often a costly operation
            # Convert to int8 to save disk space, one-hot values are simply 0s or 1s
            inputs = inputs.cpu().numpy().astype(np.int8)
            npy_inputs[offset : offset + len(inputs)] = inputs

        if fp_out_shap is not None:
            if sum_inplace:
                if div_after_sum is not None:
                    out = shap_vals + npy_shap[offset : offset + len(shap_vals)]
                    out /= div_after_sum
                    # write to the file only once
                    npy_shap[offset : offset + len(shap_vals)] = out
                else:
                    npy_shap[offset : offset + len(shap_vals)] += shap_vals
            else:
                npy_shap[offset : offset + len(shap_vals)] = shap_vals

        offset += len(shap_vals)

        yield inputs, shap_vals


def calc_contrib_scores_concatenated(
    dataloader: DataLoader,
    model_trained: cnn.BrainMagnetCNN,
    device: torch.device,
    random_state: int = 20240413,
    num_shufs: int = 10,
    avg_w_revcomp: bool = True,
    tqdm_bar: bool = True,
):
    """
    Convenience function that concatenates the inputs and shap values from the generator
    returned by `calc_contrib_scores`.
    Intended for quick tests, not recommended for larger datasets, use
    `calc_contrib_scores` instead and write to disk.
    """
    gen = calc_contrib_scores(
        dataloader=dataloader,
        model_trained=model_trained,
        device=device,
        random_state=random_state,
        num_shufs=num_shufs,
        avg_w_revcomp=avg_w_revcomp,
        tqdm_bar=tqdm_bar,
        fp_out_shap=None,
        fp_out_inputs=None,
    )
    inputs, shap_vals = zip(*gen)
    inputs = torch.cat(inputs)
    shap_vals = np.concatenate(shap_vals)
    return inputs, shap_vals
