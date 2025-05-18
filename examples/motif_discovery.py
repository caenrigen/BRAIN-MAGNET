from functools import partial
from typing import List, Optional
import numpy as np
import warnings
from importlib import reload
import torch

# Original repo:
# https://github.com/kundajelab/shap/commit/29d2ffab405619340419fc848de6b53e2ef0f00c
# Mirror over here:
# https://github.com/caenrigen/shap/commit/29d2ffab405619340419fc848de6b53e2ef0f00c
import shap

# https://github.com/kundajelab/deeplift/commit/0201a218965a263b9dd353099feacbb6f6db0051
import deeplift.dinuc_shuffle as ds

import cnn_starr as cnn

# Reload in reverse order to be sure it works as intended
reload(shap.explainers.deep.deep_pytorch)
reload(shap.explainers.deep)
reload(shap.explainers)
reload(shap)

reload(ds)

reload(cnn)

# Specify handler for Flatten used in our model, otherwise `shap_values` will emit
# warnings and (potentially) use an incorrect handler.
dpyt = shap.explainers.deep.deep_pytorch
dpyt.op_handler["Flatten"] = dpyt.passthrough


def tensor_to_onehot(t):
    # Detach is because we won't need the gradients
    return t.detach().transpose(1, 0).cpu().numpy()


def onehot_to_tensor_shape(one_hot: np.ndarray):
    return one_hot.transpose(1, 0)


def make_shuffled_1hot_seqs(
    inp: List[torch.Tensor],
    # Accoriding to Avanti Shrikumar:
    # 10 should already work well, 100 is on the high side
    device: torch.device,
    num_shufs: int = 30,
    rng: Optional[np.random.RandomState] = np.random.RandomState(913),
):
    # Assuming len(inp) == 1 because this function is designed for models with one
    # input mode (i.e. just sequence as the input mode)
    assert inp is None or len(inp) == 1, inp

    # Internally the `DeepExplainer` performs some checks by using a quick sample and
    # requires this function to accept `None` and return some sample ref data (zeros).
    if inp is None:
        num_bp = 10
        return torch.tensor(np.zeros((1, 4, num_bp), dtype=np.float32)).to(device)

    # Some reshaping/transposing, onehot_dinuc_shuffle expects (length x 4)
    seq_1hot = tensor_to_onehot(inp[0])
    # Expectes (length x 4) for a one-hot encoded sequence
    shufs = ds.dinuc_shuffle(seq_1hot, num_shufs=num_shufs, rng=rng)
    shufs = map(onehot_to_tensor_shape, shufs)
    to_return = torch.tensor(np.array(list(shufs), dtype=np.float32)).to(device)
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
    assert len(mult) == len(orig_inp) == len(bg_data) == 1
    to_return = []

    # Perform some reshaping/transposing because the code was designed
    # for inputs that are in the format (length x 4)
    # List[(num_shufs, 4, N)] -> List[(num_shufs, N, 4)]
    mult = [x.transpose(0, 2, 1) for x in mult]
    # List[(num_shufs, 4, N)] -> List[(num_shufs, N, 4)]
    bg_data = [x.transpose(0, 2, 1) for x in bg_data]
    # List[(4, N)] -> List[(N, 4)]
    orig_inp = [x.transpose(1, 0) for x in orig_inp]

    for l_idx in range(len(mult)):
        len_one_hot, num_bp = 4, orig_inp[l_idx].shape[0]

        assert len(orig_inp[l_idx].shape) == 2, orig_inp[l_idx].shape
        assert orig_inp[l_idx].shape[-1] == len_one_hot, orig_inp[l_idx].shape

        # We don't need zeros, these will be overwritten
        projected_hyp_contribs = np.empty_like(bg_data[l_idx], dtype=np.float32)
        hyp_contribs = np.empty_like(bg_data[l_idx], dtype=np.float32)

        ident = np.eye(len_one_hot, dtype=np.float32)
        # Iterate over 4 hypothetical sequences, each made of the same base,
        # e.g. for idx_col_1hot == 0: "AAAA....AAAA" (but one hot encoded of course)
        for idx_col_1hot in range(len_one_hot):
            # ##########################################################################
            # These two lines allocate extra memory
            # // hyp_seq_1hot = np.zeros_like(orig_inp[l_idx], dtype=np.float32)
            # // hyp_seq_1hot[:, idx_col_1hot] = 1.0
            # This trick avoids memory allocation
            hyp_seq_1hot = np.broadcast_to(ident[idx_col_1hot], (num_bp, 4))
            # ##########################################################################

            # `hyp_seq_1hot[None, :, :]` shapes it such that it can match the
            # shape of `bg_data[l_idx]` that has the extra dimension of num_shufs.
            # It is only a view of the underlying memory, so it is efficient.
            np.subtract(hyp_seq_1hot[None, :, :], bg_data[l_idx], out=hyp_contribs)
            np.multiply(hyp_contribs, mult[l_idx], out=hyp_contribs)

            # Sum on the one-hot axis, save directly to `projected_hyp_contribs`.
            # The sum is to get the total hypothetical contribution (at that bp)
            hyp_contribs.sum(axis=-1, out=projected_hyp_contribs[:, :, idx_col_1hot])

        # Average on the num_shufs axis to arrive to the final hypothetical
        # contribution scores (at each bp).
        p_h_cbs_mean = onehot_to_tensor_shape(projected_hyp_contribs.mean(axis=0))
        to_return.append(torch.tensor(p_h_cbs_mean))
    return to_return


# Silence an warning that is not relevant for this code
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="torch.nn.modules.module",
    message=".*register_full_backward_hook.*",
)


def calc_contrib_scores(dataloader, model_trained: cnn.CNNSTARR, device: torch.device):
    inputs_all = []
    shap_vals_all = []

    for batch, data in enumerate(dataloader):
        inputs, _targets = data
        inputs = inputs.to(device)
        # targets = targets.to(device) # not needed for shap

        # calculate shap
        e = shap.DeepExplainer(
            model=model_trained,
            data=partial(make_shuffled_1hot_seqs, device=device),
            combine_mult_and_diffref=partial(
                combine_multipliers_and_diff_from_ref, device=device
            ),
        )

        # These will be consumed by TF-MoDISco
        shap_vals = e.shap_values(inputs)
        inputs = inputs.detach()

        inputs_all.append(inputs)
        shap_vals_all.append(shap_vals)

    inputs_all = torch.cat(inputs_all, dim=0).cpu().numpy()

    shap_vals_all = np.concatenate(shap_vals_all, axis=0)

    return inputs_all, shap_vals_all
