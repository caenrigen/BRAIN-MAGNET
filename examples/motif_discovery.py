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

# Original repo:
# https://github.com/kundajelab/deeplift/commit/0201a218965a263b9dd353099feacbb6f6db0051
# Mirror over here:
# https://github.com/caenrigen/deeplift/commit/0201a218965a263b9dd353099feacbb6f6db0051
import deeplift.dinuc_shuffle as ds

import utils as ut
import cnn_starr as cnn

# Reload in reverse order to be sure it works as intended
reload(shap.explainers.deep.deep_pytorch)
reload(shap.explainers.deep)
reload(shap.explainers)
reload(shap)

reload(ds)

reload(ut)
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
        return torch.tensor(np.zeros((1, 4, num_bp), dtype=np.float32), device=device)

    # dinuc_shuffle expects (length x 4) for a one-hot encoded sequence
    seq_1hot = tensor_to_onehot(inp[0])
    # With the implementation of dinuc_shuffle used here, if not unpadded,
    # the shuffled sequences will "move" into the padded region,
    # since we are not experts on the SHAP DeepExplainer code, we stay safe and unpad
    # before shuffling. Note that the shuffled sequences are not really passed through
    # the model. These are used only for hypothetical contributions estimate.
    shufs = ds.dinuc_shuffle(ut.unpad_one_hot(seq_1hot), num_shufs=num_shufs, rng=rng)
    # Pad back to the original length
    shufs = map(partial(ut.pad_one_hot, to=seq_1hot.shape[0]), shufs)
    shufs = map(onehot_to_tensor_shape, shufs)
    # Putting this data to a device != "cpu" is futile and likely add overhead,
    # but the deep_pytorch code is spaghetti and won't work otherwise
    to_return = torch.tensor(np.array(list(shufs), dtype=np.float32), device=device)
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

    # Perform some reshaping/transposing because the code was designed
    # for inputs that are in the format (length x 4)
    # List[(num_shufs, 4, N)] -> List[(num_shufs, N, 4)]
    mult_ = mult_.transpose(0, 2, 1)
    # List[(num_shufs, 4, N)] -> List[(num_shufs, N, 4)]
    bg_data_ = bg_data_.transpose(0, 2, 1)
    # List[(4, N)] -> List[(N, 4)]
    orig_inp_ = orig_inp_.transpose(1, 0)

    len_one_hot, num_bp = 4, orig_inp_.shape[0]

    assert len(orig_inp_.shape) == 2, orig_inp_.shape
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
    p_h_cbs_mean = onehot_to_tensor_shape(projected_hyp_contribs.mean(axis=0))
    return [torch.tensor(p_h_cbs_mean)]


# Silence an warning that is not relevant for this code
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="torch.nn.modules.module",
    message=".*register_full_backward_hook.*",
)


def calc_contrib_scores(
    dataloader,
    model_trained: cnn.CNNSTARR,
    device: torch.device,
    random_state: int = 913,
    num_shufs: int = 30,
):
    inputs_all = []
    shap_vals_all = []

    for batch, data in enumerate(dataloader):
        inputs, _targets = data
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
                rng=np.random.RandomState(random_state),
            ),
            combine_mult_and_diffref=combine_multipliers_and_diff_from_ref,
        )

        # These will be consumed by TF-MoDISco
        shap_vals = e.shap_values(inputs)
        inputs = inputs.detach()

        inputs_all.append(inputs)
        shap_vals_all.append(shap_vals)

    inputs_all = torch.cat(inputs_all, dim=0).cpu().numpy()

    shap_vals_all = np.concatenate(shap_vals_all, axis=0)

    return inputs_all, shap_vals_all
