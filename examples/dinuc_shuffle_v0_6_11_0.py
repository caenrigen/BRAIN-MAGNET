from collections import defaultdict
from random import shuffle
import numpy as np
import torch


# compile the dinucleotide edges
def prepare_edges(s):
    edges = defaultdict(list)
    for i in range(len(s) - 1):
        edges[tuple(s[i])].append(s[i + 1])
    return edges


def shuffle_edges(edges, rng=None):
    # for each character, remove the last edge, shuffle, add edge back
    for char in edges:
        last_edge = edges[char][-1]
        edges[char] = edges[char][:-1]
        the_list = edges[char]
        if rng is None:
            shuffle(the_list)
        else:
            rng.shuffle(the_list)
        edges[char].append(last_edge)
    return edges


def traverse_edges(s, edges):
    generated = [s[0]]
    edges_queue_pointers = defaultdict(int)
    for _ in range(len(s) - 1):
        last_char = generated[-1]
        generated.append(
            edges[tuple(last_char)][edges_queue_pointers[tuple(last_char)]]
        )
        edges_queue_pointers[tuple(last_char)] += 1
    if isinstance(generated[0], str):
        return "".join(generated)
    else:
        return np.asarray(generated)


def dinuc_shuffle(s, rng=None):
    if isinstance(s, str):
        s = s.upper()
    return traverse_edges(s, shuffle_edges(prepare_edges(s), rng))


def onehot_dinuc_shuffle(s):
    s = np.squeeze(s)
    assert len(s.shape) == 2
    assert s.shape[1] == 4
    argmax_vals = "".join([str(x) for x in np.argmax(s, axis=-1)])
    shuffled_argmax_vals = [
        int(x)
        for x in traverse_edges(argmax_vals, shuffle_edges(prepare_edges(argmax_vals)))
    ]
    to_return = np.zeros_like(s)
    to_return[list(range(len(s))), shuffled_argmax_vals] = 1
    return to_return


device = "cpu"


def shuffle_several_times(inp):
    # I am assuming len(inp) == 1 because this function is designed for models with one
    # input mode (i.e. just sequence as the input mode)
    assert (inp is None) or len(inp) == 1
    if inp is None:
        return torch.tensor(
            np.zeros((1, 4, 1, 1000), dtype=np.float32), dtype=torch.float32
        ).to(device)
    else:
        # Some reshaping/transposing needs to be performed before calling
        # onehot_dinuc_shuffle becuase the input to the DeepSEA model
        # is in the format (4 x 1 x length) for each sequence, whereas
        # onehot_dinuc_shuffle expects (length x 4)
        to_return = torch.tensor(
            np.array(
                [
                    onehot_dinuc_shuffle(
                        inp[0].detach().cpu().numpy().squeeze().transpose(1, 0)
                    ).transpose((1, 0))[:, None, :]
                    for i in range(100)
                ]
            ).astype("float32")
        ).to(device)
        return to_return


def combine_mult_and_diffref(mult, orig_inp, bg_data):
    to_return = []
    # Perform some reshaping/transposing because the code was designed
    # for inputs that are in the format (length x 4), whereas the DeepSEA
    # model has inputs in the format (4 x 1 x length)
    mult = [x.squeeze().transpose((0, 2, 1)) for x in mult]
    orig_inp = [x.squeeze().transpose((1, 0)) for x in orig_inp]
    bg_data = [x.squeeze().transpose((0, 2, 1)) for x in bg_data]
    for l in range(len(mult)):
        # At each position in the input sequence, we iterate over the one-hot encoding
        # possibilities (eg: for genomic sequence, this is ACGT i.e.
        # 1000, 0100, 0010 and 0001) and compute the hypothetical
        # difference-from-reference in each case. We then multiply the hypothetical
        # differences-from-reference with the multipliers to get the hypothetical contributions.
        # For each of the one-hot encoding possibilities,
        # the hypothetical contributions are then summed across the ACGT axis to estimate
        # the total hypothetical contribution of each position. This per-position hypothetical
        # contribution is then assigned ("projected") onto whichever base was present in the
        # hypothetical sequence.
        # The reason this is a fast estimate of what the importance scores *would* look
        # like if different bases were present in the underlying sequence is that
        # the multipliers are computed once using the original sequence, and are not
        # computed again for each hypothetical sequence.
        projected_hypothetical_contribs = np.zeros_like(bg_data[l]).astype("float")
        assert len(orig_inp[l].shape) == 2, orig_inp[l].shape
        for i in range(orig_inp[l].shape[-1]):
            hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
            hypothetical_input[:, i] = 1.0
            hypothetical_difference_from_reference = (
                hypothetical_input[None, :, :] - bg_data[l]
            )
            hypothetical_contribs = hypothetical_difference_from_reference * mult[l]
            projected_hypothetical_contribs[:, :, i] = np.sum(
                hypothetical_contribs, axis=-1
            )
        to_return.append(
            torch.tensor(
                np.mean(projected_hypothetical_contribs, axis=0).transpose((1, 0))[
                    :, None, :
                ]
            )
        )
    return to_return
