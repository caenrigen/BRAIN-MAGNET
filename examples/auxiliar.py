def to_uint8(string):
    return np.frombuffer(string.encode("ascii"), dtype=np.uint8)


# function to convert only one piece of sequence to np.array
def make_one_hot_encode(alphabet: str = "ACGT", dtype=np.float32) -> np.ndarray:
    """
    One-hot encode for a sequence.
    A -> [1,0,0,0]
    C -> [0,1,0,0]
    G -> [0,0,1,0]
    T -> [0,0,0,1]
    N -> [0,0,0,0]
    """

    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    return hash_table


HASH_TABLE = make_one_hot_encode()


def one_hot_encode(sequence: str):
    return HASH_TABLE[to_uint8(sequence)]


def pad_arr(snippet: np.ndarray, arr_pre: int = 1024):
    assert len(snippet) <= arr_pre, len(snippet)
    arr_len = len(snippet)
    pad = (arr_pre - arr_len) / 2
    return np.pad(snippet, [(int(pad), math.ceil(pad)), (0, 0)], mode="constant")


def tensor_reverse_complement(x):
    # Flip along the length dimension (dim=3).
    x_rev = torch.flip(x, dims=[3])

    # Swap A <-> T and C <-> G channels.
    # Channel order is (A=0, C=1, G=2, T=3)
    channel_map = [3, 2, 1, 0]  # T, G, C, A
    x_revcomp = x_rev[:, channel_map, :, :]
    return x_revcomp
