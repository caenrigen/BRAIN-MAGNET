from collections import defaultdict
from random import shuffle
import numpy as np


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
