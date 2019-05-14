from enum import Enum
import numpy as np


class KEY(Enum):
    UP = 0,
    DOWN = 1,
    LEFT = 2,
    RIGHT = 3,
    PICKUP = 4,
    TRANSFORM = 5,
    USE_1 = 5,
    USE_2 = 6,
    USE_3 = 7,
    USE_4 = 8,
    USE_5 = 9,
    QUIT = 'q'


WHITE = (255, 255, 255)
LIGHT = (196, 196, 196)
GREEN = (80, 160, 0)
DARK = (128, 128, 128)
DARK_RED = (139, 0, 0)
BLACK = (0, 0, 0)

MOVE_ACTS = {KEY.UP, KEY.DOWN, KEY.LEFT, KEY.RIGHT}

AGENT = 0
BLOCK = 1
WATER = 2
OBJ_BIAS = 3

TYPE_PICKUP = 0
TYPE_TRANSFORM = 1


def get_id_from_ind_multihot(indexed_tensor, mapping, max_dim):
    if type(mapping) == dict:
        mapping_ = np.zeros(max(mapping.keys())+1, dtype=np.long)
        for k, v in mapping.items():
            mapping_[k] = v
        mapping = mapping_
    if indexed_tensor.ndim == 2:
        nbatch = indexed_tensor.shape[0]
        out = np.zeros(nbatch, max_dim).astype(np.byte)
        np.add.at(out, mapping.ravel(), indexed_tensor.ravel())
    else:
        out = np.zeros(max_dim).astype(np.byte)
        np.add.at(out, mapping.ravel(), indexed_tensor.ravel())

    return out
