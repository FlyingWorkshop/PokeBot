import re
from .catalogs import MoveEnum

import torch

def move_to_pred_vec_index(m):
    return MoveEnum[re.sub(r"\s|-|'", "", m.lower())].value - 1


def pred_vec_to_string(pred: torch.Tensor):
    i = pred.argmax().item()
    if i == len(MoveEnum):
        return 'switch'
    else:
        return MoveEnum(i + 1).name

