import re
from typing import List, Tuple
import os
import tqdm

import torch



def exact_match(pred, label):
    tot=0
    for i in range(len(pred)):
        if pred[i].lower() == label[i].lower():
            tot+=1
    return tot


def _is_num(t: str) -> bool:
    try:
        float(t)
        return True
    except ValueError:
        return False
    
def _to_float(text: str):
    try:
        return float(text.rstrip('%')) / 100.0 if text.endswith('%') else float(text)
    except ValueError:
        return None

def relaxed_accuracy(
    targets: List[str],
    predictions: List[str],
    tol: float = 0.05,
    unanswerable = False,
) -> Tuple[float, List[int]]:
    """
    Return (accuracy, per-item correctness flags).
    """
    flags = []
    for target, prediction in zip(targets, predictions):
        if target.lower() == "unanswerable" and unanswerable:
            flags.append(int(1))
            continue
        def _to_float(text: str):
            try:
                return float(text.rstrip('%')) / 100.0 if text.endswith('%') else float(text)
            except ValueError:
                return None
        prediction = prediction[:-1] if prediction.endswith(".") else prediction
        prediction, target = str(prediction).strip(), str(target).strip()
        p_float, t_float = _to_float(prediction), _to_float(target)

        # NB: the "and t_float" check is what prevents ZeroDivisionError
        if p_float is not None and t_float:
            rel_change = abs(p_float - t_float) / abs(t_float)
            ok = rel_change <= tol
        else:
            ok = prediction.lower() == target.lower()

        flags.append(int(ok))

    return sum(flags), flags 
