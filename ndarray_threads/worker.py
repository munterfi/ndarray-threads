import logging
import time
from typing import Tuple

import numpy as np

LOG = logging.getLogger(__name__)


def worker_thread(arr_ref: np.ndarray, chunk: Tuple[int, int], value: int) -> float:
    """Example worker thread function.

    Args:
        arr_ref (np.ndarray): Reference to array.
        chunk (Tuple[int, int]): Bounds where to manipulate the array.
        value (int): Value to set.

    Raises:
        exc: Any Exceptions are logged and raised.

    Returns:
        float: Processing time where the thread has the CPU assigned.
    """
    t0 = time.thread_time()
    try:
        arr_ref[chunk[0] : chunk[1], :] = value
    except Exception as exc:
        LOG.error(exc)
        raise exc
    return time.thread_time() - t0
