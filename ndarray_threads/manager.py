import logging
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np


LOG = logging.getLogger(__name__)


def array_apply(arr: np.ndarray, worker_function: Callable, n_worker: int = 8) -> bool:
    """Applies a worker function on a ndarray using a ThreadPoolExecutor.

    Args:
        arr (np.ndarray): Reference to the array.
        worker_function (Callable): Function to run as thread.
        n_worker (int, optional): Number of worker threads. Defaults to 8.

    Returns:
        bool: True if success, else False.
    """
    success = True
    x_size = arr.shape[0]

    chunk_size = int(x_size / n_worker)
    chunks = [
        (chunk, min(chunk + chunk_size, x_size))
        for chunk in range(0, x_size, chunk_size)
    ]  # Uncomment to check error handling: + [("error", "here")]

    with ThreadPoolExecutor(max_workers=n_worker) as executor:
        futures = {
            executor.submit(worker_function, arr, chunk, i): chunk
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                thread_cpu_time = future.result()
            except Exception as exc:
                LOG.error(
                    "Chunk %r generated an exception: %s",
                    chunk,
                    exc,
                )
                success = False
            else:
                LOG.info(
                    "Chunk %r successfully processed in %f seconds.",
                    chunk,
                    thread_cpu_time,
                )
    return success
