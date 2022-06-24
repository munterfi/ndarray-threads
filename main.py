import logging

import matplotlib.pyplot as plt
import numpy as np

from ndarray_threads.manager import array_apply
from ndarray_threads.worker import worker_thread

if __name__ == "__main__":

    # Set log level
    logging.basicConfig(
        format="[%(asctime)s %(threadName)s %(levelname)s] %(message)s (%(name)s)",
        level=logging.INFO,
    )

    # Create array
    arr = np.ndarray((167, 200))
    arr[:] = 0

    # Process
    _ = array_apply(arr, worker_thread, n_worker=8)

    # Show result
    plt.imshow(arr)
    plt.show(block=False)
    plt.pause(3)
    plt.close()
