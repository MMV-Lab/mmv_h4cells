import numpy as np
from typing import Tuple
from napari.qt.threading import thread_worker


@thread_worker
def analyse_roi(
    data: np.ndarray,
    y: Tuple[int, int],
    x: Tuple[int, int],
    threshold: int,
    paths: Tuple[str, str],
):
    pass
    # Please make sure to return the paths as well
